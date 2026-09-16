"""
ML Knowledge Base Skill - 检索与编排引擎
版本: ml-skill v1.0.0
架构兼容: >=1.0.0, <2.0.0
注意：所有领域参数从架构定义文件加载，严禁硬编码
"""

import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


class MLKnowledgeBaseSkill:
    """
    机器学习知识库检索与编排引擎
    启动时从架构定义文件加载全部参数，无硬编码领域参数
    """

    def __init__(self, architecture_path: str, knowledge_base_path: str):
        # 从架构定义文件加载参数（唯一真相源）
        with open(architecture_path, 'r', encoding='utf-8') as f:
            arch = json.load(f)

        self.namespace = arch["meta"]["namespace"]
        self.params = arch["params_override"]
        self.entity_types = {e["id"]: e for e in arch["entity_types"]}
        self.relation_types = {r["id"]: r for r in arch["relation_types"]}
        self.task_templates = {t["id"]: t for t in arch["task_templates"]}
        self.verdict_rules = {v["id"]: v for v in arch.get("verdict_rules", [])}

        # 从知识库加载实例
        with open(knowledge_base_path, 'r', encoding='utf-8') as f:
            kb = json.load(f)

        self.instances = kb["instances"]
        self.relations = kb["relations"]
        self.meta = kb["meta"]

        # 加载参数（从架构文件，非硬编码）
        self.theta_use = self.params.get("θ_use", 0.6)
        self.theta_hint = self.params.get("θ_hint", 0.4)
        self.top_k = self.params.get("top_k", 50)
        self.top_m = self.params.get("top_m", 30)
        self.rrf_k = self.params.get("rrf_k", 60)
        self.coverage_threshold = self.params.get("coverage_threshold", 0.85)

        # 策略统计（内存中，持久化另行实现）
        self.strategy_stats: Dict[str, Dict] = {}
        self._init_strategy_stats()

    def _init_strategy_stats(self):
        """从知识库初始化策略统计"""
        for template in self.task_templates.values():
            tid = template["id"]
            # 从知识库加载使用统计
            # 实际实现中从KB的strategy_templates节读取
            self.strategy_stats[tid] = {
                "n": 0, "s": 0, "success_rate": 0.0
            }

    # ========== 任务识别 ==========

    def recognize_task_type(self, user_query: str) -> str:
        """
        识别任务类型（AP-02任务类型体系）
        返回: 答疑 / 评判 / 诊断 / 生成 / 规划 / 审核 / 重组 / 因果调试 / RAG治理 / 风控选型
        """
        query_lower = user_query.lower()

        # 因果调试关键词
        if any(k in query_lower for k in ["错误", "误", "bug", "为什么", "根因", " cascade", "级联"]):
            return "因果调试"
        # RAG治理关键词
        if any(k in query_lower for k in ["向量化", "脱敏", "切片", "rag", "数据治理", "隐私"]):
            return "RAG治理"
        # 风控选型关键词
        if any(k in query_lower for k in ["选型", "推荐", "风控", "反欺诈", "合规", "模型选择"]):
            return "风控选型"
        # 答疑关键词
        if any(k in query_lower for k in ["原理", "是什么", "概念", "解释", "什么是"]):
            return "答疑"
        # 诊断关键词
        if any(k in query_lower for k in ["诊断", "问题", "定位", "原因"]):
            return "诊断"
        # 生成关键词
        if any(k in query_lower for k in ["生成", "代码", "实现", "写"]):
            return "生成"
        # 审核关键词
        if any(k in query_lower for k in ["审核", "检查", "评审"]):
            return "审核"

        return "答疑"  # 默认

    # ========== 四阶段召回 ==========

    def recall(
        self,
        task_type: str,
        query_vector: List[float],
        time_point: str = None
    ) -> Dict[str, List[Dict]]:
        """
        阶段2：双路召回
        返回: {"symbolic": [...], "vector": [...], "graph": [...]}
        """
        time_point = time_point or datetime.now().strftime("%Y-%m-%d")

        # 符号路：按任务类型+知识需求清单精确查询
        c_symbolic = self._symbolic_search(task_type, time_point)

        # 向量路：ANN Top-K
        c_vector = self._vector_search(query_vector, top_k=self.top_k)

        return {"symbolic": c_symbolic, "vector": c_vector}

    def _symbolic_search(self, task_type: str, time_point: str) -> List[Dict]:
        """符号路精确查询"""
        results = []

        # 查找与任务类型匹配的模板，提取所需知识类型
        needed_types = self._get_needed_types_for_task(task_type)

        for category, instances in self.instances.items():
            for inst in instances:
                if inst.get("validity") and len(inst["validity"]) == 2:
                    start, end = inst["validity"]
                    if not (start <= time_point and (end is None or time_point <= end)):
                        continue  # 时间过滤

                # 类型匹配
                inst_type = inst.get("type") or inst.get("model_type") or category
                if any(nt in inst_type for nt in needed_types):
                    results.append({
                        "id": inst["id"],
                        "name": inst.get("name") or inst.get("model_type", ""),
                        "category": category,
                        "confidence": inst.get("confidence", 0.9),
                        "source": inst.get("source", ""),
                        "anchor": inst.get("anchor", {}),
                        "precise_match": True
                    })

        return results

    def _vector_search(self, query_vector: List[float], top_k: int) -> List[Dict]:
        """
        向量路ANN搜索
        实际实现需接入向量数据库（Milvus / Pinecone / Weaviate 等）
        此处为接口示例
        """
        # === 方案A：Milvus ===
        # from pymilvus import MilvusClient
        # client = MilvusClient(uri="./milvus.db")
        # results = client.search(
        #     collection_name="ml_kb",
        #     query_vector=query_vector,
        #     limit=top_k
        # )
        # return [self._milvus_result_to_instance(r) for r in results]

        # === 方案B：Pinecone ===
        # import pinecone
        # index = pinecone.Index("ml-kb")
        # results = index.query(
        #     vector=query_vector,
        #     top_k=top_k,
        #     include_metadata=True
        # )
        # return [self._pinecone_result_to_instance(r) for r in results["matches"]]

        # === 方案C：OpenAI Embedding + FAISS（轻量本地）===
        # from openai import OpenAI
        # client = OpenAI()
        # response = client.embeddings.create(
        #     model="text-embedding-ada-002",
        #     input=query_text
        # )
        # query_vector = response.data[0].embedding
        # # FAISS 检索
        # import faiss
        # index = faiss.read_index("ml_kb.index")
        # D, I = index.search(np.array([query_vector]), top_k)
        # return [self._faiss_result_to_instance(idx) for idx in I[0]]

        # TODO: 根据实际选型取消注释并配置
        raise NotImplementedError(
            "请配置向量数据库：方案A(Milvus) / 方案B(Pinecone) / 方案C(FAISS)"
        )

    def graph_expand(
        self,
        candidates: Dict[str, List[Dict]],
        max_hops: int = 3,
        threshold: float = None
    ) -> List[Dict]:
        """
        阶段3：图扩展
        沿可传递关系受限扩展
        """
        threshold = threshold or self.theta_hint
        all_candidates = candidates.get("symbolic", []) + candidates.get("vector", [])
        seed_ids = [c["id"] for c in all_candidates]

        expanded = list(all_candidates)
        visited = set(seed_ids)

        for hop in range(max_hops):
            new_ids = []
            for rel in self.relations:
                if rel["subject"] in visited and rel["subject"] not in new_ids:
                    confidence = rel.get("confidence", 0.9)
                    delta = self.relation_types.get(rel["relation"], {}).get("δ", 1.0)

                    # 路径置信度 = conf(e) × δ
                    path_conf = confidence * delta

                    if path_conf >= threshold:
                        target = rel["object"]
                        if target not in visited:
                            new_ids.append(target)
                            inst = self._find_instance_by_id(target)
                            if inst:
                                expanded.append({
                                    "id": inst["id"],
                                    "name": inst.get("name", ""),
                                    "category": inst.get("category", ""),
                                    "confidence": path_conf,
                                    "source": inst.get("source", ""),
                                    "anchor": inst.get("anchor", {}),
                                    "hop": hop + 1,
                                    "path_id": rel["id"]
                                })
                                visited.add(target)

            if not new_ids:
                break

        return expanded

    def _find_instance_by_id(self, instance_id: str) -> Optional[Dict]:
        """根据ID查找实例"""
        for category, instances in self.instances.items():
            for inst in instances:
                if inst["id"] == instance_id:
                    return inst
        return None

    def fuse(
        self,
        candidates: List[Dict],
        rrf_k: int = None,
        top_m: int = None
    ) -> List[Dict]:
        """
        阶段4：分层融合
        Tier-1: 符号路精确命中且过五步校验 → 置顶
        Tier-2: RRF融合
        """
        rrf_k = rrf_k or self.rrf_k
        top_m = top_m or self.top_m

        tier1 = [c for c in candidates if c.get("precise_match")]
        tier2 = [c for c in candidates if not c.get("precise_match")]

        # Tier-1 置顶不稀释
        tier1_sorted = sorted(tier1, key=lambda x: (-x["confidence"], x.get("source", "")))

        # Tier-2 RRF融合
        tier2_ranked = self._rrf_rank(tier2, k=rrf_k)[:top_m]

        return tier1_sorted + tier2_ranked

    def _rrf_rank(self, candidates: List[Dict], k: int) -> List[Dict]:
        """
        RRF (Reciprocal Rank Fusion) 名次融合
        score_RRF(x) = Σ 1/(k + rank_lane(x))
        """
        if not candidates:
            return []

        # 按多个维度排序（简化版：只用confidence）
        sorted_by_conf = sorted(candidates, key=lambda x: -x["confidence"])

        # 计算RRF分数
        rrf_scores = {}
        for rank, c in enumerate(sorted_by_conf):
            rrf_scores[c["id"]] = 1.0 / (k + rank + 1)

        # 按RRF分数重排
        result = sorted(candidates, key=lambda x: -rrf_scores.get(x["id"], 0))
        return result

    # ========== 模板填充 ==========

    def get_template(self, template_id: str) -> Dict:
        """获取任务模板"""
        return self.task_templates.get(template_id, {})

    def fill_template(
        self,
        template: Dict,
        slots: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        模板填充 + 五步校验
        返回填充后的槽位字典
        """
        template_id = template["id"]
        slot_defs = {s["slot_id"]: s for s in template.get("slots", [])}

        filled = {}
        missing_slots = []
        degradation_events = []

        # 五步校验
        for slot_id, slot_def in slot_defs.items():
            weight = slot_def.get("weight", 0.5)
            cardinality = slot_def.get("cardinality", "1")
            degrade = slot_def.get("degrade_strategy", "留空标注")

            if slot_id in slots and slots[slot_id] is not None:
                filled[slot_id] = slots[slot_id]
            else:
                # 检查是否必填
                if cardinality.startswith("1") or cardinality.startswith("必填"):
                    # 一票规则检查
                    if weight >= self.params.get("one_vote_weight_threshold", 0.75):
                        # 硬阻断
                        missing_slots.append(slot_id)
                        filled[slot_id] = {"ERROR": f"[阻断] 缺失必填槽位 {slot_id}"}
                    else:
                        # 降级处理
                        missing_slots.append(slot_id)
                        degradation_events.append(f"降级: {slot_id} -> {degrade}")
                        filled[slot_id] = {"WARNING": f"[降级] {slot_id} {degrade}"}
                else:
                    # 可选槽
                    filled[slot_id] = None

        # 加权覆盖率计算
        coverage = self._calc_coverage(filled, slot_defs)

        filled["_meta"] = {
            "coverage": coverage,
            "missing_slots": missing_slots,
            "degradation_events": degradation_events,
            "coverage_pass": coverage >= self.coverage_threshold
        }

        # 覆盖率三档判定
        if coverage >= self.coverage_threshold:
            filled["_meta"]["coverage_decision"] = "直接出稿"
        elif coverage >= 0.6:
            filled["_meta"]["coverage_decision"] = "初稿+缺口清单"
        else:
            filled["_meta"]["coverage_decision"] = "触发补全"

        return filled

    def _calc_coverage(self, filled: Dict, slot_defs: Dict) -> float:
        """计算加权覆盖率"""
        total_weight = 0.0
        filled_weight = 0.0

        for slot_id, slot_def in slot_defs.items():
            weight = slot_def.get("weight", 0.5)
            cardinality = slot_def.get("cardinality", "1")

            # 解析基数
            if cardinality.startswith("1"):
                required = True
            else:
                required = False

            if required:
                total_weight += weight
                if slot_id in filled and filled[slot_id] is not None:
                    val = filled[slot_id]
                    if isinstance(val, dict) and "ERROR" in val:
                        pass  # 阻断槽未填
                    else:
                        filled_weight += weight

        if total_weight == 0:
            return 1.0

        return filled_weight / total_weight

    # ========== 判定执行 ==========

    def execute_verdict(
        self,
        verdict_rule_id: str,
        evidence: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        执行判定规则
        支持四种模式：确定性代码 / 双组件 / LLM评分+阈值 / 人工判定
        """
        rule = self.verdict_rules.get(verdict_rule_id)
        if not rule:
            return {"ERROR": f"未找到判定规则 {verdict_rule_id}"}

        mode = rule.get("mode", "确定性代码")
        verdict_logic = rule.get("verdict_logic", "")

        if mode == "确定性代码":
            # 直接执行判决逻辑（简化版）
            return self._execute_deterministic(verdict_logic, evidence)

        elif mode == "双组件":
            # 证据抽取器（LLM）+ 确定性判决
            evidence_slots = rule.get("evidence_slots", [])
            extracted = self._extract_evidence(evidence_slots, evidence)
            return self._execute_deterministic(verdict_logic, {**evidence, **extracted})

        elif mode == "LLM评分+阈值":
            threshold = rule.get("threshold", 0.95)
            score_feedback = self._llm_score(evidence)
            return {
                "score": score_feedback["score"],
                "feedback": score_feedback["feedback"],
                "threshold": threshold,
                "pass": score_feedback["score"] >= threshold,
                "fuzzy_marker": "含模糊判定"
            }

        elif mode == "人工判定":
            return {"status": "待人工判定", "evidence": evidence}

        return {"ERROR": f"未知判定模式 {mode}"}

    def _execute_deterministic(self, logic: str, evidence: Dict) -> Dict:
        """执行确定性判决"""
        # 简化实现：实际应解析逻辑表达式执行
        return {"verdict": "确定性判决结果", "logic": logic, "evidence": evidence}

    def _extract_evidence(self, slots: List[Dict], evidence: Dict) -> Dict:
        """证据抽取器（LLM实现）"""
        # TODO: 实际实现调用LLM API
        return {}

    def _llm_score(self, evidence: Dict) -> Dict:
        """LLM评分"""
        # TODO: 实际实现调用LLM API
        return {"score": 0.9, "feedback": "质量良好"}

    # ========== 反馈闭环 ==========

    def record_success_signal(
        self,
        template_id: str,
        采纳: bool = False,
        异议: bool = False,
        执行: bool = False,
        评价: str = ""
    ):
        """
        采集成功信号，更新策略统计
        """
        if template_id not in self.strategy_stats:
            self.strategy_stats[template_id] = {"n": 0, "s": 0, "success_rate": 0.0}

        stats = self.strategy_stats[template_id]
        stats["n"] += 1

        if 采纳 or (执行 and not 异议):
            stats["s"] += 1

        # 贝叶斯平滑
        alpha = self.params.get("bayesian_alpha", 8)
        beta = self.params.get("bayesian_beta", 2)
        stats["success_rate"] = (stats["s"] + alpha) / (stats["n"] + alpha + beta)

        return stats

    # ========== 工具方法 ==========

    def get_instance_by_id(self, instance_id: str) -> Optional[Dict]:
        """根据ID获取实例完整信息"""
        return self._find_instance_by_id(instance_id)

    def get_relation_by_id(self, relation_id: str) -> Optional[Dict]:
        """根据ID获取关系实例"""
        for rel in self.relations:
            if rel["id"] == relation_id:
                return rel
        return None

    def time_filter(self, instance: Dict, time_point: str) -> bool:
        """时间有效性过滤"""
        validity = instance.get("validity", [None, None])
        if not validity or len(validity) != 2:
            return True

        start, end = validity
        if start and time_point < start:
            return False
        if end and time_point > end:
            return False
        return True

    def check_conflict(self, instance: Dict) -> bool:
        """冲突检测"""
        return instance.get("status") == "冲突中"


# ========== Skill工厂函数 ==========

def create_skill(architecture_path: str, knowledge_base_path: str) -> MLKnowledgeBaseSkill:
    """
    创建ML知识库Skill实例
    入口函数，供外部调用
    """
    return MLKnowledgeBaseSkill(
        architecture_path=architecture_path,
        knowledge_base_path=knowledge_base_path
    )


# ========== 示例用法 ==========

if __name__ == "__main__":
    skill = create_skill(
        architecture_path="ml-arch-v1.0.0.json",
        knowledge_base_path="ml-kb-v1.0.0.json"
    )

    # 示例：答疑任务
    task_type = skill.recognize_task_type("K-Means聚类的原理是什么？")
    print(f"识别任务类型: {task_type}")

    template = skill.get_template("ml:template_qa_explanation")
    print(f"获取模板: {template['id']}")

    # 示例：反馈回写
    stats = skill.record_success_signal(
        template_id="ml:template_causal_debug",
        采纳=True
    )
    print(f"策略统计更新: {stats}")
