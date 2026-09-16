# ML Knowledge Base Skill

**版本**: ml-skill v1.1.0
**架构兼容声明**: `arch_compat: ">=1.0.0, <2.0.0"`
**维护者**: ML知识库团队
**更新日期**: 2026-09-16
**Phase B扩展**: 簇C经典深度学习（CNN/RNN/Transformer/优化器）

---

## 第一部分：操作规程（面向挂载本Skill的AI）

### 1. 接入方式

```python
# 通过API调用接入
from ml_kb_skill import MLKnowledgeBaseSkill

skill = MLKnowledgeBaseSkill(
    knowledge_base_path="ml-kb-v1.0.0.json",
    architecture_path="ml-arch-v1.0.0.json"
)
```

### 2. 知识库类型速查

| 类型前缀 | 说明 |
|---------|------|
| `ml:Concept` | 概念/原理知识（如：过拟合、交叉熵） |
| `ml:Model` | 模型实体（如：RandomForest、PCA） |
| `ml:Dataset` | 数据集（如：Iris、Wine） |
| `ml:Pipeline` | 标准流程（如：聚类Pipeline、文本分类Pipeline） |
| `ml:VerdictRule` | 判定规则（如：模型质量判定、参数合理性判定） |
| `ml:TaskTemplate` | 任务模板（如：问题诊断、模型选型） |

### 3. 四阶段召回管线调用

```python
# 阶段1：任务解析
task_type = skill.recognize_task_type(
    user_query="K-Means聚类的原理是什么？"
)
# → 返回: "答疑" / "问题诊断" / "模型选型" / "代码生成" 等

# 阶段2：双路召回
candidates = skill.recall(
    task_type=task_type,
    query_vector=query_embedding,
    time_point="2026-09-15"
)
# → 返回: {"symbolic": [...], "vector": [...], "graph": [...]}

# 阶段3：图扩展
extended = skill.graph_expand(
    candidates=candidates,
    max_hops=3,
    threshold=0.4
)

# 阶段4：分层融合
ranked = skill.fuse(
    candidates=extended,
    rrf_k=60,
    top_m=30
)
# → 返回排序后的最终候选列表
```

### 4. 逐任务工作流

#### 任务类型A：问题诊断

```python
# 识别 → 取模板 → 检索填槽 → 带锚点输出

# 识别
task = skill.recognize_task_type("为什么模型在测试集上表现很好但线上效果很差？")
assert task == "问题诊断"

# 取模板
template = skill.get_template("ml:template_diagnosis")

# 填充槽位
filled = skill.fill_template(
    template=template,
    slots={
        "slot_problem_description": "模型测试集准确率95%，但线上实际准确率只有72%",
        "slot_model_info": "RandomForest, n_estimators=200, max_depth=None",
        "slot_data_info": "训练集10万条，测试集2万条"
    }
)

# 输出（带锚点）
report = filled["slot_diagnosis_report"]
print(report["root_cause"])        # 可能原因：过拟合/数据泄露/分布差异
print(report["causal_chain"])    # 诊断链路
print(report["fix_suggestions"])  # 修复建议
```

#### 任务类型B：模型选型

```python
task = skill.recognize_task_type(
    "我有10万条表格数据，需要做二分类，应该选什么模型？"
)
template = skill.get_template("ml:template_model_selection")

filled = skill.fill_template(
    template=template,
    slots={
        "slot_data_description": "10万条表格数据，50个特征，含缺失值",
        "slot_task_type": "二分类",
        "slot_constraints": ["可解释性要求中等", "推理延迟<100ms"]
    }
)

# 输出
print(filled["slot_recommendation"])
# 推荐：RandomForest（首选）/ XGBoost（备选）/ LogReg（可选）
```

#### 任务类型C：代码生成

```python
task = skill.recognize_task_type(
    "帮我写一个K-Means聚类的完整代码"
)
template = skill.get_template("ml:template_code_generation")

filled = skill.fill_template(
    template=template,
    slots={
        "slot_task": "K-Means聚类",
        "slot_requirements": "数据标准化，K=3，可视化结果"
    }
)

# 输出
print(filled["slot_code"])           # 完整代码
print(filled["slot_explanation"])    # 代码解释
```

#### 任务类型D：答疑

```python
task = skill.recognize_task_type("K-Means的原理是什么？")
template = skill.get_template("ml:template_qa_explanation")

filled = skill.fill_template(
    template=template,
    slots={"slot_question": "K-Means的原理是什么？"}
)

# 输出带锚点的答疑报告
print(filled["slot_answer"]["content"])   # 原理说明
print(filled["slot_answer"]["anchors"])  # 可点击的原文锚点
```

### 5. 当前库缺口的呈现要求

知识库当前缺口必须在输出中显式可见，不得悄悄隐藏：

- **簇D/簇E空缺**：提示"本期A暂未覆盖深度学习和强化学习内容，B期补充"
- **待核对项**：显示状态标记（如：`[待核对]`）
- **冲突中项**：双面呈现原文主张+反例，不得单方断言
- **语义碎分风险**：降级输出时标注`[降级]`并说明原因

---

## 第二部分：维护者章节（面向人类维护者）

### 1. 检索与编排的实现规范

#### 1.1 向量化规范

- 嵌入模型：`text-embedding-ada-002`（可配置）
- 嵌入文本模板：
  - 事实类-概念：`标准名称+别名+定义/核心内容+所属领域`
  - 事实类-Model：`名称+模型类型+输入数据类型+适用任务`
  - 过程类-Pipeline：`名称+适用场景+输入/输出工件类型名+步骤描述摘要`
  - 规则类：`规则名称+条件+结论+适用边界`
- 向量化时机：入库时生成，更新时重算

#### 1.2 混合召回管线

```python
# 四阶段召回伪代码
def recall(task_type, query_vector, time_point):
    # 阶段1：任务解析
    task_obj = parse_task(task_type)

    # 阶段2：双路召回
    c_symbolic = symbolic_search(task_obj)  # 符号路
    c_vector = vector_search(query_vector, top_k=50)  # 向量路

    # 阶段3：图扩展
    c_graph = graph_expand(
        seeds=c_symbolic + c_vector,
        max_hops=3,
        threshold=params["θ_hint"]  # 0.4
    )

    # 阶段4：分层融合
    return layered_fusion(c_symbolic, c_vector, c_graph)

def layered_fusion(c_s, c_v, c_g):
    tier1 = [x for x in c_s if x["precise_match"] and x["five_step_pass"]]
    tier2 = ranked_rrf(c_v, c_g, k=60, top_m=30)
    return tier1 + tier2
```

#### 1.3 RRF融合公式

```
score_RRF(x) = Σ 1/(k + rank_lane(x))  (k=60)
```

- Tier-1：符号路精确命中且过五步校验 → 置顶不稀释
- Tier-2：向量路按sim、图路按conf×rel×δ各自排序后RRF融合

### 2. 判决模式落点

| 模式 | 使用场景 | 实现方式 |
|------|---------|---------|
| 确定性代码 | 模型参数合理性判定（n_estimators>0） | `if/else` 条件表达式 |
| 双组件 | 问题根因诊断（证据LLM抽→程序判决） | EvidenceExtractor + DeterministicVerdict |
| LLM评分+阈值 | 代码质量判定 | `{score, feedback}` + 阈值判定 |
| 人工判定 | 无自动验证手段时 | UI入口（人工自证/仲裁） |

### 3. 设计决策与实测修正

| 决策 | 理由 | 实测修正 |
|------|------|---------|
| 问题诊断一票规则权重0.75 | 关键信息缺失即阻断是防止无依据乱诊断 | 实测：用户常漏填问题描述，需在模板说明中强化前置说明 |
| 模型选型降级策略为"留空标注" | 部分约束缺失时留空由用户补充 | 实测：约束不完整时应给出推荐并说明前提 |
| 覆盖率阈值0.85（L3更严格） | ML场景不容忍低覆盖率直接出稿 | 实测：有效防止不完整答案流出 |

### 4. 参数调优规程

#### 4.1 调优节奏

1. **默认参数运行**：积累<50任务，使用默认参数
2. **调参循环**：积累≥50任务后，调优 RRF k / Top-M / fresh阈值
3. **LTR引入**：标注≥500对后评估LTR重排

#### 4.2 关键参数

| 参数 | 默认值 | 说明 | 调优范围 |
|------|-------|------|---------|
| `θ_use` | 0.6 | 可直接编排填充的置信度阈值 | 0.5-0.7 |
| `θ_hint` | 0.4 | 推理增强的置信度截断阈值 | 0.3-0.5 |
| `Top-K` | 50 | 向量路ANN召回数 | 30-100 |
| `Top-M` | 30 | Tier-2截断数 | 20-50 |
| `RRF_k` | 60 | RRF融合常数 | 30-100 |
| `coverage_threshold` | 0.85 | L3档覆盖率达标阈值 | 0.8-0.9 |

**调参即改架构参数，走黄色回流，禁止应用侧直接改。**

### 5. 运行治理清单

| 周期 | 内容 |
|------|------|
| 日 | 审核队列消化（AI生成内容） |
| 周 | 待核对答案核销、冲突仲裁结论回写 |
| 月 | 画像数据质量抽查、兼容矩阵维护 |
| 触发 | 失效级联监测（三类） |

### 6. 回流协议

| 颜色 | 触发 | 受理方 |
|------|------|--------|
| 🔵蓝 | 检索精度/编排逻辑/判决实现/参数适配缺陷 | SOP-3受理 |
| 🟢绿 | 实例错误/答案争议/关系缺边/密度不足 | SOP-2治理队列 |
| 🟡黄 | 子类/属性/关系/槽位/判决契约不够用 | SOP-1受理 |
| 🔴红 | 业务目标变化 | SOP-0受理 |

### 7. 架构版本兼容声明

```yaml
arch_compat: ">=1.0.0, <2.0.0"
```

- 架构升**次版本**：默认区间内兼容，重载定义文件即可
- 架构升**主版本**：默认区间外，需回归自测通过后扩区间
- 兼容矩阵由机器判定，版本变化时自动提示回归范围

### 8. 反馈闭环

```python
# 成功信号采集
skill.record_success_signal(
    template_id="ml:template_diagnosis",
   采纳=True,
    异议=False,
    执行=True,
    评价="准确"
)

# 贝叶斯回写
# conf_new = (s + α) / (n + α + β)
# α=8, β=2

# 知识级微调
# 正反馈采纳：置信度 += 学习率 * 0.5
# 负反馈删改：置信度 -= 学习率 * 0.25
```

### 9. 失效级联监测

三类触发：

1. **知识失效** → 扫描以其为客体的fillWith/checkBy/deriveFrom，主体标"待复核"
2. **锚点漂移** → 原文切片哈希比对不符，标记该叶子知识及证据链含该锚点的物化推理结论
3. **工件结构Schema变更** → 触发所有consume该类型的下游步骤兼容性复核

---

## 附录：版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.1.0 | 2026-09-16 | Phase B扩展：簇C经典深度学习（LeNet/AlexNet/VGG/ResNet/DenseNet/LSTM/GRU/Transformer/注意力机制/优化器） |
| v1.0.0 | 2026-09-15 | 首版，基于ml-arch v1.0.0和ml-kb v1.0.0 |
