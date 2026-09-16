# ML知识库 - 文件说明

## 概述

本目录包含基于SOP体系构建的机器学习知识库完整产物。

**领域**：金融合规风控 + 机器学习（Phase A：传统ML / Phase B：深度学习 / Phase C：AI大模型）
**Skill形态**：可挂载到风控多智能体Agent的AI助手
**能力档位**：L3 严格审查档

---

## 文件清单

| 文件 | 类型 | 说明 |
|------|------|------|
| `ml-arch-v1.0.0.json` | **真相源** | 架构定义文件，机器可读，唯一的架构权威 |
| `ml-kb-v1.0.0.json` | **真相源** | 知识库文件，机器可读，包含全部知识实例 |
| `SKILL.md` | 操作规程 | Skill的操作规程与维护手册 |
| `engine.py` | 算法代码 | 检索与编排引擎实现（从架构文件加载参数） |
| `案例集.md` | 视图 | 应用案例正文（含5段trace + 两张核销映射表） |
| `执行台账.md` | 视图 | 执行台账，记录全链路执行过程 |
| `README.md` | 说明 | 本文件 |
| `build_index.py` | 入库脚本 | 将知识库向量化并写入向量数据库 |
| `llm_mount.py` | 挂载脚本 | 三种LLM挂载方案（OpenAI/LangChain/直接API） |

---

## 架构定义文件 (ml-arch-v1.0.0.json) 结构

```json
{
  "meta": { "version": "1.0.0", "conformsTo": "v3.0", "namespace": "ml:" },
  "entity_types": [ ... ],      // 实体类型定义（四大类骨架）
  "relation_types": [ ... ],    // 关系类型定义（五大关系族）
  "task_templates": [ ... ],   // 任务模板（含槽位定义）
  "verdict_rules": [ ... ],    // 判定规则（含四种判决模式）
  "source_levels": [ ... ],    // 来源权威度等级
  "params_override": { ... },  // 参数覆盖表
  "change_propagation_matrix": [ ... ]  // 变更传导矩阵
}
```

---

## 知识库文件 (ml-kb-v1.0.0.json) 结构

```json
{
  "meta": {
    "conformsTo": "ml-arch v1.0.0",
    "stats": { "事实类实例": 140, "过程类实例": 35, "规则类实例": 25, "策略类实例": 8 },
    "验收": { "总结论": "通过" }
  },
  "instances": {
    "facts_concepts": [ ... ],     // 概念知识
    "facts_models": [ ... ],        // 模型实体
    "facts_datasets": [ ... ],     // 数据集
    "facts_attributes": [ ... ],     // 属性事实
    "process_pipelines": [ ... ],   // 标准流程
    "rule_causal": [ ... ],        // 因果规则
    "rule_verdicts": [ ... ],      // 判定规则
    "strategy_templates": [ ... ]   // 策略模板
  },
  "relations": [ ... ]              // 关系实例
}
```

---

## Skill调用示例

```python
from engine import create_skill

skill = create_skill(
    architecture_path="ml-arch-v1.0.0.json",
    knowledge_base_path="ml-kb-v1.0.0.json"
)

# 识别任务类型
task = skill.recognize_task_type("K-Means聚类的原理是什么？")

# 获取模板
template = skill.get_template("ml:template_qa_explanation")

# 填充模板
filled = skill.fill_template(
    template=template,
    slots={"slot_question": "K-Means的原理是什么？"}
)
```

---

## 版本体系

| 版本 | 日期 | 说明 |
|------|------|------|
| v1.0.0 | 2026-09-15 | 首版，基于scikit-learn v1.9.1 User Guide |

---

## 知识覆盖范围

**五大簇**：

| 簇 | 覆盖内容 |
|----|---------|
| 簇A 统计与线性 | 线性模型、逻辑回归、Ridge/Lasso、SVM、朴素贝叶斯、高斯过程 |
| 簇B 树与集成 | 决策树、随机森林、GBDT、XGBoost、LightGBM、CatBoost |
| 簇C 经典深度 | MLP、PCA/Kernel PCA、流行学习、聚类（GMM/K-Means/DBSCAN等） |
| 簇D 决策类 | **本期A暂不覆盖，B期补充** |
| 簇E 模型生成 | **本期A暂不覆盖，C期补充** |

**核心任务模板**：

| 模板 | 用途 |
|------|------|
| ml:template_causal_debug | 多智能体因果调试（定位"看似正确但逻辑违规"的边缘Case） |
| ml:template_rag_governance | RAG数据管道治理（语义切片+隐私脱敏） |
| ml:template_risk_selection | 风控模型选型推荐 |
| ml:template_qa_explanation | 机器学习答疑解释 |

---

## 挂载到LLM的三种方案

### 方案1：OpenAI Assistant API（最简，推荐起步）

```bash
pip install openai
python llm_mount.py openai
```

```python
from openai import OpenAI
client = OpenAI()

# 创建Thread
thread = client.beta.threads.create()
client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="K-Means聚类的原理是什么？"
)

# 运行Assistant
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant_id
)
```

### 方案2：LangChain Agent（更灵活的编排）

```bash
pip install langchain langchain-openai
python llm_mount.py langchain
```

```python
executor = create_langchain_agent()
result = executor.invoke({"input": "为什么反欺诈模型误报率升高了？"})
print(result["output"])
```

### 方案3：直接API调用（最轻量）

```python
python llm_mount.py api
```

---

## 依赖安装

```bash
pip install openai          # OpenAI API
pip install langchain       # LangChain框架
pip install faiss-cpu      # 向量索引（轻量本地）
pip install numpy          # 数值计算
```

---

## 后续扩展

- **B期（深度学习）**：补充簇C扩展（CNN/RNN/Transformer）和簇D（强化学习）知识
- **C期（AI大模型）**：补充簇E（大语言模型、扩散模型、Agent）知识
- **增量维护**：资源包更新 → 走总控增量启动协议 → SOP-2增量扩批 → SOP-4增量回归

---

## 约束与限制

1. 所有参数从架构定义文件加载，**禁止在engine.py中硬编码领域参数**
2. 知识库和架构定义文件为**机器可读唯一权威**，架构书由定义文件生成
3. 案例集为视图，**禁止手写**，由真相源脚本生成
4. 变更走**变更传导矩阵**，禁止口头改动冻结的架构
