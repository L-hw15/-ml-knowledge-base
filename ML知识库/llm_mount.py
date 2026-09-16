"""
ML Knowledge Base Skill - LLM挂载脚本
展示如何将该Skill挂载到主流LLM Agent框架

支持：OpenAI Assistant / LangChain / Claude API
"""

import json
from typing import List, Dict, Any, Optional


# ============================================================
# 方案1：OpenAI Assistant API（最简挂载方式）
# ============================================================

def create_openai_assistant(
    skill_md_path: str = "SKILL.md",
    kb_path: str = "ml-kb-v1.0.0.json"
):
    """
    将ML Skill作为OpenAI Assistant的Tool挂载
    """
    from openai import OpenAI

    client = OpenAI()

    # 读取SKILL.md作为Assistant的system prompt
    with open(skill_md_path, 'r', encoding='utf-8') as f:
        skill_prompt = f.read()

    # 读取知识库摘要（实际生产中可作为file_search的知识库）
    with open(kb_path, 'r', encoding='utf-8') as f:
        kb_summary = f.read()[:2000]  # 截取前2000字符作为摘要

    # 创建Assistant
    assistant = client.beta.assistants.create(
        name="ML Knowledge Assistant",
        instructions=f"""
你是一个专业的机器学习知识助手，基于通用机器学习知识库提供服务。

你必须严格遵循以下规范：

1. 只基于知识库中的知识回答，不在库外编造
2. 答案必须标注锚点来源
3. AI生成的内容必须标注"AI生成"
4. 遇到不确定的问题，明确告知用户"知识库中无相关信息"
5. 问题诊断必须给出完整的因果链路图

你掌握的知识库摘要：
{kb_summary}
""",
        model="gpt-4o",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "problem_diagnosis",
                    "description": "执行机器学习问题诊断，定位模型效果差的根因",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "problem_description": {
                                "type": "string",
                                "description": "问题描述"
                            },
                            "model_info": {
                                "type": "string",
                                "description": "模型信息（类型、参数配置）"
                            },
                            "data_info": {
                                "type": "string",
                                "description": "数据信息（样本量、特征数）"
                            }
                        },
                        "required": ["problem_description", "model_info"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "model_selection",
                    "description": "机器学习模型选型推荐",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "data_description": {
                                "type": "string",
                                "description": "数据描述（样本量、特征类型）"
                            },
                            "task_type": {
                                "type": "string",
                                "description": "任务类型（二分类/多分类/回归/聚类）"
                            },
                            "constraints": {
                                "type": "string",
                                "description": "约束条件（如：延迟、可解释性要求）"
                            }
                        },
                        "required": ["data_description", "task_type"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "code_generation",
                    "description": "生成机器学习代码",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "任务名称（如：K-Means聚类、随机森林分类）"
                            },
                            "requirements": {
                                "type": "string",
                                "description": "需求描述（如：包含可视化、交叉验证）"
                            }
                        },
                        "required": ["task"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "ml_qa",
                    "description": "机器学习知识答疑",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "用户问题"
                            }
                        },
                        "required": ["question"]
                    }
                }
            }
        ],
        tool_resources={
            "code_interpreter": {
                "file_ids": []
            }
        }
    )

    print(f"✅ Assistant创建成功: {assistant.id}")
    return assistant


# ============================================================
# 方案2：LangChain Agent（更灵活的编排）
# ============================================================

def create_langchain_agent(
    architecture_path: str = "ml-arch-v1.0.0.json",
    knowledge_base_path: str = "ml-kb-v1.0.0.json"
):
    """
    将ML Skill作为LangChain Agent的Tool挂载
    """
    try:
        from langchain.agents import AgentExecutor, create_openai_functions_agent
        from langchain_openai import ChatOpenAI
        from langchain.tools import Tool
        from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    except ImportError:
        print("❌ 请安装 langchain: pip install langchain langchain-openai")
        return None

    # 加载skill引擎
    from engine import create_skill
    skill = create_skill(architecture_path, knowledge_base_path)

    # 定义Tool函数
    def problem_diagnosis_tool(query: str) -> str:
        """
        问题诊断Tool
        输入格式：JSON字符串 {error_signal, workflow_def, agent_traces?}
        """
        import json
        try:
            params = json.loads(query)
        except json.JSONDecodeError:
            return "输入格式错误，请提供JSON格式参数"

        template = skill.get_template("ml:template_diagnosis")
        filled = skill.fill_template(
            template,
            slots={
                "slot_error_signal": params.get("error_signal", ""),
                "slot_workflow_def": params.get("workflow_def", ""),
                "slot_agent_traces": params.get("agent_traces", "")
            }
        )

        if "_meta" in filled:
            meta = filled["_meta"]
            if not meta.get("coverage_pass"):
                return f"[阻断] 缺失必填槽位: {meta['missing_slots']}"

        # 执行判定
        verdict = skill.execute_verdict(
            "ml:verdict_001",
            evidence={
                "error_signal": params.get("error_signal", ""),
                "workflow_def": params.get("workflow_def", ""),
                "agent_traces": params.get("agent_traces", "")
            }
        )

        return json.dumps({
            "filled_slots": {k: v for k, v in filled.items() if not k.startswith("_")},
            "verdict": verdict,
            "coverage": filled.get("_meta", {}).get("coverage", 0)
        }, ensure_ascii=False, indent=2)

    def ml_qa_tool(query: str) -> str:
        """答疑Tool"""
        task_type = skill.recognize_task_type(query)
        template = skill.get_template("ml:template_qa_explanation")

        filled = skill.fill_template(
            template,
            slots={"slot_question": query}
        )

        # 检索相关知识
        candidates = skill.recall(
            task_type=task_type,
            query_vector=[],  # 简化版，实际需真实向量
            time_point="2026-09-15"
        )

        ranked = skill.fuse(
            candidates.get("symbolic", []) + candidates.get("vector", [])
        )

        # 生成答案（此处为简化版，实际可接LLM生成）
        answer = f"关于您的问题，我检索到了{len(ranked)}条相关知识：\n"
        for item in ranked[:3]:
            answer += f"- [{item['name']}] (置信度: {item['confidence']})\n"

        return answer

    def model_selection_tool(query: str) -> str:
        """模型选型Tool"""
        import json
        try:
            params = json.loads(query)
        except json.JSONDecodeError:
            return "输入格式错误，请提供JSON格式参数"

        template = skill.get_template("ml:template_model_selection")
        filled = skill.fill_template(
            template,
            slots={
                "slot_scenario": params.get("scenario", ""),
                "slot_constraints": params.get("constraints", "")
            }
        )

        return json.dumps({
            "filled_slots": {k: v for k, v in filled.items() if not k.startswith("_")},
            "coverage": filled.get("_meta", {}).get("coverage", 0)
        }, ensure_ascii=False, indent=2)

    # 注册Tools
    tools = [
        Tool(
            name="problem_diagnosis",
            func=problem_diagnosis_tool,
            description="执行问题诊断，定位模型效果差的根因。输入为JSON: {error_signal, workflow_def, agent_traces?}"
        ),
        Tool(
            name="ml_qa",
            func=ml_qa_tool,
            description="机器学习知识答疑，如算法原理、模型选择、代码实现等"
        ),
        Tool(
            name="model_selection",
            func=model_selection_tool,
            description="模型选型推荐，输入为JSON: {scenario, constraints}"
        ),
        Tool(
            name="rag_governance",
            func=lambda x: "[TODO] RAG治理需接入文件系统和脱敏组件",
            description="RAG数据治理：语义切片+隐私脱敏"
        )
    ]

    # 创建Agent
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的ML知识助手。当用户提问时：
1. 识别问题类型（答疑/问题诊断/模型选型/RAG治理）
2. 调用对应的Tool获取知识
3. 综合Tool返回结果，用自然语言回答
4. 所有回答必须附带知识锚点来源

你擅长的领域：
- 机器学习算法原理（监督/无监督/深度学习）
- 模型选型推荐
- 问题诊断
- RAG数据治理
"""),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    agent = create_openai_functions_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    print("✅ LangChain Agent创建成功")
    return executor


# ============================================================
# 方案3：直接API调用（最轻量）
# ============================================================

def direct_api_call():
    """
    不使用Agent框架，直接通过API调用skill引擎
    适用于自有UI或自定义Agent场景
    """
    from engine import create_skill

    skill = create_skill(
        architecture_path="ml-arch-v1.0.0.json",
        knowledge_base_path="ml-kb-v1.0.0.json"
    )

    # 示例1：答疑
    question = "K-Means聚类的原理是什么？"
    task_type = skill.recognize_task_type(question)
    template = skill.get_template("ml:template_qa_explanation")
    result = skill.fill_template(template, slots={"slot_question": question})
    print(f"任务类型: {task_type}")
    print(f"填充结果: {result}")

    # 示例2：问题诊断
    debug_params = {
        "error_signal": "分类模型准确率突然从95%降至75%",
        "workflow_def": json.dumps({
            "agents": ["RetrievalAgent", "ReasoningAgent", "FinalDecisionAgent"],
            "connections": ["input->Retrieval", "Retrieval->Reasoning", "Reasoning->Final"]
        }),
        "agent_traces": [
            {"agent": "RetrievalAgent", "input": "样本特征", "output": "预测结果", "confidence": 0.6}
        ]
    }

    template = skill.get_template("ml:template_diagnosis")
    result = skill.fill_template(template, slots={
        "slot_error_signal": debug_params["error_signal"],
        "slot_workflow_def": debug_params["workflow_def"],
        "slot_agent_traces": str(debug_params["agent_traces"])
    })

    verdict = skill.execute_verdict("ml:verdict_001", evidence=debug_params)
    print(f"问题诊断结果: {verdict}")

    # 示例3：反馈回写
    stats = skill.record_success_signal(
        template_id="ml:template_diagnosis",
        采纳=True,
        评价="准确"
    )
    print(f"策略统计更新: {stats}")

    return skill


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法:")
        print("  python llm_mount.py openai    - 创建OpenAI Assistant")
        print("  python llm_mount.py langchain - 创建LangChain Agent")
        print("  python llm_mount.py api      - 直接API调用示例")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "openai":
        create_openai_assistant()
    elif mode == "langchain":
        create_langchain_agent()
    elif mode == "api":
        direct_api_call()
    else:
        print(f"未知模式: {mode}")
