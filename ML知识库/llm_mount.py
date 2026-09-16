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
你是一个专业的机器学习知识助手，基于金融合规风控场景的知识库提供服务。

你必须严格遵循以下规范：

1. 只基于知识库中的知识回答，不在库外编造
2. 合规建议类问题必须标注锚点来源
3. AI生成的内容必须标注"AI生成"
4. 遇到不确定的问题，明确告知用户"知识库中无相关信息"
5. 因果调试问题必须给出完整的因果链路图

你掌握的知识库摘要：
{kb_summary}
""",
        model="gpt-4o",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "causal_debug",
                    "description": "执行多智能体因果调试，定位边缘Case的根因",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "error_signal": {
                                "type": "string",
                                "description": "错误信号描述"
                            },
                            "workflow_def": {
                                "type": "string",
                                "description": "工作流结构定义（JSON格式）"
                            },
                            "agent_traces": {
                                "type": "string",
                                "description": "Agent追踪日志（可选）"
                            }
                        },
                        "required": ["error_signal", "workflow_def"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "rag_governance",
                    "description": "执行RAG数据治理：语义切片+隐私脱敏",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "raw_data_path": {
                                "type": "string",
                                "description": "原始数据路径"
                            },
                            "instruction": {
                                "type": "string",
                                "description": "自然语言指令"
                            }
                        },
                        "required": ["raw_data_path", "instruction"]
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
            },
            {
                "type": "function",
                "function": {
                    "name": "risk_selection",
                    "description": "风控模型选型推荐",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "scenario": {
                                "type": "string",
                                "description": "业务场景"
                            },
                            "constraints": {
                                "type": "string",
                                "description": "约束条件（JSON格式）"
                            }
                        },
                        "required": ["scenario", "constraints"]
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
    def causal_debug_tool(query: str) -> str:
        """
        因果调试Tool
        输入格式：JSON字符串 {error_signal, workflow_def, agent_traces?}
        """
        import json
        try:
            params = json.loads(query)
        except json.JSONDecodeError:
            return "输入格式错误，请提供JSON格式参数"

        template = skill.get_template("ml:template_causal_debug")
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

    def risk_selection_tool(query: str) -> str:
        """风控选型Tool"""
        import json
        try:
            params = json.loads(query)
        except json.JSONDecodeError:
            return "输入格式错误，请提供JSON格式参数"

        template = skill.get_template("ml:template_risk_selection")
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
            name="causal_debug",
            func=causal_debug_tool,
            description="执行多智能体因果调试，定位边缘Case根因。输入为JSON: {error_signal, workflow_def, agent_traces?}"
        ),
        Tool(
            name="ml_qa",
            func=ml_qa_tool,
            description="机器学习知识答疑，如算法原理、模型选择、代码实现等"
        ),
        Tool(
            name="risk_selection",
            func=risk_selection_tool,
            description="风控模型选型推荐，输入为JSON: {scenario, constraints}"
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
1. 识别问题类型（答疑/因果调试/风控选型/RAG治理）
2. 调用对应的Tool获取知识
3. 综合Tool返回结果，用自然语言回答
4. 所有回答必须附带知识锚点来源

你擅长的领域：
- 机器学习算法原理（监督/无监督/深度学习）
- 金融风控模型选型
- 多智能体系统因果调试
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

    # 示例2：因果调试
    debug_params = {
        "error_signal": "反欺诈模型误报率突然从2%升至15%",
        "workflow_def": json.dumps({
            "agents": ["RetrievalAgent", "ReasoningAgent", "FinalDecisionAgent"],
            "connections": ["input->Retrieval", "Retrieval->Reasoning", "Reasoning->Final"]
        }),
        "agent_traces": [
            {"agent": "RetrievalAgent", "input": "长期债券", "output": "短期国债", "confidence": 0.6}
        ]
    }

    template = skill.get_template("ml:template_causal_debug")
    result = skill.fill_template(template, slots={
        "slot_error_signal": debug_params["error_signal"],
        "slot_workflow_def": debug_params["workflow_def"],
        "slot_agent_traces": str(debug_params["agent_traces"])
    })

    verdict = skill.execute_verdict("ml:verdict_001", evidence=debug_params)
    print(f"因果调试结果: {verdict}")

    # 示例3：反馈回写
    stats = skill.record_success_signal(
        template_id="ml:template_causal_debug",
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
