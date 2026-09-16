"""
ML Knowledge Base Skill - 知识入库脚本
将 ml-kb-v1.0.0.json 中的知识实例向量化并写入向量数据库
"""

import json
import hashlib
from typing import List, Dict, Any
import numpy as np


def generate_embedding(text: str, model: str = "text-embedding-ada-002") -> List[float]:
    """
    生成文本向量
    实际实现接入 OpenAI / Cohere / HuggingFace 等嵌入API
    """
    # === 方案A：OpenAI ===
    # from openai import OpenAI
    # client = OpenAI()
    # response = client.embeddings.create(
    #     model=model,
    #     input=text
    # )
    # return response.data[0].embedding

    # === 方案B：Cohere ===
    # import cohere
    # co = cohere.Client("YOUR_API_KEY")
    # response = co.embed(texts=[text])
    # return response.embeddings[0]

    raise NotImplementedError("请配置嵌入模型API")


def build_embedding_text(instance: Dict, category: str) -> str:
    """
    按模板大类生成嵌入文本
    对应 AP-13 嵌入文本模板
    """
    if category == "facts_concepts":
        parts = [
            instance.get("name", ""),
            instance.get("definition", ""),
            instance.get("domain", "")
        ]
    elif category == "facts_models":
        parts = [
            instance.get("name", ""),
            instance.get("model_type", ""),
            instance.get("input_data", ""),
            " ".join(instance.get("tasks", []))
        ]
    elif category == "process_pipelines":
        parts = [
            instance.get("name", ""),
            instance.get("scenario", ""),
            "输入: " + ", ".join(instance.get("input_artifacts", [])),
            "输出: " + ", ".join(instance.get("output_artifacts", [])),
            instance.get("steps", "")
        ]
    elif category == "rule_causal":
        parts = [
            instance.get("name", ""),
            "条件: " + instance.get("condition", ""),
            "结论: " + instance.get("conclusion", ""),
            "范围: " + instance.get("scope", "")
        ]
    else:
        parts = [instance.get("name", ""), instance.get("definition", "")]

    return " | ".join([p for p in parts if p])


def index_knowledge_base(
    kb_path: str,
    output_index_path: str = "ml_kb.index",
    output_meta_path: str = "ml_kb_metadata.json"
):
    """
    将知识库向量化并写入FAISS索引
    （轻量本地方案；生产环境建议用 Milvus / Pinecone）
    """
    import faiss

    # 加载知识库
    with open(kb_path, 'r', encoding='utf-8') as f:
        kb = json.load(f)

    embeddings = []
    metadata = []

    # 遍历所有实例类别
    for category, instances in kb["instances"].items():
        for inst in instances:
            text = build_embedding_text(inst, category)
            vec = generate_embedding(text)

            embeddings.append(vec)
            metadata.append({
                "id": inst["id"],
                "category": category,
                "name": inst.get("name", ""),
                "instance": inst
            })

    # 转换为numpy数组
    embedding_matrix = np.array(embeddings).astype('float32')

    # 建立FAISS索引（内积索引，适合cosine相似度）
    dim = embedding_matrix.shape[1]
    index = faiss.IndexFlatIP(dim)

    # L2归一化（使内积等价于cosine相似度）
    faiss.normalize_L2(embedding_matrix)
    index.add(embedding_matrix)

    # 保存索引和元数据
    faiss.write_index(index, output_index_path)

    with open(output_meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"✅ 索引构建完成: {len(metadata)} 条知识向量")
    print(f"   索引文件: {output_index_path}")
    print(f"   元数据: {output_meta_path}")

    return metadata


if __name__ == "__main__":
    metadata = index_knowledge_base(
        kb_path="ml-kb-v1.0.0.json",
        output_index_path="ml_kb.index",
        output_meta_path="ml_kb_metadata.json"
    )
