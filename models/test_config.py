"""
测试GraphRAG配置和客户端
验证DeepSeek LLM和阿里云Embedding是否能正常工作
"""

import asyncio
import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from config import get_default_config, load_config_from_env
from graphrag_builder import (
    build_documents_from_sources,
    build_text_units,
    build_entities_and_relationships,
    build_communities,
    OpenAIClient,
    DeepSeekClient,
    OpenAIEmbeddingClient,
    AliyunEmbeddingClient
)


async def test_config():
    """测试配置加载"""
    print("=" * 60)
    print("测试GraphRAG配置")
    print("=" * 60)
    
    # 测试默认配置
    print("\n1. 测试默认配置...")
    default_config = get_default_config()
    print(f"   LLM提供商: {default_config.llm_config.provider.value}")
    print(f"   LLM模型: {default_config.llm_config.model}")
    print(f"   Embedding提供商: {default_config.embedding_config.provider.value}")
    print(f"   Embedding模型: {default_config.embedding_config.model}")
    
    # 测试环境变量配置
    print("\n2. 测试环境变量配置...")
    env_config = load_config_from_env()
    print(f"   LLM提供商: {env_config.llm_config.provider.value}")
    print(f"   LLM模型: {env_config.llm_config.model}")
    print(f"   Embedding提供商: {env_config.embedding_config.provider.value}")
    print(f"   Embedding模型: {env_config.embedding_config.model}")
    
    return default_config


async def test_clients(config):
    """测试客户端初始化"""
    print("\n" + "=" * 60)
    print("测试客户端初始化")
    print("=" * 60)
    
    # 测试DeepSeek客户端
    print("\n1. 测试DeepSeek客户端...")
    try:
        deepseek_client = DeepSeekClient(config)
        print(f"   ✅ DeepSeek客户端初始化成功")
        print(f"   模型: {deepseek_client.llm_config.model}")
        print(f"   API密钥: {deepseek_client.llm_config.api_key[:10]}...")
    except Exception as e:
        print(f"   ❌ DeepSeek客户端初始化失败: {e}")
        deepseek_client = None
    
    # 测试阿里云Embedding客户端
    print("\n2. 测试阿里云Embedding客户端...")
    try:
        aliyun_client = AliyunEmbeddingClient(config)
        print(f"   ✅ 阿里云Embedding客户端初始化成功")
        print(f"   模型: {aliyun_client.embedding_config.model}")
        print(f"   API密钥: {aliyun_client.embedding_config.api_key[:10]}...")
    except Exception as e:
        print(f"   ❌ 阿里云Embedding客户端初始化失败: {e}")
        aliyun_client = None
    
    return deepseek_client, aliyun_client


async def test_embedding(client, text):
    """测试嵌入向量生成"""
    if client is None:
        print("   ⚠️ 客户端未初始化，跳过测试")
        return None
    
    try:
        print(f"   生成文本嵌入: '{text[:50]}...'")
        embedding = await client.embed_text(text)
        print(f"   ✅ 嵌入向量生成成功")
        print(f"   向量维度: {len(embedding)}")
        print(f"   向量示例: {embedding[:3]}...")
        return embedding
    except Exception as e:
        print(f"   ❌ 嵌入向量生成失败: {e}")
        return None


async def test_llm(client, prompt):
    """测试LLM调用"""
    if client is None:
        print("   ⚠️ 客户端未初始化，跳过测试")
        return None
    
    try:
        print(f"   调用LLM: '{prompt[:50]}...'")
        response = await client._call_llm(prompt)
        print(f"   ✅ LLM调用成功")
        print(f"   响应长度: {len(response)} 字符")
        print(f"   响应示例: {response[:100]}...")
        return response
    except Exception as e:
        print(f"   ❌ LLM调用失败: {e}")
        return None


async def test_full_pipeline():
    """测试完整构建流程"""
    print("\n" + "=" * 60)
    print("测试完整构建流程")
    print("=" * 60)
    
    # 获取配置
    config = get_default_config()
    
    # 构建文档
    print("\n1. 构建文档...")
    base_dir = Path(__file__).resolve().parent
    default_pdf = base_dir / "附件：学员操作手册.pdf"
    
    if default_pdf.exists():
        print(f"   找到PDF文件: {default_pdf.name}")
        pdf_paths = [str(default_pdf)]
        
        try:
            documents = await build_documents_from_sources(
                pdf_paths=pdf_paths,
                config=config
            )
            print(f"   ✅ 文档构建成功: {len(documents)} 个文档")
            
            # 显示文档信息
            for doc in documents[:2]:  # 只显示前2个文档
                print(f"     文档: {doc.title}")
                print(f"     文本长度: {len(doc.text)} 字符")
                print(f"     元数据: {doc.metadata}")
            
        except Exception as e:
            print(f"   ❌ 文档构建失败: {e}")
            documents = []
    else:
        print(f"   ⚠️ 未找到PDF文件，跳过文档构建")
        documents = []
    
    # 构建文本单元
    print("\n2. 构建文本单元...")
    if documents:
        try:
            text_units = await build_text_units(documents, config, None)  # 先不测试Embedding
            print(f"   ✅ 文本单元构建成功: {len(text_units)} 个单元")
            
            # 显示文本单元信息
            for unit in text_units[:2]:  # 只显示前2个单元
                print(f"     单元 {unit.index}: {len(unit.text)} 字符")
                print(f"     分词: {unit.tokens[:5]}...")
                
        except Exception as e:
            print(f"   ❌ 文本单元构建失败: {e}")
            text_units = []
    else:
        print(f"   ⚠️ 无文档数据，跳过文本单元构建")
        text_units = []
    
    # 构建实体和关系
    print("\n3. 构建实体和关系...")
    if text_units:
        try:
            entities, relationships = await build_entities_and_relationships(
                text_units, config, None, None  # 先不测试LLM和Embedding
            )
            print(f"   ✅ 实体关系构建成功")
            print(f"     实体数: {len(entities)}")
            print(f"     关系数: {len(relationships)}")
            
            # 显示实体信息
            if entities:
                print(f"     前5个实体:")
                for entity in entities[:5]:
                    print(f"       {entity.name} ({entity.entity_type}): {entity.frequency} 次")
            
            # 显示关系信息
            if relationships:
                print(f"     前5个关系:")
                for rel in relationships[:5]:
                    print(f"       {rel.relation_type}({rel.source}→{rel.target}): 权重 {rel.weight}")
                    
        except Exception as e:
            print(f"   ❌ 实体关系构建失败: {e}")
            entities, relationships = [], []
    else:
        print(f"   ⚠️ 无文本单元数据，跳过实体关系构建")
        entities, relationships = [], []
    
    # 构建社区
    print("\n4. 构建社区...")
    if entities and relationships:
        try:
            communities = await build_communities(entities, relationships, config, None)
            print(f"   ✅ 社区构建成功: {len(communities)} 个社区")
            
            # 显示社区信息
            for community in communities[:2]:  # 只显示前2个社区
                print(f"     社区: {community.title}")
                print(f"     实体数: {len(community.entity_ids)}")
                print(f"     关系数: {len(community.relationship_ids)}")
                print(f"     摘要: {community.summary[:100]}...")
                
        except Exception as e:
            print(f"   ❌ 社区构建失败: {e}")
            communities = []
    else:
        print(f"   ⚠️ 无实体关系数据，跳过社区构建")
        communities = []
    
    return {
        "documents": len(documents),
        "text_units": len(text_units),
        "entities": len(entities),
        "relationships": len(relationships),
        "communities": len(communities)
    }


async def main():
    """主测试函数"""
    print("GraphRAG配置和客户端测试")
    print("=" * 60)
    
    # 测试配置
    config = await test_config()
    
    # 测试客户端
    llm_client, embedding_client = await test_clients(config)
    
    # 测试嵌入向量生成
    print("\n" + "=" * 60)
    print("测试嵌入向量生成")
    print("=" * 60)
    
    test_text = "Python是一种高级编程语言，广泛用于数据科学和人工智能领域。"
    await test_embedding(embedding_client, test_text)
    
    # 测试LLM调用
    print("\n" + "=" * 60)
    print("测试LLM调用")
    print("=" * 60)
    
    test_prompt = "请简要介绍Python编程语言的特点。"
    await test_llm(llm_client, test_prompt)
    
    # 测试完整流程
    results = await test_full_pipeline()
    
    # 输出总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"文档数: {results['documents']}")
    print(f"文本单元数: {results['text_units']}")
    print(f"实体数: {results['entities']}")
    print(f"关系数: {results['relationships']}")
    print(f"社区数: {results['communities']}")
    
    if results['entities'] > 0 and results['relationships'] > 0:
        print("\n✅ GraphRAG构建流程测试成功！")
    else:
        print("\n⚠️ GraphRAG构建流程测试完成，但未生成实体关系数据。")
        print("   请检查PDF文件内容和配置设置。")


if __name__ == "__main__":
    asyncio.run(main())