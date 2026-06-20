"""
简化API测试脚本
验证DeepSeek LLM和阿里云Embedding API调用
"""

import asyncio
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from config import get_default_config
from graphrag_builder import DeepSeekClient, AliyunEmbeddingClient


async def test_deepseek_api():
    """测试DeepSeek API"""
    print("=" * 60)
    print("测试DeepSeek API")
    print("=" * 60)
    
    config = get_default_config()
    
    try:
        # 初始化客户端
        client = DeepSeekClient(config)
        print(f"✅ DeepSeek客户端初始化成功")
        print(f"   模型: {client.llm_config.model}")
        print(f"   API密钥: {client.llm_config.api_key[:10]}...")
        
        # 测试简单调用
        prompt = "请简要介绍Python编程语言的特点。"
        print(f"\n   调用DeepSeek API...")
        print(f"   提示词: {prompt}")
        
        response = await client._call_llm(prompt)
        print(f"   ✅ DeepSeek API调用成功")
        print(f"   响应长度: {len(response)} 字符")
        print(f"   响应内容: {response[:200]}...")
        
        return True
    except Exception as e:
        print(f"   ❌ DeepSeek API调用失败: {e}")
        return False


async def test_aliyun_embedding():
    """测试阿里云Embedding API"""
    print("\n" + "=" * 60)
    print("测试阿里云Embedding API")
    print("=" * 60)
    
    config = get_default_config()
    
    try:
        # 初始化客户端
        client = AliyunEmbeddingClient(config)
        print(f"✅ 阿里云Embedding客户端初始化成功")
        print(f"   模型: {client.embedding_config.model}")
        print(f"   API密钥: {client.embedding_config.api_key[:10]}...")
        
        # 测试嵌入向量生成
        text = "Python是一种高级编程语言，广泛用于数据科学和人工智能领域。"
        print(f"\n   生成文本嵌入...")
        print(f"   文本: {text}")
        
        embedding = await client.embed_text(text)
        print(f"   ✅ 阿里云Embedding API调用成功")
        print(f"   向量维度: {len(embedding)}")
        print(f"   向量示例: {embedding[:3]}...")
        
        # 检查是否为全零向量
        if all(v == 0.0 for v in embedding[:10]):
            print(f"   ⚠️  警告: 向量可能为全零，请检查API密钥和配置")
        
        return True
    except Exception as e:
        print(f"   ❌ 阿里云Embedding API调用失败: {e}")
        return False


async def main():
    """主测试函数"""
    print("GraphRAG API测试")
    print("=" * 60)
    
    # 测试DeepSeek API
    deepseek_success = await test_deepseek_api()
    
    # 测试阿里云Embedding API
    aliyun_success = await test_aliyun_embedding()
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"DeepSeek API: {'✅ 成功' if deepseek_success else '❌ 失败'}")
    print(f"阿里云Embedding API: {'✅ 成功' if aliyun_success else '❌ 失败'}")
    
    if deepseek_success and aliyun_success:
        print("\n🎉 所有API测试成功！GraphRAG配置正确。")
    else:
        print("\n⚠️  部分API测试失败，请检查配置和网络连接。")


if __name__ == "__main__":
    asyncio.run(main())