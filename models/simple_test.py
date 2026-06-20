"""
最简单的API测试
"""

import asyncio
import os
from openai import AsyncOpenAI, OpenAI

# DeepSeek API配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# 阿里云Embedding配置
ALIYUN_API_KEY = os.getenv("ALIYUN_EMBEDDING_API_KEY")
ALIYUN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


async def test_deepseek():
    """测试DeepSeek API"""
    print("测试DeepSeek API...")
    
    try:
        if not DEEPSEEK_API_KEY:
            print("DeepSeek错误: 未设置环境变量 DEEPSEEK_API_KEY")
            return False

        client = AsyncOpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL
        )
        
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": "你好，请说'测试成功'"}
            ],
            max_tokens=50
        )
        
        content = response.choices[0].message.content
        print(f"DeepSeek响应: {content}")
        return True
    except Exception as e:
        print(f"DeepSeek错误: {e}")
        return False


def test_aliyun():
    """测试阿里云Embedding API"""
    print("\n测试阿里云Embedding API...")
    
    try:
        if not ALIYUN_API_KEY:
            print("阿里云错误: 未设置环境变量 ALIYUN_EMBEDDING_API_KEY")
            return False

        client = OpenAI(
            api_key=ALIYUN_API_KEY,
            base_url=ALIYUN_BASE_URL
        )
        
        response = client.embeddings.create(
            model="text-embedding-v2",
            input="测试文本",
            dimensions=1536
        )
        
        embedding = response.data[0].embedding
        print(f"阿里云Embedding维度: {len(embedding)}")
        print(f"前3个值: {embedding[:3]}")
        return True
    except Exception as e:
        print(f"阿里云错误: {e}")
        return False


async def main():
    print("=" * 60)
    print("简单API测试")
    print("=" * 60)
    
    # 测试DeepSeek
    deepseek_ok = await test_deepseek()
    
    # 测试阿里云
    aliyun_ok = test_aliyun()
    
    print("\n" + "=" * 60)
    print("测试结果:")
    print(f"DeepSeek: {'✅ 成功' if deepseek_ok else '❌ 失败'}")
    print(f"阿里云Embedding: {'✅ 成功' if aliyun_ok else '❌ 失败'}")


# 当作为脚本运行时
if __name__ == "__main__":
    asyncio.run(main())
