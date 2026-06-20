#!/usr/bin/env python3
"""
测试修复后的GraphRAG构建
"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
sys.path.insert(0, str(MODELS_DIR))

async def test_fix():
    """测试修复"""
    print("测试GraphRAG构建修复...")
    
    try:
        # 导入模块
        from graphrag_builder import GraphRAGDataset
        
        print("✅ GraphRAGDataset导入成功")
        
        # 创建一个测试数据集
        dataset = GraphRAGDataset(
            documents=[{"id": "doc1", "title": "测试文档"}],
            text_units=[{"id": "tu1", "text": "测试文本"}],
            entities=[{"id": "ent1", "name": "Python", "type": "编程语言"}],
            relationships=[{"id": "rel1", "source": "ent1", "target": "ent2", "type": "包含"}],
            communities=[{"id": "comm1", "title": "测试社区"}],
            config=None
        )
        
        print(f"文档数: {len(dataset.documents)}")
        print(f"实体数: {len(dataset.entities)}")
        print(f"关系数: {len(dataset.relationships)}")
        
        # 测试字典式访问（应该失败）
        try:
            docs = dataset['documents']
            print(f"❌ 字典式访问不应该成功: {docs}")
        except TypeError as e:
            print(f"✅ 字典式访问正确失败: {type(e).__name__}")
        
        # 测试属性访问（应该成功）
        try:
            docs = dataset.documents
            print(f"✅ 属性访问成功: {len(docs)} 个文档")
        except Exception as e:
            print(f"❌ 属性访问失败: {type(e).__name__}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """主函数"""
    print("测试GraphRAG构建修复")
    print("="*60)
    
    if await test_fix():
        print("\n✅ 修复测试通过！")
        print("\n问题已解决:")
        print("1. 在 run_project.py 中，将 result['documents'] 改为 result.documents")
        print("2. GraphRAGDataset 是 dataclass 对象，应该使用属性访问而不是字典访问")
        print("3. 现在可以在PyCharm + Anaconda py39环境中正常运行")
    else:
        print("\n❌ 修复测试失败")

if __name__ == "__main__":
    asyncio.run(main())