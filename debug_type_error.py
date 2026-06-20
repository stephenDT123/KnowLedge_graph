#!/usr/bin/env python3
"""
调试 'type' object is not subscriptable 错误
"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
sys.path.insert(0, str(MODELS_DIR))

async def test_graphrag_builder():
    """测试GraphRAG构建器"""
    print("测试GraphRAG构建器...")
    
    try:
        # 导入模块
        from graphrag_builder import (
            EntityExtractionResult, 
            RelationExtractionResult,
            GraphRAGDataset,
            build_graphrag_dataset
        )
        
        print("✅ 模块导入成功")
        
        # 测试EntityExtractionResult
        print("\n测试EntityExtractionResult...")
        test_entities = [
            {"name": "Python", "type": "编程语言"},
            {"name": "变量", "type": "概念"}
        ]
        
        entity_result = EntityExtractionResult(
            entities=test_entities,
            confidence=0.95
        )
        
        print(f"   实体数: {len(entity_result.entities)}")
        for i, entity in enumerate(entity_result.entities):
            print(f"   实体{i+1}: name={entity.get('name')}, type={entity.get('type')}")
        
        # 测试RelationExtractionResult
        print("\n测试RelationExtractionResult...")
        test_relations = [
            {"source": "Python", "target": "变量", "type": "包含"},
            {"source": "变量", "target": "赋值", "type": "先修"}
        ]
        
        relation_result = RelationExtractionResult(
            relationships=test_relations,
            confidence=0.88
        )
        
        print(f"   关系数: {len(relation_result.relationships)}")
        for i, rel in enumerate(relation_result.relationships):
            print(f"   关系{i+1}: {rel.get('source')} → {rel.get('target')} ({rel.get('type')})")
        
        # 测试GraphRAGDataset访问
        print("\n测试GraphRAGDataset访问...")
        
        # 创建一个简单的GraphRAGDataset对象
        dataset = GraphRAGDataset(
            documents=[],
            text_units=[],
            entities=[],
            relationships=[],
            communities=[],
            config=None
        )
        
        # 测试属性访问
        print(f"   文档数: {len(dataset.documents)}")
        print(f"   实体数: {len(dataset.entities)}")
        
        # 测试字典式访问（这是错误的）
        print("\n测试字典式访问（应该会失败）...")
        try:
            doc_count = dataset['documents']
            print(f"   dataset['documents'] = {doc_count}")
        except Exception as e:
            print(f"   ❌ 字典式访问失败: {type(e).__name__}: {e}")
        
        # 测试属性访问（这是正确的）
        print("\n测试属性访问（应该成功）...")
        try:
            doc_count = len(dataset.documents)
            print(f"   len(dataset.documents) = {doc_count}")
            print("   ✅ 属性访问成功")
        except Exception as e:
            print(f"   ❌ 属性访问失败: {type(e).__name__}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_build_function():
    """测试构建函数"""
    print("\n" + "="*60)
    print("测试build_graphrag_dataset函数...")
    
    try:
        from graphrag_builder import build_graphrag_dataset
        
        # 使用模拟数据测试
        result = await build_graphrag_dataset(
            output_dir=str(MODELS_DIR / "test_output"),
            pdf_paths=None,  # 不使用实际PDF
            config=None
        )
        
        print(f"✅ 构建成功")
        print(f"   文档数: {len(result.documents)}")
        print(f"   文本单元数: {len(result.text_units)}")
        print(f"   实体数: {len(result.entities)}")
        print(f"   关系数: {len(result.relationships)}")
        print(f"   社区数: {len(result.communities)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 构建测试失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """主函数"""
    print("调试 'type' object is not subscriptable 错误")
    print("="*60)
    
    # 测试1：模块导入和基本功能
    test1_ok = await test_graphrag_builder()
    
    # 测试2：构建函数
    test2_ok = await test_build_function()
    
    print("\n" + "="*60)
    print("测试结果:")
    print(f"  测试1（模块导入）: {'✅ 通过' if test1_ok else '❌ 失败'}")
    print(f"  测试2（构建函数）: {'✅ 通过' if test2_ok else '❌ 失败'}")
    
    if test1_ok and test2_ok:
        print("\n✅ 所有测试通过！")
        print("\n问题诊断:")
        print("1. 在 run_project.py 第148行，代码使用了 dataset['documents']")
        print("2. 但 GraphRAGDataset 是一个 dataclass 对象，不是字典")
        print("3. 应该使用 dataset.documents 来访问属性")
    else:
        print("\n❌ 测试失败，需要进一步调试")

if __name__ == "__main__":
    asyncio.run(main())