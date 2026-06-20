"""
调试GraphRAG构建问题
"""

import asyncio
import sys
from pathlib import Path

# 添加models目录到路径
sys.path.insert(0, str(Path.cwd() / 'models'))

async def debug_type_error():
    """调试类型错误"""
    print("调试'type' object is not subscriptable错误...")
    print("=" * 60)
    
    try:
        # 1. 测试导入
        print("1. 测试模块导入...")
        from config import get_default_config
        from graphrag_builder import EntityExtractionResult, RelationExtractionResult
        print("   ✅ 模块导入成功")
        
        # 2. 测试EntityExtractionResult
        print("\n2. 测试EntityExtractionResult...")
        test_entities = [
            {"name": "Python", "type": "编程语言"},
            {"name": "变量", "type": "概念"}
        ]
        
        entity_result = EntityExtractionResult(
            entities=test_entities,
            confidence=0.95
        )
        print(f"   ✅ EntityExtractionResult创建成功")
        print(f"     实体数量: {len(entity_result.entities)}")
        
        # 3. 测试访问实体数据
        print("\n3. 测试访问实体数据...")
        for i, entity in enumerate(entity_result.entities):
            print(f"   实体{i+1}: name={entity.get('name')}, type={entity.get('type')}")
        
        # 4. 测试RelationExtractionResult
        print("\n4. 测试RelationExtractionResult...")
        test_relationships = [
            {"source": "Python", "target": "变量", "type": "包含"}
        ]
        
        relation_result = RelationExtractionResult(
            relationships=test_relationships,
            confidence=0.90
        )
        print(f"   ✅ RelationExtractionResult创建成功")
        
        # 5. 测试JSON序列化
        print("\n5. 测试JSON序列化...")
        entity_json = entity_result.model_dump_json()
        print(f"   ✅ JSON序列化成功: {entity_json[:100]}...")
        
        # 6. 测试从JSON反序列化
        print("\n6. 测试JSON反序列化...")
        entity_result2 = EntityExtractionResult.model_validate_json(entity_json)
        print(f"   ✅ JSON反序列化成功")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 调试失败: {e}")
        import traceback
        print("详细错误信息:")
        traceback.print_exc()
        return False

async def test_simple_build():
    """测试简单构建"""
    print("\n" + "=" * 60)
    print("测试简单GraphRAG构建...")
    
    try:
        from config import get_default_config
        from graphrag_builder import build_graphrag_dataset
        
        print("✅ 模块导入成功")
        
        # 获取默认配置
        config = get_default_config()
        print(f"✅ 配置加载成功")
        print(f"  LLM提供商: {config.llm_config.provider.value}")
        print(f"  Embedding提供商: {config.embedding_config.provider.value}")
        
        # 设置输出目录
        output_dir = Path.cwd() / "models" / "graphrag_output"
        output_dir.mkdir(exist_ok=True)
        
        print(f"📁 输出目录: {output_dir}")
        
        # 使用模拟数据构建
        print("\n🚀 开始构建GraphRAG...")
        dataset = await build_graphrag_dataset(
            output_dir=output_dir,
            pdf_paths=None,  # 使用模拟数据
            video_paths=None,  # 使用模拟数据
            config=config
        )
        
        print(f"\n✅ GraphRAG构建成功!")
        print(f"📊 生成的数据集:")
        print(f"  - 文本单元: {len(dataset.text_units)} 个")
        print(f"  - 实体: {len(dataset.entities)} 个")
        print(f"  - 关系: {len(dataset.relationships)} 个")
        print(f"  - 社区: {len(dataset.communities)} 个")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 构建失败: {e}")
        import traceback
        print("详细错误信息:")
        traceback.print_exc()
        return False

async def main():
    """主函数"""
    print("开始调试GraphRAG构建问题...")
    
    # 调试类型错误
    debug_ok = await debug_type_error()
    
    if debug_ok:
        # 测试简单构建
        build_ok = await test_simple_build()
        
        print("\n" + "=" * 60)
        print("调试结果:")
        print(f"类型错误调试: {'✅ 成功' if debug_ok else '❌ 失败'}")
        print(f"简单构建测试: {'✅ 成功' if build_ok else '❌ 失败'}")
        
        if debug_ok and build_ok:
            print("\n🎉 所有测试通过!")
            print("\n现在可以正常运行项目:")
            print("1. 在PyCharm中打开 run_project.py")
            print("2. 使用Anaconda py39环境运行")
            print("3. 选择选项4: 仅构建GraphRAG")
        else:
            print("\n⚠️  部分测试失败，请查看详细错误信息")
    else:
        print("\n❌ 类型错误调试失败，请查看详细错误信息")

if __name__ == "__main__":
    asyncio.run(main())