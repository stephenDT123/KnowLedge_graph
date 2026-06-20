"""
测试GraphRAG构建功能
"""

import asyncio
import sys
from pathlib import Path

# 添加models目录到路径
sys.path.insert(0, str(Path.cwd() / 'models'))

async def test_graphrag_build():
    """测试GraphRAG构建"""
    print("=" * 60)
    print("测试GraphRAG构建功能")
    print("=" * 60)
    
    try:
        # 导入必要的模块
        from config import get_default_config
        from graphrag_builder import build_graphrag_dataset
        
        print("✅ 模块导入成功")
        
        # 获取默认配置
        config = get_default_config()
        print(f"✅ 配置加载成功:")
        print(f"  - LLM: {config.llm_config.provider.value}")
        print(f"  - Embedding: {config.embedding_config.provider.value}")
        
        # 设置输出目录
        output_dir = Path.cwd() / "models" / "graphrag_output"
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n📁 输出目录: {output_dir}")
        
        # 使用模拟数据构建GraphRAG
        print("\n🚀 开始构建GraphRAG...")
        
        # 构建数据集
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
        
        # 显示一些示例数据
        if dataset.text_units:
            print(f"\n📝 文本单元示例:")
            for i, unit in enumerate(dataset.text_units[:2]):
                print(f"  {i+1}. {unit.text[:100]}...")
        
        if dataset.entities:
            print(f"\n🏷️ 实体示例:")
            for i, entity in enumerate(dataset.entities[:3]):
                print(f"  {i+1}. {entity.name} ({entity.type})")
        
        if dataset.relationships:
            print(f"\n🔗 关系示例:")
            for i, relation in enumerate(dataset.relationships[:3]):
                print(f"  {i+1}. {relation.source} → {relation.target} ({relation.type})")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"❌ GraphRAG构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """主函数"""
    print("开始测试GraphRAG构建功能...")
    success = await test_graphrag_build()
    
    if success:
        print("\n🎉 GraphRAG构建测试完成!")
        print("下一步:")
        print("1. 运行完整项目: python run_project.py")
        print("2. 查看输出结果: models/graphrag_output/")
        print("3. 启动API服务: python models/main.py")
    else:
        print("\n❌ GraphRAG构建测试失败")
        print("请检查:")
        print("1. 依赖是否安装: pip install openai")
        print("2. API密钥是否正确")
        print("3. 网络连接是否正常")

if __name__ == "__main__":
    asyncio.run(main())