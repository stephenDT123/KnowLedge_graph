#!/usr/bin/env python3
"""
快速测试脚本 - 验证GraphRAG构建修复
"""

import asyncio
import sys
import os
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
sys.path.insert(0, str(MODELS_DIR))

def print_header(title):
    """打印标题"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

async def quick_test():
    """快速测试"""
    print_header("快速测试GraphRAG构建修复")
    
    print("1. 检查Python环境...")
    print(f"   Python版本: {sys.version}")
    print(f"   工作目录: {os.getcwd()}")
    
    print("\n2. 检查模块导入...")
    try:
        from graphrag_builder import GraphRAGDataset
        print("   ✅ GraphRAGDataset导入成功")
    except Exception as e:
        print(f"   ❌ 导入失败: {e}")
        return False
    
    print("\n3. 测试属性访问...")
    try:
        # 创建测试对象
        dataset = GraphRAGDataset(
            documents=[{"id": "doc1", "title": "测试"}],
            text_units=[{"id": "tu1", "text": "测试"}],
            entities=[{"id": "ent1", "name": "Python"}],
            relationships=[{"id": "rel1", "source": "a", "target": "b"}],
            communities=[{"id": "comm1", "title": "测试社区"}],
            config=None
        )
        
        # 测试属性访问
        doc_count = len(dataset.documents)
        ent_count = len(dataset.entities)
        
        print(f"   ✅ 属性访问成功")
        print(f"   文档数: {doc_count}")
        print(f"   实体数: {ent_count}")
        
    except Exception as e:
        print(f"   ❌ 属性访问失败: {type(e).__name__}: {e}")
        return False
    
    print("\n4. 测试字典访问（应该失败）...")
    try:
        dataset = GraphRAGDataset(
            documents=[], text_units=[], entities=[], 
            relationships=[], communities=[], config=None
        )
        docs = dataset['documents']
        print(f"   ❌ 字典访问不应该成功")
        return False
    except TypeError:
        print("   ✅ 字典访问正确失败（这是预期的）")
    except Exception as e:
        print(f"   ⚠️  其他错误: {e}")
    
    print("\n5. 检查PDF文件...")
    pdf_file = MODELS_DIR / "附件：学员操作手册.pdf"
    if pdf_file.exists():
        print(f"   ✅ 找到PDF文件: {pdf_file.name}")
        print(f"   文件大小: {pdf_file.stat().st_size:,} 字节")
    else:
        print(f"   ⚠️  PDF文件不存在: {pdf_file.name}")
        print("   将使用模拟数据进行构建测试")
    
    return True

async def test_build():
    """测试构建功能"""
    print_header("测试GraphRAG构建")
    
    try:
        from graphrag_builder import build_graphrag_dataset
        
        print("开始构建测试...")
        
        # 使用模拟数据测试
        result = await build_graphrag_dataset(
            output_dir=str(MODELS_DIR / "quick_test_output"),
            pdf_paths=None,  # 不使用实际PDF
            config=None
        )
        
        print(f"✅ 构建测试成功")
        print(f"   文档数: {len(result.documents)}")
        print(f"   文本单元数: {len(result.text_units)}")
        print(f"   实体数: {len(result.entities)}")
        print(f"   关系数: {len(result.relationships)}")
        print(f"   社区数: {len(result.communities)}")
        
        # 清理测试输出
        import shutil
        test_dir = MODELS_DIR / "quick_test_output"
        if test_dir.exists():
            shutil.rmtree(test_dir)
            print(f"   已清理测试输出目录")
        
        return True
        
    except Exception as e:
        print(f"❌ 构建测试失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """主函数"""
    print("GraphRAG构建修复验证")
    print("="*60)
    
    # 运行快速测试
    test1_ok = await quick_test()
    
    if test1_ok:
        print("\n" + "="*60)
        print("✅ 基础测试通过！")
        print("="*60)
        
        # 询问是否运行构建测试
        response = input("\n是否运行完整的构建测试？(y/n): ").strip().lower()
        if response == 'y':
            test2_ok = await test_build()
            if test2_ok:
                print("\n🎉 所有测试通过！")
                print("\n现在可以:")
                print("1. 运行 python run_project.py 选择选项4构建GraphRAG")
                print("2. 使用自己的PDF教材进行测试")
                print("3. 访问 http://localhost:8000/docs 查看API")
            else:
                print("\n⚠️  构建测试失败，请检查API配置")
        else:
            print("\n✅ 基础功能正常")
            print("可以运行 python run_project.py 进行完整测试")
    else:
        print("\n❌ 基础测试失败，请检查环境配置")
    
    print("\n" + "="*60)
    print("运行指南:")
    print("1. 确保使用Anaconda py39环境")
    print("2. 安装依赖: pip install PyPDF2 jieba pillow openai numpy pydantic")
    print("3. 配置API密钥在 .env 文件中")
    print("4. 运行: python run_project.py")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())