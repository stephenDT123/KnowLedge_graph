"""
本科《Python程序设计》教育知识图谱项目 - 简化运行脚本
避免异步循环嵌套问题
"""

import os
import sys
import asyncio
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"


def print_header(title):
    """打印标题"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def check_environment():
    """检查Python环境"""
    print_header("1. 检查Python环境")
    
    try:
        import platform
        python_version = platform.python_version()
        print(f"✅ Python版本: {python_version}")
        
        # 检查是否在Anaconda环境中
        conda_prefix = os.environ.get('CONDA_PREFIX', '')
        if conda_prefix:
            print(f"✅ 检测到Anaconda环境: {conda_prefix}")
        else:
            print("⚠️  未检测到Anaconda环境，建议使用Anaconda py39环境")
        
        return True
    except Exception as e:
        print(f"❌ 环境检查失败: {e}")
        return False


async def build_and_test():
    """构建和测试GraphRAG"""
    print_header("2. 构建GraphRAG知识图谱")
    
    try:
        # 导入GraphRAG构建模块
        sys.path.insert(0, str(MODELS_DIR))
        from graphrag_builder import build_graphrag_dataset
        
        # 检查PDF文件
        pdf_file = MODELS_DIR / "附件：学员操作手册.pdf"
        if not pdf_file.exists():
            print(f"⚠️  PDF文件不存在: {pdf_file}")
            print("将使用模拟数据进行测试...")
            pdf_paths = None
        else:
            print(f"✅ 找到PDF文件: {pdf_file.name}")
            pdf_paths = [str(pdf_file)]
        
        # 构建知识图谱
        print("开始构建知识图谱...")
        result = await build_graphrag_dataset(
            output_dir=str(MODELS_DIR / "graphrag_output"),
            pdf_paths=pdf_paths,
            config=None  # 使用默认配置
        )
        
        print(f"✅ GraphRAG构建完成")
        print(f"   文档数: {result['documents']}")
        print(f"   文本单元数: {result['text_units']}")
        print(f"   实体数: {result['entities']}")
        print(f"   关系数: {result['relationships']}")
        print(f"   社区数: {result['communities']}")
        
        # 显示前5个实体
        entities_file = MODELS_DIR / "graphrag_output" / "entities.jsonl"
        if entities_file.exists():
            import json
            with open(entities_file, 'r', encoding='utf-8') as f:
                entities = [json.loads(line) for line in f.readlines()[:5]]
            print(f"\n📊 前5个实体:")
            for entity in entities:
                print(f"  - {entity['name']} ({entity['type']}): {entity['count']}次")
        
        return True
    except Exception as e:
        print(f"❌ GraphRAG构建失败: {e}")
        return False


def show_instructions():
    """显示使用说明"""
    print_header("使用说明")
    
    print("🎯 项目已成功运行！")
    print("\n📁 输出文件位置:")
    print(f"   {MODELS_DIR / 'graphrag_output'}")
    
    print("\n🚀 下一步操作:")
    print("1. 查看生成的实体:")
    print("   python -c \"")
    print("   import json")
    print("   with open('models/graphrag_output/entities.jsonl', 'r', encoding='utf-8') as f:")
    print("       for line in f.readlines()[:10]:")
    print("           data = json.loads(line)")
    print("           print(f'{data[\\\"name\\\"]} ({data[\\\"type\\\"]}): {data[\\\"count\\\"]}次')")
    print("   \"")
    
    print("\n2. 启动FastAPI服务:")
    print("   python -m uvicorn models.main:app --reload --host 0.0.0.0 --port 8000")
    
    print("\n3. 访问API文档:")
    print("   http://localhost:8000/docs")


async def main():
    """主函数"""
    print_header("本科《Python程序设计》教育知识图谱项目")
    print("简化运行脚本")
    print("=" * 60)
    
    try:
        # 检查环境
        if not check_environment():
            print("❌ 环境检查失败，请修复问题后重试")
            return
        
        # 构建和测试
        success = await build_and_test()
        
        if success:
            print("\n" + "=" * 60)
            print("🎉 项目运行成功！")
            print("=" * 60)
            
            show_instructions()
        else:
            print("\n" + "=" * 60)
            print("⚠️  项目运行遇到问题")
            print("=" * 60)
            print("请检查:")
            print("1. API密钥是否正确")
            print("2. 网络连接是否正常")
            print("3. 依赖包是否已安装")
            
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n❌ 程序运行异常: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 运行主程序
    asyncio.run(main())