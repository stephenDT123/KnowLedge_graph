"""
本科《Python程序设计》教育知识图谱项目 - 运行脚本
提供完整的项目运行指南和自动化脚本
"""

import os
import sys
import subprocess
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


def install_dependencies():
    """安装项目依赖"""
    print_header("2. 安装项目依赖")
    
    dependencies = [
        "PyPDF2",
        "jieba", 
        "pillow",
        "pytesseract",
        "pymupdf",
        "openai",
        "neo4j",
        "numpy",
        "pydantic",
        "fastapi",
        "uvicorn"
    ]
    
    print("需要安装的依赖包:")
    for dep in dependencies:
        print(f"  - {dep}")
    
    # 检查是否使用Anaconda环境
    python_exe = sys.executable
    if "anaconda3" in python_exe.lower():
        print(f"\n使用Anaconda Python: {python_exe}")
    else:
        print(f"\n使用系统Python: {python_exe}")
        print("⚠️  建议使用Anaconda py39环境: & \"C:\\ProgramData\\anaconda3\\envs\\py39\\python.exe\"")
    
    # 询问是否安装
    response = input("\n是否安装依赖包？(y/n): ").strip().lower()
    if response != 'y':
        print("跳过依赖安装")
        return True
    
    try:
        # 安装依赖
        for dep in dependencies:
            print(f"安装 {dep}...")
            subprocess.run([python_exe, "-m", "pip", "install", dep], check=True)
        
        print("✅ 依赖安装完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖安装失败: {e}")
        return False


async def test_apis():
    """测试API配置"""
    print_header("3. 测试API配置")
    
    try:
        print("运行API测试...")
        
        # 直接导入并运行测试函数
        sys.path.insert(0, str(MODELS_DIR))
        from simple_test import test_deepseek, test_aliyun
        
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
        
        return deepseek_ok and aliyun_ok
        
    except Exception as e:
        print(f"❌ API测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False


async def build_graphrag():
    """构建GraphRAG知识图谱"""
    print_header("4. 构建GraphRAG知识图谱")
    
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
        print(f"   文档数: {len(result.documents)}")
        print(f"   文本单元数: {len(result.text_units)}")
        print(f"   实体数: {len(result.entities)}")
        print(f"   关系数: {len(result.relationships)}")
        print(f"   社区数: {len(result.communities)}")
        
        # 显示前5个实体
        entities_file = MODELS_DIR / "graphrag_output" / "entities.jsonl"
        if entities_file.exists():
            import json
            with open(entities_file, 'r', encoding='utf-8') as f:
                entities = [json.loads(line) for line in f.readlines()[:5]]
            print(f"\n前5个实体:")
            for entity in entities:
                print(
                    f"  - {entity.get('name', '未知实体')} "
                    f"({entity.get('entity_type', '未知类型')}): "
                    f"{entity.get('frequency', 0)}次"
                )
        
        return True
    except Exception as e:
        print(f"❌ GraphRAG构建失败: {e}")
        return False


def start_fastapi():
    """启动FastAPI服务"""
    print_header("5. 启动FastAPI服务")
    
    print("启动命令:")
    print(f"  cd \"{PROJECT_ROOT}\"")
    print(f"  & \"{sys.executable}\" -m uvicorn models.main:app --reload --host 0.0.0.0 --port 8000")
    
    print("\n访问地址:")
    print("  - 主页: http://localhost:8000")
    print("  - API文档: http://localhost:8000/docs")
    print("  - 获取项目信息: http://localhost:8000/items/1?q=test")
    print("  - 创建项目: POST http://localhost:8000/items/")
    
    # 询问是否启动服务
    response = input("\n是否启动FastAPI服务？(y/n): ").strip().lower()
    if response != 'y':
        print("跳过服务启动")
        return True
    
    try:
        # 切换到项目目录
        os.chdir(PROJECT_ROOT)
        
        # 启动FastAPI服务
        print("启动FastAPI服务...")
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "models.main:app", 
            "--reload", 
            "--host", "0.0.0.0", 
            "--port", "8000"
        ])
        
        return True
    except KeyboardInterrupt:
        print("\n服务已停止")
        return True
    except Exception as e:
        print(f"❌ 服务启动失败: {e}")
        return False


def view_outputs():
    """查看输出结果"""
    print_header("6. 查看输出结果")
    
    output_dir = MODELS_DIR / "graphrag_output"
    
    if not output_dir.exists():
        print(f"⚠️  输出目录不存在: {output_dir}")
        print("请先运行GraphRAG构建")
        return False
    
    print("输出文件:")
    for file in output_dir.glob("*.jsonl"):
        print(f"  - {file.name}")
    
    # 查看实体数据
    entities_file = output_dir / "entities.jsonl"
    if entities_file.exists():
        try:
            import json
            with open(entities_file, 'r', encoding='utf-8') as f:
                entities = [json.loads(line) for line in f.readlines()[:5]]
            
            print(f"\n前5个实体数据:")
            print(json.dumps(entities, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"读取实体数据失败: {e}")
    
    return True


def import_to_neo4j():
    """将 GraphRAG 输出导入 Neo4j"""
    print_header("7. 导入Neo4j数据库")

    output_dir = MODELS_DIR / "graphrag_output"
    if not output_dir.exists():
        print(f"⚠️  GraphRAG 输出目录不存在: {output_dir}")
        print("请先运行 GraphRAG 构建")
        return False

    entities_file = output_dir / "entities.jsonl"
    relationships_file = output_dir / "relationships.jsonl"
    if not entities_file.exists() or not relationships_file.exists():
        print("⚠️  缺少核心输出文件，无法导入 Neo4j")
        print(f"需要存在: {entities_file.name} 和 {relationships_file.name}")
        return False

    try:
        sys.path.insert(0, str(MODELS_DIR))
        from neo4j_importer import Neo4jGraphImporter, normalize_neo4j_uri
    except Exception as e:
        print(f"❌ 无法加载 Neo4j 导入模块: {e}")
        print("请先安装依赖: pip install neo4j")
        return False

    default_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    default_user = os.getenv("NEO4J_USER", "neo4j")
    default_database = os.getenv("NEO4J_DATABASE", "neo4j")

    print("请输入 Neo4j 连接信息，直接回车可使用默认值。")
    uri = input(f"Neo4j URI [{default_uri}]: ").strip() or default_uri
    user = input(f"用户名 [{default_user}]: ").strip() or default_user
    print("请输入 Neo4j 密码。为兼容 PyCharm 控制台，这里采用可见输入。")
    password = input("密码: ").strip()
    if not password:
        password = os.getenv("NEO4J_PASSWORD", "").strip()

    if not password:
        print("❌ 未提供 Neo4j 密码")
        return False

    database = input(f"数据库名 [{default_database}]: ").strip() or default_database
    include_embeddings = input("是否导入 embedding 向量？(y/n, 默认n): ").strip().lower() == "y"
    clear_existing = input("导入前是否清空已有图谱节点？(y/n, 默认n): ").strip().lower() == "y"

    normalized_uri = normalize_neo4j_uri(uri)
    if normalized_uri != uri:
        print(f"检测到本地单机 Neo4j，已将连接协议从 {uri} 自动调整为 {normalized_uri}")
    uri = normalized_uri

    print("\n开始导入 Neo4j...")
    print(f"  URI: {uri}")
    print(f"  用户名: {user}")
    print(f"  数据库: {database}")
    print(f"  输入目录: {output_dir}")
    print(f"  导入 embedding: {'是' if include_embeddings else '否'}")
    print(f"  清空旧图谱: {'是' if clear_existing else '否'}")

    importer = None
    try:
        importer = Neo4jGraphImporter(
            uri=uri,
            user=user,
            password=password,
            database=database,
            include_embeddings=include_embeddings,
        )
        summary = importer.import_from_directory(
            input_dir=output_dir,
            clear_existing=clear_existing
        )

        print("✅ Neo4j 导入完成")
        print(f"   文档数: {summary['documents']}")
        print(f"   文本单元数: {summary['text_units']}")
        print(f"   实体数: {summary['entities']}")
        print(f"   关系数: {summary['relationships']}")
        print(f"   社区数: {summary['communities']}")
        print("\n可在 Neo4j Browser 中执行示例查询:")
        print("  MATCH (n:Entity) RETURN count(n);")
        print("  MATCH ()-[r]->() RETURN type(r), count(r) ORDER BY count(r) DESC;")
        return True
    except Exception as e:
        print(f"❌ Neo4j 导入失败: {e}")
        print("提示: 本地 Neo4j Desktop/单机服务通常使用 bolt://127.0.0.1:7687")
        print("提示: 若数据库名不存在，请将数据库名改为默认的 neo4j 后重试")
        return False
    finally:
        if importer is not None:
            try:
                importer.close()
            except Exception:
                pass


async def main_menu():
    """主菜单"""
    print_header("教育知识图谱项目")
    print("项目运行指南")
    print("\n请选择操作:")
    print("1. 完整运行（环境检查→依赖安装→API测试→构建→服务）")
    print("2. 仅安装依赖")
    print("3. 仅测试API")
    print("4. 仅构建GraphRAG")
    print("5. 仅启动FastAPI服务")
    print("6. 查看输出结果")
    print("7. 导入Neo4j数据库")
    print("8. 退出")
    
    choice = input("\n请输入选项 (1-8): ").strip()
    
    if choice == "1":
        # 完整运行
        check_environment()
        install_dependencies()
        await test_apis()
        
        # 异步运行GraphRAG构建
        await build_graphrag()
        
        start_fastapi()
        view_outputs()
        import_response = input("\n是否继续导入 Neo4j 数据库？(y/n): ").strip().lower()
        if import_response == "y":
            import_to_neo4j()
        
    elif choice == "2":
        install_dependencies()
    elif choice == "3":
        await test_apis()
    elif choice == "4":
        await build_graphrag()
    elif choice == "5":
        start_fastapi()
    elif choice == "6":
        view_outputs()
    elif choice == "7":
        import_to_neo4j()
    elif choice == "8":
        print("退出程序")
        return False
    else:
        print("无效选项")
    
    return True


async def main():
    """主函数"""
    try:
        # 显示欢迎信息
        print_header("欢迎使用教育知识图谱项目")
        print("项目位置: d:\\code\\Knowledge_Graph")
        print("Python环境: " + sys.executable)
        
        # 运行主菜单
        while await main_menu():
            pass
        
        print("\n🎉 项目运行完成！")
        print("\n下一步建议:")
        print("1. 访问 http://localhost:8000/docs 查看API文档")
        print("2. 修改 models/main.py 添加更多API功能")
        print("3. 使用自己的PDF教材进行测试")
        print("4. 集成Neo4j数据库进行持久化存储")
        
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n❌ 程序运行异常: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 运行主程序
    asyncio.run(main())
