# 本科《Python程序设计》教育知识图谱项目

## 项目简介
本项目为本科《Python程序设计》课程构建基于GraphRAG的教育知识图谱系统，集成GraphXR实现多维可视化分析。项目从传统RAG方案演进为GraphRAG方案，提供更强大的检索、摘要和可解释问答能力。

## 核心特性
- **GraphRAG架构**：文本单元→实体→关系→社区→摘要的完整知识图谱构建流程
- **多模态处理**：支持中文PDF教材（分词+OCR）和视频课程数据处理
- **教育本体设计**：概念、技能、评测、资源等教育领域实体关系模型
- **可视化集成**：计划集成GraphXR实现多维教育图谱可视化
- **可解释问答**：基于知识图谱的答案生成，支持来源引用和推理路径

## 项目结构
```
Knowledge_Graph/
├── docs/                          # 项目文档
│   ├── 项目开发计划_本科Python知识图谱_RAG_GraphXR.md
│   ├── 项目架构文档.md
│   └── graphxr_scene.mmd
├── models/                        # 核心代码模块
│   ├── pdf_data_processor.py      # PDF中文处理（分词+OCR）
│   ├── graphrag_builder.py        # GraphRAG构建核心
│   ├── video_data_processor.py    # 视频数据处理
│   ├── main.py                    # FastAPI服务框架
│   ├── stopwords-zh.json          # 中文停用词
│   └── graphrag_output/           # GraphRAG输出数据
└── README.md                      # 本文件
```

## 🚀 快速开始

### 方法一：使用自动化脚本（推荐）

#### 1. 运行完整项目（一键式）
```bash
# 使用PowerShell（推荐）
cd "d:\code\Knowledge_Graph"
.\run.ps1

# 或使用批处理文件
.\run.bat
```

#### 2. 使用交互式菜单
```bash
cd "d:\code\Knowledge_Graph"
python run_project.py
```

### 方法二：手动分步运行

#### 步骤1：环境准备
```bash
# 切换到项目目录
cd "d:\code\Knowledge_Graph"

# 安装所有依赖（使用Anaconda py39环境）
& "C:\ProgramData\anaconda3\envs\py39\python.exe" -m pip install PyPDF2 jieba pillow pytesseract pymupdf openai numpy pydantic fastapi uvicorn
```

#### 步骤2：验证API配置
```bash
# 测试DeepSeek LLM和阿里云Embedding API
& "C:\ProgramData\anaconda3\envs\py39\python.exe" models/simple_test.py
```

#### 步骤3：构建知识图谱
```bash
# 构建GraphRAG知识图谱
& "C:\ProgramData\anaconda3\envs\py39\python.exe" -c "
import asyncio
import sys
sys.path.insert(0, 'models')
from graphrag_builder import build_graphrag_dataset

async def main():
    result = await build_graphrag_dataset(
        output_dir='models/graphrag_output',
        pdf_paths=['models/附件：学员操作手册.pdf']
    )
    print(f'✅ 构建完成：{result[\"documents\"]}个文档，{result[\"entities\"]}个实体，{result[\"relationships\"]}个关系')

asyncio.run(main())
"
```

#### 步骤4：启动API服务
```bash
# 启动FastAPI服务
& "C:\ProgramData\anaconda3\envs\py39\python.exe" -m uvicorn models.main:app --reload --host 0.0.0.0 --port 8000
```

#### 步骤5：访问服务
- **主页**: http://localhost:8000
- **API文档**: http://localhost:8000/docs
- **获取项目信息**: http://localhost:8000/items/1?q=test
- **创建项目**: POST http://localhost:8000/items/

### 方法三：使用Python脚本

#### 1. 创建运行脚本 `run_my_project.py`
```python
import asyncio
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent / "models"))

from graphrag_builder import build_graphrag_dataset

async def main():
    # 构建知识图谱
    result = await build_graphrag_dataset(
        output_dir="models/graphrag_output",
        pdf_paths=["models/附件：学员操作手册.pdf"]
    )
    
    print(f"✅ 构建完成：")
    print(f"   文档数: {result['documents']}")
    print(f"   实体数: {result['entities']}")
    print(f"   关系数: {result['relationships']}")
    
    # 查看前5个实体
    import json
    with open("models/graphrag_output/entities.jsonl", "r", encoding="utf-8") as f:
        entities = [json.loads(line) for line in f.readlines()[:5]]
    
    print(f"\n📊 前5个实体:")
    for entity in entities:
        print(f"   - {entity['name']} ({entity['type']}): {entity['count']}次")

asyncio.run(main())
```

#### 2. 运行脚本
```bash
cd "d:\code\Knowledge_Graph"
python run_my_project.py
```

### 📁 项目文件说明

| 文件/目录 | 说明 |
|-----------|------|
| `models/graphrag_builder.py` | GraphRAG构建核心模块 |
| `models/pdf_data_processor.py` | PDF中文处理（分词+OCR） |
| `models/main.py` | FastAPI服务框架 |
| `models/graphrag_output/` | 知识图谱输出数据 |
| `run_project.py` | 交互式运行脚本 |
| `run.ps1` | PowerShell一键运行脚本 |
| `run.bat` | 批处理一键运行脚本 |
| `docs/项目架构文档.md` | 完整技术架构说明 |

### 🔧 常见问题

#### Q1: 如何修改API密钥？
请通过环境变量配置 API 密钥，例如：
```powershell
$env:DEEPSEEK_API_KEY="your_deepseek_api_key"
$env:ALIYUN_EMBEDDING_API_KEY="your_aliyun_embedding_api_key"
```

#### Q2: 如何使用自己的PDF文件？
```python
result = await build_graphrag_dataset(
    output_dir="models/graphrag_output",
    pdf_paths=["path/to/your/file.pdf"]  # 替换为您的PDF文件路径
)
```

#### Q3: 如何查看生成的实体关系？
```bash
# 查看实体数据
python -c "
import json
with open('models/graphrag_output/entities.jsonl', 'r', encoding='utf-8') as f:
    for line in f.readlines()[:10]:
        data = json.loads(line)
        print(f'{data[\"name\"]} ({data[\"type\"]}): {data[\"count\"]}次')
"

# 查看关系数据
python -c "
import json
with open('models/graphrag_output/relationships.jsonl', 'r', encoding='utf-8') as f:
    for line in f.readlines()[:10]:
        data = json.loads(line)
        print(f'{data[\"source\"]} → {data[\"type\"]} → {data[\"target\"]}')
"
```

### 🎯 运行验证
运行以下命令验证项目是否正常工作：
```bash
cd "d:\code\Knowledge_Graph"
& "C:\ProgramData\anaconda3\envs\py39\python.exe" models/simple_test.py
```

预期输出：
```
✅ DeepSeek API调用成功
✅ 阿里云Embedding API调用成功
```

## 技术栈
- **核心语言**：Python 3.9
- **数据处理**：PyPDF2, PyMuPDF, jieba, pytesseract
- **知识图谱**：GraphRAG架构，RDF三元组，Neo4j（计划）
- **服务框架**：FastAPI, LangChain
- **可视化**：GraphXR（计划集成）

## 开发状态
- ✅ PDF中文教材处理模块
- ✅ GraphRAG核心构建流程
- ✅ 实体关系抽取基础实现
- 🔄 Neo4j集成（开发中）
- 🔄 GraphXR可视化（计划中）
- 🔄 API服务完善（开发中）

## 文档
- [项目开发计划](docs/项目开发计划_本科Python知识图谱_RAG_GraphXR.md) - 详细的技术架构和开发流程
- [项目架构文档](docs/项目架构文档.md) - 完整的系统架构说明

## 项目价值
- **教学应用**：课程知识结构化、学习路径推荐、教学评估
- **技术示范**：教育领域GraphRAG应用实践
- **研究价值**：教育知识图谱构建方法论探索

## 许可证
本项目为教育改革项目开发，具体许可证信息待定。

## 联系方式
项目开发中，如有问题请联系项目负责人。

---
*最后更新：2026-06-20*
