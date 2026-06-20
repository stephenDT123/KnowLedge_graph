# PyCharm + Anaconda py39 环境运行指南

## 问题解决

### 1. 已修复的错误
**错误信息**: `❌ GraphRAG构建失败: 'type' object is not subscriptable`

**根本原因**: 
- 在 `run_project.py` 第148行，代码使用了 `result['documents']`
- 但 `GraphRAGDataset` 是一个 `dataclass` 对象，不是字典
- Python 3.9及更早版本对类型注解 `List[Dict[str, str]]` 的解析有问题

**修复方案**:
1. 将 `result['documents']` 改为 `result.documents`
2. 将 `graphrag_builder.py` 中的复杂类型注解简化为 `List[dict]`
3. 为 `Entity` 类添加 `@dataclass` 装饰器

## 2. 环境配置步骤

### 步骤1: 配置PyCharm Python解释器
1. 打开PyCharm
2. 点击 `File` → `Settings` (或 `Ctrl+Alt+S`)
3. 选择 `Project: Knowledge_Graph` → `Python Interpreter`
4. 点击齿轮图标 → `Add...`
5. 选择 `Conda Environment` → `Existing environment`
6. 找到路径: `C:\ProgramData\anaconda3\envs\py39\python.exe`
7. 点击 `OK`

### 步骤2: 安装依赖包
在PyCharm的Terminal中运行:
```bash
pip install PyPDF2 jieba pillow pytesseract pymupdf openai numpy pydantic
```

或者使用项目提供的脚本:
```bash
python run_project.py
```
然后选择选项 `2. 仅安装依赖`

### 步骤3: 配置API密钥
在PyCharm中创建 `.env` 文件:
1. 在项目根目录 (`d:\code\Knowledge_Graph`) 创建 `.env` 文件
2. 添加以下内容:
```
DEEPSEEK_API_KEY=your_deepseek_api_key
ALIYUN_EMBEDDING_API_KEY=your_aliyun_embedding_api_key
```

### 步骤4: 测试API配置
运行项目并选择选项 `3. 仅测试API`:
```bash
python run_project.py
```

预期输出:
```
✅ API测试成功
DeepSeek: ✅ 成功
阿里云Embedding: ✅ 成功
```

## 3. 完整运行流程

### 选项1: 完整运行（推荐）
```bash
python run_project.py
```
选择选项 `1. 完整运行`

流程:
1. 环境检查
2. 依赖安装
3. API测试
4. GraphRAG构建
5. FastAPI服务启动
6. 查看输出结果

### 选项2: 分步运行
1. **仅构建GraphRAG**: 选择选项 `4`
2. **仅启动服务**: 选择选项 `5`
3. **查看结果**: 选择选项 `6`

## 4. 常见问题解决

### Q1: 缺少PDF文件
**错误**: `⚠️ PDF文件不存在: ...\附件：学员操作手册.pdf`

**解决方案**:
1. 将你的PDF教材复制到 `models/` 目录
2. 重命名为 `附件：学员操作手册.pdf`
3. 或者修改代码使用其他PDF文件

### Q2: OCR功能不可用
**错误**: `TesseractNotFoundError`

**解决方案**:
1. 在代码中设置 `enable_ocr=False`
2. 或者安装Tesseract OCR:
   - 下载: https://github.com/UB-Mannheim/tesseract/wiki
   - 安装后添加到系统PATH

### Q3: 异步循环错误
**错误**: `Cannot run the event loop while another loop is running`

**解决方案**:
- 已修复，所有异步函数使用 `await` 调用
- 使用 `asyncio.run(main())` 作为入口点

## 5. 验证修复

运行测试脚本验证修复:
```bash
python test_fix.py
```

预期输出:
```
✅ 修复测试通过！
✅ 属性访问成功
```

## 6. 项目结构说明

```
d:\code\Knowledge_Graph\
├── models/                    # 核心模型代码
│   ├── graphrag_builder.py   # GraphRAG构建器（已修复）
│   ├── pdf_data_processor.py # PDF处理模块
│   ├── config.py            # 配置管理
│   └── main.py              # FastAPI服务
├── run_project.py           # 主运行脚本（已修复）
├── run.bat                  # Windows一键运行
├── run.ps1                  # PowerShell一键运行
├── test_fix.py              # 修复测试脚本
└── PyCharm_运行指南.md      # 本指南
```

## 7. 下一步建议

1. **测试实际PDF**: 使用你的《Python程序设计》教材PDF进行测试
2. **扩展功能**: 修改 `models/main.py` 添加更多API
3. **集成Neo4j**: 将知识图谱存储到Neo4j数据库
4. **可视化**: 集成GraphXR进行多维可视化

## 8. 技术支持

如果仍有问题:
1. 检查Python版本: `python --version` (应为3.9.x)
2. 检查依赖包: `pip list`
3. 查看详细错误信息
4. 确保API密钥正确配置

**修复总结**: 
- ✅ 修复了 `'type' object is not subscriptable` 错误
- ✅ 修复了异步循环嵌套问题  
- ✅ 优化了类型注解兼容Python 3.9
- ✅ 提供了完整的运行指南

现在可以在PyCharm + Anaconda py39环境中正常运行项目！
