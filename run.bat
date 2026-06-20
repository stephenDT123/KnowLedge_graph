@echo off
echo ========================================
echo 本科《Python程序设计》教育知识图谱项目
echo ========================================
echo.

REM 设置Python路径
set PYTHON_PATH=C:\ProgramData\anaconda3\envs\py39\python.exe

REM 检查Python环境
echo 1. 检查Python环境...
if exist "%PYTHON_PATH%" (
    echo   使用Anaconda Python: %PYTHON_PATH%
) else (
    echo   ⚠️  Anaconda Python未找到，使用系统Python
    set PYTHON_PATH=python
)

REM 安装依赖
echo.
echo 2. 安装项目依赖...
echo   安装PyPDF2, jieba, pillow, pytesseract, pymupdf, openai, numpy, pydantic...
%PYTHON_PATH% -m pip install PyPDF2 jieba pillow pytesseract pymupdf openai numpy pydantic fastapi uvicorn

REM 测试API
echo.
echo 3. 测试API配置...
%PYTHON_PATH% models\simple_test.py

REM 构建GraphRAG
echo.
echo 4. 构建GraphRAG知识图谱...
%PYTHON_PATH% -c "
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, 'models')
from graphrag_builder import build_graphrag_dataset

async def main():
    result = await build_graphrag_dataset(
        output_dir='models/graphrag_output',
        pdf_paths=['models/附件：学员操作手册.pdf']
    )
    print(f'构建完成：{result[\"documents\"]}个文档，{result[\"entities\"]}个实体，{result[\"relationships\"]}个关系')

asyncio.run(main())
"

REM 启动FastAPI服务
echo.
echo 5. 启动FastAPI服务...
echo   访问地址: http://localhost:8000
echo   按Ctrl+C停止服务
echo.
%PYTHON_PATH% -m uvicorn models.main:app --reload --host 0.0.0.0 --port 8000

pause