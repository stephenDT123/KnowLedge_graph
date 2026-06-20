# 本科《Python程序设计》教育知识图谱项目 - PowerShell运行脚本

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "本科《Python程序设计》教育知识图谱项目" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 设置Python路径
$PythonPath = "C:\ProgramData\anaconda3\envs\py39\python.exe"

# 检查Python环境
Write-Host "1. 检查Python环境..." -ForegroundColor Yellow
if (Test-Path $PythonPath) {
    Write-Host "   使用Anaconda Python: $PythonPath" -ForegroundColor Green
} else {
    Write-Host "   ⚠️  Anaconda Python未找到，使用系统Python" -ForegroundColor Yellow
    $PythonPath = "python"
}

# 安装依赖
Write-Host ""
Write-Host "2. 安装项目依赖..." -ForegroundColor Yellow
Write-Host "   安装PyPDF2, jieba, pillow, pytesseract, pymupdf, openai, numpy, pydantic..." -ForegroundColor Gray
& $PythonPath -m pip install PyPDF2 jieba pillow pytesseract pymupdf openai numpy pydantic fastapi uvicorn

# 测试API
Write-Host ""
Write-Host "3. 测试API配置..." -ForegroundColor Yellow
& $PythonPath models\simple_test.py

# 构建GraphRAG
Write-Host ""
Write-Host "4. 构建GraphRAG知识图谱..." -ForegroundColor Yellow
& $PythonPath -c @"
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
    print(f'构建完成：{result["documents"]}个文档，{result["entities"]}个实体，{result["relationships"]}个关系')

asyncio.run(main())
"@

# 启动FastAPI服务
Write-Host ""
Write-Host "5. 启动FastAPI服务..." -ForegroundColor Yellow
Write-Host "   访问地址: http://localhost:8000" -ForegroundColor Green
Write-Host "   按Ctrl+C停止服务" -ForegroundColor Gray
Write-Host ""
& $PythonPath -m uvicorn models.main:app --reload --host 0.0.0.0 --port 8000

Read-Host "按Enter键退出"