@echo off
echo 测试GraphRAG构建修复...
echo.

REM 使用Anaconda py39环境
"C:\ProgramData\anaconda3\envs\py39\python.exe" test_fix.py

echo.
pause