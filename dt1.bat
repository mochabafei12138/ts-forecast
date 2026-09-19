@echo off
rem ============================================================
rem  dt1.bat — 用 dt1 环境运行 Python 的便捷封装
rem  用法: dt1.bat <python 参数...>
rem  例  : dt1.bat main.py --exp exp_baseline
rem         dt1.bat streamlit run streamlit_app.py
rem  作用: 清空全局 PYTHONHOME/PYTHONPATH（避免误用灵犀 python-env 导致 re 模块冲突）
rem ============================================================
setlocal
set "PYTHONHOME="
set "PYTHONPATH="
"C:\Users\Miracle\anaconda3\envs\dt1\python.exe" %*
endlocal
