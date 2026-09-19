@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "WITH_DL="
if "%1"=="--with-dl" set "WITH_DL=1"

echo ==============================================
echo  一键运行：统计 - ML - DL(可选) - 最终对比
echo  用法：双击运行；或在 cmd 中：run_all.bat [--with-dl]
echo ==============================================

echo.
echo [1/4] 统计项目 ...
cd /d "%~dp0time-series-forecast"
python main.py --exp exp_baseline
if errorlevel 1 goto :fail

REM 定位最新 run 的 test_set_predictions.csv
set "STAT_PRED="
for /f "delims=" %%d in ('dir /b /o-d /ad runs') do (
  if exist "runs\%%d\test_set_predictions.csv" (
    set "STAT_PRED=runs\%%d\test_set_predictions.csv"
    goto :foundstat
  )
)
:foundstat
if not defined STAT_PRED (
  echo [错误] 统计项目未生成 test_set_predictions.csv
  goto :fail
)
echo   统计侧 2021 预测: %STAT_PRED%

echo.
echo [2/4] ML 项目 ...
cd /d "%~dp0ml_project"
python main.py
if errorlevel 1 goto :fail
set "ML_PRED=%~dp0ml_project\output\ml_test_predictions.csv"

echo.
if "%WITH_DL%"=="1" (
  echo [3/4] DL 项目 ^(N-BEATS, 耗时较长^) ...
  cd /d "%~dp0dl_project"
  python main.py
  if errorlevel 1 goto :fail
  set "DL_PRED=%~dp0dl_project\output\dl_test_predictions.csv"
  echo   DL 2021 预测: !DL_PRED!
) else (
  echo [3/4] 跳过 DL ^(如需深度时序对比，加 --with-dl^)
)

echo.
echo [4/4] 最终对比 ...
cd /d "%~dp0final_compare"
set "STAT_ABS=%~dp0%STAT_PRED%"
set "TEST_CSV=%~dp0time-series-forecast\data\test.csv"

if "%WITH_DL%"=="1" (
  python main.py --stat "%STAT_ABS%" --ml "%ML_PRED%" --dl "!DL_PRED!" --test "%TEST_CSV%"
) else (
  python main.py --stat "%STAT_ABS%" --ml "%ML_PRED%" --test "%TEST_CSV%"
)
if errorlevel 1 goto :fail

echo.
echo ==============================================
echo  全部完成！对比结果见 final_compare\output\
echo ==============================================
exit /b 0

:fail
echo.
echo 运行失败，请检查上方错误信息。
exit /b 1
