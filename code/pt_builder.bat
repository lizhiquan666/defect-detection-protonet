@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul
title 启智杯 - 高敏感原型生成 (K=75)
echo ================================================
echo 【高敏感模式】K=75 已启动
echo ================================================

for %%I in ("%~dp0..") do set "ROOT=%%~fI"

set K=75
set RATIO=0.2
if not defined CLASS_SET set "CLASS_SET=1 2 3 4 5 6 7 8 9 10"
if not defined DATA_OK_SUFFIX set "DATA_OK_SUFFIX=-ok"
set "EXE=%~dp0prototype_generator.exe"
if not exist "%EXE%" (
    set "EXE_FOUND="
    for /f "delims=" %%I in ('where prototype_generator.exe 2^>nul') do (
        if not defined EXE_FOUND (
            set "EXE=%%~fI"
            set "EXE_FOUND=1"
        )
    )
)
if not exist "%EXE%" set "EXE=%ROOT%\bin\prototype_generator.exe"
if not exist "%EXE%" set "EXE=%ROOT%\extra\build\prototype_generator.exe"
if not defined DATASET_DIR_NAME set "DATASET_DIR_NAME=第四届启智杯算法初赛数据集"
if not defined OUT_ROOT set "OUT_ROOT=%ROOT%\prototypes"
if not defined BIN_OUT_ROOT set "BIN_OUT_ROOT=%ROOT%\bin\prototypes"
if not defined BUILD_OUT_ROOT set "BUILD_OUT_ROOT=%ROOT%\extra\build\prototypes"

REM DATA_ROOT resolution priority:
REM 1) Command argument (first non --no-pause argument)
REM 2) Existing DATA_ROOT environment variable
REM 3) Auto-detect under project root / parent / current directory
if not "%~1"=="" if /I not "%~1"=="--no-pause" (
    set "DATA_ROOT=%~1"
) else if not "%~2"=="" if /I not "%~2"=="--no-pause" (
    set "DATA_ROOT=%~2"
)

if not defined DATA_ROOT (
    if exist "%ROOT%\%DATASET_DIR_NAME%" set "DATA_ROOT=%ROOT%\%DATASET_DIR_NAME%"
)
if not defined DATA_ROOT (
    if exist "%ROOT%\..\%DATASET_DIR_NAME%" set "DATA_ROOT=%ROOT%\..\%DATASET_DIR_NAME%"
)
if not defined DATA_ROOT (
    if exist "%CD%\%DATASET_DIR_NAME%" set "DATA_ROOT=%CD%\%DATASET_DIR_NAME%"
)

if defined DATA_ROOT for %%I in ("%DATA_ROOT%") do set "DATA_ROOT=%%~fI"

if not exist "%EXE%" (
    echo [ERROR] prototype_generator.exe not found: %EXE%
    exit /b 1
)

for %%I in ("%EXE%") do set "EXE=%%~fI"
for %%I in ("%OUT_ROOT%") do set "OUT_ROOT=%%~fI"
for %%I in ("%BIN_OUT_ROOT%") do set "BIN_OUT_ROOT=%%~fI"
for %%I in ("%BUILD_OUT_ROOT%") do set "BUILD_OUT_ROOT=%%~fI"

if not exist "%DATA_ROOT%" (
    echo [ERROR] DATA_ROOT not found.
    echo [HINT] Keep dataset folder name as "%DATASET_DIR_NAME%" and place it near project root,
    echo [HINT] or pass explicit path: pt_builder.bat "<数据集根目录>\%DATASET_DIR_NAME%"
    exit /b 1
)

if not exist "%OUT_ROOT%" mkdir "%OUT_ROOT%" >nul 2>&1
if not exist "%BIN_OUT_ROOT%" mkdir "%BIN_OUT_ROOT%" >nul 2>&1
if not exist "%BUILD_OUT_ROOT%" mkdir "%BUILD_OUT_ROOT%" >nul 2>&1

for %%c in (%CLASS_SET%) do (
    echo.
    echo [正在生成 Class %%c 原型 ... K=75]
    pushd "%OUT_ROOT%"
    "%EXE%" --class %%c --data "%DATA_ROOT%\%%c\%%c%DATA_OK_SUFFIX%" --K %K% --ratio %RATIO%
    set "RET=!errorlevel!"
    popd

    if not "!RET!"=="0" (
        echo  Class %%c 失败
    ) else (
        set "SRC_PT=%OUT_ROOT%\%%c\prototype_generator.pt"
        if exist "!SRC_PT!" (
            if not exist "%BIN_OUT_ROOT%\%%c" mkdir "%BIN_OUT_ROOT%\%%c" >nul 2>&1
            if not exist "%BUILD_OUT_ROOT%\%%c" mkdir "%BUILD_OUT_ROOT%\%%c" >nul 2>&1
            copy /Y "!SRC_PT!" "%BIN_OUT_ROOT%\%%c\prototype_generator.pt" >nul
            copy /Y "!SRC_PT!" "%BUILD_OUT_ROOT%\%%c\prototype_generator.pt" >nul
            echo  Class %%c 完成，已复制到 prototypes(根/bin/build)
        ) else (
            echo  Class %%c 完成，但未找到输出文件: !SRC_PT!
        )
    )
)

echo.
echo 【生成完成】K=75 已生成所有原型
set "NO_PAUSE=0"
for %%A in (%*) do (
    if /I "%%~A"=="--no-pause" set "NO_PAUSE=1"
)
if "%NO_PAUSE%"=="0" pause