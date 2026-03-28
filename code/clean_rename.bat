@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul

for %%I in ("%~dp0..") do set "ROOT=%%~fI"

if not defined BASE set "BASE=%ROOT%\res"
if not defined SETS set "SETS=1 2 3 4 5 6 7 8 9 10"
if not defined KINDS set "KINDS=ng ok"

for %%I in ("%BASE%") do set "BASE=%%~fI"

set TOTAL_RENAME=0
set TOTAL_DELETE=0
set TOTAL_KEEP=0
set TOTAL_SKIP=0
set TOTAL_DIR=0

echo ================================================
echo   多目录整理版 - 改名并清理无关文件
echo ================================================
echo 开始处理 1-10 下的 ng/ok 目录...
echo.

for %%s in (%SETS%) do (
    for %%k in (%KINDS%) do (
        set "dir=%BASE%\%%s\%%k"
        if exist "!dir!\" (
            set /a TOTAL_DIR+=1
            echo [目录] !dir!
            call :ProcessDir "!dir!"
            echo.
        ) else (
            echo [跳过] 目录不存在: !dir!
            echo.
        )
    )
)

echo.
echo ================================================
echo 全部完成！
echo 已处理目录 !TOTAL_DIR! 个
echo 共改名 !TOTAL_RENAME! 个文件
echo 共删除 !TOTAL_DELETE! 个无关或重复文件
echo 共保留 !TOTAL_KEEP! 个合规文件
echo 共跳过 !TOTAL_SKIP! 个异常项
echo 规则: 仅保留 .txt / .bmp
echo 规则: 已有 _rst.bmp / _rst.txt 直接保留
echo 规则: result_*.bmp / result_*.txt 统一改为 *_rst.bmp / *_rst.txt
echo 规则: 不生成任何 _log.txt
echo ================================================
if /I not "%~1"=="--no-pause" pause

exit /b

:ProcessDir
set "DIR=%~1"
for %%f in ("%DIR%\*") do (
    if exist "%%~ff" if not exist "%%~ff\" call :HandleFile "%%~ff"
)
exit /b

:HandleFile
set "FILE=%~1"
set "NAME=%~n1"
set "EXT=%~x1"
set "FILENAME=%~nx1"
set "TARGET_BASE="
set "TARGET_NAME="
set "TARGET_FILE="

if /I "!EXT!"==".bmp" goto HandleBmp
if /I "!EXT!"==".txt" goto HandleTxt

echo [删除] 非目标类型: !FILENAME!
del /f /q "!FILE!" >nul 2>&1
if not errorlevel 1 (
    set /a TOTAL_DELETE+=1
) else (
    set /a TOTAL_SKIP+=1
    echo [跳过] 删除失败: !FILENAME!
)
exit /b

:HandleBmp
if /I "!NAME:~-4!"=="_rst" (
    set /a TOTAL_KEEP+=1
    echo [保留] 已存在规范 BMP: !FILENAME!
    exit /b
)

if /I "!NAME:~0,7!"=="result_" (
    set "TARGET_BASE=!NAME:~7!"
    set "TARGET_NAME=!TARGET_BASE!_rst!EXT!"
    goto RenameOrDeleteDuplicate
)

set /a TOTAL_KEEP+=1
exit /b

:HandleTxt
if /I "!NAME:~-4!"=="_rst" (
    set /a TOTAL_KEEP+=1
    echo [保留] 已存在规范 TXT: !FILENAME!
    exit /b
)

if /I "!NAME:~-4!"=="_log" (
    echo [删除] 不需要的日志文本: !FILENAME!
    del /f /q "!FILE!" >nul 2>&1
    if not errorlevel 1 (
        set /a TOTAL_DELETE+=1
    ) else (
        set /a TOTAL_SKIP+=1
        echo [跳过] 删除失败: !FILENAME!
    )
    exit /b
)

if /I "!NAME:~0,7!"=="result_" (
    set "TARGET_BASE=!NAME:~7!"
    if /I "!TARGET_BASE:~-4!"=="_log" set "TARGET_BASE=!TARGET_BASE:~0,-4!"
    set "TARGET_NAME=!TARGET_BASE!_rst!EXT!"
    goto RenameOrDeleteDuplicate
)

set /a TOTAL_KEEP+=1
exit /b

:RenameOrDeleteDuplicate
set "TARGET_FILE=%~dp1!TARGET_NAME!"

if /I "!FILENAME!"=="!TARGET_NAME!" (
    set /a TOTAL_KEEP+=1
    exit /b
)

if exist "!TARGET_FILE!" (
    echo [保留] 已存在目标文件: !TARGET_NAME!
    echo [删除] 重复源文件: !FILENAME!
    del /f /q "!FILE!" >nul 2>&1
    if not errorlevel 1 (
        set /a TOTAL_DELETE+=1
    ) else (
        set /a TOTAL_SKIP+=1
        echo [跳过] 删除失败: !FILENAME!
    )
    exit /b
)

echo [改名] !FILENAME! ^> !TARGET_NAME!
ren "!FILE!" "!TARGET_NAME!" >nul 2>&1
if not errorlevel 1 (
    set /a TOTAL_RENAME+=1
) else (
    set /a TOTAL_SKIP+=1
    echo [跳过] 改名失败: !FILENAME!
)
exit /b