@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "PROTOTYPES_ONLY=0"
for %%A in (%*) do (
	if /I "%%~A"=="--prototypes-only" set "PROTOTYPES_ONLY=1"
)

REM Auto-elevate to Administrator for SYSTEM Path updates.
if /I "%~1"=="--elevated" goto :main
if "%PROTOTYPES_ONLY%"=="1" goto :main

net session >nul 2>&1
if not "%errorlevel%"=="0" (
	echo Requesting Administrator permission...
	powershell -NoProfile -ExecutionPolicy Bypass -Command "^$p=Start-Process -FilePath '%~f0' -ArgumentList '--elevated' -Verb RunAs -Wait -PassThru; exit ^$p.ExitCode"
	if errorlevel 1 (
		echo [ERROR] Administrator permission was not granted.
		exit /b 1
	)
	exit /b %errorlevel%
)

:main

REM Resolve the directory where this script is located.
for %%I in ("%~dp0.") do set "SCRIPT_DIR=%%~fI"

if not defined LIBTORCH_URL set "LIBTORCH_URL=https://download.pytorch.org/libtorch/cu128/libtorch-win-shared-with-deps-debug-2.10.0%%2Bcu128.zip"
if not defined LIBTORCH_ZIP set "LIBTORCH_ZIP=%SCRIPT_DIR%\libtorch-win-shared-with-deps-debug-2.10.0+cu128.zip"
if not defined LIBTORCH_DIR set "LIBTORCH_DIR=%SCRIPT_DIR%\libtorch"
if not defined LIBTORCH_STAGE set "LIBTORCH_STAGE=%SCRIPT_DIR%\_tmp_libtorch_extract"

if not defined OPENCV_URL set "OPENCV_URL=https://github.com/opencv/opencv/releases/download/4.12.0/opencv-4.12.0-windows.exe"
if not defined OPENCV_EXE set "OPENCV_EXE=%SCRIPT_DIR%\opencv-4.12.0-windows.exe"
if not defined OPENCV_DIR set "OPENCV_DIR=%SCRIPT_DIR%\opencv"
if not defined OPENCV_STAGE set "OPENCV_STAGE=%SCRIPT_DIR%\_tmp_opencv_extract"
if not defined WORKSPACE_ROOT set "WORKSPACE_ROOT=%SCRIPT_DIR%\.."
if not defined PROTOTYPES_REPO_URL set "PROTOTYPES_REPO_URL=https://github.com/lizhiquan666/prototypes_storage.git"
if not defined PROTOTYPES_TMP_DIR set "PROTOTYPES_TMP_DIR=%SCRIPT_DIR%\_tmp_prototypes_repo"
if not defined PROTOTYPES_ROOT set "PROTOTYPES_ROOT=%WORKSPACE_ROOT%\code\prototypes"
if not defined ENABLE_PROTOTYPES_DOWNLOAD set "ENABLE_PROTOTYPES_DOWNLOAD=1"
for %%I in ("%WORKSPACE_ROOT%") do set "WORKSPACE_ROOT=%%~fI"
if not defined EXTRA_VSCODE_DIR set "EXTRA_VSCODE_DIR=%SCRIPT_DIR%\.vscode"
if not defined ROOT_VSCODE_DIR set "ROOT_VSCODE_DIR=%WORKSPACE_ROOT%\.vscode"
if not defined MAX_RETRIES set "MAX_RETRIES=3"
if not defined RETRY_WAIT set "RETRY_WAIT=3"

for %%I in ("%LIBTORCH_ZIP%") do set "LIBTORCH_ZIP=%%~fI"
for %%I in ("%LIBTORCH_DIR%") do set "LIBTORCH_DIR=%%~fI"
for %%I in ("%LIBTORCH_STAGE%") do set "LIBTORCH_STAGE=%%~fI"
for %%I in ("%OPENCV_EXE%") do set "OPENCV_EXE=%%~fI"
for %%I in ("%OPENCV_DIR%") do set "OPENCV_DIR=%%~fI"
for %%I in ("%OPENCV_STAGE%") do set "OPENCV_STAGE=%%~fI"
for %%I in ("%PROTOTYPES_TMP_DIR%") do set "PROTOTYPES_TMP_DIR=%%~fI"
for %%I in ("%PROTOTYPES_ROOT%") do set "PROTOTYPES_ROOT=%%~fI"
for %%I in ("%EXTRA_VSCODE_DIR%") do set "EXTRA_VSCODE_DIR=%%~fI"
for %%I in ("%ROOT_VSCODE_DIR%") do set "ROOT_VSCODE_DIR=%%~fI"
set "LIBTORCH_LIB_DIR=%LIBTORCH_DIR%\lib"
set "OPENCV_BIN_DIR=%OPENCV_DIR%\build\x64\vc16\bin"

if "%PROTOTYPES_ONLY%"=="1" (
	echo [MODE] Prototypes-only download mode.
	call :downloadPrototypes
	set "RET=!errorlevel!"
	exit /b !RET!
)

echo [1/8] Target directory: %SCRIPT_DIR%

echo [2/8] Downloading libtorch...
call :downloadWithRetry "%LIBTORCH_URL%" "%LIBTORCH_ZIP%" "libtorch package"
if errorlevel 1 (
	echo [ERROR] Failed to download libtorch package.
	exit /b 1
)

if exist "%LIBTORCH_DIR%" (
	echo Removing existing libtorch directory...
	rmdir /s /q "%LIBTORCH_DIR%"
)
if exist "%LIBTORCH_STAGE%" rmdir /s /q "%LIBTORCH_STAGE%"
mkdir "%LIBTORCH_STAGE%" >nul 2>&1

echo [3/8] Extracting libtorch zip...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -LiteralPath '%LIBTORCH_ZIP%' -DestinationPath '%LIBTORCH_STAGE%' -Force"
if errorlevel 1 (
	echo [ERROR] Failed to extract libtorch zip.
	exit /b 1
)

echo [3/8] Promoting libtorch folder to %LIBTORCH_DIR% ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "^$stage='%LIBTORCH_STAGE%'; ^$dst='%LIBTORCH_DIR%'; if (Test-Path ^$dst) { Remove-Item -LiteralPath ^$dst -Recurse -Force }; ^$cand=Get-ChildItem -LiteralPath ^$stage -Directory -Recurse | Where-Object { ^$_.Name -ieq 'libtorch' } | Select-Object -First 1; if (-not ^$cand) { Write-Error 'libtorch folder not found in extracted contents.'; exit 1 }; Move-Item -LiteralPath ^$cand.FullName -Destination ^$dst -Force"
if errorlevel 1 (
	echo [ERROR] Failed to place libtorch folder into extra.
	exit /b 1
)

if not exist "%LIBTORCH_DIR%\" (
	echo [ERROR] libtorch directory was not found after folder promotion.
	exit /b 1
)

echo [4/8] Downloading OpenCV extractor...
call :downloadWithRetry "%OPENCV_URL%" "%OPENCV_EXE%" "OpenCV package"
if errorlevel 1 (
	echo [ERROR] Failed to download OpenCV package.
	exit /b 1
)

echo [5/8] Extracting OpenCV from exe...
REM 7z SFX switches: -y for yes to all, -o<dir> for output directory.
if exist "%OPENCV_STAGE%" rmdir /s /q "%OPENCV_STAGE%"
mkdir "%OPENCV_STAGE%" >nul 2>&1
"%OPENCV_EXE%" -y -o"%OPENCV_STAGE%"
if errorlevel 1 (
	echo [ERROR] OpenCV extraction failed.
	exit /b 1
)

if exist "%OPENCV_DIR%" (
	echo Removing existing opencv directory...
	rmdir /s /q "%OPENCV_DIR%"
)

echo [5/8] Promoting OpenCV folder to %OPENCV_DIR% ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "^$stage='%OPENCV_STAGE%'; ^$dst='%OPENCV_DIR%'; ^$cand=Get-ChildItem -LiteralPath ^$stage -Directory -Recurse | Where-Object { Test-Path (Join-Path ^$_.FullName 'build\x64\vc16\bin') } | Select-Object -First 1; if (-not ^$cand) { Write-Error 'OpenCV folder with build\\x64\\vc16\\bin not found.'; exit 1 }; Move-Item -LiteralPath ^$cand.FullName -Destination ^$dst -Force"
if errorlevel 1 (
	echo [ERROR] Failed to place OpenCV folder into extra.
	exit /b 1
)

if not exist "%OPENCV_BIN_DIR%" (
	echo [ERROR] OpenCV vc16 bin directory was not found after folder promotion.
	exit /b 1
)

echo [6/8] Adding libtorch/opencv paths to SYSTEM Path...
powershell -NoProfile -ExecutionPolicy Bypass -Command "^$libPath='%LIBTORCH_LIB_DIR%'; ^$opencvPath='%OPENCV_BIN_DIR%'; if (-not (Test-Path ^$libPath)) { Write-Error 'libtorch lib directory not found.'; exit 1 }; if (-not (Test-Path ^$opencvPath)) { Write-Error 'OpenCV vc16 bin directory not found.'; exit 1 }; ^$machinePath=[Environment]::GetEnvironmentVariable('Path','Machine'); ^$parts=@(); if (^$machinePath) { ^$parts=^$machinePath -split ';' | Where-Object { ^$_ -and ^$_.Trim() -ne '' } }; ^$cmp=[StringComparer]::OrdinalIgnoreCase; ^$set=New-Object 'System.Collections.Generic.HashSet[string]' (^$cmp); foreach (^$p in ^$parts) { [void]^$set.Add(^$p.Trim()) }; [void]^$set.Add(^$libPath); [void]^$set.Add(^$opencvPath); ^$newPath=([string[]]^$set) -join ';'; [Environment]::SetEnvironmentVariable('Path', ^$newPath, 'Machine'); Write-Host ('Added to SYSTEM Path: ' + ^$libPath); Write-Host ('Added to SYSTEM Path: ' + ^$opencvPath)"
if errorlevel 1 (
	echo [ERROR] Failed to update SYSTEM Path.
	echo [ERROR] Please run this bat as Administrator.
	exit /b 1
)

echo [7/8] Copying .vscode folder to workspace root...
if exist "%EXTRA_VSCODE_DIR%\" (
	if not exist "%ROOT_VSCODE_DIR%\" mkdir "%ROOT_VSCODE_DIR%" >nul 2>&1
	robocopy "%EXTRA_VSCODE_DIR%" "%ROOT_VSCODE_DIR%" /E /NFL /NDL /NJH /NJS /NP >nul
	set "ROBO_EXIT=!errorlevel!"
	if !ROBO_EXIT! GEQ 8 (
		echo [ERROR] Failed to copy .vscode folder to workspace root.
		exit /b 1
	)
	echo [INFO] .vscode copied to: %ROOT_VSCODE_DIR%
) else (
	echo [WARN] Source .vscode folder not found under extra; skipping copy.
)

call :downloadPrototypes
if errorlevel 1 exit /b 1

echo Cleaning downloaded packages and temporary folders...
if exist "%LIBTORCH_ZIP%" del /f /q "%LIBTORCH_ZIP%" >nul 2>&1
if exist "%OPENCV_EXE%" del /f /q "%OPENCV_EXE%" >nul 2>&1
if exist "%LIBTORCH_STAGE%" rmdir /s /q "%LIBTORCH_STAGE%"
if exist "%OPENCV_STAGE%" rmdir /s /q "%OPENCV_STAGE%"
if exist "%PROTOTYPES_TMP_DIR%" rmdir /s /q "%PROTOTYPES_TMP_DIR%"

echo Cleaning extra subfolders (keep only .vscode, opencv, libtorch, build)...
for /d %%D in ("%SCRIPT_DIR%\*") do (
	set "KEEP=0"
	if /I "%%~nxD"==".vscode" set "KEEP=1"
	if /I "%%~nxD"=="opencv" set "KEEP=1"
	if /I "%%~nxD"=="libtorch" set "KEEP=1"
	if /I "%%~nxD"=="build" set "KEEP=1"
	if "!KEEP!"=="0" (
		echo Removing unused folder: %%~nxD
		rmdir /s /q "%%~fD"
	)
)

echo.
echo Done. libtorch/OpenCV and optional prototypes have been prepared in:
echo %SCRIPT_DIR%
exit /b 0

:downloadPrototypes
if /I "%ENABLE_PROTOTYPES_DOWNLOAD%"=="0" (
	echo [8/8] Skipping prototypes download ^(ENABLE_PROTOTYPES_DOWNLOAD=0^).
	exit /b 0
)

echo [8/8] Downloading and preparing prototypes in %PROTOTYPES_ROOT% ...
where git >nul 2>&1
if errorlevel 1 (
	echo [ERROR] git.exe was not found. Install Git or set ENABLE_PROTOTYPES_DOWNLOAD=0.
	exit /b 1
)
if exist "%PROTOTYPES_TMP_DIR%" rmdir /s /q "%PROTOTYPES_TMP_DIR%"
git clone --depth 1 "%PROTOTYPES_REPO_URL%" "%PROTOTYPES_TMP_DIR%"
if errorlevel 1 (
	echo [ERROR] Failed to clone prototypes repository.
	exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -Command "^$repo='%PROTOTYPES_TMP_DIR%'; ^$dst='%PROTOTYPES_ROOT%'; if (Test-Path ^$dst) { Remove-Item -LiteralPath ^$dst -Recurse -Force }; New-Item -ItemType Directory -Path ^$dst -Force | Out-Null; 1..10 | ForEach-Object { ^$cls=[string]^$_; ^$outDir=Join-Path ^$dst ^$cls; New-Item -ItemType Directory -Path ^$outDir -Force | Out-Null; ^$pt=Get-ChildItem -LiteralPath ^$repo -File -Recurse | Where-Object { ^$_.Name -ieq 'prototype_generator.pt' -and ^$_.FullName -match ('[\\/]' + [Regex]::Escape(^$cls) + '[\\/]prototype_generator\\.pt^$') } | Select-Object -First 1; if (-not ^$pt) { throw ('Missing prototype_generator.pt for class ' + ^$cls) }; Copy-Item -LiteralPath ^$pt.FullName -Destination (Join-Path ^$outDir 'prototype_generator.pt') -Force }; Write-Host 'Prototypes prepared under code/prototypes/1..10.'"
if errorlevel 1 (
	echo [ERROR] Failed to prepare prototypes from repository.
	exit /b 1
)

if exist "%PROTOTYPES_TMP_DIR%" rmdir /s /q "%PROTOTYPES_TMP_DIR%"
exit /b 0

:downloadWithRetry
set "DL_URL=%~1"
set "DL_OUT=%~2"
set "DL_NAME=%~3"
set "DL_OK=0"

for /L %%R in (1,1,%MAX_RETRIES%) do (
	echo [INFO] Downloading !DL_NAME! (attempt %%R/%MAX_RETRIES%)...
	if exist "!DL_OUT!" del /f /q "!DL_OUT!" >nul 2>&1
	curl.exe -L --fail --connect-timeout 20 --retry 0 "!DL_URL!" -o "!DL_OUT!"
	if not errorlevel 1 (
		set "DL_OK=1"
		goto :download_done
	)
	if %%R LSS %MAX_RETRIES% (
		echo [WARN] Download failed. Retrying in %RETRY_WAIT%s...
		timeout /t %RETRY_WAIT% /nobreak >nul
	)
)

:download_done
if "!DL_OK!"=="1" exit /b 0
echo [ERROR] Failed to download !DL_NAME! after %MAX_RETRIES% attempts.
exit /b 1