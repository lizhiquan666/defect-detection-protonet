@echo off
setlocal EnableExtensions EnableDelayedExpansion

for %%I in ("%~dp0..") do set "ROOT=%%~fI"
if not defined LIBTORCH_BAT set "LIBTORCH_BAT=%ROOT%\extra\libtorch.bat"
if not defined INFERENCE_BAT set "INFERENCE_BAT=%ROOT%\code\inference.bat"
if not defined CLEAN_RENAME_BAT set "CLEAN_RENAME_BAT=%ROOT%\code\clean_rename.bat"
if not defined PT_BUILDER_BAT set "PT_BUILDER_BAT=%ROOT%\code\pt_builder.bat"

for %%I in ("%LIBTORCH_BAT%") do set "LIBTORCH_BAT=%%~fI"
for %%I in ("%INFERENCE_BAT%") do set "INFERENCE_BAT=%%~fI"
for %%I in ("%CLEAN_RENAME_BAT%") do set "CLEAN_RENAME_BAT=%%~fI"
for %%I in ("%PT_BUILDER_BAT%") do set "PT_BUILDER_BAT=%%~fI"

set "DRY_RUN=0"
for %%A in (%*) do (
	if /I "%%~A"=="--dry-run" set "DRY_RUN=1"
	if /I "%%~A"=="/dryrun" set "DRY_RUN=1"
	if /I "%%~A"=="-n" set "DRY_RUN=1"
)

if "%DRY_RUN%"=="1" (
	echo [INFO] DRY-RUN mode: commands will NOT be executed.
)

if not exist "%LIBTORCH_BAT%" (
	echo [ERROR] Missing file: %LIBTORCH_BAT%
	exit /b 1
)
if not exist "%INFERENCE_BAT%" (
	echo [ERROR] Missing file: %INFERENCE_BAT%
	exit /b 1
)
if not exist "%CLEAN_RENAME_BAT%" (
	echo [ERROR] Missing file: %CLEAN_RENAME_BAT%
	exit /b 1
)

set "TOTAL_STEPS=3"
set "CUR_STEP=0"
set "DISABLE_INFERENCE_PT_BUILDER=0"

echo.
call :showProgress !CUR_STEP! !TOTAL_STEPS! "Starting"

call :runStep "Download prototypes from GitHub" "%LIBTORCH_BAT%" "--prototypes-only"
if errorlevel 1 (
	echo [WARN] Prototypes download failed. Trying pt_builder fallback for .pt generation...
	if not exist "%PT_BUILDER_BAT%" (
		echo [ERROR] Missing file: %PT_BUILDER_BAT%
		goto :failed
	)
	set /a TOTAL_STEPS+=1
	call :runStep "Fallback: run pt_builder" "%PT_BUILDER_BAT%" "--no-pause"
	if errorlevel 1 goto :failed
	set "DISABLE_INFERENCE_PT_BUILDER=1"
)

call :runStep "Run inference" "%INFERENCE_BAT%"
if errorlevel 1 goto :failed

call :runStep "Run clean_rename" "%CLEAN_RENAME_BAT%" "--no-pause"
if errorlevel 1 goto :failed

echo.
echo All steps completed successfully.
exit /b 0

:failed
echo.
echo Workflow stopped because a step failed.
exit /b 1

:runStep
set /a CUR_STEP+=1
set "STEP_NAME=%~1"
set "STEP_FILE=%~2"
set "STEP_ARGS=%~3"

echo.
echo [Step !CUR_STEP!/!TOTAL_STEPS!] !STEP_NAME!
if "!DRY_RUN!"=="1" (
	echo [DRY-RUN] Would execute: !STEP_FILE! !STEP_ARGS!
) else (
	if defined STEP_ARGS (
		call "!STEP_FILE!" !STEP_ARGS!
	) else (
		call "!STEP_FILE!"
	)
	if errorlevel 1 (
		echo [ERROR] !STEP_NAME! failed.
		exit /b 1
	)
)

call :showProgress !CUR_STEP! !TOTAL_STEPS! "!STEP_NAME! completed"
exit /b 0

:showProgress
set "PB_CURRENT=%~1"
set "PB_TOTAL=%~2"
set "PB_TEXT=%~3"

set /a PB_PERCENT=(PB_CURRENT*100)/PB_TOTAL
set /a PB_FILLED=(PB_CURRENT*30)/PB_TOTAL
set "PB_BAR="

for /L %%I in (1,1,30) do (
	if %%I LEQ !PB_FILLED! (
		set "PB_BAR=!PB_BAR!#"
	) else (
		set "PB_BAR=!PB_BAR!-"
	)
)

echo [!PB_BAR!] !PB_PERCENT!%% - !PB_TEXT!
exit /b 0