@echo off
setlocal enabledelayedexpansion

for %%I in ("%~dp0..") do set "ROOT=%%~fI"

chcp 65001 >nul
title 启智杯 - 全自动推理（干净版 - 只输出_rst文件）
echo ================================================
echo 启智杯 - 全自动推理（只输出 rst.bmp + rst.txt + heat.png）
echo ================================================

if not defined IN_DATASET_DIR_NAME set "IN_DATASET_DIR_NAME=内部数据_无真值标签"
if not defined CLASS_SET set "CLASS_SET=2 4 6 8 10"
if not defined EXE_BUILD set "EXE_BUILD=%ROOT%\extra\build\defect_detect.exe"
if not defined EXE_BIN set "EXE_BIN=%ROOT%\bin\defect_detect.exe"
if not defined PT_CODE_ROOT set "PT_CODE_ROOT=%ROOT%\code\prototypes"
if not defined PT_MAIN_ROOT set "PT_MAIN_ROOT=%ROOT%\prototypes"
if not defined PT_BIN_ROOT set "PT_BIN_ROOT=%ROOT%\bin\prototypes"
if not defined PT_BUILD_ROOT set "PT_BUILD_ROOT=%ROOT%\extra\build\prototypes"
if not defined PT_BUILDER set "PT_BUILDER=%ROOT%\code\pt_builder.bat"
if not defined INPUT_KINDS set "INPUT_KINDS=ok ng"
if not defined INPUT_SUBDIR_PREFIX set "INPUT_SUBDIR_PREFIX=验证集"
if not defined DISABLE_INFERENCE_PT_BUILDER set "DISABLE_INFERENCE_PT_BUILDER=0"
if not defined DET_THRESH set "DET_THRESH=0.24"
if not defined DET_MIN_AREA set "DET_MIN_AREA=220"
if not defined DET_EDGE_MARGIN set "DET_EDGE_MARGIN=0.034"
if not defined DET_OPEN_ITER set "DET_OPEN_ITER=2"
if not defined DET_NMS_IOU set "DET_NMS_IOU=0.3"
if not defined DET_IGNORE_TL_W set "DET_IGNORE_TL_W=0.43"
if not defined DET_IGNORE_TL_H set "DET_IGNORE_TL_H=0.40"
if not defined DET_IGNORE_X0 set "DET_IGNORE_X0=0.96"
if not defined DET_IGNORE_Y0 set "DET_IGNORE_Y0=0.00"
if not defined DET_IGNORE_X1 set "DET_IGNORE_X1=1.00"
if not defined DET_IGNORE_Y1 set "DET_IGNORE_Y1=0.40"

if exist "%EXE_BUILD%" if not exist "%EXE_BIN%" copy /Y "%EXE_BUILD%" "%EXE_BIN%" >nul
if exist "%EXE_BIN%" if not exist "%EXE_BUILD%" copy /Y "%EXE_BIN%" "%EXE_BUILD%" >nul

if not defined EXE (
  if exist "%EXE_BUILD%" (
    set "EXE=%EXE_BUILD%"
  ) else if exist "%EXE_BIN%" (
    set "EXE=%EXE_BIN%"
  )
)
if not defined EXE (
  for /f "delims=" %%I in ('where defect_detect.exe 2^>nul') do (
    if not defined EXE set "EXE=%%~fI"
  )
)
if not defined OUT_ROOT set "OUT_ROOT=%ROOT%\res"
if not defined WEIGHTS_ROOT set "WEIGHTS_ROOT=%ROOT%\code"

for %%I in ("%EXE_BUILD%") do set "EXE_BUILD=%%~fI"
for %%I in ("%EXE_BIN%") do set "EXE_BIN=%%~fI"
for %%I in ("%PT_CODE_ROOT%") do set "PT_CODE_ROOT=%%~fI"
for %%I in ("%PT_MAIN_ROOT%") do set "PT_MAIN_ROOT=%%~fI"
for %%I in ("%PT_BIN_ROOT%") do set "PT_BIN_ROOT=%%~fI"
for %%I in ("%PT_BUILD_ROOT%") do set "PT_BUILD_ROOT=%%~fI"
for %%I in ("%PT_BUILDER%") do set "PT_BUILDER=%%~fI"
for %%I in ("%OUT_ROOT%") do set "OUT_ROOT=%%~fI"
for %%I in ("%WEIGHTS_ROOT%") do set "WEIGHTS_ROOT=%%~fI"

if not defined IN_ROOT (
  if exist "%ROOT%\%IN_DATASET_DIR_NAME%" set "IN_ROOT=%ROOT%\%IN_DATASET_DIR_NAME%"
)
if not defined IN_ROOT (
  if exist "%ROOT%\..\%IN_DATASET_DIR_NAME%" set "IN_ROOT=%ROOT%\..\%IN_DATASET_DIR_NAME%"
)
if not defined IN_ROOT (
  if exist "%CD%\%IN_DATASET_DIR_NAME%" set "IN_ROOT=%CD%\%IN_DATASET_DIR_NAME%"
)

if not "%~1"=="" set "IN_ROOT=%~1"
if not "%~2"=="" set "OUT_ROOT=%~2"

if defined IN_ROOT for %%I in ("%IN_ROOT%") do set "IN_ROOT=%%~fI"
if defined OUT_ROOT for %%I in ("%OUT_ROOT%") do set "OUT_ROOT=%%~fI"

if not exist "%EXE%" (
  echo EXE 不存在: %EXE%
  exit /b 1
)

for %%I in ("%EXE%") do set "EXE=%%~fI"
if exist "%EXE%" (
  if not exist "%EXE_BIN%" copy /Y "%EXE%" "%EXE_BIN%" >nul
  if not exist "%EXE_BUILD%" copy /Y "%EXE%" "%EXE_BUILD%" >nul
)

if not defined IN_ROOT (
  echo IN_ROOT 未设置，且未自动发现目录 "%IN_DATASET_DIR_NAME%"
  echo 可手动传参: inference.bat "<数据集根目录>\%IN_DATASET_DIR_NAME%" [OUT_ROOT]
  exit /b 1
)

if not exist "%IN_ROOT%" (
  echo IN_ROOT 不存在: %IN_ROOT%
  exit /b 1
)

echo.
echo === 处理验证集 ===

for %%c in (%CLASS_SET%) do (
  set "WEIGHTS="
  set "PT_DIR=%PT_CODE_ROOT%\%%c"
  if exist "!PT_DIR!\" (
    for /f "delims=" %%P in ('dir /b /a:-d "!PT_DIR!\*.pt" 2^>nul') do (
      if not defined WEIGHTS set "WEIGHTS=!PT_DIR!\%%P"
    )
  )

  set "PT_DIR=%PT_MAIN_ROOT%\%%c"
  if not defined WEIGHTS if exist "!PT_DIR!\" (
    for /f "delims=" %%P in ('dir /b /a:-d "!PT_DIR!\*.pt" 2^>nul') do (
      if not defined WEIGHTS set "WEIGHTS=!PT_DIR!\%%P"
    )
  )
  set "PT_DIR=%PT_BIN_ROOT%\%%c"
  if not defined WEIGHTS if exist "!PT_DIR!\" (
    for /f "delims=" %%P in ('dir /b /a:-d "!PT_DIR!\*.pt" 2^>nul') do (
      if not defined WEIGHTS set "WEIGHTS=!PT_DIR!\%%P"
    )
  )
  set "PT_DIR=%PT_BUILD_ROOT%\%%c"
  if not defined WEIGHTS if exist "!PT_DIR!\" (
    for /f "delims=" %%P in ('dir /b /a:-d "!PT_DIR!\*.pt" 2^>nul') do (
      if not defined WEIGHTS set "WEIGHTS=!PT_DIR!\%%P"
    )
  )

  if not defined WEIGHTS (
    if /I "%DISABLE_INFERENCE_PT_BUILDER%"=="1" (
      echo [INFO] Class %%c 缺少可用原型，已禁用 inference 内部 pt_builder 回退。
    ) else (
      if exist "%PT_BUILDER%" (
        echo [INFO] Class %%c 缺少可用原型，尝试调用 pt_builder 生成...
        call "%PT_BUILDER%" --no-pause
        if errorlevel 1 (
          echo [WARN] pt_builder 执行失败，Class %%c 继续尝试其他权重来源。
        )
      ) else (
        echo [WARN] 未找到 pt_builder: %PT_BUILDER%
      )
    )
  )

  set "PT_DIR=%PT_MAIN_ROOT%\%%c"
  if not defined WEIGHTS if exist "!PT_DIR!\" (
    for /f "delims=" %%P in ('dir /b /a:-d "!PT_DIR!\*.pt" 2^>nul') do (
      if not defined WEIGHTS set "WEIGHTS=!PT_DIR!\%%P"
    )
  )
  set "PT_DIR=%PT_BIN_ROOT%\%%c"
  if not defined WEIGHTS if exist "!PT_DIR!\" (
    for /f "delims=" %%P in ('dir /b /a:-d "!PT_DIR!\*.pt" 2^>nul') do (
      if not defined WEIGHTS set "WEIGHTS=!PT_DIR!\%%P"
    )
  )
  set "PT_DIR=%PT_BUILD_ROOT%\%%c"
  if not defined WEIGHTS if exist "!PT_DIR!\" (
    for /f "delims=" %%P in ('dir /b /a:-d "!PT_DIR!\*.pt" 2^>nul') do (
      if not defined WEIGHTS set "WEIGHTS=!PT_DIR!\%%P"
    )
  )
  if not defined WEIGHTS if exist "%WEIGHTS_ROOT%\%%c\prnet_semisup_weights_typed_aug.pt" set "WEIGHTS=%WEIGHTS_ROOT%\%%c\prnet_semisup_weights_typed_aug.pt"
  if not defined WEIGHTS if exist "%WEIGHTS_ROOT%\%%c\prototype_generator.pt" set "WEIGHTS=%WEIGHTS_ROOT%\%%c\prototype_generator.pt"

  if not exist "!WEIGHTS!" (
    echo 跳过 Class %%c，权重不存在: !WEIGHTS!
  ) else (
    for %%W in ("!WEIGHTS!") do set "PT_NAME=%%~nxW"
    set "PT_DST_ROOT=%PT_MAIN_ROOT%\%%c\!PT_NAME!"
    set "PT_DST_BIN=%PT_BIN_ROOT%\%%c\!PT_NAME!"
    set "PT_DST_BUILD=%PT_BUILD_ROOT%\%%c\!PT_NAME!"
    if not exist "%PT_MAIN_ROOT%\%%c" mkdir "%PT_MAIN_ROOT%\%%c" >nul 2>&1
    if not exist "%PT_BIN_ROOT%\%%c" mkdir "%PT_BIN_ROOT%\%%c" >nul 2>&1
    if not exist "%PT_BUILD_ROOT%\%%c" mkdir "%PT_BUILD_ROOT%\%%c" >nul 2>&1
    if /I not "!WEIGHTS!"=="!PT_DST_ROOT!" copy /Y "!WEIGHTS!" "!PT_DST_ROOT!" >nul
    if /I not "!WEIGHTS!"=="!PT_DST_BIN!" copy /Y "!WEIGHTS!" "!PT_DST_BIN!" >nul
    if /I not "!WEIGHTS!"=="!PT_DST_BUILD!" copy /Y "!WEIGHTS!" "!PT_DST_BUILD!" >nul
    if /I not "!PT_NAME!"=="prototype_generator.pt" (
      copy /Y "!WEIGHTS!" "%PT_MAIN_ROOT%\%%c\prototype_generator.pt" >nul
      copy /Y "!WEIGHTS!" "%PT_BIN_ROOT%\%%c\prototype_generator.pt" >nul
      copy /Y "!WEIGHTS!" "%PT_BUILD_ROOT%\%%c\prototype_generator.pt" >nul
    )

    for %%s in (%INPUT_KINDS%) do (
      set "IN_DIR=%IN_ROOT%\%%c\%INPUT_SUBDIR_PREFIX%%%s"
      set "OUT_DIR=%OUT_ROOT%\%%c\%%s"

      if not exist "!IN_DIR!" (
        echo 跳过 Class %%c %%s，输入目录不存在: !IN_DIR!
      ) else (
        if not exist "!OUT_DIR!" mkdir "!OUT_DIR!"
        if exist "!IN_DIR!\*.bmp" (
          for %%f in ("!IN_DIR!\*.bmp") do (
            "%EXE%" -i "%%f" --class %%c --weights "!WEIGHTS!" --out "!OUT_DIR!" --thresh %DET_THRESH% --min_area %DET_MIN_AREA% --edge_margin %DET_EDGE_MARGIN% --open_iter %DET_OPEN_ITER% --nms_iou %DET_NMS_IOU% --ignore_tl_w %DET_IGNORE_TL_W% --ignore_tl_h %DET_IGNORE_TL_H% --ignore_x0 %DET_IGNORE_X0% --ignore_y0 %DET_IGNORE_Y0% --ignore_x1 %DET_IGNORE_X1% --ignore_y1 %DET_IGNORE_Y1%
            set "BASE=%%~nf"
            if exist "!OUT_DIR!\!BASE!_rst.bmp" del /q "!OUT_DIR!\!BASE!_rst.bmp"
            if exist "!OUT_DIR!\!BASE!_rst.txt" del /q "!OUT_DIR!\!BASE!_rst.txt"
            if exist "!OUT_DIR!\result_!BASE!.bmp" ren "!OUT_DIR!\result_!BASE!.bmp" "!BASE!_rst.bmp"
            if exist "!OUT_DIR!\result_!BASE!.txt" ren "!OUT_DIR!\result_!BASE!.txt" "!BASE!_rst.txt"
          )
        ) else (
          echo  无 BMP: !IN_DIR!
        )
      )
    )
  )
)

echo.
echo ================================================
echo 全部完成！只生成了 _rst.bmp + _rst.txt + _heat.png
echo ================================================

endlocal
exit /b 0