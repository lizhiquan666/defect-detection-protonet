# .vscode/run_cpp.ps1
# Run Code script: automatically compile two exes

param(
    [string]$cppFile
)

$projectRoot = $null
$scriptParent = Split-Path -Parent $PSScriptRoot
if ((Split-Path -Leaf $scriptParent) -ieq "extra") {
    $projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
} else {
    $projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
$codeRoot = Join-Path $projectRoot "code"
$binRoot = Join-Path $projectRoot "bin"
if (!(Test-Path $binRoot)) {
    New-Item -ItemType Directory -Path $binRoot | Out-Null
}
Set-Location $codeRoot

$libtorchRoot = if ($env:LIBTORCH_ROOT) { $env:LIBTORCH_ROOT } else { Join-Path $projectRoot "extra\libtorch" }
$opencvRoot = if ($env:OPENCV_ROOT) { $env:OPENCV_ROOT } else { Join-Path $projectRoot "extra\opencv\build" }

$vsDevCmd = $null
if ($env:VSDEVCMD_PATH -and (Test-Path $env:VSDEVCMD_PATH)) {
    $vsDevCmd = $env:VSDEVCMD_PATH
} else {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $installPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($installPath) {
            $candidate = Join-Path $installPath "Common7\Tools\VsDevCmd.bat"
            if (Test-Path $candidate) {
                $vsDevCmd = $candidate
            }
        }
    }
}

if ($vsDevCmd) {
    & $vsDevCmd -arch=x64 -host_arch=x64
} else {
    Write-Host "VsDevCmd.bat not found. Please set VSDEVCMD_PATH or open Developer PowerShell." -ForegroundColor Yellow
}

Write-Host "Starting to compile two exe files..." -ForegroundColor Green

Write-Host "Compiling defect_detect.exe ..." -ForegroundColor Cyan
cl.exe main.cpp prnet.cpp attention.cpp dataset.cpp losses.cpp modules.cpp utils.cpp `
    /utf-8 /EHsc /std:c++17 /MDd /Zi /FS `
    /I "$libtorchRoot/include" `
    /I "$libtorchRoot/include/torch/csrc/api/include" `
    /I "$opencvRoot/include" `
    /link `
    /LIBPATH:"$libtorchRoot/lib" `
    /LIBPATH:"$opencvRoot/x64/vc16/lib" `
    torch_cpu.lib c10.lib opencv_world4120d.lib `
    /OUT:"$binRoot/defect_detect.exe" /DEBUG

Write-Host "Compiling prototype_generator.exe ..." -ForegroundColor Cyan
cl.exe prototype.cpp `
    /utf-8 /EHsc /std:c++17 /MDd /Zi /FS `
    /I "$libtorchRoot/include" `
    /I "$libtorchRoot/include/torch/csrc/api/include" `
    /I "$opencvRoot/include" `
    /link `
    /LIBPATH:"$libtorchRoot/lib" `
    /LIBPATH:"$opencvRoot/x64/vc16/lib" `
    torch_cpu.lib c10.lib opencv_world4120d.lib `
    /OUT:"$binRoot/prototype_generator.exe" /DEBUG

if ($LASTEXITCODE -eq 0) {
    Write-Host "Compile SUCCESS! Two files generated:" -ForegroundColor Green
    Write-Host "  defect_detect.exe          <-- main inference program" -ForegroundColor Yellow
    Write-Host "  prototype_generator.exe    <-- prototype generator" -ForegroundColor Yellow
    $env:PATH += ";$libtorchRoot/lib;$opencvRoot/x64/vc16/bin"
} else {
    Write-Host "Compile FAILED! Please check the error messages above." -ForegroundColor Red
}