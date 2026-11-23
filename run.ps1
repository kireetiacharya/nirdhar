# Nirdhar: build data, train, and serve.  Usage:  .\run.ps1   (add -SkipTrain to just serve)
param([switch]$SkipTrain)
Set-Location $PSScriptRoot
if (-not $SkipTrain) {
    if ((Test-Path ..\western-zone.gpkg) -and (Test-Path ..\Indian_housing_Pune_data.csv)) {
        python -m nirdhar.build_real_pune --gpkg ..\western-zone.gpkg --rents ..\Indian_housing_Pune_data.csv
    }
    python -m nirdhar.generate
    if ($LASTEXITCODE -ne 0) { exit 1 }
    python -W ignore -m nirdhar.train
    if ($LASTEXITCODE -ne 0) { exit 1 }
}
Write-Host "Nirdhar running at http://localhost:8000"
python -W ignore -m uvicorn app:app --port 8000
