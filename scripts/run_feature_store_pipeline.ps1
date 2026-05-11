Write-Host ""
Write-Host "========================================"
Write-Host " FEATURE STORE PIPELINE - FEAST"
Write-Host "========================================"
Write-Host ""

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
$FeatureStorePath = Join-Path $ProjectRoot "feature_store"

Set-Location $ProjectRoot

Write-Host ">> Etapa 1 - Consumindo eventos Kafka e atualizando Parquet" -ForegroundColor Cyan

python .\consumers\kafka_predictions_to_parquet.py `
  --from-beginning `
  --max-messages 100 `
  --idle-timeout-seconds 20

Write-Host ""
Write-Host ">> Etapa 2 - Aplicando definicoes do Feast" -ForegroundColor Cyan

Set-Location $FeatureStorePath
feast apply

Write-Host ""
Write-Host ">> Etapa 3 - Materializando Online Store" -ForegroundColor Cyan

$ts = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss")
feast materialize-incremental $ts

Write-Host ""
Write-Host ">> Etapa 4 - Validando Offline Retrieval" -ForegroundColor Cyan

Set-Location $ProjectRoot
python .\scripts\validate_feast_offline.py

Write-Host ""
Write-Host ">> Etapa 5 - Validando Online Retrieval" -ForegroundColor Cyan

python .\scripts\validate_feast_online.py

Write-Host ""
Write-Host "========================================"
Write-Host " FEATURE STORE PIPELINE FINALIZADO"
Write-Host "========================================"
Write-Host ""