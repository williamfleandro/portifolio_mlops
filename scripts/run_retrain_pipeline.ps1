Write-Host ""
Write-Host "========================================"
Write-Host " RETRAIN PIPELINE - DRIFT + MLFLOW"
Write-Host "========================================"
Write-Host ""

$ErrorActionPreference = "Stop"

# Caminho robusto: pega a pasta scripts/ e volta um nível para a raiz do projeto
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")

Set-Location $ProjectRoot

Write-Host ">> Etapa 1 - Validando metricas de drift" -ForegroundColor Cyan

$DriftMetricsPath = Join-Path $ProjectRoot "reports\latest_drift_metrics.json"

if (!(Test-Path $DriftMetricsPath)) {
    throw "Arquivo latest_drift_metrics.json nao encontrado. Execute antes: .\scripts\run_drift_pipeline.ps1"
}

Get-Content $DriftMetricsPath

Write-Host ""
Write-Host ">> Etapa 2 - Configurando variaveis do MLflow e MinIO" -ForegroundColor Cyan

$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
$env:MLFLOW_S3_ENDPOINT_URL = "http://127.0.0.1:9000"
$env:AWS_ACCESS_KEY_ID = "minioadmin"
$env:AWS_SECRET_ACCESS_KEY = "minioadmin123"

# Mantemos candidate para nao trocar producao automaticamente.
# Depois podemos criar uma etapa separada para promover candidate -> champion.
$env:MODEL_ALIAS = "candidate"

# Thresholds usados pelo retrain_if_needed.py
$env:DATA_DRIFT_THRESHOLD = "0.30"
$env:PREDICTION_DRIFT_THRESHOLD = "0.30"

# URL do backend para recarregar o modelo apos retreino
$env:BACKEND_RELOAD_URL = "http://localhost:30081/model/reload"
$env:RUN_MODEL_RELOAD = "true"

Write-Host "MLFLOW_TRACKING_URI=$env:MLFLOW_TRACKING_URI"
Write-Host "MLFLOW_S3_ENDPOINT_URL=$env:MLFLOW_S3_ENDPOINT_URL"
Write-Host "MODEL_ALIAS=$env:MODEL_ALIAS"
Write-Host "BACKEND_RELOAD_URL=$env:BACKEND_RELOAD_URL"

Write-Host ""
Write-Host ">> Etapa 3 - Executando retreinamento se necessario" -ForegroundColor Cyan

python .\monitoring\retrain_if_needed.py

Write-Host ""
Write-Host ">> Etapa 4 - Validando modelo carregado no backend" -ForegroundColor Cyan

curl.exe --max-time 30 http://localhost:30081/model

Write-Host ""
Write-Host ">> Etapa 5 - Validando schema do backend" -ForegroundColor Cyan

curl.exe --max-time 30 http://localhost:30081/schema

Write-Host ""
Write-Host ">> Etapa 6 - Validando endpoint /health" -ForegroundColor Cyan

curl.exe --max-time 30 http://localhost:30081/health

Write-Host ""
Write-Host ">> Etapa 7 - Validando historico de retreinamento" -ForegroundColor Cyan

$RetrainHistoryPath = Join-Path $ProjectRoot "reports\retrain_history.jsonl"

if (Test-Path $RetrainHistoryPath) {
    Get-Content $RetrainHistoryPath -Tail 5
} else {
    Write-Host "Arquivo retrain_history.jsonl ainda nao existe."
}

Write-Host ""
Write-Host "========================================"
Write-Host " RETRAIN PIPELINE FINALIZADO"
Write-Host "========================================"
Write-Host ""