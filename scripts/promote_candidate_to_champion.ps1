Write-Host ""
Write-Host "========================================"
Write-Host " PROMOTION PIPELINE - CANDIDATE -> CHAMPION"
Write-Host "========================================"
Write-Host ""

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")

Set-Location $ProjectRoot

Write-Host ">> Etapa 1 - Configurando variaveis" -ForegroundColor Cyan

$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
$env:MODEL_NAME = "apartment-price-regression"
$env:CANDIDATE_ALIAS = "candidate"
$env:CHAMPION_ALIAS = "champion"

$env:BACKEND_MODEL_LOAD_URL = "http://localhost:30081/model/load"
$env:BACKEND_MODEL_URL = "http://localhost:30081/model"
$env:BACKEND_HEALTH_URL = "http://localhost:30081/health"

# Gates de qualidade
$env:PROMOTION_MIN_TEST_R2 = "0.98"
$env:PROMOTION_MAX_TEST_MAPE = "6.0"
$env:PROMOTION_MAX_TEST_RMSE = "100000"
$env:PROMOTION_MAX_SMOKE_SECONDS = "0.1"

Write-Host "MLFLOW_TRACKING_URI=$env:MLFLOW_TRACKING_URI"
Write-Host "MODEL_NAME=$env:MODEL_NAME"
Write-Host "CANDIDATE_ALIAS=$env:CANDIDATE_ALIAS"
Write-Host "CHAMPION_ALIAS=$env:CHAMPION_ALIAS"

Write-Host ""
Write-Host ">> Etapa 2 - Validando candidate e promovendo se aprovado" -ForegroundColor Cyan

python .\scripts\promote_candidate_to_champion.py

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "[OK] Candidate promovido para champion com sucesso." -ForegroundColor Green
}
elseif ($LASTEXITCODE -eq 2) {
    Write-Host ""
    Write-Host "[BLOQUEADO] Candidate nao passou nos criterios de promocao." -ForegroundColor Yellow
    Write-Host "Consulte: reports\promotion_result.json" -ForegroundColor Yellow
    exit 2
}
else {
    Write-Host ""
    Write-Host "[ERRO] Falha no pipeline de promocao." -ForegroundColor Red
    Write-Host "Consulte: reports\promotion_result.json" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host ">> Etapa 3 - Validacao final do backend" -ForegroundColor Cyan

curl.exe --max-time 30 http://localhost:30081/model
curl.exe --max-time 30 http://localhost:30081/health

Write-Host ""
Write-Host "========================================"
Write-Host " PROMOTION PIPELINE FINALIZADO"
Write-Host "========================================"
Write-Host ""