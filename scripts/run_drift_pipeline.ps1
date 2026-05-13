Write-Host ""
Write-Host "========================================"
Write-Host " DRIFT PIPELINE - EVIDENTLY + BACKEND"
Write-Host "========================================"
Write-Host ""

$ErrorActionPreference = "Stop"

# Caminho robusto: pega a pasta scripts/ e volta um nível para a raiz do projeto
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")

Set-Location $ProjectRoot

Write-Host ">> Etapa 1 - Executando analise de drift com Evidently" -ForegroundColor Cyan

python .\monitoring\check_drift.py

Write-Host ""
Write-Host ">> Etapa 2 - Validando arquivo latest_drift_metrics.json" -ForegroundColor Cyan

$DriftMetricsPath = Join-Path $ProjectRoot "reports\latest_drift_metrics.json"

if (!(Test-Path $DriftMetricsPath)) {
    throw "Arquivo de metricas de drift nao encontrado: $DriftMetricsPath"
}

Get-Content $DriftMetricsPath

Write-Host ""
Write-Host ">> Etapa 3 - Atualizando metricas de drift no backend" -ForegroundColor Cyan

curl.exe --max-time 30 -X POST http://localhost:30081/drift/refresh

Write-Host ""
Write-Host ">> Etapa 4 - Validando status de drift no backend" -ForegroundColor Cyan

curl.exe --max-time 30 http://localhost:30081/drift/status

Write-Host ""
Write-Host ">> Etapa 5 - Validando metricas Prometheus expostas pelo backend" -ForegroundColor Cyan

curl.exe --max-time 30 http://localhost:30081/metrics | findstr /I "drift"

Write-Host ""
Write-Host "========================================"
Write-Host " DRIFT PIPELINE FINALIZADO"
Write-Host "========================================"
Write-Host ""