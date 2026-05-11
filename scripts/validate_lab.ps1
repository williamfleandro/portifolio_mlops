# scripts/validate_lab.ps1

Write-Host ""
Write-Host "========================================"
Write-Host " VALIDACAO DO LABORATORIO MLOPS"
Write-Host "========================================"
Write-Host ""

$ErrorActionPreference = "Continue"

function Test-Command {
    param (
        [string]$Description,
        [scriptblock]$Command
    )

    Write-Host ""
    Write-Host ">> $Description" -ForegroundColor Cyan

    try {
        & $Command
        if ($LASTEXITCODE -eq 0 -or $null -eq $LASTEXITCODE) {
            Write-Host "[OK] $Description" -ForegroundColor Green
        } else {
            Write-Host "[ERRO] $Description" -ForegroundColor Red
        }
    } catch {
        Write-Host "[ERRO] $Description" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
    }
}

Test-Command "Docker ativo" {
    docker ps
}

Test-Command "Kubernetes respondendo" {
    kubectl get nodes
}

Test-Command "Namespace mlops-demo" {
    kubectl get pods -n mlops-demo
}

Test-Command "Argo CD ativo" {
    kubectl get pods -n argocd
}

Test-Command "Argo Rollouts ativo" {
    kubectl get pods -n argo-rollouts
}

Test-Command "Backend /health" {
    curl.exe --max-time 10 http://localhost:30081/health
}

Test-Command "Backend /model" {
    curl.exe --max-time 10 http://localhost:30081/model
}

Test-Command "Backend /schema" {
    curl.exe --max-time 10 http://localhost:30081/schema
}

Test-Command "Backend /metrics" {
    curl.exe --max-time 10 http://localhost:30081/metrics
}

Test-Command "Frontend acessivel" {
    curl.exe --max-time 10 http://localhost:30080
}

Test-Command "MLflow acessivel" {
    curl.exe --max-time 10 http://localhost:5000
}

Test-Command "MinIO Console acessivel" {
    curl.exe --max-time 10 http://localhost:9001
}

Test-Command "Feast offline retrieval" {
    python .\scripts\validate_feast_offline.py
}

Test-Command "Feast online retrieval" {
    python .\scripts\validate_feast_online.py
}

Write-Host ""
Write-Host "========================================"
Write-Host " VALIDACAO FINALIZADA"
Write-Host "========================================"
Write-Host ""