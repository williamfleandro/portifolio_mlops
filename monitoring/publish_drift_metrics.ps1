Write-Host "1. Running Evidently drift check..." -ForegroundColor Cyan
python .\monitoring\check_drift.py

if (!(Test-Path ".\reports\latest_drift_metrics.json")) {
    Write-Host "Erro: arquivo .\reports\latest_drift_metrics.json nao foi encontrado." -ForegroundColor Red
    exit 1
}

Write-Host "2. Updating Kubernetes ConfigMap..." -ForegroundColor Cyan
kubectl create configmap drift-metrics-runtime-config `
  -n mlops-demo `
  --from-file=latest_drift_metrics.json=.\reports\latest_drift_metrics.json `
  --dry-run=client -o yaml | kubectl apply -f -

Write-Host "3. Restarting backend rollout to reload drift metrics..." -ForegroundColor Cyan
kubectl argo rollouts restart apartment-price-backend -n mlops-demo

Write-Host "4. Waiting rollout..." -ForegroundColor Cyan
kubectl argo rollouts status apartment-price-backend -n mlops-demo --timeout 180s

Write-Host "5. Refreshing drift metrics endpoint..." -ForegroundColor Cyan
curl.exe -X POST http://localhost:30081/drift/refresh

Write-Host "6. Checking drift status..." -ForegroundColor Cyan
curl.exe http://localhost:30081/drift/status

Write-Host "7. Checking Prometheus metrics..." -ForegroundColor Cyan
curl.exe http://localhost:30081/metrics | findstr drift