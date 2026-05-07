# Portifólio MLOps — MLflow, FastAPI, Kubernetes, Argo CD, Prometheus, Grafana e Evidently AI

Projeto de MLOps completo para previsão de preço de apartamentos, com versionamento de modelo no MLflow, armazenamento de artefatos no MinIO, serving via FastAPI, frontend React, deploy em Kubernetes, GitOps com Argo CD, rollout canário com Argo Rollouts, observabilidade com Prometheus/Grafana, detecção de drift com Evidently AI e retreinamento orientado por métricas.

---

## 1. Objetivo do projeto

O objetivo deste laboratório é demonstrar um ciclo MLOps próximo de um ambiente real de produção:

```text
Train → Register → Deploy → Monitor → Detect Drift → Retrain → Promote → Reload
```

O projeto não se limita a treinar e publicar um modelo. Ele inclui monitoramento operacional, métricas de inferência, métricas de drift, dashboards no Grafana, estratégia de retreinamento e promoção automática de uma nova versão do modelo no MLflow Registry.

---

## 2. Problema de negócio

O modelo realiza a previsão de preço de apartamentos com base em características do imóvel, como:

```text
cidade
área em m²
quartos
banheiros
andar
vagas
nota do bairro
valor do condomínio
idade do imóvel
distância ao centro
```

A aplicação simula um cenário real no qual um modelo de regressão precisa ficar disponível para consulta via API e interface web, mantendo rastreabilidade, versionamento e observabilidade contínua.

---

## 3. Arquitetura geral

```text
Frontend React/Nginx
        ↓
Backend FastAPI
        ↓
MLflow Registry → models:/apartment-price-regression@champion
        ↓
MinIO Artifact Store

Backend FastAPI
        ↓
/metrics
        ↓
Prometheus
        ↓
Grafana

Evidently AI
        ↓
reports/latest_drift_metrics.json
        ↓
ConfigMap runtime no Kubernetes
        ↓
Backend /drift/refresh
        ↓
Prometheus / Grafana

Drift detectado
        ↓
monitoring/retrain_if_needed.py
        ↓
apartament-price-regression/train.py
        ↓
MLflow Registry
        ↓
Novo alias champion
        ↓
Backend /model/reload
```

---

## 4. Tecnologias utilizadas

| Camada | Tecnologia |
|---|---|
| Linguagem | Python |
| Modelo | Scikit-learn / RandomForestRegressor |
| API | FastAPI |
| Frontend | React + Nginx |
| Registry | MLflow Model Registry |
| Artifact Store | MinIO |
| Containerização | Docker |
| Orquestração | Kubernetes |
| GitOps | Argo CD |
| Rollout | Argo Rollouts |
| Métricas | Prometheus |
| Dashboards | Grafana |
| Drift | Evidently AI |
| Retreinamento | Script Python orientado por métricas |
| Automação | PowerShell + kubectl |

---

## 5. Estrutura principal do projeto

```text
portifolio_mlops/
│
├── apartament-price-regression/
│   ├── train.py
│   ├── data/
│   ├── local_artifacts/
│   ├── mlflow.db
│   └── requirements.txt
│
├── backend/
│   ├── app/
│   │   └── main.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── nginx.conf
│   ├── Dockerfile
│   └── package.json
│
├── k8s/
│   ├── backend-rollout.yaml
│   ├── frontend-rollout.yaml
│   └── drift-metrics-configmap.yaml
│
├── monitoring/
│   ├── check_drift.py
│   ├── publish_drift_metrics.ps1
│   ├── retrain_if_needed.py
│   └── models.yaml
│
├── reports/
│   ├── data_drift_report.html
│   ├── drift_result.json
│   ├── latest_drift_metrics.json
│   └── retrain_history.jsonl
│
├── minio-data/
├── mlruns/
├── docker-compose.yml
├── README.md
└── requirements.txt
```

---

## 6. Modelo de Machine Learning

O modelo principal é:

```text
apartment-price-regression
```

O backend carrega o modelo ativo usando o alias:

```text
models:/apartment-price-regression@champion
```

O fluxo usa o MLflow Registry para controlar versões do modelo. Quando o retreinamento é executado com sucesso, uma nova versão é registrada e o alias `champion` é atualizado.

Exemplo de resposta do backend após o reload:

```json
{
  "status": "loaded",
  "model_uri": "models:/apartment-price-regression@champion",
  "model_name": "apartment-price-regression",
  "model_version": "4",
  "model_alias": "champion",
  "tracking_uri": "http://host.docker.internal:5000"
}
```

---

## 7. MLflow e MinIO

O MLflow é usado como plataforma central de experiment tracking e registry.

O MinIO é usado como artifact store para armazenar artefatos do modelo.

Exemplo de variáveis utilizadas:

```powershell
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
$env:MLFLOW_S3_ENDPOINT_URL="http://127.0.0.1:9000"
$env:AWS_ACCESS_KEY_ID="minioadmin"
$env:AWS_SECRET_ACCESS_KEY="minioadmin"
```

Acesso local:

```text
MLflow: http://localhost:5000
MinIO Console: http://localhost:9001
```

---

## 8. Backend FastAPI

O backend expõe endpoints para predição, status do modelo, reload do modelo e métricas Prometheus.

| Endpoint | Método | Função |
|---|---:|---|
| `/health` | GET | Verifica saúde da API |
| `/model` | GET | Mostra modelo carregado |
| `/model/load` | POST | Carrega modelo |
| `/model/reload` | POST | Recarrega modelo champion |
| `/predict` | POST | Realiza predição |
| `/invocations` | POST | Endpoint compatível com serving |
| `/metrics` | GET | Métricas Prometheus |
| `/drift/status` | GET | Mostra métricas de drift carregadas |
| `/drift/refresh` | POST | Recarrega métricas de drift do arquivo JSON |

---

## 9. Logging de predições

O backend registra logs de predição em:

```text
/app/data/predictions/predictions_log.csv
```

Variável configurada no Kubernetes:

```yaml
- name: PREDICTION_LOG_PATH
  value: "/app/data/predictions/predictions_log.csv"
```

Esse arquivo armazena informações úteis para auditoria e monitoramento:

```text
timestamp
endpoint
status
model_uri
model_name
model_version
model_alias
prediction
score
latency_ms
input_features
error_message
```

---

## 10. Métricas Prometheus do backend

O backend expõe métricas no endpoint:

```text
http://localhost:30081/metrics
```

Principais métricas:

```text
model_prediction_total
model_prediction_latency_seconds
model_prediction_batch_size
model_last_prediction_value
model_data_drift_score
model_prediction_drift_score
model_drift_detected
http_requests_total
http_request_duration_seconds
process_resident_memory_bytes
process_cpu_seconds_total
```

---

## 11. Kubernetes e Argo Rollouts

O backend é controlado por um `Rollout` do Argo Rollouts com estratégia canária:

```yaml
strategy:
  canary:
    steps:
      - setWeight: 20
      - pause:
          duration: 30s
      - setWeight: 50
      - pause:
          duration: 30s
      - setWeight: 100
```

Aplicar manifesto:

```powershell
kubectl apply -f .\k8s\backend-rollout.yaml
```

Acompanhar rollout:

```powershell
kubectl argo rollouts get rollout apartment-price-backend -n mlops-demo --watch
```

Promover se pausar no canário:

```powershell
kubectl argo rollouts promote apartment-price-backend -n mlops-demo --full
```

---

## 12. Services do backend

O backend possui dois Services:

```text
apartment-price-backend          ClusterIP
apartment-price-backend-metrics  NodePort 30081
```

O `NodePort 30081` permite que o Prometheus externo colete o endpoint `/metrics`.

Exemplo:

```powershell
curl.exe http://localhost:30081/metrics
```

---

## 13. Prometheus

O Prometheus coleta as métricas do backend pelo target:

```text
192.168.56.1:30081
```

Exemplo de configuração:

```yaml
scrape_configs:
  - job_name: apartment-price-backend
    metrics_path: /metrics
    scrape_interval: 15s
    static_configs:
      - targets:
          - 192.168.56.1:30081
```

Validar configuração:

```bash
sudo docker exec prometheus promtool check config /etc/prometheus/prometheus.yml
```

Reiniciar Prometheus:

```bash
sudo docker restart prometheus
```

Acessar targets:

```text
http://192.168.56.50:9090/targets
```

---

## 14. Grafana

O Grafana é usado para visualizar métricas operacionais, métricas de inferência e métricas de drift.

Acesso:

```text
http://192.168.56.50:3000
```

### Painéis principais de inferência

| Painel | Query PromQL |
|---|---|
| Backend UP | `up{job="apartment-price-backend"}` |
| Total Predictions | `sum(model_prediction_total)` |
| Predictions per Minute | `sum(rate(model_prediction_total[5m])) * 60` |
| Average Model Latency ms | `(rate(model_prediction_latency_seconds_sum[5m]) / rate(model_prediction_latency_seconds_count[5m])) * 1000` |
| Model Latency p95 ms | `histogram_quantile(0.95, sum(rate(model_prediction_latency_seconds_bucket[5m])) by (le)) * 1000` |
| Last Prediction Value | `model_last_prediction_value` |
| HTTP Requests por endpoint | `sum by (handler, method, status) (rate(http_requests_total{job="apartment-price-backend"}[5m]))` |
| Memory MB | `process_resident_memory_bytes{job="apartment-price-backend"} / 1024 / 1024` |

---

## 15. Evidently AI para detecção de drift

O script principal de drift é:

```text
monitoring/check_drift.py
```

Ele compara os datasets:

```text
data/reference/reference_apartments.csv
data/current/current_apartments.csv
```

E gera:

```text
reports/data_drift_report.html
reports/drift_result.json
reports/latest_drift_metrics.json
```

Executar:

```powershell
python .\monitoring\check_drift.py
```

Exemplo de saída consolidada:

```json
{
  "data_drift_score": 0.5556,
  "prediction_drift_score": 0.0,
  "drift_detected": true,
  "drifted_features_count": 5,
  "drifted_features_ratio": 0.5556,
  "total_features_count": 9,
  "max_feature_drift_score": 0.7002,
  "mean_feature_drift_score": 0.62096,
  "current_dataset_rows": 800,
  "reference_dataset_rows": 2500,
  "retrain_required": true
}
```

---

## 16. Publicação das métricas de drift no Kubernetes

O arquivo gerado pelo Evidently é publicado em um ConfigMap runtime:

```text
drift-metrics-runtime-config
```

Comando manual:

```powershell
kubectl create configmap drift-metrics-runtime-config `
  -n mlops-demo `
  --from-file=latest_drift_metrics.json=.\reports\latest_drift_metrics.json `
  --dry-run=client -o yaml | kubectl apply -f -
```

Esse ConfigMap é montado no backend em:

```text
/app/data/monitoring/latest_drift_metrics.json
```

O `backend-rollout.yaml` usa:

```yaml
volumeMounts:
  - name: drift-monitoring
    mountPath: /app/data/monitoring
    readOnly: true

volumes:
  - name: drift-monitoring
    configMap:
      name: drift-metrics-runtime-config
```

---

## 17. Automação da publicação de drift

Script:

```text
monitoring/publish_drift_metrics.ps1
```

Executar:

```powershell
.\monitoring\publish_drift_metrics.ps1
```

Esse script realiza:

```text
1. Executa o Evidently
2. Gera reports/latest_drift_metrics.json
3. Atualiza o ConfigMap runtime
4. Reinicia o backend Rollout
5. Chama /drift/refresh
6. Valida /drift/status
7. Valida /metrics
```

---

## 18. Métricas de drift no Grafana

Foram criadas 10 visualizações para drift:

| Nº | Visualização | PromQL |
|---:|---|---|
| 1 | Data Drift Score | `model_data_drift_score{job="apartment-price-backend"}` |
| 2 | Prediction Drift Score | `model_prediction_drift_score{job="apartment-price-backend"}` |
| 3 | Drift Detected | `model_drift_detected{job="apartment-price-backend"}` |
| 4 | Data Drift Trend | `model_data_drift_score{job="apartment-price-backend"}` |
| 5 | Prediction Drift Trend | `model_prediction_drift_score{job="apartment-price-backend"}` |
| 6 | Drift Status Timeline | `model_drift_detected{job="apartment-price-backend"}` |
| 7 | Maximum Drift Score | `max({__name__=~"model_data_drift_score|model_prediction_drift_score", job="apartment-price-backend"})` |
| 8 | Max Data Drift - Last 1h | `max_over_time(model_data_drift_score{job="apartment-price-backend"}[1h])` |
| 9 | Drift Refresh Count - Last 1h | `increase(http_requests_total{job="apartment-price-backend", handler="/drift/refresh", status="2xx"}[1h])` |
| 10 | Drift Refresh Latency p95 ms | `histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{job="apartment-price-backend", handler="/drift/refresh"}[5m])) by (le)) * 1000` |

Thresholds sugeridos:

```text
0.00 até 0.20 = verde
0.20 até 0.30 = amarelo
acima de 0.30 = vermelho
```

Para `model_drift_detected`:

```text
0 = No Drift
1 = Drift Detected
```

---

## 19. Retreinamento orientado por drift

O script responsável por avaliar se o modelo deve ser retreinado é:

```text
monitoring/retrain_if_needed.py
```

Ele lê:

```text
reports/latest_drift_metrics.json
```

Critérios atuais:

```text
retrain_required = true
ou drift_detected = true
ou data_drift_score >= 0.30
ou prediction_drift_score >= 0.30
```

Executar:

```powershell
python .\monitoring\retrain_if_needed.py
```

Fluxo executado:

```text
1. Carrega métricas de drift
2. Avalia thresholds
3. Executa apartament-price-regression/train.py
4. Registra nova versão no MLflow
5. Atualiza alias champion
6. Chama /model/reload no backend
7. Salva histórico em reports/retrain_history.jsonl
```

---

## 20. Histórico de retreinamento

O histórico é gravado em:

```text
reports/retrain_history.jsonl
```

Validar:

```powershell
Get-Content .\reports\retrain_history.jsonl
```

Esse arquivo registra:

```text
timestamp
métricas de drift
thresholds usados
decisão de retreinamento
script executado
stdout
stderr
status
resultado do reload do backend
```

---

## 21. Reload do modelo no backend

Após a promoção de uma nova versão no MLflow, o backend pode ser recarregado com:

```powershell
curl.exe -X POST http://localhost:30081/model/reload
```

Validar modelo ativo:

```powershell
curl.exe http://localhost:30081/model
```

Exemplo esperado:

```json
{
  "status": "loaded",
  "model_name": "apartment-price-regression",
  "model_version": "4",
  "model_alias": "champion"
}
```

---

## 22. Fluxo GitOps com Argo CD

A aplicação `portifolio-mlops` é controlada pelo Argo CD a partir da branch `main` do GitHub.

Após alterações em manifests ou scripts versionados:

```powershell
git status
git add .
git commit -m "Describe change"
git push origin main
```

Sincronizar pelo Argo CD:

```text
portifolio-mlops → REFRESH → SYNC → SYNCHRONIZE
```

Ou forçar refresh:

```powershell
kubectl annotate application portifolio-mlops `
  -n argocd `
  argocd.argoproj.io/refresh=hard `
  --overwrite
```

---

## 23. Comandos úteis

### Ver Pods

```powershell
kubectl get pods -n mlops-demo
```

### Ver Services

```powershell
kubectl get svc -n mlops-demo
```

### Ver Rollout

```powershell
kubectl argo rollouts get rollout apartment-price-backend -n mlops-demo
```

### Ver imagem ativa do backend

```powershell
kubectl get rollout apartment-price-backend -n mlops-demo -o jsonpath="{.spec.template.spec.containers[0].image}"
```

### Testar API

```powershell
curl.exe http://localhost:30081/health
curl.exe http://localhost:30081/model
curl.exe http://localhost:30081/metrics
```

### Ver métricas de drift

```powershell
curl.exe http://localhost:30081/metrics | findstr drift
```

### Publicar drift

```powershell
.\monitoring\publish_drift_metrics.ps1
```

### Retreinar se necessário

```powershell
python .\monitoring\retrain_if_needed.py
```

---

## 24. Boas práticas adotadas

```text
Versionamento de modelo com MLflow
Alias champion para produção
Separação entre infraestrutura GitOps e dados runtime de drift
ConfigMap runtime para métricas dinâmicas
Observabilidade via Prometheus
Dashboard operacional no Grafana
Deploy canário com Argo Rollouts
Retreinamento orientado por métricas
Reload explícito do modelo em produção
Histórico de retreinamento em JSONL
```

---

## 25. Próximas evoluções

Possíveis melhorias futuras:

```text
1. Criar alertas no Grafana para drift alto.
2. Criar alerta quando /drift/refresh não for executado.
3. Enviar eventos de predição para Kafka.
4. Criar consumer para montar dataset current automaticamente.
5. Adicionar Feast como Feature Store.
6. Adicionar pipeline CI/CD para build automático de imagens.
7. Criar análise automática candidate vs champion antes da promoção.
8. Adicionar aprovação manual para promoção em produção.
9. Criar job agendado para drift check.
10. Criar job agendado para retreinamento controlado.
```

---

## 26. Status atual

```text
MLflow Registry funcionando
MinIO funcionando
Backend FastAPI funcionando
Frontend React funcionando
Kubernetes funcionando
Argo CD sincronizando
Argo Rollouts funcionando
Prometheus coletando métricas
Grafana exibindo inferência e drift
Evidently detectando drift
Retreinamento acionado por drift
Novo champion carregado no backend
```

---

## 27. Conclusão

Este projeto demonstra um fluxo MLOps completo e prático, indo além do treinamento e deploy tradicional de modelos.

A arquitetura implementa observabilidade, versionamento, detecção de drift, retreinamento e atualização controlada do modelo em produção. Esse tipo de estrutura é essencial para ambientes reais de Machine Learning, nos quais modelos precisam ser continuamente monitorados, avaliados e atualizados com segurança.

## 👨‍💻 Autor

William Ferreira Leandro
linkedin: https://www.linkedin.com/in/william-ferreira-leandro-5b75a925/
