# Projeto simples: previsão de preço de apartamentos com MLflow e Docker

Este projeto cria um modelo de regressão para previsão de preço de apartamentos, registra parâmetros, métricas e artefatos no MLflow e exporta o modelo para servir previsões em um container Docker.

## 1. Criar ambiente virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## 2. Configurar o tracking do MLflow

### Opção local mais simples

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

### Subir a interface do MLflow

Em outro terminal:

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 127.0.0.1 \
  --port 5000
```

Abra no navegador:

```text
http://127.0.0.1:5000
```

## 3. Treinar o modelo e gerar artefatos

```bash
python train.py
```

Após o treino, você terá:

- métricas no MLflow
- modelo salvo como artifact no run
- `local_artifacts/feature_importances.csv`
- `local_artifacts/predictions_preview.csv`
- `local_artifacts/real_vs_predicted.png`
- `exported_model/` pronto para serving
- `run_summary.json` com `run_id` e `model_uri`

## 4. Servir localmente com MLflow sem Docker

Se quiser testar primeiro:

```bash
mlflow models serve -m ./exported_model -h 0.0.0.0 -p 8080 --env-manager local
```

Teste a API:

```bash
curl http://127.0.0.1:8080/invocations \
  -H "Content-Type: application/json" \
  --data @sample_request.json
```

## 5. Build e deploy com Docker

### Build

```bash
docker build -t apartment-price-mlflow:latest .
```

### Run

```bash
docker run --rm -p 8080:8080 apartment-price-mlflow:latest
```

### Teste no container

```bash
curl http://127.0.0.1:8080/invocations \
  -H "Content-Type: application/json" \
  --data @sample_request.json
```

## 6. Alternativa nativa do MLflow para gerar a imagem

Depois de treinar, usando o `run_id` do `run_summary.json`:

```bash
mlflow models build-docker -m runs:/SEU_RUN_ID/model -n apartment-price-mlflow
```

## 7. Próximos passos recomendados

- trocar o dataset sintético por um CSV real da sua empresa ou do Kaggle
- registrar múltiplos modelos e comparar runs no MLflow
- criar uma etapa de validação automática antes do deploy
- integrar o build da imagem em CI/CD
