# 🧠 MLOps Apartment Price Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![React](https://img.shields.io/badge/React-Frontend-blue)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![MLflow](https://img.shields.io/badge/MLflow-Model%20Registry-orange)
![MinIO](https://img.shields.io/badge/MinIO-S3%20Storage-red)
![Kubernetes](https://img.shields.io/badge/Kubernetes-k3d-blue)
![MLOps](https://img.shields.io/badge/MLOps-End--to--End-success)

Pipeline completo de Machine Learning para previsão de preços de apartamentos, incluindo modelagem, versionamento, serving e frontend.

---

## 🎯 Problema de Negócio

Estimativa automática de preços de imóveis com base em características estruturais e localização.

---

## 📊 Dados

Dataset sintético com 10.000 registros simulando capitais brasileiras.

---

## 🤖 Modelagem

Pipeline:
city → OneHotEncoder → RandomForestRegressor

Métricas:
- MAE
- RMSE
- R²
- MAPE

---

## 🧾 MLflow

- Tracking de experimentos
- Model Registry
- Alias: champion

---

## 🗄️ MinIO

Storage S3 local para artefatos do MLflow.

---

## ⚙️ Backend (FastAPI)

Endpoints:
- /invocations
- /model
- /model/reload

Suporte a reload dinâmico sem restart.

---

## 🌐 Frontend (React)

- Formulário de entrada
- Consumo da API
- Exibição de predição

---

## 🧩 Arquitetura

Frontend → Backend → MLflow → MinIO

---

## 🚀 Próximos passos

- Argo CD
- Canary Deployment
- Monitoramento de drift

---

## 👨‍💻 Autor

William Ferreira Leandro
