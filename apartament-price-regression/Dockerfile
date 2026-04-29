FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir "mlflow>=3.1"

COPY exported_model /app/exported_model

RUN if [ -f /app/exported_model/requirements.txt ]; then \
      pip install --no-cache-dir -r /app/exported_model/requirements.txt; \
    fi

EXPOSE 8080

CMD ["mlflow", "models", "serve", "-m", "/app/exported_model", "-h", "0.0.0.0", "-p", "8080", "--env-manager", "local"]
