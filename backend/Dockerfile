FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir mlflow fastapi uvicorn boto3 botocore

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]