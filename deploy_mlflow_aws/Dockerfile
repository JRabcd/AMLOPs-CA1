FROM python:3.12-slim

# mlflow: the server; pymysql + cryptography: MySQL (RDS) driver; boto3: S3 for model files
RUN pip install --no-cache-dir "mlflow==3.18.0" "pymysql==1.2.4" "cryptography==50.0.2" "boto3==1.44.13"

WORKDIR /mlflow
COPY start_server.py .

EXPOSE 5000

# Storage settings (RDS + S3) come from Elastic Beanstalk environment properties; see start_server.py
CMD ["python", "start_server.py"]
