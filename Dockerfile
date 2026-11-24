FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir scikit-learn==1.5.2 joblib==1.4.2 numpy==2.1.3
COPY src/ src/
COPY model/model.joblib model/metrics.json model/
CMD ["python", "-m", "src.serve"]
