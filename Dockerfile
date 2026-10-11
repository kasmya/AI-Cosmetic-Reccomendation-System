FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY cosmetic_rec ./cosmetic_rec
COPY data ./data
COPY frontend ./frontend
RUN mkdir -p models && python -m cosmetic_rec.train
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "cosmetic_rec.api:app", "--host", "0.0.0.0", "--port", "8000"]
