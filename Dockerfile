FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY cosmetic_rec ./cosmetic_rec
COPY static ./static
COPY skindataall.csv .
EXPOSE 8000
CMD ["uvicorn", "cosmetic_rec.api:app", "--host", "0.0.0.0", "--port", "8000"]
