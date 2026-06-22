FROM python:3.11-slim

# Inštalácia systémových nástrojov
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Vytvorenie štruktúry priečinkov
RUN mkdir -p /app/data/notebooks /app/data/models_cache
ENV HF_HOME=/app/data/models_cache

# Inštalácia requirements (obsahujúce torch-cpu)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Skopírovanie zvyšku appky
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]