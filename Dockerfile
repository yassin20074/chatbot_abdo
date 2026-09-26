FROM python:3.11-slim

 
WORKDIR /app_root

    
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
 
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

 
COPY requirements.txt /app_root/requirements.txt

 
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app_root/requirements.txt

 
COPY . /app_root/

 
EXPOSE 8000

  
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
