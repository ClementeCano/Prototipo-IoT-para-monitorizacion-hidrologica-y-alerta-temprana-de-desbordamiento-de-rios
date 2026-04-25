# Imagen base
FROM python:3.11-slim

# Variables de entorno (muy importantes)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Carpeta de trabajo
WORKDIR /app

# 🔥 Certificados + utilidades básicas
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && update-ca-certificates && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero (mejor cache)
COPY requirements.txt .

# 🔥 Actualizar pip + instalar deps
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# 🔍 (Opcional) debug para ver archivos en logs
# RUN ls -R /app

# Comando por defecto (modo no bufferizado)
CMD ["python", "-u", "app/scheduler.py"]