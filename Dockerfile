FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY mlruns /app/mlruns

# Permisos de ejecución para el script de la aplicación
RUN chmod +x /app/entrypoint.sh

# Exponer el puerto 8000 para FastAPI.
EXPOSE 8000

# Variables de entorno.
ENV PYTHONUNBUFFERED=1

# Comando para ejecutar la aplicación.
ENTRYPOINT ["/app/entrypoint.sh"]