#!/bin/bash
set -e
echo "Entrenando modelo..."
# Ejecutar el script de entrenamiento
cd /app
python -m src.main || true

echo "Iniciando API en puerto 8000..."
# Ejecutar la API
exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000
