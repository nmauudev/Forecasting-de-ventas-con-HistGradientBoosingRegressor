# ============================================================
# Dockerfile - Simulador de Ventas (Streamlit)
# ============================================================
# Imagen base oficial de Python (slim para menor tamaño)
FROM python:3.10-slim

# Mantener metadatos del autor
LABEL maintainer="Forecasting de Ventas"
LABEL description="Simulador de Ventas Noviembre 2025 - HistGradientBoostingRegressor"
LABEL version="1.0"

# Variables de entorno para Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar requirements primero para aprovechar la caché de Docker
# (si no cambian los requirements, esta capa se reutiliza)
COPY app/requirements.txt ./requirements.txt

# Instalar dependencias del sistema necesarias para algunas librerías
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copiar el código de la aplicación
COPY app/app.py ./app/app.py

# Copiar el modelo entrenado
COPY models/ ./models/

# Copiar los datos procesados necesarios para inferencia
COPY data/processed/ ./data/processed/

# Copiar los tests (necesario para poder ejecutarlos dentro del contenedor si se desea)
COPY tests/ ./tests/

# Instalar pytest para poder correr los tests dentro del contenedor si se necesita
RUN pip install pytest

# Exponer el puerto de Streamlit
EXPOSE 8501

# Healthcheck: verifica que la app responde
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Punto de entrada: ejecutar Streamlit
# --server.address=0.0.0.0 para que sea accesible desde fuera del contenedor
# --server.headless=true para modo sin navegador
CMD ["streamlit", "run", "app/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
