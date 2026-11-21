FROM python:3.11-slim

WORKDIR /app

# Dependencias primero, para aprovechar la caché de Docker si solo cambia el código
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código, configuración, modelos ya entrenados y los datos históricos que la
# app necesita (no se conecta a SIVIGILA/GEE en vivo, ver app/main.py)
COPY src/ src/
COPY app/ app/
COPY models/ models/
COPY data/ data/
COPY config.yaml .

EXPOSE 8501

# Parametrización vía variables de entorno (ver src/config.py); valores por
# defecto, se pueden sobreescribir al correr el contenedor con --env-set, ej.:
#   docker run -e MUNICIPIO_DEFAULT=50001 ...
ENV MUNICIPIO_DEFAULT=73001

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
