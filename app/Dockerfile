FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# 1. Dependencias base del sistema (como root)
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# 2. Repositorio Microsoft para ODBC (Debian 12 Bookworm)
RUN curl -sSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/microsoft.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install -y msodbcsql18 && \
    rm -rf /var/lib/apt/lists/*

# 3. Dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. App
COPY --chown=user . /app

# HF Spaces usa puerto 7860
ENV STREAMLIT_SERVER_PORT=7860
EXPOSE 7860

HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
