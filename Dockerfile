FROM python:3.11-slim

# ------------------------------------------------------------------
# System setup
# ------------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# ------------------------------------------------------------------
# Install Python deps
# ------------------------------------------------------------------
COPY pyproject.toml ./

RUN pip install --no-cache-dir \
    "numpy<2" \
    torch==2.2.0+cpu --index-url https://download.pytorch.org/whl/cpu




# ------------------------------------------------------------------
# Copy source code
# ------------------------------------------------------------------
COPY . .

RUN pip install --no-cache-dir .

EXPOSE 8501

CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]