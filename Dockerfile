# Minimal Dockerfile for the ml-app
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    if [ -f /app/requirements.txt ]; then pip install --no-cache-dir -r /app/requirements.txt; fi

# Copy source
COPY src/ /app/src/
COPY models/ /app/models/
COPY README.md /app/README.md

# Default command: show python version and list files (placeholder)
CMD ["python", "-c", "import sys, pathlib; print(sys.version); print(list(pathlib.Path('/app').glob('**/*')))"]

