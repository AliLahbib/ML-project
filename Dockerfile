# Improved Dockerfile for the ml-app
FROM python:3.12-slim

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies required for scientific Python packages (scikit-learn, numpy)
# Keep image small by cleaning apt lists
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       gcc \
       gfortran \
       libopenblas-dev \
       liblapack-dev \
       ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create app user and working directory
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt /app/requirements.txt

# Upgrade pip and install Python dependencies
RUN python -m pip install --upgrade pip setuptools wheel \
    && if [ -f /app/requirements.txt ]; then pip install --no-cache-dir -r /app/requirements.txt; fi

# Copy application code
COPY . /app

# Set PYTHONPATH so imports like `from model import ...` work (src is on PYTHONPATH)
ENV PYTHONPATH=/app/src

# Give ownership to non-root user and switch
RUN chown -R appuser:appuser /app
USER appuser

# Expose nothing in particular; this image is mainly for running the small CLI prediction example
# Default command: run the example prediction script
CMD ["python", "src/predict.py"]
