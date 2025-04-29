FROM python:3.11-slim

# Install system-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libffi-dev \
    liblapack-dev \
    libblas-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /flask-app

# Copy requirements and install Python packages (including cython)
COPY artifacts/requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install --prefer-binary --no-cache-dir cython \
    && pip install --prefer-binary --no-cache-dir -r requirements.txt

COPY . /flask-app/

CMD ["python", "-m", "flask", "--app", "hello_model.py", "run", "--host=0.0.0.0", "--port=8000"]
