# Stage 1: Builder
FROM python:3.9-slim as builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY req_prod.txt .
RUN pip install --user -r req_prod.txt

# Stage 2: Final Image
FROM python:3.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH
ENV STREAMLIT_SERVER_PORT=8501

EXPOSE 8502

CMD ["streamlit", "run", "webapp/app.py", "--server.port=8502", "--server.address=0.0.0.0"]
