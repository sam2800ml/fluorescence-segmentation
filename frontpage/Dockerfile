FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    streamlit \
    requests \
    numpy \
    torch \
    matplotlib \
    pillow \
    opencv-python-headless

COPY . .

EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "webpage.py", "--server.address=0.0.0.0"]