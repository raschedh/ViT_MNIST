FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/your-username/your-repo.git /app

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        streamlit \
        torch \
        torchvision \
        tqdm \
        matplotlib

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
