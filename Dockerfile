FROM python:3.10-slim
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 wget unzip && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir -p /app/models && wget -qO /app/models/buffalo_l.zip https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip && unzip -q /app/models/buffalo_l.zip -d /app/models/ && rm /app/models/buffalo_l.zip
COPY src/ ./src/
ENV PYTHONUNBUFFERED=1
EXPOSE 5000
CMD ["gunicorn", "--workers=4", "--bind=0.0.0.0:5000", "src.app:app"]
