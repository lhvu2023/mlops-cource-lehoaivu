FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/ ./scripts/

ENV PYTHONPATH=/app

CMD ["python3", "./scripts/session_3/api.py"]