FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

CMD ["pytest", "tests/", "-v", "--cov=src", "--cov=strategies", "--cov=data", "--cov-report=term-missing"]
