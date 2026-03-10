FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source files
COPY main.py .
COPY model.pkl .
COPY model_quiz.pkl .

# Serve HTML ด้วย FastAPI static files
COPY real.html ./static/index.html

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
