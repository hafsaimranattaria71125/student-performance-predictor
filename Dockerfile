FROM python:3.9-slim

WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py .
COPY model.py .
COPY schemas.py .
COPY train_model.py .

# Copy model artifacts
COPY model.pkl .
COPY scaler.pkl .
COPY model_meta.pkl .
COPY test_indices.npy .

# Copy dataset
COPY StudentPerformanceFactors.csv .

# Expose port (Hugging Face uses 7860)
EXPOSE 7860

# Run FastAPI with host 0.0.0.0 for external access
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
