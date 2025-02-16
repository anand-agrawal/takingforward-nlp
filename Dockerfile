FROM python:3.9-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Procfile and application code
COPY Procfile .
COPY app.py .

# Set environment variable
ENV PORT=8080

# Expose the port
EXPOSE 8080

# Use the Procfile command
CMD ["python", "app.py"]