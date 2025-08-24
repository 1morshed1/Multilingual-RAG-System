# Use Python 3.9 as base image
FROM python:3.9

# Set working directory
WORKDIR /app

# Set Python buffering to 0 to ensure logs are displayed immediately
ENV PYTHONUNBUFFERED=1

# Install system dependencies needed for some Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Expose port 7860 for Gradio interface
EXPOSE 7860

# Expose port 8000 for FastAPI
EXPOSE 8000

# Command to run the Gradio application by default
CMD ["python", "gradio_app.py"]
