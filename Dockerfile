FROM python:3.11-slim

# Install system dependencies for building C-based libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure the execution script is ready
RUN chmod +x start.sh

# Expose ports for internal and external traffic
EXPOSE 8000
EXPOSE 8501

# Run the process manager
CMD ["bash", "start.sh"]