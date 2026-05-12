FROM python:3.11-slim

# Install essential system dependencies only
# Removed software-properties-common for Debian Trixie compatibility
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install requirements[cite: 3]
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure the execution script is ready[cite: 1]
RUN chmod +x start.sh

# Expose ports for internal (8000) and external (8501) traffic[cite: 1]
EXPOSE 8000
EXPOSE 8501

# Run the process manager[cite: 1]
CMD ["bash", "start.sh"]