# Use the official Python 3.9 slim image as the base image
FROM python:3.9-slim-bullseye

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (e.g., gcc for compiling Python packages, curl for health checks)
RUN apt-get update && apt-get install -y gcc curl && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies from the requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code into the container
COPY . .

# Ensure the worker.py script is executable
RUN chmod +x /app/worker.py

# Expose the worker's metrics port
EXPOSE 2112

# Set the default command to run the worker
CMD ["python", "-u", "app.py"]