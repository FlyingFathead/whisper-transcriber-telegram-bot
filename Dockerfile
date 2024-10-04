FROM python:slim-bookworm

# Install dependencies & clean up after to reduce Docker file size
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application files
COPY src src
COPY config config

# Set `PYTHONUNBUFFERED` to `1` for better logging/debugging
ENV PYTHONUNBUFFERED=1

# Set environment variable to indicate running in Docker
ENV RUNNING_IN_DOCKER=true

# Optional: List files and print working directory for debugging
# (Comment these out in production)
# RUN ls -lsa
# RUN pwd

# Define the default command to run the application
CMD ["python3", "src/main.py"]