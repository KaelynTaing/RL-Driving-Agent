# Compatible w/ CARLA
FROM python:3.11-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    git wget curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python dependencies
COPY requirements.txt /workspace/requirements.txt
RUN pip install -r requirements.txt

# Copy your project files
COPY . /workspace

# Default command
CMD ["python", "train_test.py"]
