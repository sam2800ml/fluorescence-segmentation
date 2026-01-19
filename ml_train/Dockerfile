FROM python:3.11

# Set the working directory
WORKDIR /workdir

# 1. Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy EVERYTHING from the ml_train folder into the image
# This ensures train_model.py, dataset_loader.py, etc., are all there
COPY . .

# 3. Tell Python to look in /workdir for your imports
ENV PYTHONPATH="/workdir"