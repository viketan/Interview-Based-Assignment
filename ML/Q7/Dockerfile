# Use a Python base image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy the necessary files to the container
COPY requirements.txt .
COPY genre_model.pkl .
COPY scaler.pkl .
COPY encoder.pkl .
COPY app.py .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the command to run when the container starts
CMD ["python", "app.py"]
