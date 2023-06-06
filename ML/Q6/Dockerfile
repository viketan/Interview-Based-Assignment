# Use a Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Flask app files to the container
COPY app.py .
COPY model.pkl .
COPY scaler.pkl .
COPY X_encoder.pkl .
COPY y_encoder.pkl .

# Expose the port on which the Flask app will run
EXPOSE 5000

# Define the command to run the Flask app
CMD ["python", "app.py"]
