# Use an official Python image as a base
FROM python:3.9-slim

# Install build tools (gcc, make, etc.) to handle compiling native extensions
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the dependencies listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application files into the container
COPY . /app

# Expose port 8501 to communicate with the Streamlit app
EXPOSE 8501

# Set the default command to run your Streamlit app
CMD ["streamlit", "run", "app.py"]
