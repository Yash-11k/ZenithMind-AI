# Step 1: Use an official light Python version as base image
FROM python:3.10-slim

# Step 2: Set the working directory inside the docker container
WORKDIR /app

# Step 3: Copy your project files into the container
COPY . /app

# Step 4: Install all the libraries listed in your requirements.txt

RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Step 5: Expose the port Streamlit usually runs on (8501)
EXPOSE 8501

# Step 6: Command to run your Streamlit app automatically when container starts
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]