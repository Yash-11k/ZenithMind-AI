# ZenithMind-AI
AI-powered student burnout prediction system using machine learning and behavioral analytics. Built with XGBoost, Streamlit, and interactive visualizations to detect academic burnout risk.




## 🚀 How to Run This Project Locally

You can run this project on your local machine using **Docker**. This ensures that all dependencies, including Streamlit, XGBoost, and Scikit-learn, work perfectly without setting up a local virtual environment manually.

### Prerequisites
Make sure you have [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running on your system.

### Step-by-Step Guide

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/Yash-11k/ZenithMind-AI.git](https://github.com/Yash-11k/ZenithMind-AI.git)
   cd ZenithMind-AI

# Build the Docker Image:
Run the following command to build the production-ready Docker container:

Bash
docker build -t zenithmind-app .
Run the Container:
Start the application by mapping the ports:

Bash
docker run -p 8501:8501 zenithmind-app
Access the App:
Open your web browser and go to:

http://localhost:8501
