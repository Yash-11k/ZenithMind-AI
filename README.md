# 🧠 ZenithMind AI — Student Mental Health & Burnout Prediction System

An end-to-end Machine Learning project that predicts student burnout risk using academic, lifestyle, and behavioral data.

ZenithMind AI helps identify patterns that may be linked to student burnout. It uses **Random Forest, XGBoost, Python, and Streamlit** to analyze student-related features and provide easy-to-understand predictions through a web application.

The project works with a dataset containing **1M+ rows** and uses 20+ academic and lifestyle features. The trained models achieve **79% classification accuracy** in the reported evaluation.

🚀 **Live Demo:** [Try ZenithMind AI](https://zenithmind-ai-yash11k.streamlit.app/)

💻 **GitHub Repository:** [Yash-11k/ZenithMind-AI](https://github.com/Yash-11k/ZenithMind-AI)

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Why This Project?](#-why-this-project)
- [Key Features](#-key-features)
- [How It Works](#-how-it-works)
- [Machine Learning Models](#-machine-learning-models)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Run with Docker](#-run-with-docker)
- [Run Without Docker](#-run-without-docker)
- [Model Performance](#-model-performance)
- [Example Use Case](#-example-use-case)
- [Future Improvements](#-future-improvements)
- [Limitations and Responsible Use](#-limitations-and-responsible-use)
- [Author](#-author)

---

## 🎯 Project Overview

Student burnout is a growing concern. Academic pressure, poor sleep, lack of physical activity, and an unhealthy daily routine can affect a student's well-being and academic performance.

However, it can be difficult to understand how different factors are related to burnout risk.

**ZenithMind AI uses machine learning to analyze student-related data and estimate burnout risk based on learned patterns.**

The system combines trained machine learning models with a simple Streamlit interface. Users can interact with the application, provide the required information, and view predictions without needing to understand the underlying machine learning code.

### Project Highlights

- **Dataset:** 1M+ rows
- **Features:** 20+ academic and lifestyle features
- **Machine Learning:** Random Forest and XGBoost
- **Reported Classification Accuracy:** 79%
- **Web Interface:** Streamlit
- **Deployment:** Streamlit Community Cloud
- **Containerization:** Docker
- **Application Type:** End-to-end Machine Learning project

---

## 💡 Why This Project?

Many students face challenges related to academic workload, sleep schedules, daily habits, and personal well-being.

ZenithMind AI explores how machine learning can help analyze these factors and identify patterns associated with burnout risk.

The main goals are to:

- Understand the relationship between academic and lifestyle factors.
- Apply machine learning to a real-world style prediction problem.
- Compare different classification models.
- Make machine learning predictions accessible through a web interface.
- Demonstrate an end-to-end workflow, from trained models to application deployment.

The project is also a practical demonstration of how Python, machine learning, and web deployment can work together in one application.

---

## ✨ Key Features

### 1. Student Burnout Risk Prediction

Uses trained machine learning models to estimate burnout risk from the features supported by the application.

### 2. Multiple Machine Learning Models

Includes Random Forest and XGBoost models, allowing the project to explore different approaches to classification.

### 3. Large Dataset Support

The project is designed around a dataset containing more than one million rows, with 20+ academic and lifestyle features.

### 4. Interactive Web Application

The Streamlit interface makes it easier to use the prediction system without running Python scripts manually.

### 5. Real-Time Predictions

The deployed application can generate predictions from user-provided inputs without requiring users to interact directly with the model files.

### 6. Docker Support

A Dockerfile packages the application and its dependencies into a container, helping make local setup and execution more consistent.

### 7. Saved Machine Learning Models

The repository includes serialized model files that the application can load for prediction.

### 8. End-to-End Implementation

The project combines machine learning, application development, dependency management, and deployment in one workflow.

---

## ⚙️ How It Works

The project follows a typical machine learning application workflow.

1. **Input Data:** Academic and lifestyle information is provided in the format supported by the application.
2. **Data Processing:** The application prepares the input features for the trained model.
3. **Model Loading:** The saved Random Forest or XGBoost model is loaded.
4. **Prediction:** The selected model processes the input and generates a classification.
5. **Result Display:** The Streamlit interface displays the prediction to the user.

### Workflow Diagram

```text
        Student Information
                 |
                 v
        Input Feature Processing
                 |
                 v
          Trained ML Model
           /           \
          v             v
    Random Forest     XGBoost
           \           /
                 |
                 v
        Burnout Risk Prediction
                 |
                 v
          Streamlit Interface
                 |
                 v
          Display the Result
```

*Note: The exact preprocessing steps, model selection logic, and output labels depend on the implementation in the repository.*

---

## 🤖 Machine Learning Models

### Random Forest

Random Forest is an ensemble learning algorithm that combines predictions from multiple decision trees.

Instead of relying on a single tree, it combines the results of many trees to make a final prediction.

**Why use it?**

- Can learn complex patterns in tabular data.
- Can model relationships between different input features.
- Provides a strong baseline for classification tasks.

### XGBoost

XGBoost is a gradient-boosting algorithm that builds decision trees sequentially. Each new tree helps improve the model by learning from the errors made by previous trees.

**Why use it?**

- Works well with structured and tabular datasets.
- Can learn complex relationships between features.
- Offers useful training and regularization options.

### Model Comparison

Both algorithms are included in the project to explore different machine learning approaches to student burnout prediction.

The reported 79% accuracy should be understood in the context of the evaluation data and metrics used during model development.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning utilities and Random Forest |
| XGBoost | Gradient-boosting classification |
| Streamlit | Interactive web application |
| Pickle | Saving and loading trained model artifacts |
| Docker | Application containerization |
| Git & GitHub | Version control and project hosting |
| Streamlit Community Cloud | Live application deployment |

*The exact libraries and versions are defined by the project's dependency files.*

---

## 📂 Project Structure

```text
ZenithMind-AI/
│
├── api.py              # API-related application code
├── app.py              # Main Streamlit application
├── rf_model.pkl        # Saved Random Forest model
├── xgb_model.pkl       # Saved XGBoost model
├── features.pkl        # Saved feature-related artifact
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker image configuration
├── .dockerignore       # Files excluded from Docker builds
├── README.md           # Project documentation
└── yk.txt              # Additional project file
```

**Important:** The descriptions above identify files by their names. Check the source code for the exact responsibility of `api.py`, `features.pkl`, and `yk.txt`.

---

## 🚀 Getting Started

You can run ZenithMind AI locally using Docker or a Python environment.

### Prerequisites

Before starting, make sure you have:

- Git installed.
- Python 3.10 or another version compatible with the project's dependencies, if running without Docker.
- Docker Desktop installed and running, if using Docker.
- An internet connection for cloning the repository and installing dependencies.

### 1. Clone the Repository

Open your terminal or command prompt and run:

```bash
git clone https://github.com/Yash-11k/ZenithMind-AI.git
```

### 2. Open the Project Folder

```bash
cd ZenithMind-AI
```

You are now inside the project directory.

---

## 🐳 Run with Docker

Docker is the recommended option for creating a consistent local runtime environment.

### Step 1: Install Docker

Download and install Docker Desktop:

https://www.docker.com/products/docker-desktop/

Start Docker Desktop before continuing.

### Step 2: Build the Docker Image

Run this command from the project directory, where the Dockerfile is located:

```bash
docker build -t zenithmind-app .
```

This command builds a Docker image named `zenithmind-app`.

### Step 3: Start the Application

```bash
docker run --rm -p 8501:8501 zenithmind-app
```

This command starts the container and maps port `8501` on your computer to port `8501` inside the container.

### Step 4: Open the Application

Open your browser and visit:

http://localhost:8501

The application should now be available locally.

### Stop the Application

Press `Ctrl + C` in the terminal running the container.

If you started the container in detached mode, you can stop it using:

```bash
docker ps
docker stop <container_id>
```

Replace `<container_id>` with the ID displayed by `docker ps`.

---

## 🐍 Run Without Docker

If you prefer to run the project directly with Python, follow these steps.

### Step 1: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 2: Activate the Environment

**Windows:**

```bash
venv\Scripts\activate
```

**macOS or Linux:**

```bash
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Launch the Streamlit Application

```bash
streamlit run app.py
```

The application will usually open at:

http://localhost:8501

If your terminal reports a missing package or incompatible Python version, check `requirements.txt` and the application's imports.

---

## 📊 Model Performance

The project reports **79% classification accuracy** for its machine learning implementation.

Accuracy measures the proportion of predictions that match the actual labels in the evaluation dataset.

For example, an accuracy of 79% means that 79 out of every 100 evaluated predictions were correct, assuming the reported metric was calculated on that evaluation set.

However, accuracy alone does not provide a complete picture of model quality.

For a more complete evaluation, useful additional metrics include:

- **Precision:** How many predicted positive cases were actually positive?
- **Recall:** How many actual positive cases did the model identify?
- **F1-score:** A combined measure of precision and recall.
- **Confusion Matrix:** Shows correct and incorrect predictions for each class.

These metrics are particularly important when different prediction errors have different consequences.

*The reported accuracy is a project result, not a guarantee of performance on new or unseen populations. Additional metrics should only be reported when supported by actual evaluation results.*

---

## 🧪 Example Use Case

Imagine a student experiencing heavy academic workload and irregular sleep.

A prediction system such as ZenithMind AI can process the supported academic and lifestyle features and return a model-generated burnout risk classification.

The output may help demonstrate how multiple factors can be analyzed together rather than looking at a single factor in isolation.

The prediction should be treated as an estimate based on the model's training data, not as a medical conclusion.

---

## 🔮 Future Improvements

Possible improvements for future versions include:

- Add more evaluation metrics and a model comparison dashboard.
- Improve explainability using SHAP or feature importance visualizations.
- Add data validation and stronger error handling.
- Improve the interface for mobile devices.
- Add model monitoring and version tracking.
- Evaluate performance on independent datasets.
- Improve accessibility and user experience.
- Add clear explanations of model limitations alongside predictions.
- Add automated tests for preprocessing, predictions, and application endpoints.

These are potential improvements and are not claims about features already implemented.

---

## ⚠️ Limitations and Responsible Use

ZenithMind AI is an educational machine learning project.

- Predictions depend on the quality and representativeness of the training data.
- A model may perform differently on students whose characteristics differ from the training population.
- Classification accuracy does not establish that the model is suitable for clinical or individual decision-making.
- The output should not be used as a medical diagnosis or as the sole basis for important decisions about a student.
- Student information should be handled carefully, with appropriate privacy and data protection practices.

The system is intended to demonstrate machine learning concepts and should not replace qualified professional support.

---

## 👨‍💻 Author

**Yash**

Final-Year Computer Science Engineering Student | Aspiring Data Scientist

- **GitHub:** [Yash-11k](https://github.com/Yash-11k)
- **Project Repository:** [ZenithMind-AI](https://github.com/Yash-11k/ZenithMind-AI)
- **Live Demo:** [ZenithMind AI on Streamlit](https://zenithmind-ai-yash11k.streamlit.app/)
  

---

## ⭐ Support

If you find this project useful or interesting, consider giving the repository a star on GitHub.

Feedback, suggestions, and contributions that improve the project's documentation, usability, and machine learning workflow are welcome.

**Built with Python, Machine Learning, and Streamlit.**
