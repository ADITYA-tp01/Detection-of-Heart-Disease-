# Heart Disease Prediction App

This project is a Streamlit application designed to predict the likelihood of heart disease based on various medical attributes. It utilizes a machine learning model trained on patient data to provide instant predictions.

## Overview

The application takes user inputs for standard health metrics (like age, cholesterol, blood pressure, etc.) and uses a trained **Support Vector Machine (SVM)** model to classify whether a patient is at high risk of heart disease or not.

The project includes:
-   **Data Analysis:** A Jupyter notebook (`Capstone Project - Detection of Heart Disease.ipynb`) performing EDA, preprocessing, and model comparison (DT, RF, LR, SVM).
-   **Web App:** An interactive `app.py` built with Streamlit.
-   **Model Persistence:** Trained models and scalers are saved as pickle files in the `artifacts/` directory.

## Features

-   **Interactive Sidebar:** Easy-to-use sliders and dropdowns for inputting patient data.
-   **Real-time Prediction:** Instant classification results.
-   **Probability Estimation:** Displays the probability of having vs. not having heart disease.
-   **High Recall Model:** The SVM model was selected specifically for its high recall score, prioritizing the detection of potential positive cases (minimizing false negatives).

## Installation

1.  **Clone the repository** (or download the files):
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Create and activate a virtual environment** (recommended):
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # Mac/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Generate Artifacts (First Time Only):**
    If the `artifacts/` folder is missing, run the Jupyter Notebook to train and save the models:
    -   Open `Capstone Project - Detection of Heart Disease.ipynb`
    -   Run all cells.

2.  **Run the App:**
    ```bash
    streamlit run app.py
    ```

3.  **Access:**
    Open your browser and navigate to the URL shown in the terminal (usually `http://localhost:8501`).

## Project Structure

-   `app.py`: Main Streamlit application file.
-   `requirements.txt`: List of Python dependencies.
-   `Capstone Project...ipynb`: Jupyter notebook for data science workflow.
-   `artifacts/`: Directory containing saved models (`best_model.pkl`), scalers (`scaler.pkl`), and column names (`model_columns.pkl`).
-   `heart_1.csv`: Dataset used for training.

## Model Details

-   **Algorithm:** Support Vector Machine (SVM)
-   **Preprocessing:** One-Hot Encoding for categorical variables, Standard Scaling for numerical features.
-   **Metric:** Optimized for Recall (Class 1).
