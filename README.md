# Apartment_price_prediction_AI_Assistant_llm
🏢 Apartment Price Prediction AI Assistant (Gemini LLM + ML + FastAPI)
📌 Project Overview

This project is an AI-powered assistant that predicts apartment prices based on size (sqft) and possession year. It combines a Machine Learning model for price prediction with Google’s Gemini LLM for generating human-like, conversational responses. The assistant runs on a FastAPI web server and provides an intuitive web-based interface for users to interact with.

⚙️ Tech Stack
🔹 Backend

Python 3.x – Core programming language

FastAPI – Web framework for serving the application and APIs

Uvicorn – ASGI server for running FastAPI

🔹 Machine Learning

Scikit-learn – For Linear Regression and Random Forest models

Joblib – Model persistence (saving and loading trained models)

Pandas – Data handling and preprocessing

NumPy – Numerical computations

🔹 LLM Integration

Google Generative AI SDK (google-generativeai) – To integrate Gemini 1.5 Flash LLM

Gemini LLM – Generates conversational, human-like responses explaining ML predictions

🔹 Frontend

HTML + CSS + JavaScript (served via FastAPI routes)

Custom UI Enhancements:

Modern design with banner and logo

Styled textarea and response box

User-friendly input and output interface

📊 ML Model Details

The ML model predicts price per square foot using:

Features:

Avg_Size (average of min/max apartment size)

Possession (year of possession)

Target:

Price_sqft (price per square foot from dataset)

Models Used

Linear Regression (LR) – For baseline prediction

Random Forest (RF) – For better accuracy

Final Prediction = Average of LR and RF outputs

The predicted price per square foot is then multiplied by the apartment size to compute the final price in lakhs.

🧠 How Gemini is Used

The ML model generates raw predictions (price per sqft & estimated total price).

These results are passed into Gemini LLM, which generates friendly, human-like explanations instead of plain numbers.

Example:
“Based on your query, the estimated price for a 1660 sqft apartment in 2026 is ₹92.4 lakhs (≈ ₹5565 per sqft). This is derived from our ML model combining Linear Regression and Random Forest predictions.”

🛠️ Tools & Libraries

Python

FastAPI

Uvicorn

Scikit-learn

Pandas

NumPy

Joblib

Google Generative AI (Gemini)

🚀 Running the Project

Follow these steps to set up and run the project locally:

Open CMD and go to your project folder

cd path\to\Apartment


Create virtual environment

python -m venv venv


Activate virtual environment

.\venv\Scripts\activate


Install multipart dependency (for form handling)

pip install python-multipart


Install all required libraries

pip install fastapi uvicorn scikit-learn joblib pandas numpy google-generativeai


Set Gemini API Key

set GEMINI_API_KEY=AIzaSyB7-97jF1vcneYCU_YfKAvfOMU8aGl8ic4


Run the FastAPI server

uvicorn apartment_ai_assistant_llm_gemini_ui:app --reload


Open in browser

http://127.0.0.1:8000

🎯 Usage

Type queries like:

“What is the price of a 3BHK 1660 sqft in 2026?”

“Estimate price of 1200 sqft apartment in 2025”

The assistant will:

Extract size and year from your query.

Predict the price using the ML model.

Generate a human-like explanation using Gemini LLM.

Display the response on the web page.
