from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import re
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import google.generativeai as genai

# ----------------- Gemini Setup -----------------
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("Please set GEMINI_API_KEY environment variable")
genai.configure(api_key=API_KEY)

gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# ----------------- ML Model -----------------
class PricePredictor:
    def __init__(self):
        self.lr = None
        self.rf = None
        self.df = None
        self.features = ["Avg_Size", "Possession"]
        self.target = "Price_sqft"

    def prepare_data(self):
        data = {
            "Project": ["Sai fortune heights","Concerte avasa","Element rocktown","SV casa",
                        "Pallavi Gardenia","North kishore","Mahanagar ecopolise","Namishree aria",
                        "VK grand","Pallavi Gardenia","Krushi gardenia","SV Height",
                        "Maruti rudra","Buildwel acropolis","Mauthi Samantha","Prred gretel",
                        "Anjani Residency","Spectro metro Height","Environ towers","Value landmark",
                        "Yasodas gokuilam","Sai niwas apartment","Sr ram nagar"],
            "Possession": [2027,2023,2020,2025,2025,2021,2019,2028,
                           2025,2025,2025,2026,2023,2023,2024,2024,
                           2015,2024,2015,2020,2027,2015,2022],
            "Size_min": [1750,1300,2100,1400,1400,800,1600,1600,
                         1300,1400,1200,100,1300,1400,1600,2000,
                         1650,1600,1300,1500,1400,1350,2400],
            "Size_max": [1750,2000,2100,1400,1800,1500,2200,2100,
                         1800,1800,2500,1300,2300,1400,1600,2000,
                         1650,1600,1500,1700,2400,1350,2400],
            "Price_sqft": [7400,6250,5800,5500,5400,4500,5300,6390,
                           5700,5400,6800,5470,6300,4900,6500,6000,
                           4500,5580,4800,5000,5000,6300,6700]
        }
        self.df = pd.DataFrame(data)
        self.df["Avg_Size"] = (self.df["Size_min"] + self.df["Size_max"]) / 2

    def train_model(self):
        self.prepare_data()
        X = self.df[self.features]
        y = self.df[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.lr = LinearRegression()
        self.lr.fit(X_train, y_train)
        self.rf = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf.fit(X_train, y_train)
        joblib.dump({"lr": self.lr, "rf": self.rf}, "price_models.pkl")

    def load_model(self):
        if os.path.exists("price_models.pkl"):
            models = joblib.load("price_models.pkl")
            self.lr = models.get("lr")
            self.rf = models.get("rf")
        else:
            self.train_model()

    def predict_price_lakhs(self, possession_year: int, size_sqft: int):
        if self.lr is None or self.rf is None:
            self.load_model()
        features = np.array([[size_sqft, possession_year]])
        pred_lr = float(self.lr.predict(features)[0])
        pred_rf = float(self.rf.predict(features)[0])
        final_price_sqft = (pred_lr + pred_rf) / 2.0
        final_estimated_price_lakhs = final_price_sqft * size_sqft / 100000.0
        return round(final_estimated_price_lakhs, 2), round(final_price_sqft, 2)

# ----------------- AI Assistant -----------------
def parse_query(query: str):
    year_match = re.search(r"(20[2-3][0-9])", query)
    size_match = re.search(r"(\d{3,4})\s*(sqft|sq ft|square feet)?", query.lower())
    year = int(year_match.group(1)) if year_match else 2025
    size = int(size_match.group(1)) if size_match else 1200
    return year, size

def generate_response_with_gemini(query: str, price_lakhs: float, price_sqft: float):
    prompt = f"""
    The user asked: "{query}".
    The ML model predicted:
    - Estimated apartment price: ₹{price_lakhs} lakhs
    - Price per sqft: ₹{price_sqft}

    Please generate a friendly, human-like response that explains this estimate clearly.
    """
    response = gemini_model.generate_content(prompt)
    return response.text

# ----------------- FastAPI App -----------------
app = FastAPI(title="Apartment AI Assistant (Gemini LLM)")
predictor = PricePredictor()
predictor.load_model()

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Apartment AI Assistant (Gemini)</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f0f4f8; margin: 0; padding: 0; }
            header { background: #2c3e50; color: white; padding: 20px; text-align: center; }
            header img { height: 50px; vertical-align: middle; margin-right: 15px; }
            .container { max-width: 800px; margin: 30px auto; padding: 20px; background: white; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
            h1 { margin: 0; font-size: 26px; }
            .hero { text-align: center; margin-bottom: 20px; }
            .hero img { width: 100%; max-height: 200px; object-fit: cover; border-radius: 10px; }
            textarea { width: 100%; padding: 12px; border: 1px solid #ccc; border-radius: 5px; font-size: 16px; }
            button { margin-top: 10px; padding: 12px 25px; background: #3498db; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
            button:hover { background: #2980b9; }
            .response { margin-top: 20px; padding: 20px; background: #ecf9ff; border-left: 5px solid #3498db; border-radius: 5px; font-size: 18px; line-height: 1.5; }
        </style>
    </head>
    <body>
        <header>
            <img src="https://cdn-icons-png.flaticon.com/512/619/619034.png" alt="Logo">
            <span style="font-size:24px; font-weight:bold;">Apartment AI Assistant</span>
        </header>
        <div class="container">
            <div class="hero">
                <img src="https://images.unsplash.com/photo-1560448075-bb4f0b0b2bde" alt="Real Estate Banner">
            </div>
            <h2>Ask about apartment prices 🏢</h2>
            <p>Type your query below and get AI-powered price predictions, explained in a human-like manner using Gemini LLM.</p>
            <textarea id="userQuery" rows="3" placeholder="e.g., What is the price of 3BHK 1660 sqft in 2026?"></textarea>
            <button onclick="sendQuery()">Ask</button>
            <div id="response" class="response" style="display:none;"></div>
        </div>

        <script>
            async function sendQuery() {
                const query = document.getElementById('userQuery').value;
                if (!query) return;
                document.getElementById('response').style.display = 'block';
                document.getElementById('response').innerHTML = 'Thinking...';

                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                    body: 'query=' + encodeURIComponent(query)
                });
                const data = await response.json();
                if (data.status === 'success') {
                    document.getElementById('response').innerHTML = data.answer;
                } else {
                    document.getElementById('response').innerHTML = 'Error: ' + data.error;
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/ask")
async def ask(query: str = Form(...)):
    try:
        year, size = parse_query(query)
        price_lakhs, price_sqft = predictor.predict_price_lakhs(year, size)
        answer = generate_response_with_gemini(query, price_lakhs, price_sqft)
        return {"status": "success", "answer": answer}
    except Exception as e:
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
