from flask import Flask, render_template, request
import pickle
import numpy as np
import os


app = Flask(__name__)

model_path = os.path.join(
    "Insurance-Buyer-Prediction-Using-Age-Income",
    "Model",
    "insurance__buyer.pkl"
)

if not os.path.isfile(model_path):
    raise FileNotFoundError("Model file not found")

with open(model_path,"rb") as f:
    model = pickle.load(f)
    
@app.route("/")
def home():
    return render_template("UI_.html")

@app.route("/predict",methods=["GET","POST"])
def predict():
    if request.method == "POST":
        try:
            age = int(request.form["age"])
            income = float(request.form["income"])
        
            input_data = np.array([[age,income]])
        
            prediction = model.predict(input_data)[0]
        
            return render_template(
                "UI_.html",
                result = "Yes, customer will buy insurance" if prediction == 1 else "No, customer will not buy insurance"

            )
        except Exception as e:
            return render_template(
                "index.html",
                result=f"Error: {str(e)}"
            )
    return render_template("UI_.html")
        
if __name__ == "__main__":
    app.run(debug=True)
