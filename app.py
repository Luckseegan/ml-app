from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os

app = Flask(__name__)

class SimpleMLModel:
    def __init__(self):
        self.model = LinearRegression()
        self.is_trained = False
        self.training_history = []
    
    def train_model(self):
        # Generate sample data: y = 2x + 1 + noise
        np.random.seed(42)
        X = np.random.rand(100, 1) * 10
        y = 2 * X.flatten() + 1 + np.random.randn(100) * 2
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate metrics
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        results = {
            "train_score": round(train_score, 3),
            "test_score": round(test_score, 3),
            "coefficients": [round(coef, 3) for coef in self.model.coef_],
            "intercept": round(self.model.intercept_, 3),
            "equation": f"y = {round(self.model.coef_[0], 3)}x + {round(self.model.intercept_, 3)}"
        }
        
        self.training_history.append(results)
        return results
    
    def predict(self, x_value):
        if not self.is_trained:
            return {"error": "Model not trained yet"}
        
        prediction = self.model.predict([[x_value]])[0]
        return {
            "prediction": round(prediction, 3),
            "input": x_value,
            "equation": f"y = {round(self.model.coef_[0], 3)}x + {round(self.model.intercept_, 3)}"
        }

# Initialize model
ml_model = SimpleMLModel()

@app.route('/')
def home():
    return render_template('index.html', 
                         model_trained=ml_model.is_trained,
                         history=ml_model.training_history)

@app.route('/train', methods=['POST'])
def train():
    try:
        results = ml_model.train_model()
        return jsonify({
            "status": "success",
            "message": "Model trained successfully!",
            "results": results
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        x_value = float(request.form['x_value'])
        prediction = ml_model.predict(x_value)
        
        if 'error' in prediction:
            return jsonify({"status": "error", "message": prediction['error']})
        
        return jsonify({
            "status": "success",
            "prediction": prediction
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/model-info')
def model_info():
    return jsonify({
        "is_trained": ml_model.is_trained,
        "model_type": "Linear Regression",
        "training_count": len(ml_model.training_history),
        "last_training": ml_model.training_history[-1] if ml_model.training_history else None
    })

@app.route('/api/train', methods=['POST'])
def api_train():
    """API endpoint for training"""
    try:
        results = ml_model.train_model()
        return jsonify({
            "status": "success",
            "results": results
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        x_value = float(data['x_value'])
        prediction = ml_model.predict(x_value)
        
        if 'error' in prediction:
            return jsonify({"status": "error", "message": prediction['error']})
        
        return jsonify({
            "status": "success",
            "prediction": prediction
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True)