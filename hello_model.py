from flask import Flask, render_template_string, jsonify
import pickle
import numpy as np

# Load the saved model
with open('iris_model.pkl', 'rb') as pk:
    model = pickle.load(pk)

app = Flask(__name__)

# Welcome page with styled HTML
@app.route('/')
def welcome():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Welcome to My ML App</title>
        <style>
            body {
                background: linear-gradient(to right, #83a4d4, #b6fbff);
                font-family: Arial, sans-serif;
                text-align: center;
                padding-top: 100px;
                color: #333;
            }
            h1 {
                font-size: 3em;
                margin-bottom: 20px;
            }
            p {
                font-size: 1.5em;
            }
            .button {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 15px 30px;
                text-align: center;
                font-size: 16px;
                margin-top: 30px;
                border-radius: 8px;
                cursor: pointer;
                text-decoration: none;
            }
        </style>
    </head>
    <body>
        <h1>🚀 Welcome to the Machine Learning App!</h1>
        <p>Your intelligent assistant awaits...</p>
        <a href="/predict" class="button">Run Model</a>
    </body>
    </html>
    """
    return render_template_string(html)

# Model prediction route
@app.route('/predict', methods=['GET'])
def predict():
    # Example input — this would be dynamic in a real app
    sample_input = np.array([[5.1, 3.5, 1.4, 0.2]])  # Shape (1, 4)

    # Make prediction
    prediction = model.predict(sample_input)
    
    # Return prediction as JSON
    return jsonify({
        'prediction': int(prediction[0])  # Make sure it's JSON serializable
    })

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
