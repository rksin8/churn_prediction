from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import sys

from models.preprocessing import preprocess_user_input

app = Flask(__name__)

# Load the model objects
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs from form
    #data = [float(x) for x in request.form.values()]
    #final_input = np.array([data])

    # Get form data
    user_input = request.form.to_dict()
    
    # Convert form data to DataFrame
    data = pd.DataFrame([user_input])

    print(data.info(), file=sys.stdout)

    # save input to csv to test pre-processing steps - one time precesss comment it later
    # data.to_csv("input.csv", index=False)

    # Preprocess user input
    processed_input = preprocess_user_input(data)

    ## Make prediction using the preprocessed input
    # Make a prediction
    prediction = model.predict(processed_input.values)

    print(prediction[0], file=sys.stdout)
    
    # Get the model accuracy
    #accuracy = accuracy_rf * 100  # assuming accuracy_rf is the model's accuracy
    accuracy = 81.48
    
    # Display the result
    return render_template('index.html', prediction_text=f'Churn Prediction: {"Yes" if prediction[0] == 1 else "No"}',
                           accuracy_text=f'Model Accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    app.run(debug=True)
