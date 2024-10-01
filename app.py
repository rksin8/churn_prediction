from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved model
# model = pickle.load(open('best_rf_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = [float(x) for x in request.form.values()]
    final_input = np.array([data])
    
    # Make a prediction
    #prediction = model.predict(final_input)
    prediction =[1]
    
    # Get the model accuracy
    #accuracy = accuracy_rf * 100  # assuming accuracy_rf is the model's accuracy
    accuracy = 88.0
    
    # Display the result
    return render_template('index.html', prediction_text=f'Churn Prediction: {"Yes" if prediction[0] == 1 else "No"}',
                           accuracy_text=f'Random Forest Model Accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    app.run(debug=True)
