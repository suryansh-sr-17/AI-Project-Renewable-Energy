
from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained solar prediction model
model = pickle.load(open(r"C:\Users\Suryansh SR\Desktop\SuryanshSR\STUDII\[AI] CS307\F A N U M\Reference Codes\Deployment\wind_model.pkl", 'rb'))

@app.route('/')
def home():
    return render_template("wind_prediction.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input values from the form
        features = [float(x) for x in request.form.values()]
        # Ensure the input is in the correct shape for the model
        input_array = np.array(features).reshape(1, -1)
        
        # Make a prediction
        prediction = model.predict(input_array)
        output = prediction[0]  # Get the first (and only) prediction value
        
        # Return the result to the HTML page
        return render_template('wind_prediction.html',
                               prediction_text=f"Predicted Wind Energy Generation: {output:.2f} kW")
    except Exception as e:
        return render_template('wind_prediction.html',
                               prediction_text=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
