
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
import pickle

# Load the dataset
dataset = pd.read_csv(r"C:\Users\Suryansh SR\Desktop\SuryanshSR\STUDII\[AI] CS307\F A N U M\Reference Codes\S_W - Deployment\dataset.csv", parse_dates=[0], index_col=0)

# Splitting the data into training and testing sets
train = dataset[:6098]
test = dataset[6098:]

# Feature selection
X_train = train[['SWTDN', 'SWGDN', 'T']]
y_train = train['DE_solar_generation_actual']
X_test = test[['SWTDN', 'SWGDN', 'T']]
y_test = test['DE_solar_generation_actual']

# Initialize and train the model
hgbr = HistGradientBoostingRegressor()
hgbr.fit(X_train, y_train)

# Save the trained model into a .pkl file
# pickle.dump(hgbr, open("solar_model.pkl", "wb"))

pickle.dump(hgbr, open(r"C:\Users\Suryansh SR\Desktop\SuryanshSR\STUDII\[AI] CS307\F A N U M\Reference Codes\Deployment\solar_model.pkl", "wb"))

# Load the model to verify serialization
loaded_model = pickle.load(open(r"C:\Users\Suryansh SR\Desktop\SuryanshSR\STUDII\[AI] CS307\F A N U M\Reference Codes\Deployment\solar_model.pkl", "rb"))

# Make a test prediction to confirm the model works as expected
sample_input = pd.DataFrame([[200, 150, 25]], columns=['SWTDN', 'SWGDN', 'T'])  # Replace with realistic values for SWTDN, SWGDN, and T
predicted_output = loaded_model.predict(sample_input)

print("Sample Prediction:", predicted_output)
