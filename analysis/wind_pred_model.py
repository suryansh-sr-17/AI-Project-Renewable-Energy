
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import pickle

# Load the dataset
dataset = pd.read_csv(r"C:\Users\Suryansh SR\Desktop\SuryanshSR\STUDII\[AI] CS307\F A N U M\Reference Codes\S_W - Deployment\dataset.csv", parse_dates=[0], index_col=0)

# Splitting the data into training and testing sets
train = dataset[:6098]
test = dataset[6098:]

# Feature selection
X_train = train[['v1', 'v2', 'v_50m', 'z0']]
y_train = train['DE_wind_generation_actual']
X_test = test[['v1', 'v2', 'v_50m', 'z0']]
y_test = test['DE_wind_generation_actual']

# Initialize and train the model
gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)

# Save the trained model into a .pkl file
# pickle.dump(gbr, open("solar_model.pkl", "wb"))

pickle.dump(gbr, open(r"C:\Users\Suryansh SR\Desktop\SuryanshSR\STUDII\[AI] CS307\F A N U M\Reference Codes\Deployment\wind_model.pkl", "wb"))

# Load the model to verify serialization
loaded_model = pickle.load(open(r"C:\Users\Suryansh SR\Desktop\SuryanshSR\STUDII\[AI] CS307\F A N U M\Reference Codes\Deployment\wind_model.pkl", "rb"))

# Make a test prediction to confirm the model works as expected
sample_input = pd.DataFrame([[200, 150, 25, 10]], columns=['v1', 'v2', 'v_50m', 'z0'])  
predicted_output = loaded_model.predict(sample_input)

print("Sample Prediction:", predicted_output)