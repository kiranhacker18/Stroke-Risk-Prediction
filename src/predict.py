import pickle
import pandas as pd
from data_preprocessing import load_data, preprocess_data

# Load model
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load and preprocess data to get correct columns
df = load_data()
df = preprocess_data(df)

# Remove target column
X = df.drop('stroke', axis=1)

def predict_stroke(sample_index=0):
    sample = X.iloc[sample_index].values.reshape(1, -1)
    
    prediction = model.predict(sample)
    
    if prediction[0] == 1:
        return "High Risk of Stroke"
    else:
        return "Low Risk of Stroke"


# Test
if __name__ == "__main__":
    result = predict_stroke(0)
    print("Prediction:", result)