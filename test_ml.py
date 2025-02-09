import pytest
import pandas as pd
from ml.model import load_model, performance_on_categorical_slice, compute_model_metrics, inference
from ml.data import process_data
from sklearn.preprocessing import LabelBinarizer 
import joblib

# Load necessary components
model = load_model("./model/model.pkl")
encoder = load_model("./model/encoder.pkl")
test_data = pd.read_csv("data/census.csv")  # Assuming test data is available

# Define categorical features based on dataset
categorical_features = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]

lb = LabelBinarizer()
lb.fit(test_data["salary"]) 

# Process test data to retrieve encoder and label binarizer
X_test, y_test, encoder, _ = process_data(test_data, categorical_features, label="salary", training=False, encoder=encoder, lb=lb)

# Test model loading
def test_model_loading():
    """
    Test if the model loads correctly.
    """
    assert model is not None, "Model should be loaded successfully"

# Test model performance on a categorical slice
def test_performance_on_categorical_slice():
    """
    Test model performance on a categorical slice (education='Bachelors').
    """
    col = "education"
    slice_value = "Bachelors"
    slice_data = test_data[test_data[col] == slice_value]
    X_slice, y_slice, _, _ = process_data(slice_data, categorical_features=categorical_features, label="salary", training=False, encoder=encoder, lb=lb)
    
    preds = inference(model, X_slice)
    
    y_slice_numerical = lb.transform(y_slice).ravel()
    
    p, r, fb = compute_model_metrics(y_slice_numerical, preds)
    
    assert p is not None, "Precision should not be None"
    assert r is not None, "Recall should not be None"
    assert fb is not None, "F-beta score should not be None"

# Test data processing
def test_data_processing():
    """
    Test if data processing works correctly and returns expected output shapes.
    """
    assert X_test.shape[0] == y_test.shape[0], "Feature and label arrays should have the same number of rows"
    assert encoder is not None, "Encoder should not be None"