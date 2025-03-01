import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from xgboost import XGBClassifier
import joblib
from sklearn.preprocessing import LabelEncoder

thresholds = [0, 30, 50, 70, 110, float('inf')]
saturasyon_labels = ['Kumlu', 'Tınlı', 'Killi tınlı', 'Killi', 'Ağır Killi']

# ------------------------------
    # Load Models and Label Encoders
    # ------------------------------
    # Load label encoder for 'ilce'
with open('label_encoder_ilce.pkl', 'rb') as file:
    le_ilce = pickle.load(file)
if not isinstance(le_ilce, LabelEncoder):
    le_ilce = LabelEncoder()
    le_ilce.classes_ = pickle.load(open('label_encoder_ilce.pkl', 'rb'))

# Load label encoder for 'urun'
label_encoder_urun = joblib.load('label_encoder_urun.pkl')
if not isinstance(label_encoder_urun, LabelEncoder):
    raise Exception("label_encoder_urun is not a LabelEncoder instance")

# Load the saturasyon Booster model
model_saturasyon = xgb.Booster()
model_saturasyon.load_model('xgb_final_model_saturasyon.model')

# Load the classification model (XGBClassifier) for predicting 'urun'
xgb_model = XGBClassifier()
xgb_model.load_model('xgb_model_urun.model')

def predict_all(input_data):
    """
    Given input_data as a dictionary with keys:
        - 'ilce'      : (string) District name.
        - 'potasyum'  : (numeric) Potasyum value.
        - 'fosfor'    : (numeric) Fosfor value.
        - 'ph'        : (numeric) pH value.

    The function performs the following steps:
      1. Loads the required models and label encoders.
      2. Encodes the 'ilce' value.
      3. Predicts 'saturasyon' using the saturasyon Booster model.
      4. Classifies the predicted saturasyon into a categorical saturasyon_durumu.
      5. Constructs a feature set (including saturasyon) for the classification model.
      6. Uses the classification model to predict the top 2 'urun' (product) labels.
      7. Returns a dictionary with both the original and encoded values, the predicted saturasyon,
         its category, and the top 2 predicted 'urun'.
    """
    

    # ------------------------------
    # Encode Input and Build DataFrame
    # ------------------------------
    # Store original ilce value
    original_ilce = input_data['ilce']

    # Encode 'ilce'
    input_data['ilce_encoded'] = le_ilce.transform([input_data['ilce']])[0]

    # Convert input_data dictionary to DataFrame
    input_df = pd.DataFrame([input_data])

    # ------------------------------
    # Step 1: Predict 'saturasyon'
    # ------------------------------
    # Use only the features: ilce_encoded, potasyum, fosfor, and ph
    features_saturasyon = input_df[['ilce_encoded', 'potasyum', 'fosfor', 'ph']]
    dmatrix_input = xgb.DMatrix(features_saturasyon)

    # Predict saturasyon value (assuming a single prediction)
    predicted_saturasyon = model_saturasyon.predict(dmatrix_input)
    saturasyon_value = predicted_saturasyon[0]

    # Determine saturasyon category using pd.cut
    saturasyon_category = pd.cut(
        [saturasyon_value],
        bins=thresholds,
        labels=saturasyon_labels,
        include_lowest=True
    )[0]

    # Add predicted saturasyon to the DataFrame for classification input
    input_df['saturasyon'] = saturasyon_value

    # ------------------------------
    # Step 2: Predict 'urun'
    # ------------------------------
    # Build feature set for classification:
    # using ilce_encoded, potasyum, fosfor, ph, and saturasyon
    classification_features = input_df[['ilce_encoded', 'potasyum', 'fosfor', 'ph', 'saturasyon']]

    # Get class probabilities from the classifier
    class_probabilities = xgb_model.predict_proba(classification_features)

    # Extract indices of the top 2 classes (largest probabilities)
    top_2_indices = np.argsort(class_probabilities[0])[-2:][::-1]

    # Decode the top 2 'urun' labels using the label encoder
    top_2_urun = label_encoder_urun.inverse_transform(top_2_indices)

    # ------------------------------
    # Prepare and Return Output
    # ------------------------------
    result = {
        'ilce': original_ilce,
        'ilce_encoded': int(input_data['ilce_encoded']),
        'potasyum': input_data['potasyum'],
        'fosfor': input_data['fosfor'],
        'ph': input_data['ph'],
        'saturasyon': float(saturasyon_value),
        'saturasyon_durumu': saturasyon_category,
        'urun': top_2_urun.tolist()  # Top 2 predicted products
    }

    return result