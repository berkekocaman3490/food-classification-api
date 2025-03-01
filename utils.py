import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from xgboost import XGBClassifier
import joblib
from sklearn.preprocessing import LabelEncoder

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

# Load the kireç Booster model
model_kirec = xgb.Booster()
model_kirec.load_model('xgb_final_model_kirec.model')

# Load the organik madde Booster model
model_organik = xgb.Booster()
model_organik.load_model('xgb_final_model_organik.model')

# Load the classification model (XGBClassifier) for predicting 'urun'
xgb_model = XGBClassifier()
xgb_model.load_model('xgb_model_urun.model')

# ------------------------------
# Define Classification Thresholds and Labels
# ------------------------------
# For saturasyon
saturasyon_thresholds = [0, 30, 50, 70, 110, float('inf')]
saturasyon_labels = ['Kumlu', 'Tınlı', 'Killi tınlı', 'Killi', 'Ağır Killi']

# For kireç
kirec_thresholds = [-float('inf'), 1, 4, 15, 25, float('inf')]
kirec_labels = ['az kireçli', 'kireçli', 'orta kireçli', 'fazla kireçli', 'çok fazla kireçli']

# For organik madde
organik_thresholds = [-float('inf'), 1, 2, 3, 4, float('inf')]
organik_labels = ['ççok az', 'az', 'orta', 'iyi', 'yüksek']

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
      3. Predicts 'saturasyon', 'kireç', and 'organik_madde' using their respective Booster models.
      4. Classifies each prediction into its categorical status.
      5. Constructs a feature set (including saturasyon) for the product classification model.
      6. Uses the classification model to predict the top 2 'urun' (product) labels.
      7. Returns a dictionary with the original and encoded values, predicted continuous values,
         their categories, and the top 2 predicted 'urun'.
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

    # Use features: ilce_encoded, potasyum, fosfor, ph for all predictions
    features = input_df[['ilce_encoded', 'potasyum', 'fosfor', 'ph']]
    dmatrix_input = xgb.DMatrix(features)

    # ------------------------------
    # Step 1: Predict 'saturasyon'
    # ------------------------------
    predicted_saturasyon = model_saturasyon.predict(dmatrix_input)
    saturasyon_value = predicted_saturasyon[0]
    saturasyon_category = pd.cut(
        [saturasyon_value],
        bins=saturasyon_thresholds,
        labels=saturasyon_labels,
        include_lowest=True
    )[0]
    # Add saturasyon to DataFrame
    input_df['saturasyon'] = saturasyon_value

    # ------------------------------
    # Step 2: Predict 'kireç'
    # ------------------------------
    predicted_kirec = model_kirec.predict(dmatrix_input)
    kirec_value = predicted_kirec[0]
    kirec_category = pd.cut(
        [kirec_value],
        bins=kirec_thresholds,
        labels=kirec_labels,
        include_lowest=True
    )[0]
    # Add kireç to DataFrame
    input_df['kireç'] = kirec_value

    # ------------------------------
    # Step 3: Predict 'organik madde'
    # ------------------------------
    predicted_organik = model_organik.predict(dmatrix_input)
    organik_value = predicted_organik[0]
    organik_category = pd.cut(
        [organik_value],
        bins=organik_thresholds,
        labels=organik_labels,
        include_lowest=True
    )[0]
    # Add organik madde to DataFrame
    input_df['organik_madde'] = organik_value

    # ------------------------------
    # Step 4: Predict 'urun'
    # ------------------------------
    # Build feature set for product classification:
    # using ilce_encoded, potasyum, fosfor, ph, and saturasyon
    classification_features = input_df[['ilce_encoded', 'potasyum', 'fosfor', 'ph', 'saturasyon']]
    class_probabilities = xgb_model.predict_proba(classification_features)
    top_2_indices = np.argsort(class_probabilities[0])[-2:][::-1]
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
        'kireç': float(kirec_value),
        'kireç_durumu': kirec_category,
        'organik_madde': float(organik_value),
        'organik_madde_durumu': organik_category,
        'urun': top_2_urun.tolist()  # Top 2 predicted products
    }

    return result

# ------------------------------
# Example usage:
# ------------------------------
new_input_data = {
    'ilce': 'merkez',    # This value must have been seen during training
    'potasyum': 35,
    'fosfor': 120,
    'ph': 6.5
}

# Run the prediction
results = predict_all(new_input_data)
print("🔍 Prediction Results:")
for key, value in results.items():
    print(f"{key}: {value}")
