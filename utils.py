from xgboost import XGBClassifier
import xgboost as xgb
import joblib
import numpy as np
import pandas as pd

try:
    model_organik = xgb.Booster()
    model_organik.load_model('./weights/xgb_model_organik_madde.model')

    model_saturasyon = xgb.Booster()
    model_saturasyon.load_model('./weights/xgb_model_saturasyon.model')

    model_ph = xgb.Booster()
    model_ph.load_model('./weights/xgb_model_ph.model')

    xgb_model = XGBClassifier()
    xgb_model.load_model('./weights/xgb_classifier.model')
    
except FileNotFoundError:
    raise Exception("Model file not found. Make sure 'xgb_best_model.pkl' exists.")

# Load the encoders
label_encoder_ilce = joblib.load('./weights/label_encoder_ilce.pkl')
label_encoder_tarim = joblib.load('./weights/label_encoder_tarim.pkl')
label_encoder_urun = joblib.load('./weights/label_encoder_urun.pkl')


def predict(req_body):
    # Load the trained model and encoders
    
    # Extract fields from req_body
    ilce = req_body['ilce']
    tarim_sekli = req_body['tarimSekli']
    fosfor = req_body['fosfor']
    potasyum = req_body['potasyum']
    kirec = req_body['kirec']
    toplam_tuz = req_body['toplamTuz']

    # Apply label encoding for categorical fields
    ilce_encoded = label_encoder_ilce.transform([ilce])[0]
    tarim_sekli_encoded = label_encoder_tarim.transform([tarim_sekli])[0]

    # Construct the input data for model
    input_data = {
        'ilce': ilce,
        'ilce_encoded': ilce_encoded,
        'tarim_sekli': tarim_sekli,
        'tarim_sekli_encoded': tarim_sekli_encoded, 
        'potasyum': potasyum,
        'fosfor': fosfor,
        'kirec': kirec,
        'toplam_tuz': toplam_tuz,
    }

    # Convert the input_data dictionary to a DataFrame for compatibility with DMatrix
    input_df = pd.DataFrame([input_data])

    # Step 1: Predict 'organik_madde'
    dmatrix_input = xgb.DMatrix(input_df[['ilce_encoded', 'tarim_sekli_encoded', 'potasyum', 'fosfor', 'kirec', 'toplam_tuz']])
    predicted_organik_madde = model_organik.predict(dmatrix_input)

    # Step 2: Predict 'saturasyon' using predicted 'organik_madde'
    input_df['organik_madde'] = predicted_organik_madde
    dmatrix_input_with_organik = xgb.DMatrix(input_df[['ilce_encoded', 'tarim_sekli_encoded', 'potasyum', 'fosfor', 'kirec', 'toplam_tuz', 'organik_madde']])
    predicted_saturasyon = model_saturasyon.predict(dmatrix_input_with_organik)

    # Step 3: Predict 'ph' using predicted 'saturasyon'
    input_df['saturasyon'] = predicted_saturasyon
    dmatrix_input_with_saturasyon = xgb.DMatrix(input_df[['ilce_encoded', 'tarim_sekli_encoded', 'potasyum', 'fosfor', 'kirec', 'toplam_tuz', 'saturasyon']])
    predicted_ph = model_ph.predict(dmatrix_input_with_saturasyon)

    # Add the predictions back to the input_data dictionary
    input_data['organikMadde'] = predicted_organik_madde[0]
    input_data['saturasyon'] = predicted_saturasyon[0]
    input_data['ph'] = predicted_ph[0]

    # Step 4: Use the classification model to predict 'urun'
    input_df['ph'] = predicted_ph
    # Create the full feature set for classification
    classification_features = input_df[['ilce_encoded', 'tarim_sekli_encoded', 'potasyum', 'fosfor', 'kirec', 'toplam_tuz', 'organik_madde', 'saturasyon', 'ph']]

    # Predict the encoded product (urun) and get the probabilities
    class_probabilities = xgb_model.predict_proba(classification_features)

    # Get the indices of the top 2 classes with highest probabilities
    top_2_indices = np.argsort(class_probabilities[0])[-2:][::-1]

    # Decode the top 2 predicted 'urun' labels
    top_2_urun = label_encoder_urun.inverse_transform(top_2_indices)

    # Add the top 2 predicted 'urun' to the input_data dictionary
    input_data['urun'] = top_2_urun.tolist()
    
    return_data = {
        'ilce_encoded': input_data['ilce_encoded'].item(),
        'ilce': ilce,
        'tarim_sekli_encoded': input_data['tarim_sekli_encoded'].item(),
        'tarim_sekli': tarim_sekli,
        'potasyum': input_data['potasyum'],
        'fosfor': input_data['fosfor'],
        'kirec': input_data['kirec'],
        'toplam_tuz': input_data['toplam_tuz'],
        'organikMadde': input_data['organikMadde'].item(),
        'saturasyon': input_data['saturasyon'].item(),
        'ph': input_data['ph'].item(),
        'urun': input_data['urun']  # Top 2 predicted products
    }
    return return_data

# Input order is critical: ensure it's consistent with the order used during training

# input_data = {
#     "ilce": "keciborlu",
#     "koy": "cumhuriyet",
#     "tarimSekli": "sulu",
#     "fosfor": 5,
#     "potasyum": 10,
#     "organikMadde": 1.5,
#     "ph": 7.1,
#     "kirec": 2,
#     "toplamTuz": 0.1,
#     "saturasyon": 70
# }

# Get the top 2 predicted classes and their probabilities
# top_2_predictions = preprocess_and_predict(input_data)
# print(f"The top 2 predictions are: {top_2_predictions}")