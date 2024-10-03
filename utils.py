from xgboost import XGBClassifier
import joblib
import numpy as np

try:
        xgb_loaded_model = XGBClassifier()
        xgb_loaded_model.load_model('./weights/xgbModel.model')
except FileNotFoundError:
    raise Exception("Model file not found. Make sure 'xgb_best_model.pkl' exists.")

# Load the encoders
label_encoder_ilce = joblib.load('./weights/label_encoder_ilce.pkl')
label_encoder_koy = joblib.load('./weights/label_encoder_koy.pkl')
label_encoder_tarim = joblib.load('./weights/label_encoder_tarim.pkl')
label_encoder_urun = joblib.load('./weights/label_encoder_urun.pkl')


def predict(input_data):
    # Load the trained model and encoders
    
    # Extract fields from input_data
    ilce = input_data['ilce']
    koy = input_data['koy']
    tarim_sekli = input_data['tarimSekli']
    fosfor = input_data['fosfor']
    potasyum = input_data['potasyum']
    organik_madde = input_data['organikMadde']
    ph = input_data['ph']
    kirec = input_data['kirec']
    toplam_tuz = input_data['toplamTuz']
    saturasyon = input_data['saturasyon']

    # Apply label encoding for categorical fields
    ilce_encoded = label_encoder_ilce.transform([ilce])[0]
    koy_encoded = label_encoder_koy.transform([koy])[0]
    tarim_sekli_encoded = label_encoder_tarim.transform([tarim_sekli])[0]

    # Create feature vector in the correct order as a NumPy array (without feature names)
    feature_vector = np.array([[ilce_encoded, kirec, saturasyon, tarim_sekli_encoded, koy_encoded, ph, organik_madde, fosfor, potasyum, toplam_tuz]])

    # Predict class probabilities
    class_probabilities = xgb_loaded_model.predict_proba(feature_vector)

    # Get the indices of the top 2 classes
    top_2_indices = np.argsort(class_probabilities[0])[-2:][::-1]

    # Get the class labels for the top 2 indices
    top_2_classes = label_encoder_urun.inverse_transform(top_2_indices)

    # Return only the class labels in array form
    return top_2_classes.tolist()

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