import os
import json
import numpy as np
import tensorflow as tf
import joblib
import firebase_admin
from firebase_admin import credentials, auth
from firebase_admin import firestore
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from google import genai
from google.genai import types
import logging
from dotenv import load_dotenv
import datetime 
import traceback # Import for better error logging

# --- Initialization ---
load_dotenv() 
app = Flask(__name__)
# Allow all origins for development, consider restricting in production
CORS(app, resources={r"/predict": {"origins": "*"}})
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Firebase Admin SDK Initialization ---
db = None # Initialize db variable
try:
    # Try getting credentials from ENV VAR (for deployment like Hugging Face)
    cred_json_string = os.getenv('FIREBASE_SERVICE_ACCOUNT_JSON')
    if cred_json_string:
        logging.info("Initializing Firebase Admin using environment variable.")
        cred_dict = json.loads(cred_json_string)
        cred = credentials.Certificate(cred_dict)
    else:
        # Fallback for local development
        logging.info("Initializing Firebase Admin using local serviceAccountKey.json file.")
        cred_path = "serviceAccountKey.json" # Assumes file is in the same directory as app.py
        if not os.path.exists(cred_path):
            raise FileNotFoundError(
                "serviceAccountKey.json not found locally. "
                "Ensure the file is in the backend directory or set FIREBASE_SERVICE_ACCOUNT_JSON env var."
            )
        cred = credentials.Certificate(cred_path)

    # Initialize the app with the credentials found
    # Check if app is already initialized to prevent errors on hot-reload
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    else:
        logging.warning("Firebase Admin App already initialized.")
        
    db = firestore.client() # Initialize Firestore client
    logging.info("Firebase Admin SDK and Firestore client initialized successfully.")

except Exception as e:
    logging.error(f"FATAL: Failed to initialize Firebase Admin SDK: {e}")
# --- End Firebase Init ---


# --- Configure Gemini API (Original Working Version) ---
gemini_client = None
gemini_enabled = False
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        raise ValueError("GOOGLE_API_KEY was NOT found in environment. Check your .env file.")
    
    # Force the API version to 'v1'
    client_options = types.HttpOptions(api_version='v1')
    # Create the client, passing the key and options
    gemini_client = genai.Client(api_key=api_key, http_options=client_options) 
    
    # Optional: List models to confirm
    # logging.info("--- AVAILABLE MODELS ---")
    # for m in gemini_client.models.list():
    #     logging.info(f"Model Name: {m.name}") 
    # logging.info("------------------------")
    
    gemini_enabled = True
    logging.info("Gemini client object created successfully (original method).") 

except Exception as e:
    logging.error(f"Error creating Gemini client object: {e}")
    logging.warning("Gemini summary will be disabled. Check your .env file and API key permissions.")
    gemini_enabled = False


# --- Load Models and Preprocessors ---
image_model = None
clinical_model = None
scaler = None
feature_order = None
IMAGE_SIZE = (350, 350)
IMAGE_CLASS_LABELS = ['Adenocarcinoma', 'Large Cell Carcinoma', 'Normal', 'Squamous Cell Carcinoma']
CLINICAL_CLASS_LABELS = ['Low', 'Medium', 'High']

try:
    image_model_path = 'models/trained_lung_cancer_model_finetuned.keras'
    if os.path.exists(image_model_path):
        image_model = tf.keras.models.load_model(image_model_path)
        logging.info(f"Image model loaded successfully from {image_model_path}")
    else:
         logging.error(f"Image model file not found at {image_model_path}.")

    clinical_model_path = 'models/random_forest_model.pkl'
    if os.path.exists(clinical_model_path): clinical_model = joblib.load(clinical_model_path)
    else: logging.error(f"Clinical model file not found at {clinical_model_path}.")

    scaler_path = 'models/scaler.pkl'
    if os.path.exists(scaler_path): scaler = joblib.load(scaler_path)
    else: logging.error(f"Scaler file not found at {scaler_path}.")

    features_path = 'models/features.pkl'
    if os.path.exists(features_path):
        feature_order = joblib.load(features_path)
        logging.info(f"Feature order loaded: {feature_order}")
    else:
        logging.error(f"Features file not found at {features_path}.")

    if all([clinical_model, scaler, feature_order]):
        logging.info(f"Clinical model and preprocessors loaded successfully.")
except Exception as e:
    logging.error(f"FATAL: Could not load one or more models. Error: {e}")
    logging.error(traceback.format_exc())
# --- End Load Models ---

@app.route('/')
def home():
    # A simple health check route
    return jsonify({"status": "API is running!"}), 200

# --- API Endpoint for Predictions ---
@app.route('/predict', methods=['POST'])
def predict():

    uid = None # Define uid outside try block
    # --- Token Verification ---
    try:
        if not firebase_admin._apps: raise Exception("Firebase Admin SDK is not initialized.")
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            logging.warning("Missing or invalid Authorization header.")
            return jsonify({'error': 'Unauthorized: Missing or invalid token'}), 401
        id_token = auth_header.split("Bearer ").pop()
        if not id_token: raise ValueError("Bearer token is empty")
        decoded_token = auth.verify_id_token(id_token)
        uid = decoded_token['uid'] # <-- Get the User ID
        logging.info(f"Token verified for user UID: {uid}")
    except Exception as e:
        logging.error(f"Auth error: {e}")
        logging.error(traceback.format_exc())
        return jsonify({'error': f'Unauthorized: {e}'}), 403
    
    if not all([image_model, clinical_model, scaler, feature_order]):
        logging.error("Prediction failed: model/preprocessor not loaded.")
        return jsonify({'error': 'A model or preprocessor failed to load.'}), 500

    image_file = request.files.get('image')
    clinical_data_form = request.form
    role = clinical_data_form.get('role', 'Medical Professional')
    logging.info(f"Received prediction request for user {uid} with role: {role}")

    if not image_file or not clinical_data_form:
        return jsonify({'error': 'Missing image or clinical data.'}), 400

    try:
        # --- Image Processing ---
        img = Image.open(image_file.stream).convert('RGB')
        img = img.resize(IMAGE_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        logging.info("Running image model prediction...")
        image_prediction = image_model.predict(img_array)
        image_pred_index = np.argmax(image_prediction[0])
        image_pred_label = IMAGE_CLASS_LABELS[image_pred_index]
        image_confidence = float(np.max(image_prediction[0]))
        logging.info(f"Image prediction: {image_pred_label} ({image_confidence:.2%})")

        # --- Clinical Data Processing (FIXED) ---
        
        # 1. Create the dictionary first 
        input_features_dict = {feature: float(clinical_data_form.get(feature, 0)) for feature in feature_order}
        
        # 2. Create the list *from* the dictionary to ensure correct order
        input_features_list = [input_features_dict[feature] for feature in feature_order]

        logging.info("Scaling clinical features...")
        scaled_features = scaler.transform([input_features_list]) # Use the list for scaling
        
        logging.info("Running clinical model prediction...")
        clinical_prediction = clinical_model.predict(scaled_features)
        clinical_proba = clinical_model.predict_proba(scaled_features)
        clinical_pred_label = CLINICAL_CLASS_LABELS[clinical_prediction[0]]
        clinical_confidence = float(np.max(clinical_proba[0]))
        logging.info(f"Clinical prediction: {clinical_pred_label} Risk ({clinical_confidence:.2%})")

        gemini_summary = "AI summary is disabled. Check API key."
        
        if gemini_enabled and gemini_client:
            logging.info("Generating Gemini summary (original method)...")
            try:
                # --- Dynamic persona prompt based on the role ---
                persona_prompt = ""
                if role.lower() == 'patient':
                    persona_prompt = """
                    As an AI assistant, explain the following results in simple, clear language for a **patient**. 
                    Your tone should be supportive and easy to understand. Do not use complex medical jargon. 
                    Your goal is to help them understand what the models found, but you MUST emphasize that this is NOT a final diagnosis and they MUST consult their doctor.

                    **Instructions for Patient Explanation:**
                    1.  **If the CT Scan shows a potential issue (carcinoma) AND the Clinical Risk is 'High' or 'Medium':** Explain that both checks (the scan and the survey) suggest there might be an issue. Reassure them that this is an early signal and their doctor will order more specific tests, like a biopsy, to be certain.
                    2.  **If the CT Scan shows a potential issue BUT the Clinical Risk is 'Low':** Explain that the scan spotted something, but the survey answers didn't. Emphasize that the scan is the more important finding here, and their doctor will focus on that.
                    3.  **If the CT Scan is 'Normal' BUT the Clinical Risk is 'High':** Explain that while the scan looks clear right now, their survey answers suggest they are at a higher risk. This is good information for their doctor, who will likely want to monitor them closely with regular check-ups.
                    4.  **If the CT Scan is 'Normal' AND the Clinical Risk is 'Low':** Explain that this is good news, as both the scan and the survey came back clear, indicating a low likelihood of any issues at this time.
                    
                    **Crucially, end EVERY summary with a clear statement:** "Please discuss these results with your doctor to determine the next steps. This AI summary is for informational purposes only and is not a medical diagnosis."
                    """
                elif role.lower() == 'student':
                    persona_prompt = """
                    As an AI teaching assistant, synthesize the following outputs for a **medical student**. 
                    Your goal is to be educational. Explain the "why" behind the synthesis.
                    For example, 'Adenocarcinoma' is a type of non-small cell lung cancer. 
                    Focus on the agreement or disagreement between the two models (radiological vs. clinical data) and what that implies for a differential diagnosis.

                    **Instructions for Student Explanation:**
                    1.  **If the CT Scan shows a carcinoma (e.g., '{image_pred_label}') AND the Clinical Risk is 'High' or 'Medium':** "The radiological and clinical data are consistent, strongly indicating malignancy. The {image_pred_label} finding on the scan is corroborated by the {clinical_pred_label} risk profile. This convergence significantly increases the post-test probability. Next logical step: biopsy for histopathology."
                    2.  **If the CT Scan shows a carcinoma BUT the Clinical Risk is 'Low':** "Note the discrepancy: positive radiological findings ({image_pred_label}) despite a 'Low' clinical risk. This highlights a limitation of the clinical risk model, which may not capture all risk factors (e.g., genetic predisposition). The radiological evidence is the primary driver for diagnosis here. Investigate patient history for uncaptured risks."
                    3.  **If the CT Scan is 'Normal' BUT the Clinical Risk is 'High':** "Discrepancy observed. The 'Normal' CT scan provides a negative finding, but the 'High' clinical risk profile suggests the patient remains at elevated risk. This could imply a very early-stage (sub-radiological) lesion or that the patient's risk factors are significant enough to warrant a more aggressive screening schedule (e.g., shorter-interval CT) than for a low-risk patient."
                    4.  **If the CT Scan is 'Normal' AND the Clinical Risk is 'Low':** "Consistent negative findings. Both the imaging and clinical risk models are in agreement, indicating a low probability of malignancy. Standard screening protocols apply."
                    """
                else: # Default to 'Medical Professional'
                    persona_prompt = """
                    As an AI medical assistant, synthesize the following outputs into a clear, actionable insight for a **medical professional**. 
                    Be concise and direct. Focus on consistency or discrepancy.

                    **Instructions for Professional Synthesis:**
                    1.  **If CT Scan shows carcinoma AND Clinical Risk is 'High' or 'Medium':** "Findings are consistent, reinforcing each other and increasing the likelihood of malignancy. Recommend confirmatory biopsy."
                    2.  **If CT Scan shows carcinoma BUT Clinical Risk is 'Low':** "Discrepancy noted. Radiological evidence (strong indicator) should be prioritized. The low clinical risk score is unusual; recommend review of uncaptured risk factors (e.g., family history)."
                    3.  **If CT Scan is 'Normal' BUT Clinical Risk is 'High':** "Discrepancy noted. While current scan is clear, high clinical risk profile warrants close monitoring and consideration for a shorter follow-up scan interval."
                    4.  **If CT Scan is 'Normal' AND Clinical Risk is 'Low':** "Both models agree, indicating low probability of malignancy. Recommend standard follow-up."
                    """

                # Build the final prompt dynamically
                prompt = f"""
                **Target Audience (Role):** {role}
                
                {persona_prompt}

                **Model Outputs to Analyze:**
                - **CT Scan Analysis (Image Model):** Predicted '{image_pred_label}' with {image_confidence:.2%} confidence.
                - **Clinical Data Analysis (Risk Model):** Predicted a '{clinical_pred_label}' risk level based on patient survey data, with {clinical_confidence:.2%} confidence.

                **Generate the Summary based on the provided outputs and the instructions for the selected role:**
                """
                
                response = gemini_client.models.generate_content(
                    model='models/gemini-3.5-flash', 
                    contents=[prompt]
                )
                gemini_summary = response.text
                logging.info("Gemini summary generated successfully.")
            except Exception as e:
                logging.error(f"Gemini API call failed: {e}")
                logging.error(traceback.format_exc()) # Log the full traceback
                gemini_summary = f"CT: **{image_pred_label}**, Risk: **{clinical_pred_label}**. (AI summary failed: {e})"

        # --- Prepare result data ---
        result_data = {
            'image_prediction': {'label': image_pred_label, 'confidence': image_confidence},
            'clinical_prediction': {'label': clinical_pred_label, 'confidence': clinical_confidence},
            'ai_summary': gemini_summary
        }

        # === SAVE TO FIRESTORE ===
        if db and uid: # Only save if Firestore is initialized and user is known
            try:
                prediction_entry = {
                    'userId': uid,
                    'timestamp': datetime.datetime.now(datetime.timezone.utc), # Record time
                    'role': role,
                    'clinicalInputs': input_features_dict, # <-- NOW THIS IS DEFINED
                    'results': result_data # Save the prediction results
                }
                # Add data to the user's subcollection
                doc_ref = db.collection('predictions').document(uid).collection('userPredictions').add(prediction_entry)
                logging.info(f"Prediction saved to Firestore for user {uid}, doc ID: {doc_ref[1].id}")
            except Exception as e:
                logging.error(f"Error saving prediction to Firestore for user {uid}: {e}")
                logging.error(traceback.format_exc())
                # Don't fail the request, just log the error
        else:
             if not db: logging.warning("Firestore client (db) not initialized, cannot save prediction.")
             if not uid: logging.warning("User ID (uid) not available, cannot save prediction.")
        # === END SAVE TO FIRESTORE ===

        return jsonify(result_data) # Return original results to frontend
        
    except Exception as e:
        logging.error(f"Prediction processing error for user {uid}: {e}")
        logging.error(traceback.format_exc())
        return jsonify({'error': f'Internal error during prediction processing: {e}'}), 500

# --- Run the Flask App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

