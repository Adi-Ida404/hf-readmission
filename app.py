from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import re
from joblib import load
from ast import literal_eval
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class TfidfWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, **tfidf_kwargs):
        self.tfidf = TfidfVectorizer(**tfidf_kwargs)
    def fit(self, X, y=None):
        # Convert all to str, fill NaNs with empty string
        data = X.fillna('').astype(str).values.ravel()
        self.tfidf.fit(data)
        return self
    def transform(self, X):
        data = X.fillna('').astype(str).values.ravel()
        return self.tfidf.transform(data)


app = Flask(__name__)

# Load both models
try:
    # Load models with custom class
    with open('readmission_yes_no_pipeline.joblib', 'rb') as f:
        readmission_model = load(f)
    with open('readmission_days.joblib', 'rb') as f:
        days_model = load(f)
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    readmission_model = None
    days_model = None

def remove_punc(text):
    if not isinstance(text, str):
        return text
    return re.sub(r'[^\w\s]', '', text)

def prepare_input(form_data):
    data = form_data.to_dict()
    
    # Create DataFrame with same structure
    df = pd.DataFrame([{
        'admission_type': data.get('admission_type'),
        'discharge_location': data.get('discharge_location'),
        'ethnicity': data.get('ethnicity'),
        'cpt_cd': data.get('cpt_cd', '').split(','),
        'all_diagnosis': data.get('all_diagnosis', ''),
        'gender': data.get('gender'),
        'age': int(data.get('age', 0)),
        'drg_type': data.get('drg_type'),
        'drg_code': float(data.get('drg_code', 0)),
        'description': data.get('description', ''),
        'drg_severity': int(data.get('drg_severity', 0)),
        'drg_mortality': int(data.get('drg_mortality', 0)),
        'procedure_pairs': data.get('procedure_pairs', '[]'),
        'lab_events': data.get('lab_events', '[]')
    }])
    
    # Apply friend's exact preprocessing steps
    df['cpt_cd'] = df['cpt_cd'].apply(lambda x: ','.join(map(str, x)))
    df['cpt_cd'] = df['cpt_cd'].apply(remove_punc)
    df['all_diagnosis'] = df['all_diagnosis'].apply(remove_punc)
    df['description'] = df['description'].apply(remove_punc)
    df['procedure_pairs'] = df['procedure_pairs'].apply(remove_punc)
    
    try:
        lab_events = literal_eval(df['lab_events'].iloc[0])
        df['lab_events'] = [','.join(map(str, lab_events))]
    except:
        df['lab_events'] = ['']
    
    df['lab_events'] = df['lab_events'].apply(remove_punc)
    
    return df

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not readmission_model or not days_model:
        return jsonify({"status": "error", "message": "Models not loaded"}), 500

    try:
        # Prepare input data exactly as friend's code
        input_df = prepare_input(request.form)
        
        # Make predictions
        readmission_pred = readmission_model.predict(input_df)[0]
        days_pred = days_model.predict(input_df)[0] if readmission_pred == 1 else None
        
        # Format results
        if readmission_pred == 1:
            result = {
                "status": "success",
                "readmission": "Yes",
                "days": int(days_pred),
                "message": f"Need of readmission within 30 days: Yes. Days in readmission: {int(days_pred)}"
            }
        else:
            result = {
                "status": "success",
                "readmission": "No",
                "days": None,
                "message": "Need of readmission within 30 days: No"
            }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "received_data": dict(request.form)
        }), 400

if __name__ == '__main__':
    app.run(debug=True)