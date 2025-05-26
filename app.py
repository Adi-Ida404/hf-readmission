from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import pandas as pd
import re
import json
from joblib import load
from ast import literal_eval
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os

class TfidfWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, **tfidf_kwargs):
        self.tfidf = TfidfVectorizer(**tfidf_kwargs)
    def fit(self, X, y=None):
        data = X.fillna('').astype(str).values.ravel()
        self.tfidf.fit(data)
        return self
    def transform(self, X):
        data = X.fillna('').astype(str).values.ravel()
        return self.tfidf.transform(data)


def to_dense(x):
    return x.toarray()

app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Needed for session

# Load both models
try:
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
    # Handle lab events as comma separated values
    lab_items = data.get('lab_events', '').strip()
    if lab_items:
        try:
            items = [item.strip() for item in lab_items.split(',')]
            result_items = []
            for i in range(0, len(items), 4):
                if i + 3 < len(items):
                    try:
                        result_items.append([
                            int(items[i]),
                            items[i + 1].strip(),
                            items[i + 2].strip(),
                            float(items[i + 3].strip()) if items[i + 3].strip().replace('.', '').isdigit() else items[i + 3].strip()
                        ])
                    except (ValueError, TypeError) as e:
                        print(f"Error processing items at index {i}: {e}")
                        continue
        except Exception as e:
            print(f"Error processing lab events: {e}")
            result_items = []
    else:
        result_items = []
    
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
        'lab_events': result_items
    }])
    
    df['cpt_cd'] = df['cpt_cd'].apply(lambda x: ','.join(map(str, x)))
    df['cpt_cd'] = df['cpt_cd'].apply(remove_punc)
    df['all_diagnosis'] = df['all_diagnosis'].apply(remove_punc)
    df['description'] = df['description'].apply(remove_punc)
    df['procedure_pairs'] = df['procedure_pairs'].apply(remove_punc)
    df['lab_events'] = df['lab_events'].apply(lambda x: ','.join(map(str, x)) if x else '')
    df['lab_events'] = df['lab_events'].apply(remove_punc)
    
    return df

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not readmission_model or not days_model:
        return render_template('error.html', message="Models not loaded"), 500
    
    try:
        input_df = prepare_input(request.form)
        
        # Make predictions
        readmission_pred = readmission_model.predict(input_df)[0]
        days_pred = days_model.predict(input_df)[0] if readmission_pred == 1 else None
        
        # Prepare all data for results template
        patient_data = request.form.to_dict()
        patient_data['lab_events'] = input_df['lab_events'].iloc[0]  # Use processed lab events
        
        # Format the message
        message = f"Need of readmission within 30 days: {'Yes' if readmission_pred == 1 else 'No'}"
        if readmission_pred == 1:
            message += f"Intervene their discharge !   Days in readmission: {int(days_pred)}"

        return render_template('result.html',
            patient_data=patient_data,
            readmission="Yes" if readmission_pred == 1 else "No",
            days=int(days_pred) if readmission_pred == 1 else None,
            message=message)
        
    except Exception as e:
        return render_template('error.html', message=str(e)), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)), debug=False)
