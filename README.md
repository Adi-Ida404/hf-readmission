# 🏥 Heart Failure Readmission Predictor

## 🎯 Hackathon Project Overview

A machine learning-powered web application that predicts **heart failure readmission risk** within 30 days of discharge, helping healthcare providers make informed decisions and improve patient outcomes.

### 🏆 Problem Statement : 
Heart failure is a prevalent ailment leading to fatalities if not
attended to promptly. Even for the patients who get proper treatment, hospital readmissions result in a significant risk of death and a financial burden for patients, their
families, as well as the already overburdened healthcare systems. Prediction of at-risk
patients for readmission allows for targeted interventions that reduce morbidity and
mortality: Develop a machine learning model with the end objective to predict readmission of
heart-failure patients within 30 days of discharge from the hospital.

---

**Deployed project link**: https://hf-readmission.onrender.com

**Figma Presentation link:** https://www.figma.com/design/Vp7zistaSGs37c0Ea14XQ7/ZeroBias?node-id=0-1&t=IC8NPKEahgbZ6DLs-1

**Demo Video link:** https://drive.google.com/drive/folders/1coATZvklLUUBkMYSICcdGmpnMIS1405L?usp=sharing

**Presentation:** https://www.figma.com/slides/rsPAHMOrk6QzEhwmeAyL2l/Zero-Bias?node-id=1-42&t=TsKkuE3OXrspaO9I-1

**PowerBI file:** https://drive.google.com/file/d/1UPLe8Oo10HTvhCLWUahSimUP1CFWN3Zl/view?usp=drive_link

---
## ✨ Features

- 🔮 **Dual Prediction System**: 
  - Readmission probability (Yes/No)
  - Expected readmission duration (days)
- 📊 **Comprehensive Input Processing**: Demographics, clinical data, ICD9/CPT codes, lab results
- 🎨 **User-Friendly Interface**: Clean, medical-grade web form
- ⚡ **Real-time Predictions**: Instant results with detailed patient summary

---

## 🔍 **Baseline Model: Performance Summary**

### ✅ Strengths:

* **High precision for class 0** (non-positive class): 94% — good at correctly identifying majority class.
* **Decent recall for class 1** (positive class): 46% — this is a good sign; the model does find some of the minority class.
* **Feature importance is clear**, and all top features are numeric, which gives clear insight for feature engineering.

### ⚠️ Weaknesses:

* **Low precision for class 1**: Only 13% → high false positive rate.
* **Low F1-score for class 1**: Just 0.21 → poor minority class performance.
* **Overall accuracy is misleadingly high (72%)** due to imbalance (194 positive vs. 2206 negative).

---

## 📊 Model Evaluation Metrics:

| Metric                   | Value  |
| ------------------------ | ------ |
| F1 score                 | 68%    |
| Minority class recall    | 63%    |
| Minority class precision | 63%    |
| Support (class 1)        | 334    |

These values suggest that the model is **struggling with the minority class**, which is **common in imbalanced datasets**.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Adi-Ida404/hf-readmission
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your trained models**
   Place these files in the project root:
   - `readmission_yes_no_pipeline.joblib`
   - `readmission_days.joblib`

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:5000`

---

## 🔧 How It Works

### Input Data Processing
The application accepts:
- **Demographics**: Age, gender, ethnicity, admission type
- **Clinical Info**: DRG codes, severity scores, mortality risk
- **Medical Codes**: CPT procedures, ICD9 diagnoses
- **Lab Results**: Structured lab event data
- **Procedure Pairs**: Sequential procedure mappings

### ML Pipeline
1. **Data Preprocessing**: Text cleaning, feature encoding
2. **Feature Engineering**: TF-IDF vectorization for medical codes
3. **Dual Model Prediction**:
   - Binary classifier for readmission risk
   - Regression model for readmission duration

### Output
- Risk assessment (Yes/No)
- Predicted readmission days (if high risk)
- Comprehensive patient data summary

---

## 📁 Project Structure

```
heart-failure-predictor/
├── app.py                              # Flask application
├── requirements.txt                    # Dependencies
├── readmission_yes_no_pipeline.joblib  # Binary classification model
├── readmission_days.joblib             # Regression model
├── templates/
│   ├── form.html                       # Input form
│   ├── result.html                     # Prediction results
│   └── error.html                      # Error handling
└── README.md                           # Project documentation
```

---

## 🎨 User Interface

### Input Form Features
- **Responsive Design**: Mobile-friendly medical interface
- **Smart Validation**: Real-time input validation
- **Structured Data Entry**: 
  - Dropdowns for standardized fields
  - Textareas for code lists and JSON data
  - Numerical inputs with proper constraints

### Results Display
- **Clear Risk Assessment**: Visual indicators for readmission probability
- **Detailed Summary**: Complete patient data overview
- **Actionable Insights**: Duration predictions for care planning

---

## 🛠️ Technical Stack

| Component | Technology |
|-----------|------------|
| **Backend** | Flask (Python) |
| **ML Framework** | scikit-learn |
| **Data Processing** | pandas, numpy |
| **Model Persistence** | joblib |
| **Frontend** | HTML5, CSS3 |
| **Text Processing** | TF-IDF Vectorization |

---

## 📊 Model Performance

Our dual-model approach provides:
- **Binary Classification**: Predicts readmission likelihood
- **Regression Prediction**: Estimates readmission duration
- **Feature Engineering**: Advanced text processing for medical codes
- **Robust Preprocessing**: Handles missing values and data inconsistencies

---

## 🔮 Future Enhancements

- 📈 **Real-time Dashboard**: Hospital-wide readmission monitoring
- 🤖 **Advanced ML Models**: Deep learning integration
- 📱 **Mobile App**: Native iOS/Android applications
- 🔗 **EHR Integration**: Direct hospital system connectivity
- 📊 **Analytics Suite**: Population health insights
- 🛡️ **HIPAA Compliance**: Enhanced security features

---

## 🤝 Team & Hackathon

**Built for**: Veersa Hackathon -2025  
**Team**: Zero Bias 
**Team Members:** Aditya Singh Rawat, Vipul Gupta, Riya Sharma and Kartik Bhatia
**Category**: Healthcare AI / Medical Technology

### Why This Matters
- **Cost Reduction**: Prevent unnecessary readmissions
- **Better Outcomes**: Proactive patient care planning
- **Resource Optimization**: Efficient hospital resource allocation
- **Data-Driven Healthcare**: Evidence-based medical decisions

---

## 🚨 Important Notes

⚠️ **Disclaimer**: This is a prototype developed for hackathon purposes. Not intended for actual medical diagnosis or treatment decisions.

🔒 **Data Security**: Ensure HIPAA compliance before using with real patient data.

📋 **Model Files**: Trained models not included in repository. Contact team for access.

---

## 📞 Contact & Support

For questions, suggestions, or collaboration:
- **Email**: adityaworks2004@gmail.com 
- **GitHub**: Adi-Ida404
- **LinkedIn**: https://www.linkedin.com/in/aditya-rawat-27ab59292/

---

## 📄 License

This project is developed for educational and hackathon purposes. Please ensure proper licensing for production use.

---

*Built with ❤️ for better healthcare outcomes*
