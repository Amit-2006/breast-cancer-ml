# Breast Cancer Prediction (ML Project)

## Objective
Tumor ki measurements se predict karna ki wo Malignant hai ya Benign.

## Tech
Python, pandas, scikit-learn, matplotlib, Streamlit

## Dataset
Wisconsin Breast Cancer (569 samples, 30 features), sklearn me built-in.

## Run
    pip install -r requirements.txt
    python train.py          # models train + compare, model.pkl banata hai
    streamlit run app.py     # web app

## Models compared
Logistic Regression, KNN, SVM, Random Forest (StandardScaler pipeline, 5-fold CV)

## Output
model.pkl, confusion_matrix.png, model_comparison.png
