# Customer Churn Analysis
This project predicts customer churn using the Telco Customer Churn dataset. It applies preprocessing, visualization, and a Random Forest model to classify whether a customer will leave (churn = 1) or stay (churn = 0).

## 📊 Dataset
Dataset: Telco-Customer-Churn from Kaggle

## ⚙️ Requirements
pandas  
numpy  
matplotlib  
seaborn  
scikit-learn  

## ▶️ Usage
- Place Telco-Customer-Churn.csv in the project folder
- Run the script:
```bash
python churn_analysis.py
```
OR open the notebook:

jupyter notebook churn_analysis.ipynb

## 📈 Results
- Visualizes churn distribution (class imbalance)
- Handles missing values in TotalCharges
- Random Forest Classifier → ~80% accuracy
- Confusion matrix shows performance on churn vs non-churn

📂 Structure
├── churn_analysis.ipynb
├── churn_analysis.py
├── Telco-Customer-Churn.csv
├── requirements.txt
└── README.md
