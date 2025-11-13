# 🧭 Titanic Survival Prediction  
### Machine Learning Challenge — Ironhack Berlin (Data Analytics Bootcamp)

This project implements an end-to-end **Machine Learning pipeline** to predict passenger survival in the Titanic disaster using the famous Kaggle dataset.  
It follows professional ML workflow standards including cleaning, feature engineering, model tuning and final predictions on unseen data.

---

## 🚀 Project Overview

This challenge simulates a full ML workflow:

1. Exploratory Data Analysis (EDA)  
2. Data cleaning & handling missing values  
3. Feature engineering  
4. Preprocessing pipeline (Label Encoding + One-Hot Encoding)  
5. Model training (Random Forest)  
6. Hyperparameter tuning with **GridSearchCV**  
7. Imbalance-handling experiments (Oversampling, SMOTE, TomekLinks)  
8. Final model training  
9. Predictions on the test set  
10. Submission file generation (`submission.csv`)

---

## 🧹 Data Cleaning & Feature Engineering

The following transformations were applied:

### Handling Missing Values
- `Age` → median  
- `Embarked` → mode  
- `Fare` → median  

### Feature Engineering
- **Title extraction** from passenger names  
- **FamilySize** = SibSp + Parch + 1  
- **Deck** extracted from the first Cabin letter, unknown → `"U"`  

### Columns removed
`Name`, `Ticket`, `Cabin`, `PassengerId`  

---

## 🧠 Machine Learning Models

Two Random Forest models were trained:

### 1️⃣ Baseline Random Forest  
- `n_estimators = 200`  
- Accuracy: **0.8109**

---

### 2️⃣ Optimized Random Forest (GridSearchCV)
Best parameters:
max_depth = 10
min_samples_leaf = 2
min_samples_split = 2
n_estimators = 100

- Final Accuracy: **0.8324**  
- Selected as the **final model**

---

## ⚖️ Imbalance Handling Experiments

Three sampling techniques were tested:

### ✔ Oversampling  
Improved recall but introduced noise.

### ✔ SMOTE  
Stable performance, very close to baseline accuracy.

### ✔ TomekLinks  
Best recall for the minority class (survivors).

➡ These were used **only for experimentation**, not in the final pipeline.

---

## 🏁 Final Predictions

The final pipeline was applied to the instructor-provided **test.csv** dataset to generate:

### Example output:
```
| PassengerId | Survived |
|-------------|----------|
| 892         | 0        |
| 893         | 0        |
| 894         | 0        |
| 895         | 0        |
| 896         | 1        |
```
---

## 📁 Repository Structure
```
├── Final_Model_Titanic.ipynb # Final pipeline + predictions
├── Titanic-Dataset.csv # Original dataset used for exploration
├── train.csv # Provided training dataset
├── test.csv # Provided test dataset
├── submission.csv # Final predictions
├── README.md # Project documentation
```

---

## 🧩 Tools & Libraries

- Python  
- pandas  
- numpy  
- scikit-learn  
- imblearn  
- matplotlib / seaborn  
- Jupyter Notebook  

---

## 👩‍💻 Author

**Irma Fernández Wiechers**  
Data Analyst — Ironhack Berlin  
GitHub: [@irmacfw](https://github.com/irmacfw)



