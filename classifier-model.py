import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import Counter

df = pd.read_csv('skindataall.csv')
print(df.head())
print("Data description:\n", df.describe())
print("Unique columns: \n", df.columns.unique())
print("Data shape: ", df.shape)
print("Missing values:\n", df.isnull().sum())
print("Data types:\n", df.dtypes)

# If your 'Review_Cleaned' column is a list of words, join them first
df['Review_Cleaned_str'] = df['Review_Cleaned'].apply(lambda x: ' '.join(eval(x)) if isinstance(x, str) else ' '.join(x))
print("Sample cleaned review text:\n", df['Review_Cleaned_str'].head())
    
# --- Data Preparation with Text Features ---
feature_columns = [
    'Dry', 'Normal', 'Oily', 'Sensitive', 'Combination', 'Price', 'Rating'
]

categorical_cols = ['Skin_Tone', 'Skin_Type', 'Eye_Color', 'Hair_Color']

text_col = 'Review_Cleaned_str'

# Combine all feature columns into one list for splitting
all_feature_cols = feature_columns + categorical_cols + [text_col]

X = df[all_feature_cols]
Y = df['Good_Stuff']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Combine all transformed features
X_train_num = X_train[feature_columns].values
X_test_num = X_test[feature_columns].values

# Fit the encoder ONLY on the training data
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_cat_encoded = ohe.fit_transform(X_train[categorical_cols])
X_test_cat_encoded = ohe.transform(X_test[categorical_cols])

# Handle Text Features (TF-IDF) - slightly improved
print("Vectorizing review text")
tfidf = TfidfVectorizer(max_features=2500, stop_words='english')  # Slightly increased features
X_train_text = tfidf.fit_transform(X_train[text_col])
X_test_text = tfidf.transform(X_test[text_col])

print("Combining all feature sets")
# Use hstack to combine the sparse text matrix with the dense numpy arrays
X_train_combined = hstack([X_train_num, X_train_cat_encoded, X_train_text])
X_test_combined = hstack([X_test_num, X_test_cat_encoded, X_test_text])

print(f"\nShape of combined training data: {X_train_combined.shape}")
print(f"Shape of combined test data: {X_test_combined.shape}")

# Use SMOTE with specific sampling strategy to oversample minority class more
smote = SMOTE(
    sampling_strategy='auto',  # Balance classes
    k_neighbors=5,  # Increased neighbors for better synthetic samples
    random_state=42
)
X_train_balanced, Y_train_balanced = smote.fit_resample(X_train_combined, Y_train)
print(f"After SMOTE, training data shape: {X_train_balanced.shape}")
print(f"After SMOTE class distribution: {pd.Series(Y_train_balanced).value_counts()}")

# Parameter grid for Random Forest
rf_param_grid = {
    'n_estimators': [200, 300, 400], 
    'max_depth': [20, 25, 30, None],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced', 'balanced_subsample', None]  # Added class weights
}

# Parameter grid for XGBoost
xgb_param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# Create a base model
rfc = RandomForestClassifier(random_state=42)
xgb = XGBClassifier(random_state=42)

# Improved random search models with correct parameter grids
rfc_random_search = RandomizedSearchCV(estimator=rfc,
                                       param_distributions=rf_param_grid,
                                       n_iter=30,  # Reasonable number of iterations
                                       cv=3,
                                       verbose=2,
                                       random_state=42,
                                       n_jobs=-1)

xgb_random_search = RandomizedSearchCV(estimator=xgb,
                                       param_distributions=xgb_param_grid,
                                       n_iter=30,  # Reasonable number of iterations
                                       cv=3,
                                       verbose=2,
                                       random_state=42,
                                       n_jobs=-1)

# Fit the random search on the BALANCED and COMBINED data
print("\nStarting hyperparameter tuning on combined features")
rfc_random_search.fit(X_train_balanced, Y_train_balanced)
xgb_random_search.fit(X_train_balanced, Y_train_balanced)

# Get the best hyperparameters
print("\nBest hyperparameters found:")
print(f"Random Forest: {rfc_random_search.best_params_}")
print(f"XGBoost: {xgb_random_search.best_params_}")

# Get the best model from the search
best_rfc = rfc_random_search.best_estimator_
print(f"\nBest model - Random Forest : {best_rfc}")
best_xgb = xgb_random_search.best_estimator_
print(f"\nBest model - XGBoost : {best_xgb}")

# Making predictions on the COMBINED test data
Y_pred_rfc = best_rfc.predict(X_test_combined)
Y_pred_xgb = best_xgb.predict(X_test_combined)

print("\nRandom Forest Classifier Evaluation:\n")
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred_rfc))
print("Classification Report:\n", classification_report(Y_test, Y_pred_rfc))
print("Accuracy Score: ", accuracy_score(Y_test, Y_pred_rfc))

print("\nXGBoost Classifier Evaluation\n")
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred_xgb))
print("Classification Report:\n", classification_report(Y_test, Y_pred_xgb))
print("Accuracy Score: ", accuracy_score(Y_test, Y_pred_xgb))

# Compare with Random Forest
rf_accuracy = accuracy_score(Y_test, Y_pred_rfc)
xgb_accuracy = accuracy_score(Y_test, Y_pred_xgb)

print(f"\n--- MODEL COMPARISON ---")
print(f"Random Forest Accuracy: {rf_accuracy}")  # 0.8225433526011561
print(f"XGBoost Accuracy:       {xgb_accuracy}") # 0.8369942196531792

