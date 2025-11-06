# train_models.py
#
# This script trains SVM, KNN, Decision Tree, Random Forest on gestures.csv.
# Evaluates via cross-validation accuracy, picks the best model, saves as best_model.pkl.
# Requires scikit-learn: pip install scikit-learn
# Run after collecting data with extract_features.py.

import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

GESTURE_CSV = "gestures.csv"
MODEL_FILE = "best_model.pkl"

def load_data():
    data = np.loadtxt(GESTURE_CSV, delimiter=',', dtype=str)
    if len(data) == 0:
        raise ValueError("No data in CSV.")
    X = data[:, :-1].astype(float)
    y = data[:, -1]
    le = LabelEncoder()
    y = le.fit_transform(y)
    return X, y, le

def train_and_evaluate():
    X, y, le = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'SVM': SVC(kernel='rbf', C=1.0),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'DT': DecisionTreeClassifier(max_depth=5),
        'RF': RandomForestClassifier(n_estimators=100, max_depth=5)
    }
    
    best_model = None
    best_score = 0.0
    best_name = ""
    
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        cv_score = np.mean(scores)
        model.fit(X_train, y_train)
        test_score = accuracy_score(y_test, model.predict(X_test))
        avg_score = (cv_score + test_score) / 2
        print(f"{name}: CV Acc={cv_score:.4f}, Test Acc={test_score:.4f}, Avg={avg_score:.4f}")
        if avg_score > best_score:
            best_score = avg_score
            best_model = model
            best_name = name
    
    print(f"Best model: {best_name} with score {best_score:.4f}")
    
    # Save model and label encoder
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump({'model': best_model, 'le': le}, f)
    print(f"Saved best model to {MODEL_FILE}")

if __name__ == "__main__":
    train_and_evaluate()