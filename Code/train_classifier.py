import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

class GestureClassifierTrainer:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        
        # Gesture label mapping
        self.gesture_names = {
            '1': 'Play',
            '2': 'Pause',
            '3': 'Volume Up',
            '4': 'Volume Down'
        }
        
    def load_data(self):
        """Load and prepare dataset"""
        print("Loading data from:", self.csv_file)
        df = pd.read_csv(self.csv_file)
        
        print(f"\nDataset shape: {df.shape}")
        print(f"\nSamples per gesture:")
        print(df['label'].value_counts())
        
        # Separate features and labels
        X = df.drop('label', axis=1)
        y = df['label']
        
        print(f"\nFeatures used: {list(X.columns)}")
        
        return X, y
    
    def prepare_data(self, X, y, test_size=0.2):
        """Split and scale data"""
        print(f"\nSplitting data: {int((1-test_size)*100)}% train, {int(test_size*100)}% test")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features (important for k-NN and SVM)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """Train multiple classifiers"""
        print("\n" + "="*50)
        print("TRAINING MODELS")
        print("="*50)
        
        # k-Nearest Neighbors
        print("\n1. Training k-NN...")
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        self.models['k-NN'] = knn
        print("   ✓ k-NN trained")
        
        # Support Vector Machine
        print("\n2. Training SVM...")
        svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
        svm.fit(X_train, y_train)
        self.models['SVM'] = svm
        print("   ✓ SVM trained")
        
        # Decision Tree
        print("\n3. Training Decision Tree...")
        dt = DecisionTreeClassifier(max_depth=10, random_state=42)
        dt.fit(X_train, y_train)
        self.models['Decision Tree'] = dt
        print("   ✓ Decision Tree trained")
    
    def evaluate_models(self, X_train, X_test, y_train, y_test):
        """Evaluate all models and find best one"""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        results = {}
        best_accuracy = 0
        
        for name, model in self.models.items():
            print(f"\n{name}:")
            print("-" * 30)
            
            # Training accuracy
            train_pred = model.predict(X_train)
            train_acc = accuracy_score(y_train, train_pred)
            
            # Testing accuracy
            test_pred = model.predict(X_test)
            test_acc = accuracy_score(y_test, test_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            cv_mean = cv_scores.mean()
            
            results[name] = {
                'train_acc': train_acc,
                'test_acc': test_acc,
                'cv_mean': cv_mean,
                'predictions': test_pred
            }
            
            print(f"  Training Accuracy:   {train_acc*100:.2f}%")
            print(f"  Testing Accuracy:    {test_acc*100:.2f}%")
            print(f"  Cross-Val Accuracy:  {cv_mean*100:.2f}% (±{cv_scores.std()*100:.2f}%)")
            
            # Track best model
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                self.best_model = model
                self.best_model_name = name
        
        print("\n" + "="*50)
        print(f"🏆 BEST MODEL: {self.best_model_name}")
        print(f"   Test Accuracy: {best_accuracy*100:.2f}%")
        print("="*50)
        
        return results, y_test
    
    def plot_confusion_matrix(self, y_test, results):
        """Plot confusion matrix for best model"""
        y_pred = results[self.best_model_name]['predictions']
        cm = confusion_matrix(y_test, y_pred)
        
        # Get gesture names for labels
        unique_labels = sorted(y_test.unique())
        labels = [self.gesture_names.get(str(lbl), str(lbl)) for lbl in unique_labels]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix - {self.best_model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save figure
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrix saved as 'confusion_matrix.png'")
        plt.show()
    
    def print_classification_report(self, y_test, results):
        """Print detailed classification report"""
        y_pred = results[self.best_model_name]['predictions']
        
        print("\n" + "="*50)
        print("DETAILED CLASSIFICATION REPORT")
        print("="*50)
        
        # Map labels to gesture names
        unique_labels = sorted(y_test.unique())
        target_names = [self.gesture_names.get(str(lbl), str(lbl)) for lbl in unique_labels]
        
        print(classification_report(y_test, y_pred, target_names=target_names))
    
    def save_model(self):
        """Save best model and scaler"""
        if self.best_model is None:
            print("No model to save!")
            return
        
        # Save model
        model_filename = f'gesture_model_{self.best_model_name.replace(" ", "_").lower()}.pkl'
        joblib.dump(self.best_model, model_filename)
        print(f"\n✓ Model saved as: {model_filename}")
        
        # Save scaler
        scaler_filename = 'gesture_scaler.pkl'
        joblib.dump(self.scaler, scaler_filename)
        print(f"✓ Scaler saved as: {scaler_filename}")
        
        # Save gesture mapping
        mapping_filename = 'gesture_mapping.pkl'
        joblib.dump(self.gesture_names, mapping_filename)
        print(f"✓ Gesture mapping saved as: {mapping_filename}")
        
        print(f"\n💾 All files saved in: {os.getcwd()}")
    
    def run(self):
        """Main training pipeline"""
        print("\n" + "="*50)
        print("GESTURE CLASSIFIER TRAINING")
        print("="*50)
        
        # Load data
        X, y = self.load_data()
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(X, y)
        
        # Train models
        self.train_models(X_train, y_train)
        
        # Evaluate models
        results, y_test = self.evaluate_models(X_train, X_test, y_train, y_test)
        
        # Print detailed report
        self.print_classification_report(y_test, results)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(y_test, results)
        
        # Save best model
        self.save_model()
        
        print("\n✅ Training complete!")
        print("\nNext step: Use the saved model in your real-time gesture recognition system!")

if __name__ == "__main__":
    # Replace with your CSV filename
    csv_file = "gesture_dataset_20251024_160809.csv"  # Change this to your actual CSV file
    
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"❌ Error: File '{csv_file}' not found!")
        print("\nAvailable CSV files in current directory:")
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        if csv_files:
            for f in csv_files:
                print(f"  - {f}")
            print(f"\nUpdate line 189 with one of these filenames")
        else:
            print("  No CSV files found!")
    else:
        trainer = GestureClassifierTrainer(csv_file)
        trainer.run()