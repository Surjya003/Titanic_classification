import pandas as pd
import tkinter as tk
from tkinter import messagebox, ttk
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess the Titanic dataset
def preprocess_data():
    data = pd.read_csv('titanic.csv')
    data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Fare'].fillna(data['Fare'].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

    label_encoder = LabelEncoder()
    data['Sex'] = label_encoder.fit_transform(data['Sex'])
    data['Embarked'] = label_encoder.fit_transform(data['Embarked'])
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = 1
    data['IsAlone'].loc[data['FamilySize'] > 1] = 0

    X = data[['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Embarked', 'FamilySize', 'IsAlone']]
    y = data['Survived']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    return X_resampled, y_resampled, X, y, scaler, data.columns


# Train models
def train_models(X_train, y_train):
    # Train Gradient Boosting Model
    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Stacking Classifier
    base_models = [
        ('gbc', GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
    ]
    meta_model = LogisticRegression()
    stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model)
    stacking_model.fit(X_train, y_train)

    # XGBoost Classifier
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)

    return model, stacking_model, xgb_model

# Evaluate models
def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred) * 100
        results[name] = accuracy
        print(f"{name} Model Accuracy: {accuracy:.2f}%")

        # Print classification report and confusion matrix
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))

        # Feature Importances
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = importances.argsort()[::-1]
            features = X.columns
            print("Feature importances:")
            for i in range(X.shape[1]):
                print(f"{features[indices[i]]}: {importances[indices[i]]}")

            # Plot feature importances
            plt.figure(figsize=(10, 6))
            sns.barplot(x=importances[indices], y=features[indices])
            plt.title(f'{name} Feature Importances')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.show()

    return results

# Load Titanic data and prepare for Tkinter UI
def load_passenger_data():
    global passenger_data
    passenger_data = pd.read_csv('titanic.csv')
    passenger_data = passenger_data.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    passenger_data['FamilySize'] = passenger_data['SibSp'] + passenger_data['Parch'] + 1
    passenger_data['IsAlone'] = 1
    passenger_data['IsAlone'].loc[passenger_data['FamilySize'] > 1] = 0
    passenger_data['Fare'].fillna(passenger_data['Fare'].median(), inplace=True)  # Handle missing Fare

    passenger_ids = passenger_data['PassengerId'].tolist()
    passenger_id_combobox['values'] = passenger_ids

def predict_survival():
    try:
        pclass = int(pclass_entry.get())
        sex = int(sex_entry.get())
        age = float(age_entry.get())
        fare = float(fare_entry.get())
        sibsp = int(sibsp_entry.get())
        parch = int(parch_entry.get())
        embarked = int(embarked_entry.get())

        # Create a DataFrame for the input
        input_data = pd.DataFrame({
            'Pclass': [pclass],
            'Sex': [sex],
            'Age': [age],
            'Fare': [fare],
            'SibSp': [sibsp],
            'Parch': [parch],
            'Embarked': [embarked],
            'FamilySize': [sibsp + parch + 1],
            'IsAlone': [1 if (sibsp + parch + 1) <= 1 else 0]
        })

        # Standardize the input
        input_scaled = scaler.transform(input_data)

        # Predict using the trained model
        prediction = model.predict(input_scaled)
        result = 'Survived' if prediction[0] == 1 else 'Did not survive'
        messagebox.showinfo("Prediction", f"The passenger {result}.")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid input values.")

# Main execution
if __name__ == "__main__":
    X_resampled, y_resampled, X, y, scaler, feature_names = preprocess_data()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Train models
    model, stacking_model, xgb_model = train_models(X_train, y_train)

    # Evaluate models
    models = {
        'Gradient Boosting': model,
        'Stacking Classifier': stacking_model,
        'XGBoost': xgb_model
    }
    evaluate_models(models, X_test, y_test)

    # Create Tkinter UI
    root = tk.Tk()
    root.title("Titanic Survival Prediction")

    tk.Label(root, text="Pclass (1, 2, or 3):").grid(row=0, column=0, padx=10, pady=5)
    pclass_entry = tk.Entry(root)
    pclass_entry.grid(row=0, column=1, padx=10, pady=5)

    tk.Label(root, text="Sex (0 for female, 1 for male):").grid(row=1, column=0, padx=10, pady=5)
    sex_entry = tk.Entry(root)
    sex_entry.grid(row=1, column=1, padx=10, pady=5)

    tk.Label(root, text="Age:").grid(row=2, column=0, padx=10, pady=5)
    age_entry = tk.Entry(root)
    age_entry.grid(row=2, column=1, padx=10, pady=5)

    tk.Label(root, text="Fare:").grid(row=3, column=0, padx=10, pady=5)
    fare_entry = tk.Entry(root)
    fare_entry.grid(row=3, column=1, padx=10, pady=5)

    tk.Label(root, text="SibSp:").grid(row=4, column=0, padx=10, pady=5)
    sibsp_entry = tk.Entry(root)
    sibsp_entry.grid(row=4, column=1, padx=10, pady=5)

    tk.Label(root, text="Parch:").grid(row=5, column=0, padx=10, pady=5)
    parch_entry = tk.Entry(root)
    parch_entry.grid(row=5, column=1, padx=10, pady=5)

    tk.Label(root, text="Embarked (0 for C, 1 for Q, 2 for S):").grid(row=6, column=0, padx=10, pady=5)
    embarked_entry = tk.Entry(root)
    embarked_entry.grid(row=6, column=1, padx=10, pady=5)

    predict_button = tk.Button(root, text="Predict Survival", command=predict_survival)
    predict_button.grid(row=7, column=0, columnspan=2, pady=20)

    root.mainloop()
