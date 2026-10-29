"""
Project 3 - Diabetes Prediction
===============================

Script Python équivalent au notebook Colab :
https://colab.research.google.com/drive/1oxnhMTlomJ4HVhPuowpPFyMt1mwuOuQo

Dataset : PIMA Diabetes Dataset (diabetes.csv)
Placez le fichier `diabetes.csv` à côté de ce script (ou dans /content/
si vous l'exécutez sur Google Colab).

Dépendances :
    pip install numpy pandas scikit-learn
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# ---------------------------------------------------------------------------
# Affichage : la console Windows utilise par défaut la page de codes 850,
# ce qui casse les accents. On force l'UTF-8 quand c'est possible.
# ---------------------------------------------------------------------------
def enable_utf8_console():
    if os.name == "nt":
        try:
            os.system("chcp 65001 > nul")
        except Exception:
            pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


enable_utf8_console()



# ---------------------------------------------------------------------------
# Localisation du dataset
# ---------------------------------------------------------------------------
def find_dataset():
    """Cherche diabetes.csv à côté du script, puis dans /content (Colab)."""
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "diabetes.csv"),
        os.path.join(os.getcwd(), "diabetes.csv"),
        "/content/diabetes.csv",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "Fichier 'diabetes.csv' introuvable. Placez-le à côté de ce script."
    )


# ---------------------------------------------------------------------------
# 1. Importing the Dependencies (voir imports ci-dessus)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 2. Data Collection and Analysis - PIMA Diabetes Dataset
# ---------------------------------------------------------------------------
diabetes_dataset = pd.read_csv(find_dataset())

# printing the first 5 rows of the dataset
print(diabetes_dataset.head())

# number of rows and Columns in this dataset
print(diabetes_dataset.shape)

# getting the statistical measures of the data
print(diabetes_dataset.describe())

print(diabetes_dataset["Outcome"].value_counts())
# 0 --> Non-Diabetic
# 1 --> Diabetic

print(diabetes_dataset.groupby("Outcome").mean())

# separating the data and labels
X = diabetes_dataset.drop(columns="Outcome")
Y = diabetes_dataset["Outcome"]

print(X)
print(Y)

# ---------------------------------------------------------------------------
# 3. Data Standardization
# ---------------------------------------------------------------------------
scaler = StandardScaler()
scaler.fit(X.values)

standardized_data = scaler.transform(X.values)
print(standardized_data)

X = standardized_data
Y = diabetes_dataset["Outcome"]

print(X)
print(Y)

# ---------------------------------------------------------------------------
# 4. Train Test Split
# ---------------------------------------------------------------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

print(X.shape, X_train.shape, X_test.shape)

# ---------------------------------------------------------------------------
# 5. Training the Model
# ---------------------------------------------------------------------------
classifier = svm.SVC(kernel="linear")

# training the support vector Machine Classifier
classifier.fit(X_train, Y_train)

# ---------------------------------------------------------------------------
# 6. Model Evaluation - Accuracy Score
# ---------------------------------------------------------------------------
# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print("Accuracy score of the training data : ", training_data_accuracy)

# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print("Accuracy score of the test data : ", test_data_accuracy)

# ---------------------------------------------------------------------------
# 7. Making a Predictive System (saisie interactive)
# ---------------------------------------------------------------------------

# ordre des colonnes attendu par le modèle
FEATURES = [
    ("Pregnancies", int),
    ("Glucose", int),
    ("BloodPressure", int),
    ("SkinThickness", int),
    ("Insulin", int),
    ("BMI", float),
    ("DiabetesPedigreeFunction", float),
    ("Age", int),
]

# valeurs proposées par défaut (appuyer sur Entrée pour les accepter)
DEFAULT_VALUES = (5, 166, 72, 19, 175, 25.8, 0.587, 51)


def ask_value(name, value_type, default):
    """Demande une valeur à l'utilisateur ; Entrée = valeur par défaut."""
    while True:
        try:
            raw = input(f"  {name} [{default}] : ").strip().replace(",", ".")
        except EOFError:
            # aucune saisie disponible (script lancé sans console interactive)
            print(default)
            return default

        if raw == "":
            return default

        try:
            return value_type(raw)
        except ValueError:
            print(f"  -> '{raw}' n'est pas un nombre valide, réessayez.")


def ask_yes_no(question):
    """Pose une question fermée ; Entrée ou 'n' = non."""
    try:
        answer = input(f"{question} (o/n) [n] : ").strip().lower()
    except EOFError:
        print("n")
        return False
    return answer in ("o", "oui", "y", "yes")


def predict_person():
    """Saisie des 8 caractéristiques puis prédiction."""
    print("\n--- Saisie des données du patient ---")
    print("Appuyez sur Entrée pour garder la valeur proposée entre crochets.\n")

    values = [
        ask_value(name, value_type, default)
        for (name, value_type), default in zip(FEATURES, DEFAULT_VALUES)
    ]

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(values, dtype=float)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # standardize the input data
    std_data = scaler.transform(input_data_reshaped)

    prediction = classifier.predict(std_data)

    print("\n--- Résultat ---")
    print("Données saisies      :", tuple(values))
    print("Données standardisées:", np.round(std_data[0], 4))
    print("Classe prédite       :", prediction[0])

    if prediction[0] == 0:
        print("The person is not diabetic")
    else:
        print("The person is diabetic")
    print()


print("\n=== Système de prédiction du diabète ===")
while True:
    predict_person()
    if not ask_yes_no("Tester une autre personne ?"):
        print("Fin du programme.")
        break

