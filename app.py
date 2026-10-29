"""
Interface web de prédiction du diabète
======================================

Lance le serveur :
    python app.py
puis ouvre http://127.0.0.1:5000

Le modèle (SVM linéaire) est entraîné une première fois sur diabetes.csv,
puis mis en cache dans model.joblib. Il est ré-entraîné automatiquement si
le fichier CSV est modifié.
"""

import os
import time

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "diabetes.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")

app = Flask(__name__)

# Champs du formulaire : clé, libellé, unité/aide, min, max, pas, valeur par défaut
FIELDS = [
    {
        "name": "Pregnancies",
        "label": "Nombre de grossesses",
        "hint": "0 si aucune",
        "min": 0, "max": 20, "step": 1, "default": 5,
    },
    {
        "name": "Glucose",
        "label": "Glycémie (Glucose)",
        "hint": "mg/dL, à jeun",
        "min": 0, "max": 250, "step": 1, "default": 166,
    },
    {
        "name": "BloodPressure",
        "label": "Pression artérielle diastolique",
        "hint": "mm Hg",
        "min": 0, "max": 150, "step": 1, "default": 72,
    },
    {
        "name": "SkinThickness",
        "label": "Épaisseur du pli cutané",
        "hint": "mm (triceps)",
        "min": 0, "max": 100, "step": 1, "default": 19,
    },
    {
        "name": "Insulin",
        "label": "Insuline sérique",
        "hint": "µU/mL, 2 h après charge",
        "min": 0, "max": 900, "step": 1, "default": 175,
    },
    {
        "name": "BMI",
        "label": "IMC (BMI)",
        "hint": "kg/m²",
        "min": 0, "max": 70, "step": 0.1, "default": 25.8,
    },
    {
        "name": "DiabetesPedigreeFunction",
        "label": "Antécédents familiaux",
        "hint": "score 0 – 2,5 (Diabetes Pedigree Function)",
        "min": 0, "max": 3, "step": 0.001, "default": 0.587,
    },
    {
        "name": "Age",
        "label": "Âge",
        "hint": "années",
        "min": 1, "max": 120, "step": 1, "default": 51,
    },
]

FEATURE_NAMES = [field["name"] for field in FIELDS]


# ---------------------------------------------------------------------------
# Entraînement / chargement du modèle
# ---------------------------------------------------------------------------
def train_model():
    """Entraîne le SVM sur diabetes.csv et met le résultat en cache."""
    dataset = pd.read_csv(DATA_PATH)
    X = dataset.drop(columns="Outcome")
    Y = dataset["Outcome"]

    scaler = StandardScaler()
    scaler.fit(X.values)
    standardized = scaler.transform(X.values)

    X_train, X_test, Y_train, Y_test = train_test_split(
        standardized, Y, test_size=0.2, stratify=Y, random_state=2
    )

    classifier = svm.SVC(kernel="linear")
    classifier.fit(X_train, Y_train)

    metrics = {
        "rows": int(dataset.shape[0]),
        "columns": int(X.shape[1]),
        "train": round(float(accuracy_score(classifier.predict(X_train), Y_train)), 4),
        "test": round(float(accuracy_score(classifier.predict(X_test), Y_test)), 4),
        "trained_at": time.strftime("%d/%m/%Y à %H:%M"),
    }

    payload = {"scaler": scaler, "classifier": classifier, "metrics": metrics}
    joblib.dump(payload, MODEL_PATH)
    return payload


def load_model():
    """Charge le modèle en cache, ou l'entraîne s'il est absent/obsolète."""
    if os.path.exists(MODEL_PATH) and os.path.exists(DATA_PATH):
        model_is_newer = os.path.getmtime(MODEL_PATH) >= os.path.getmtime(DATA_PATH)
        if model_is_newer:
            try:
                return joblib.load(MODEL_PATH)
            except Exception:
                pass
    return train_model()


MODEL = load_model()


@app.template_filter("num")
def format_number(value):
    """Affiche 5.0 -> 5, 25.8 -> 25.8 (sans zéro inutile)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return str(int(number)) if number == int(number) else f"{number:g}"



# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template(
        "index.html",
        fields=FIELDS,
        values={field["name"]: field["default"] for field in FIELDS},
        metrics=MODEL["metrics"],
        result=None,
        errors=[],
    )


def read_form(form):
    """Lit et valide les 8 champs. Retourne (valeurs, erreurs)."""
    values = {}
    errors = []

    for field in FIELDS:
        raw_value = form.get(field["name"])
        raw = "" if raw_value is None else str(raw_value).strip().replace(",", ".")
        if raw == "":
            errors.append(f"Le champ « {field['label']} » est vide.")
            values[field["name"]] = ""
            continue
        try:
            number = float(raw)
        except ValueError:
            errors.append(f"« {field['label']} » doit être un nombre (reçu : {raw}).")
            values[field["name"]] = raw
            continue
        if number < field["min"] or number > field["max"]:
            errors.append(
                f"« {field['label']} » doit être compris entre "
                f"{field['min']} et {field['max']}."
            )
        values[field["name"]] = number

    return values, errors


def predict(values):
    """Standardise les valeurs du formulaire et renvoie la prédiction."""
    sample = np.asarray([[float(values[name]) for name in FEATURE_NAMES]], dtype=float)
    standardized = MODEL["scaler"].transform(sample)
    label = int(MODEL["classifier"].predict(standardized)[0])
    score = float(MODEL["classifier"].decision_function(standardized)[0])
    # score -> indice entre 0 et 1 (non calibré, purement indicatif)
    confidence = 1.0 / (1.0 + np.exp(-score))

    return {
        "label": label,
        "is_diabetic": label == 1,
        "score": round(score, 3),
        "confidence": round(100 * (confidence if label == 1 else 1 - confidence), 1),
        "standardized": [round(float(v), 4) for v in standardized[0]],
        "values": values,
    }


@app.route("/predict", methods=["POST"])
def predict_route():
    values, errors = read_form(request.form)

    if errors:
        return render_template(
            "index.html",
            fields=FIELDS,
            values=values,
            metrics=MODEL["metrics"],
            result=None,
            errors=errors,
        ), 400

    result = predict(values)
    return render_template(
        "index.html",
        fields=FIELDS,
        values=values,
        metrics=MODEL["metrics"],
        result=result,
        errors=[],
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API JSON : POST {"Pregnancies": 5, "Glucose": 166, ...}"""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Corps JSON attendu (objet)."}), 400

    values, errors = read_form(data)
    if errors:
        return jsonify({"error": "Données invalides.", "details": errors}), 400

    result = predict(values)
    return jsonify(
        {
            "prediction": result["label"],
            "resultat": "diabétique" if result["is_diabetic"] else "non diabétique",
            "score_decision": result["score"],
            "indice_confiance_pct": result["confidence"],
        }
    )


if __name__ == "__main__":
    print("Modèle chargé :", MODEL["metrics"])
    print("Interface disponible sur http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
