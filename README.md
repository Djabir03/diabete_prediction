# diabete_prediction

Prédiction du diabète à partir du **PIMA Diabetes Dataset** (768 patients, 8 variables
cliniques), avec un **SVM linéaire** (scikit-learn). Le projet propose trois façons
d'utiliser le modèle : un notebook Colab, un script en ligne de commande et une
interface web.

## Contenu du dépôt

| Fichier | Description |
|---|---|
| `Project_3_Diabetes_Prediction.ipynb` | Notebook d'origine (Google Colab) |
| `diabetes_prediction.py` | Script CLI : entraînement, évaluation, saisie interactive |
| `app.py` | Application web Flask (formulaire + API JSON) |
| `templates/index.html` | Page de saisie et carte de résultat |
| `static/style.css` | Mise en page responsive |
| `diabetes.csv` | Dataset PIMA (768 lignes × 9 colonnes) |
| `requirements.txt` | Dépendances Python |

## Installation

```bash
git clone https://github.com/Djabir03/diabete_prediction.git
cd diabete_prediction
pip install -r requirements.txt
```

## Utilisation

### 1. Interface web (recommandé)

```bash
python app.py
```

Puis ouvrir <http://127.0.0.1:5000>. On saisit les 8 mesures du patient
(grossesses, glycémie, pression artérielle, pli cutané, insuline, IMC,
antécédents familiaux, âge) et la page affiche la classe prédite.

Le modèle est entraîné au premier lancement puis mis en cache dans
`model.joblib` ; il est ré-entraîné automatiquement si `diabetes.csv` change.

### 2. Script en ligne de commande

```bash
python diabetes_prediction.py
```

Le script entraîne le modèle, affiche les scores d'accuracy et propose une saisie
interactive (Entrée = valeur par défaut).
Les valeurs standardisées affichées par le notebook sont exactement
reproduites :

- accuracy entraînement : **0.7866**
- accuracy test : **0.7727**

### 3. API JSON

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"Pregnancies":5,"Glucose":166,"BloodPressure":72,"SkinThickness":19,
       "Insulin":175,"BMI":25.8,"DiabetesPedigreeFunction":0.587,"Age":51}'
```

```json
{
  "indice_confiance_pct": 58.4,
  "prediction": 1,
  "resultat": "diabétique",
  "score_decision": 0.338
}
```

## Méthodologie

1. Chargement du CSV avec pandas.
2. Séparation variables / étiquette (`Outcome` : 0 = non diabétique, 1 = diabétique).
3. Standardisation (`StandardScaler`).
4. Découpage 80 / 20 stratifié (`random_state=2`).
5. Entraînement d'un `SVC(kernel="linear")`.
6. Évaluation par accuracy sur les ensembles d'entraînement et de test.

## Avertissement

Projet pédagogique. Les prédictions ne constituent en aucun cas un diagnostic
médical et ne doivent pas être utilisées pour prendre une décision de santé.
