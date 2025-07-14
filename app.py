from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib, numpy as np, os
import pandas as pd

# ───── Configuración ─────
ALLOWED_ORIGIN = "https://poryectom2-1.onrender.com"  # Tu frontend
MODEL_PATH = "modelo_knn_entrenado.pkl"
INPUT_FEATURES_PATH = "input_features.pkl"
PCA_PATH = "pca_transform.pkl"

app = Flask(__name__, template_folder="templates")
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGIN}})

# ───── Cargar modelo, features y PCA ─────
try:
    model = joblib.load(MODEL_PATH)
    input_features = joblib.load(INPUT_FEATURES_PATH)
    pca = joblib.load(PCA_PATH)
    print("✅ Modelo, features y PCA cargados correctamente")
except Exception as e:
    model = None
    input_features = []
    pca = None
    print("❌ Error al cargar modelo, features o PCA:", e)

# ───── Rutas ─────

@app.route("/")
def home():
    return render_template("formulario.html")

@app.route("/formulario")
def formulario():
    return render_template("formulario.html")

@app.route("/health")
def health():
    if model is None or pca is None:
        return jsonify(status="error", message="Modelo o PCA no cargado"), 500
    return jsonify(status="ok")

@app.route("/model-info")
def model_info():
    if model is None:
        return jsonify(error="Modelo no cargado"), 500
    return jsonify(selected_features=input_features, total=len(input_features))

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or pca is None:
        return jsonify(error="Modelo o PCA no cargado"), 500

    data = request.get_json(silent=True)
    if not data or "features" not in data:
        return jsonify(error="JSON vacío o sin campo 'features'"), 400

    features = data["features"]
    if not isinstance(features, list):
        return jsonify(error="'features' debe ser lista"), 400

    if len(features) != len(input_features):
        return jsonify(
            error=f"Se requieren {len(input_features)} características",
            expected_features=len(input_features),
            received=len(features)
        ), 400

    try:
        parsed_features = [float(x) for x in features]
        X = pd.DataFrame([parsed_features], columns=input_features)
        X_pca = pca.transform(X)
        prediction = int(model.predict(X_pca)[0])
        return jsonify(predicted_class=prediction)
    except Exception as e:
        return jsonify(error=f"Error durante la predicción: {str(e)}"), 500

# ───── Arranque ─────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
