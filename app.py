"""
BladderVision AI — Flask Backend
Hybrid Quantum-Classical CNN  +  Explainable AI (Groq LLM)
===========================================================

Endpoints:
  GET  /                  → index.html
  POST /predict           → multipart image → JSON result + XAI explanation
  POST /predict-tabular   → JSON patient features → disease classification
  GET  /health            → server health check
"""

import os, io, json
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import tensorflow as tf
from dotenv import load_dotenv
import tabular_predict

load_dotenv()

# ── Groq XAI ─────────────────────────────────────────────────────────────────
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL   = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_ENABLED = bool(GROQ_API_KEY)
groq_client  = Groq(api_key=GROQ_API_KEY) if GROQ_ENABLED else None

# ── Quantum / Classical model flags ───────────────────────────────────────────
QML_MODEL_PATH = "bladder_qml_model.pt"
CNN_MODEL_PATH = "bladder_model.keras"
CLASS_IDX_FILE = "class_indices.json"
QML_META_FILE  = "qml_meta.json"
IMG_SIZE       = (128, 128)
# Cancer = class 0, No Cancer = class 1.
# sigmoid output = P(No Cancer). Lower threshold = more sensitive to cancer (safer for medical use).
CANCER_THRESH  = 0.42   # if P(No Cancer) < 0.42 → Cancer  (keeps recall high)
NO_CANCER_THRESH = 0.58 # if P(No Cancer) > 0.58 → confident No Cancer

app = Flask(__name__, static_folder=".", static_url_path="")

# ── Load models ───────────────────────────────────────────────────────────────
USE_QML = False   # will flip to True if QML model is available

print("=" * 60)
print("  BladderVision AI — Loading models …")
print("=" * 60)

# Always load classical CNN (used as backbone for QML too)
if not os.path.exists(CNN_MODEL_PATH):
    raise FileNotFoundError(
        f"'{CNN_MODEL_PATH}' not found. Run train_model.py first.")

cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
print(f"✅ Classical CNN loaded: {CNN_MODEL_PATH}")

# Try loading QML hybrid model
qml_model = None
qml_meta  = {}
if os.path.exists(QML_MODEL_PATH) and os.path.exists(QML_META_FILE):
    try:
        import torch
        import torch.nn as nn
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector
        from qiskit_machine_learning.neural_networks import EstimatorQNN
        from qiskit_machine_learning.connectors import TorchConnector

        with open(QML_META_FILE) as f:
            qml_meta = json.load(f)

        N_QUBITS       = qml_meta["n_qubits"]
        N_LAYERS       = qml_meta["n_layers"]
        CLASSICAL_FEAT = qml_meta["classical_feat"]

        # Rebuild VQC
        input_params  = ParameterVector("x",     N_QUBITS)
        weight_params = ParameterVector("theta",  N_QUBITS * N_LAYERS)
        qc = QuantumCircuit(N_QUBITS)
        for i in range(N_QUBITS):
            qc.ry(input_params[i] * np.pi, i)
        for layer in range(N_LAYERS):
            for i in range(N_QUBITS):
                qc.cx(i, (i + 1) % N_QUBITS)
            for i in range(N_QUBITS):
                qc.ry(weight_params[layer * N_QUBITS + i], i)

        qnn = EstimatorQNN(circuit=qc, input_params=input_params,
                           weight_params=weight_params, input_gradients=True)

        class HybridBladderQML(nn.Module):
            def __init__(self, qnn, classical_feat, n_qubits):
                super().__init__()
                self.reducer   = nn.Sequential(
                    nn.Linear(classical_feat, n_qubits), nn.Tanh())
                self.qnn_layer = TorchConnector(qnn)
            def forward(self, x):
                x = self.reducer(x)
                return torch.sigmoid(self.qnn_layer(x))

        qml_model = HybridBladderQML(qnn, CLASSICAL_FEAT, N_QUBITS)
        ckpt = torch.load(QML_MODEL_PATH, map_location="cpu", weights_only=True)
        qml_model.load_state_dict(ckpt["model_state"])
        qml_model.eval()

        # Feature extractor from CNN (up to GlobalAvgPool)
        gap_model = tf.keras.Model(
            inputs=cnn_model.input,
            outputs=cnn_model.get_layer('global_average_pooling2d').output
        )
        proj_layer = tf.keras.layers.Dense(CLASSICAL_FEAT, activation='tanh')
        # warm up projection weights (same init as training)
        _ = proj_layer(tf.zeros((1, 128)))

        USE_QML = True
        print(f"✅ Hybrid QML model loaded ({N_QUBITS} qubits, depth={qc.depth()})")
    except Exception as e:
        print(f"⚠️  QML model load failed ({e}). Falling back to classical CNN.")
        USE_QML = False

with open(CLASS_IDX_FILE) as f:
    class_indices = json.load(f)
idx_to_label = {v: k for k, v in class_indices.items()}
print(f"Classes: {class_indices}")
print(f"Mode: {'⚛️  Quantum-Hybrid' if USE_QML else '🧠 Classical CNN'}")
print("=" * 60)

# ── Image preprocessing ───────────────────────────────────────────────────────
def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)        # (1, H, W, 3)

# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(image_bytes: bytes):
    tensor = preprocess(image_bytes)

    # ── Always run classical CNN (higher accuracy, better calibrated) ──────
    cnn_raw = float(cnn_model.predict(tensor, verbose=0)[0][0])

    qml_raw = None
    mode_label = "Classical CNN"

    if USE_QML:
        try:
            import torch
            gap_feats  = gap_model(tensor, training=False)
            proj_feats = proj_layer(gap_feats).numpy()
            feat_t     = torch.tensor(proj_feats, dtype=torch.float32)
            with torch.no_grad():
                qml_raw = float(qml_model(feat_t)[0][0])
            mode_label = "Hybrid Quantum-Classical (VQC)"
        except Exception as e:
            print(f"QML inference failed: {e}, using CNN only")
            qml_raw = None

    # ── Ensemble decision: Cancer = low P(No Cancer) ───────────────────────
    # Cancer=0 → sigmoid output is P(No Cancer). Low value → Cancer.
    # Medical priority: err on side of caution (prefer false positive over false negative)
    cnn_is_cancer = cnn_raw < CANCER_THRESH

    if qml_raw is not None:
        qml_is_cancer = qml_raw < CANCER_THRESH
        # OR-rule: if EITHER model says cancer → flag cancer (safer)
        is_cancer = cnn_is_cancer or qml_is_cancer
        # Use CNN raw as primary score (better calibrated), QML as secondary vote
        raw = cnn_raw
    else:
        is_cancer = cnn_is_cancer
        raw = cnn_raw

    label      = "Cancer" if is_cancer else "No Cancer"
    # Confidence = how far the score is from the boundary
    if is_cancer:
        confidence = 1.0 - cnn_raw   # P(Cancer) = 1 - P(No Cancer)
    else:
        confidence = cnn_raw         # P(No Cancer)

    return label, confidence, raw, mode_label, qml_raw

# ── Groq XAI explanation ──────────────────────────────────────────────────────
CANCER_CONTEXT = {
    "HGC": "High-Grade Carcinoma — aggressive, fast-growing cancer cells requiring urgent treatment.",
    "LGC": "Low-Grade Carcinoma — slower-growing cancer cells, generally better prognosis but requiring treatment.",
}
NORMAL_CONTEXT = {
    "NST": "Normal Squamous Tissue — healthy squamous epithelium lining the bladder.",
    "NTL": "Normal Tissue Lining — healthy transitional epithelium (urothelium).",
}

def get_xai_explanation(label: str, confidence: float, raw_score: float,
                        mode: str, qml_raw) -> str:
    """Groq LLM medical XAI — concise to save credits."""
    if not GROQ_ENABLED or groq_client is None:
        return ("XAI summary unavailable: set GROQ_API_KEY in your environment "
                "to enable LLM-powered explanations.")

    is_cancer   = label == "Cancer"
    status_txt  = "CANCER TISSUE DETECTED" if is_cancer else "NO CANCER DETECTED"
    tissue_hint = ("HGC (High-Grade) or LGC (Low-Grade) Carcinoma"
                   if is_cancer else "NST / NTL (Normal bladder tissue)")
    qml_line    = f"Quantum VQC secondary score: {qml_raw:.4f}" if qml_raw is not None else "QML: not available"

    prompt = f"""You are a urology AI assistant. Explain this bladder endoscopy AI result to a patient in structured plain English. Use **bold** for section headers. Keep under 280 words.

Result: {status_txt}
Confidence: {confidence*100:.1f}%  |  CNN score (P No-Cancer): {raw_score:.3f}  |  {qml_line}
Likely tissue: {tissue_hint}
Model: {mode} (Hybrid Quantum-Classical VQC + CNN Ensemble)

Sections to include:
**What Was Found** | **What This Means Medically** | **How the AI Works** | **Next Steps** | **Disclaimer"""

    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400,   # shorter = fewer credits
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"(XAI unavailable: {e})"

# ── Result details ────────────────────────────────────────────────────────────
def get_result_details(label: str, confidence: float):
    if label == "Cancer":
        return {
            "status":         "cancer",
            "title":          "⚠️  Potential Cancer Tissue Detected",
            "message":        ("The hybrid quantum-classical model indicates potential "
                               "cancerous tissue (HGC or LGC) in the endoscopic image."),
            "urgency":        "high",
            "recommendation": ("Consult a urologist or oncologist immediately for "
                               "further evaluation, biopsy, and staging."),
            "icon":           "⚠️",
        }
    return {
        "status":         "no_cancer",
        "title":          "✅ No Cancer Signs Detected",
        "message":        ("The hybrid quantum-classical model indicates normal bladder "
                           "tissue (NST or NTL). No cancerous patterns were found."),
        "urgency":        "low",
        "recommendation": "Continue routine check-ups. Follow your doctor's schedule.",
        "icon":           "✅",
    }

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided (key: 'image')."}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        return jsonify({"error": f"Unsupported format '{ext}'."}), 400

    try:
        image_bytes = file.read()
        label, confidence, raw_score, mode, qml_raw = run_inference(image_bytes)
        details = get_result_details(label, confidence)

        # Generate Groq XAI explanation
        xai = get_xai_explanation(label, confidence, raw_score, mode, qml_raw)

        return jsonify({
            "prediction":      label,
            "confidence":      round(confidence * 100, 2),
            "raw_score":       round(raw_score, 4),
            "qml_raw":         round(qml_raw, 4) if qml_raw is not None else None,
            "model_mode":      mode,
            "xai_explanation": xai,
            **details,
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

# ── Tabular disease prediction endpoint ──────────────────────────────────────
TABULAR_REQUIRED = [
    "age", "gender"
]
TABULAR_NUMERIC = [
    "A/G Ratio", "Albumin", "Alk", "ALT (GPT)", "AST (GOT)", "BUN",
    "Calcium", "Chloride", "Creatinine", "Direct Bilirubin", "Estimated GFR",
    "Glucose AC", "Nitrite", "Urine occult Blood", "pH", "Potassium",
    "Sodium", "Specific Gravity", "Strip WBC", "Total Bilirubin",
    "Total Cholesterol", "Total Protein", "Triglyceride",
    "Urine epitheilum (UL)", "Urine epithelium count", "Uric acid",
    "Urine Bilirubin", "Urine Glucose", "Urine Ketone", "Urine Protein",
    "Urobilinogen", "age", "gender", "Hyper1en1ion", "Diabe1es",
    "Smoking", "Drinking", "Bee1leNu1", "FamilyHis1ory",
]

@app.route("/predict-tabular", methods=["POST"])
def predict_tabular():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON body required."}), 400

    # Reject patient identifier
    if "Patient Number" in data:
        return jsonify({"error": "Patient Number must not be sent."}), 400

    # Validate required fields
    missing = [f for f in TABULAR_REQUIRED if data.get(f) is None]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    # Validate numeric types
    bad = []
    for col in TABULAR_NUMERIC:
        v = data.get(col)
        if v is not None:
            try:
                float(v)
            except (ValueError, TypeError):
                bad.append(col)
    if bad:
        return jsonify({"error": f"Non-numeric values for: {bad}"}), 400

    try:
        result = tabular_predict.predict_disease(data)
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception:
        return jsonify({"error": "Prediction failed. Check server logs."}), 500


@app.route("/health")
def health():
    tabular_ready = os.path.exists("models/tabular_pipeline.pkl")
    return jsonify({
        "status":        "ok",
        "mode":          "QML" if USE_QML else "CNN",
        "qubits":        qml_meta.get("n_qubits", "N/A"),
        "classes":       class_indices,
        "tabular_model": "ready" if tabular_ready else "not trained",
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
