"""
Hybrid Quantum-Classical CNN — BladderVision QML
================================================
Architecture:
  Classical CNN backbone (frozen)  →  feature vector (8 dims)
  ↓
  Quantum Circuit (PQC / VQC)      →  classification probability
  ↓
  Binary prediction: Cancer / No Cancer

Uses:
  - Qiskit  (quantum circuit simulation via AerSimulator)
  - qiskit-machine-learning  (EstimatorQNN + gradient-based training)
  - TensorFlow (CNN backbone & training loop glue)

Run AFTER train_model.py has produced bladder_model.keras.
Produces: bladder_qml_model.pt  (full hybrid model weights)
"""

import os, json, time, numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

import torch
import torch.nn as nn
import torch.optim as optim

# ── CONFIG ────────────────────────────────────────────────────────────────────
IMG_SIZE       = (128, 128)
BATCH          = 32
SEED           = 42
N_QUBITS       = 4          # 4 qubits
N_LAYERS       = 2          # PQC depth
CLASSICAL_FEAT = 8          # features projected to quantum layer
EPOCHS_QML     = 15
LR             = 5e-4
PREPARED_DIR   = "data_binary"
CNN_MODEL_PATH = "bladder_model.keras"
QML_MODEL_SAVE = "bladder_qml_model.pt"
CLASS_IDX_FILE = "class_indices.json"

print("=" * 60)
print("  BladderVision — Hybrid QML Training")
print("=" * 60)

with open(CLASS_IDX_FILE) as f:
    class_indices = json.load(f)
print(f"Class indices: {class_indices}")

# ── Data Generators ───────────────────────────────────────────────────────────
gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_gen = gen.flow_from_directory(
    PREPARED_DIR, target_size=IMG_SIZE, batch_size=BATCH,
    class_mode='binary', subset='training', seed=SEED, shuffle=True)
val_gen = gen.flow_from_directory(
    PREPARED_DIR, target_size=IMG_SIZE, batch_size=BATCH,
    class_mode='binary', subset='validation', seed=SEED, shuffle=False)
print(f"Train: {train_gen.samples} | Val: {val_gen.samples}")

# ── Feature Extractor ─────────────────────────────────────────────────────────
print("\n==> Loading classical CNN backbone …")
cnn_full = tf.keras.models.load_model(CNN_MODEL_PATH)

# Extract everything up to GlobalAveragePooling (128-dim output)
gap_output_model = tf.keras.Model(
    inputs=cnn_full.input,
    outputs=cnn_full.get_layer('global_average_pooling2d').output
)
gap_output_model.trainable = False

# Projection Dense: 128 -> CLASSICAL_FEAT (with tanh so values in [-1,1])
proj = layers.Dense(CLASSICAL_FEAT, activation='tanh')

def get_features(imgs):
    gap = gap_output_model(imgs, training=False)   # (B, 128)
    return proj(gap)                               # (B, CLASSICAL_FEAT)

print(f"Feature size: {CLASSICAL_FEAT}")

# ── VQC Definition ────────────────────────────────────────────────────────────
print("\n==> Building Variational Quantum Circuit (VQC) …")

input_params  = ParameterVector("x", N_QUBITS)
weight_params = ParameterVector("theta", N_QUBITS * N_LAYERS)

qc = QuantumCircuit(N_QUBITS)
for i in range(N_QUBITS):
    qc.ry(input_params[i] * np.pi, i)          # angle encoding
for layer in range(N_LAYERS):
    for i in range(N_QUBITS):                  # CNOT ring entanglement
        qc.cx(i, (i + 1) % N_QUBITS)
    for i in range(N_QUBITS):                  # parameterized rotations
        qc.ry(weight_params[layer * N_QUBITS + i], i)

print(f"VQC: {N_QUBITS} qubits, depth={qc.depth()}, "
      f"{len(weight_params)} trainable params, {len(input_params)} inputs")
print(qc.draw(output='text', fold=-1))

# ── QNN ───────────────────────────────────────────────────────────────────────
qnn = EstimatorQNN(
    circuit=qc,
    input_params=input_params,
    weight_params=weight_params,
    input_gradients=True,
)
print("EstimatorQNN created ✅")

# ── Hybrid PyTorch Model ──────────────────────────────────────────────────────
class HybridBladderQML(nn.Module):
    def __init__(self, qnn, classical_feat, n_qubits):
        super().__init__()
        self.reducer   = nn.Sequential(nn.Linear(classical_feat, n_qubits), nn.Tanh())
        self.qnn_layer = TorchConnector(qnn)

    def forward(self, x):
        x   = self.reducer(x)        # (B, N_QUBITS)
        out = self.qnn_layer(x)      # (B, 1)
        return torch.sigmoid(out)

model     = HybridBladderQML(qnn, CLASSICAL_FEAT, N_QUBITS)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCELoss()

# ── Extract features from CNN ─────────────────────────────────────────────────
print("\n==> Extracting CNN features …")

def extract(generator, steps):
    Xs, ys = [], []
    for i, (imgs, lbls) in enumerate(generator):
        imgs_tf = tf.cast(imgs, tf.float32)
        feats   = get_features(imgs_tf).numpy()
        Xs.append(feats); ys.append(lbls)
        if i >= steps - 1: break
    return np.vstack(Xs), np.concatenate(ys)

tr_steps = (train_gen.samples + BATCH - 1) // BATCH
va_steps = (val_gen.samples   + BATCH - 1) // BATCH

X_train, y_train = extract(train_gen, tr_steps)
X_val,   y_val   = extract(val_gen,   va_steps)
print(f"  Train features: {X_train.shape}  |  Val: {X_val.shape}")

Xtr = torch.tensor(X_train, dtype=torch.float32)
ytr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
Xva = torch.tensor(X_val,   dtype=torch.float32)
yva = torch.tensor(y_val,   dtype=torch.float32).unsqueeze(1)

tr_dl = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(Xtr, ytr), batch_size=16, shuffle=True)
va_dl = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(Xva, yva), batch_size=16, shuffle=False)

# ── Training Loop ─────────────────────────────────────────────────────────────
print(f"\n==> Training Hybrid QML model for {EPOCHS_QML} epochs …")
best_val_acc, patience_cnt = 0.0, 0
PATIENCE = 5

for epoch in range(1, EPOCHS_QML + 1):
    model.train()
    t0 = time.time()
    tr_loss, tr_correct, tr_total = 0.0, 0, 0

    for Xb, yb in tr_dl:
        optimizer.zero_grad()
        preds = model(Xb)
        loss  = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        tr_loss    += loss.item() * Xb.size(0)
        tr_correct += ((preds > 0.5) == yb.bool()).sum().item()
        tr_total   += Xb.size(0)

    model.eval()
    va_correct, va_total = 0, 0
    with torch.no_grad():
        for Xb, yb in va_dl:
            preds      = model(Xb)
            va_correct += ((preds > 0.5) == yb.bool()).sum().item()
            va_total   += Xb.size(0)

    tr_a = tr_correct / tr_total
    va_a = va_correct / va_total
    print(f"  Epoch {epoch:2d}/{EPOCHS_QML} | "
          f"loss={tr_loss/tr_total:.4f}  train={tr_a:.4f}  val={va_a:.4f}  "
          f"[{time.time()-t0:.1f}s]")

    if va_a > best_val_acc:
        best_val_acc = va_a
        torch.save({
            "model_state": model.state_dict(),
            "class_indices": class_indices,
            "n_qubits": N_QUBITS,
            "n_layers": N_LAYERS,
            "classical_feat": CLASSICAL_FEAT,
        }, QML_MODEL_SAVE)
        print(f"    ✅ Saved (val_acc={va_a:.4f})")
        patience_cnt = 0
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"    ⏹  Early stopping")
            break

print(f"\n✅ QML model → {QML_MODEL_SAVE}  (best val_acc={best_val_acc:.4f})")
meta = {"n_qubits": N_QUBITS, "n_layers": N_LAYERS,
        "classical_feat": CLASSICAL_FEAT, "best_val_acc": best_val_acc,
        "vqc_trainable_params": len(weight_params),
        "architecture": "Hybrid CNN-VQC (EstimatorQNN)"}
with open("qml_meta.json","w") as f: json.dump(meta, f, indent=2)
print("   qml_meta.json saved. → Now run: python app.py")
