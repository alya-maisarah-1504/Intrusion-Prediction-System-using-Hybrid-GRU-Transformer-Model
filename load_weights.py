# ============================================================
# load_weights.py
# Rebuilds Heavy_Hybrid_FINAL architecture from scratch (CPU-safe,
# no CudnnRNNV3 baked in), loads ONLY the weights, verifies accuracy.
#
# Run this on your laptop from: C:\Users\alyam\Downloads\fyp_dashboard\
# ============================================================
 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
 
print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))  # should be [] on your laptop
 
SEQ_LEN = 10
N_FEATURES = 36
N_CLASSES = 4
 
WEIGHTS_PATH = "models/hybrid_weights_cpu.npz"
X_TE_PATH = "data/X_te.npy"
Y_TE_PATH = "data/y_te.npy"
 
 
# ------------------------------------------------------------
# EXACT copy of TransformerBlock from Colab Cell 9
# ------------------------------------------------------------
class TransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim, self.num_heads, self.ff_dim, self.dropout = embed_dim, num_heads, ff_dim, dropout
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation='gelu'), layers.Dense(embed_dim)])
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)
 
    def call(self, x, training=False):
        attn_out = self.drop1(self.att(x, x, training=training), training=training)
        x = self.ln1(x + attn_out)
        ffn_out = self.drop2(self.ffn(x), training=training)
        return self.ln2(x + ffn_out)
 
    def get_config(self):
        config = super().get_config()
        config.update({'embed_dim': self.embed_dim, 'num_heads': self.num_heads,
                        'ff_dim': self.ff_dim, 'dropout': self.dropout})
        return config
 
    @classmethod
    def from_config(cls, config):
        return cls(**config)
 
 
# ------------------------------------------------------------
# EXACT copy of build_heavy_hybrid from Colab Cell 9
# ------------------------------------------------------------
def build_heavy_hybrid(seq_len=SEQ_LEN, n_features=N_FEATURES, n_classes=N_CLASSES):
    inputs = keras.Input(shape=(seq_len, n_features))
    g = layers.Bidirectional(layers.GRU(256, return_sequences=True))(inputs)
    g = layers.BatchNormalization()(g); g = layers.Dropout(0.3)(g)
    g = layers.Bidirectional(layers.GRU(128, return_sequences=True))(g)
    g = layers.BatchNormalization()(g); g = layers.Dropout(0.3)(g)
    g = layers.Bidirectional(layers.GRU(64, return_sequences=True))(g)
    g = layers.BatchNormalization()(g); g = layers.Dropout(0.2)(g)
    g_attn = layers.MultiHeadAttention(num_heads=4, key_dim=32, name='gru_self_attn')(g, g)
    g_attn = layers.Dropout(0.2)(g_attn)
    g = layers.LayerNormalization(epsilon=1e-6)(g + g_attn)
    g_pool = layers.GlobalAveragePooling1D()(g)
    g_pool = layers.Dense(128, activation='relu', name='gru_proj')(g_pool)
 
    t = layers.Dense(128, activation='relu')(inputs)
    t = layers.LayerNormalization(epsilon=1e-6)(t)
    for i in range(4):
        t = TransformerBlock(embed_dim=128, num_heads=8, ff_dim=512, dropout=0.1, name=f'hybrid_trans_{i}')(t)
    t_pool = layers.GlobalAveragePooling1D()(t)
    t_pool = layers.Dense(128, activation='relu', name='trans_proj')(t_pool)
 
    g_exp = layers.Reshape((1, 128))(g_pool)
    t_exp = layers.Reshape((1, 128))(t_pool)
    gt_cross = layers.MultiHeadAttention(num_heads=4, key_dim=32, name='gt_cross')(g_exp, t_exp)
    gt_cross = layers.Reshape((128,))(gt_cross)
    tg_cross = layers.MultiHeadAttention(num_heads=4, key_dim=32, name='tg_cross')(t_exp, g_exp)
    tg_cross = layers.Reshape((128,))(tg_cross)
 
    fused = layers.Concatenate()([g_pool, t_pool, gt_cross, tg_cross])
    x = layers.Dense(512, activation='relu')(fused); x = layers.BatchNormalization()(x); x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x); x = layers.BatchNormalization()(x); x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x); x = layers.BatchNormalization()(x); x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x); x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    return keras.Model(inputs, outputs, name='Heavy_Hybrid_FINAL')
 
 
# ------------------------------------------------------------
# Build, load weights, verify
# ------------------------------------------------------------
print("\nBuilding fresh CPU-native architecture...")
model = build_heavy_hybrid()
model.summary()
 
print(f"\nLoading weights from {WEIGHTS_PATH} ...")
loaded_npz = np.load(WEIGHTS_PATH)
weight_arrays = [loaded_npz[f'arr_{i}'] for i in range(len(loaded_npz.files))]
print(f"Loaded {len(weight_arrays)} weight arrays from npz")
 
current_weights = model.get_weights()
print(f"Model expects {len(current_weights)} weight arrays")
 
if len(weight_arrays) != len(current_weights):
    raise ValueError(
        f"MISMATCH: npz has {len(weight_arrays)} arrays but model expects "
        f"{len(current_weights)}. Architecture doesn't match — stop and tell Claude."
    )
 
# Check shapes line up before setting, so we get a clear error instead of a silent mismatch
for i, (a, b) in enumerate(zip(weight_arrays, current_weights)):
    if a.shape != b.shape:
        raise ValueError(
            f"Shape mismatch at array {i}: npz has {a.shape}, model expects {b.shape}. "
            f"Stop and tell Claude this index and both shapes."
        )
 
model.set_weights(weight_arrays)
print("Weights loaded successfully via set_weights() — no CudnnRNNV3 error means this worked.")
 
print("\nCompiling (needed for evaluate())...")
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
 
print(f"\nLoading test set from {X_TE_PATH} / {Y_TE_PATH} ...")
X_te = np.load(X_TE_PATH)
y_te = np.load(Y_TE_PATH)
print("X_te shape:", X_te.shape, "| y_te shape:", y_te.shape)
 
print("\nRunning evaluation on full test set (this may take a minute on CPU)...")
loss, acc = model.evaluate(X_te, y_te, batch_size=256, verbose=1)
print(f"\n{'='*50}")
print(f"RESULT: Test accuracy = {acc*100:.2f}%")
print(f"Expected (from Colab, verified): 79.14%")
print(f"{'='*50}")
 
if abs(acc - 0.7914) < 0.02:
    print("\n✅ MATCH — weights loaded correctly. Safe to use this approach in app.py.")
else:
    print("\n⚠️ Accuracy doesn't match expected 79.14%. Do NOT use this in app.py yet — "
          "tell Claude the actual number you got so we can debug (likely a weight-shape "
          "mismatch, e.g. wrong GRU units, wrong num_heads, or wrong layer order).")
 
# ------------------------------------------------------------
# Save weights again in the newer TF 2.15+ preferred format, so
# app.py can load them cleanly without any ambiguity about format.
# ------------------------------------------------------------
model.save_weights("models/hybrid_weights_cpu_verified.weights.h5")
print("\nSaved verified weights to models/hybrid_weights_cpu_verified.weights.h5")
print("(This local re-save is safe to reload with model.load_weights() on THIS laptop,")
print(" since it's now Keras 2 saving and loading on the same machine — just don't")
print(" re-export this particular file from a different Keras version again.)")
 