import os, joblib
import numpy as np

print("=== FILE CHECK ===")
for f in ['models/heavy_hybrid_cpu_v2',
          'models/scaler_FINAL.pkl',
          'data/X_te.npy',
          'data/y_te.npy']:
    print(f"{'✅' if os.path.exists(f) else '❌'} {f}")

print("\n=== SAVEDMODEL LOAD TEST ===")
import tensorflow as tf
print(f"TF version: {tf.__version__}")

try:
    loaded = tf.saved_model.load('models/heavy_hybrid_cpu_v2')
    infer  = loaded.signatures['serving_default']
    print("✅ SavedModel loaded successfully")

    test = tf.constant(np.random.rand(1, 10, 36).astype('float32'))
    result = infer(inputs=test)
    probs  = list(result.values())[0].numpy()[0]
    print(f"   Test prediction : {probs.round(3)}")
    print(f"   Predicted class : {['DoH','NonDoH','Benign','Malicious'][probs.argmax()]}")

except Exception as e:
    print(f"❌ Load FAILED: {e}")

print("\n=== QUICK ACCURACY TEST (200 balanced samples) ===")
try:
    X_te = np.load('data/X_te.npy')
    y_te = np.load('data/y_te.npy')
    LABEL_NAMES = ["DoH","NonDoH","Benign","Malicious"]

    correct = 0
    total   = 0
    for cls in range(4):
        cls_idx = np.where(y_te == cls)[0][:50]
        for idx in cls_idx:
            seq   = tf.constant(X_te[idx].reshape(1,10,36).astype('float32'))
            result = infer(inputs=seq)
            probs  = list(result.values())[0].numpy()[0]
            if probs.argmax() == cls:
                correct += 1
            total += 1
    print(f"Accuracy: {correct/total*100:.2f}%  (expected ~79%)")

except Exception as e:
    print(f"❌ Accuracy test FAILED: {e}")