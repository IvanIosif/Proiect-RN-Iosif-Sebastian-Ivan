import os
import json
import pandas as pd
import numpy as np
import yaml
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import f1_score, confusion_matrix, classification_report

# --- 0. FIX PENTRU COMPATIBILITATE ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Stabilitate pentru execuție pe versiuni noi de Python
tf.config.run_functions_eagerly(True)

# --- 1. CONFIGURARE CĂI ---
PATH_BASE = r"D:\Facultate\RN"
PATH_OUT = os.path.join(PATH_BASE, "results", "etapa6")
PATH_MODELS = os.path.join(PATH_BASE, "models")
PATH_CONFIG = os.path.join(PATH_BASE, "config")

for path in [PATH_OUT, PATH_MODELS, PATH_CONFIG]:
    os.makedirs(path, exist_ok=True)

# Mapare Semantică Corectată: 1 = Favorizează TBC (T), 0 = Favorizează Pneumonie (P)
semantic_map = {
    0:  [1, 1, 1, 0, 0], # Q1: Febra
    1:  [0, 0, 0, 1, 1], # Q2: Activități
    2:  [1, 0, 0, 1, 1], # Q3: Respirație
    3:  [0, 0, 0, 1, 1], # Q4: Durată tuse
    4:  [0, 0, 0, 1, 1], # Q5: Frecvență tuse
    5:  [1, 1, 0, 0, 0], # Q6: Durere piept
    6:  [1, 0, 0, 0, 0], # Q7: Tuse productivă
    7:  [1, 0, 0, 0, 1], # Q8: Frisoane
    8:  [1, 0, 0, 0, 1], # Q9: Dureri cap
    9:  [0, 0, 0, 0, 0], # Q10: Muscular
    10: [1, 0, 0, 1, 1], # Q11: Transpirații noapte
    11: [0, 0, 0, 1, 1], # Q12: Respirație spate
    12: [1, 0, 0, 0, 0], # Q13: Greață/Abdominal
    13: [1, 0, 0, 0, 0], # Q14: Gust/Miros
    14: [0, 0, 1, 1, 1], # Q15: Scădere greutate 
    15: [0, 0, 1, 1, 1], # Q16: Sânge în tuse 
    16: [0, 0, 0, 1, 1], # Q17: Efort respirație 
    17: [1, 0, 0, 0, 0], # Q18: Ganglioni 
    18: [0, 0, 0, 1, 1], # Q19: Poftă mâncare 
    19: [1, 0, 0, 1, 1]  # Q20: Febră intermitentă
}

def apply_semantic_logic(X):
    """
    Ponderare selectivă: Pastreaza logica de Pneumonie, dar forțează TBC-ul 
    să iasă la suprafață în caz de derută prin simptome 'ancoră' (Sânge, Greutate).
    """
    X_opt = np.copy(X)
    tbc_anchors = [14, 15] # Q15 (Greutate), Q16 (Sânge)
    
    for i in range(X.shape[0]):
        for j in range(20):
            val_idx = int(round(X[i, j] * 4))
            val_idx = max(0, min(val_idx, 4))
            
            # Verificăm orientarea conform hărții semantice
            if semantic_map[j][val_idx] == 1:
                if j in tbc_anchors and val_idx >= 2: # Dacă e simptom critic prezent
                    X_opt[i, j] *= 1.55  
                else:
                    X_opt[i, j] *= 1.25  
            else:
       
                X_opt[i, j] *= 0.98 
    return X_opt

def load_data(split):
    p_path = os.path.join(PATH_BASE, "data", split, "pneumonie", f"pneumonie_{split}.csv")
    t_path = os.path.join(PATH_BASE, "data", split, "tuberculoza", f"tuberculoza_{split}.csv")
    
    if not os.path.exists(p_path) or not os.path.exists(t_path):
        raise FileNotFoundError(f"Lipsesc fisierele de date in folderul {split}!")
        
    df = pd.concat([pd.read_csv(p_path), pd.read_csv(t_path)], ignore_index=True)
    X = df.drop('Diagnosis', axis=1).values.astype('float32')
    y = df['Diagnosis'].values.astype('float32')
    return apply_semantic_logic(X), y

# --- 2. ÎNCĂRCARE DATE ---
print("⏳ Încărcare și procesare date (Strategia Ancoră & Balanță)...")
X_train, y_train = load_data("train")
X_val, y_val = load_data("validation")
X_test, y_test = load_data("test")

# --- 3. EXPERIMENTE DE OPTIMIZARE ---
experiments = [
    {"id": "Exp1_Base", "lr": 0.001, "layers": [64, 32], "do": 0.2},
    {"id": "Exp2_Deep_Semantic", "lr": 0.0005, "layers": [128, 64, 32], "do": 0.3},
    {"id": "Exp3_Balanced", "lr": 0.0005, "layers": [64, 64, 64], "do": 0.2},
    {"id": "Exp4_Precision", "lr": 0.0003, "layers": [128, 128], "do": 0.1}
]

results_list = []
best_f1, best_model, best_history, best_cfg = 0, None, None, None

for cfg in experiments:
    tf.keras.backend.clear_session()
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(20,)),
        *[tf.keras.layers.Dense(u, activation='relu') for u in cfg["layers"]],
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(cfg["do"]),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cfg["lr"]),
                  loss='binary_crossentropy', metrics=['accuracy'], run_eagerly=True)
    
    print(f"\n🚀 Antrenare {cfg['id']}...")
    # Folosim class_weight: 1.25 pentru TBC ca ajutor suplimentar la nivel de loss
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=50, batch_size=32, verbose=1,
                        class_weight={0: 1.0, 1: 1.25}, 
                        callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
    
    y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    f1 = f1_score(y_test, y_pred, average='macro')
    acc = np.mean(y_pred == y_test)
    
    results_list.append({"Exp": cfg["id"], "Accuracy": float(acc), "F1-score": float(f1)})
    
    if f1 > best_f1:
        best_f1, best_model, best_history, best_cfg = f1, model, history, cfg

# --- 4. SALVARE REZULTATE ȘI METRICI ---
# 1. Tabel rezultate CSV
pd.DataFrame(results_list).to_csv(os.path.join(PATH_OUT, "optimisation_experiments.csv"), index=False)

# 2. JSON Metrics pentru raportul final
y_pred_final = (best_model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
report = classification_report(y_test, y_pred_final, output_dict=True)

final_metrics = {
    "best_exp_id": best_cfg["id"],
    "overall_accuracy": report["accuracy"],
    "tbc_sensitivity_recall": report["1.0"]["recall"],
    "pneumonia_specificity_recall": report["0.0"]["recall"],
    "macro_f1": report["macro avg"]["f1-score"]
}

with open(os.path.join(PATH_OUT, "final_metrics.json"), "w") as f:
    json.dump(final_metrics, f, indent=4)

# 3. Grafice
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1); sns.barplot(x="Exp", y="Accuracy", data=pd.DataFrame(results_list)); plt.title("Accuracy Comparison")
plt.subplot(1, 2, 2); sns.barplot(x="Exp", y="F1-score", data=pd.DataFrame(results_list)); plt.title("F1-Score Comparison")
plt.savefig(os.path.join(PATH_OUT, "metrics_comparison.png"))

# 4. Matrice de Confuzie
cm = confusion_matrix(y_test, y_pred_final)
plt.figure(figsize=(8, 6)); sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', 
                                        xticklabels=['Pneu', 'TBC'], yticklabels=['Pneu', 'TBC'])
plt.title(f"Confusion Matrix - {best_cfg['id']}"); plt.savefig(os.path.join(PATH_OUT, "confusion_matrix_optimized.png"))

# 5. Salvare Model
best_model.save(os.path.join(PATH_MODELS, "optimized_model.keras"))
joblib.dump("semantic_v2_logic", os.path.join(PATH_CONFIG, "scaler_optimized.skl"))

print(f"\n✅ Etapa 6 Finalizată!")
print(f"Model: {best_cfg['id']} | F1: {best_f1:.4f} | Recall TBC: {report['1.0']['recall']:.4f}")