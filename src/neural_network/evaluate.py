import pd
import numpy as np
import os
import json
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import f1_score, confusion_matrix

# --- 0. CONFIGURARE CĂI RELATIVE (AUTOMATIZARE) ---
# Detectăm locația scriptului actual (presupunem că e în RN/src/)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Urcăm un nivel pentru a ajunge la rădăcina proiectului (RN)
PATH_BASE = os.path.abspath(os.path.join(current_dir, ".."))

# Definirea directoarelor relativ la rădăcină
PATH_DATA = os.path.join(PATH_BASE, "data", "test")
PATH_MODELS = os.path.join(PATH_BASE, "models")
PATH_SAVE_FINAL = os.path.join(PATH_BASE, "results", "etapa6")
PATH_CONFIG = os.path.join(PATH_BASE, "config")

# Ne asigurăm că directoarele de ieșire există deja
os.makedirs(PATH_SAVE_FINAL, exist_ok=True)
os.makedirs(PATH_CONFIG, exist_ok=True)

def run_evaluation():
    # --- 2. ÎNCĂRCARE DATE TEST ---
    p_path = os.path.join(PATH_DATA, "pneumonie", "pneumonie_test.csv")
    t_path = os.path.join(PATH_DATA, "tuberculoza", "tuberculoza_test.csv")
    
    if not os.path.exists(p_path) or not os.path.exists(t_path):
        print(f"❌ Datele de test nu au fost găsite la: {PATH_DATA}")
        return

    df_p = pd.read_csv(p_path)
    df_t = pd.read_csv(t_path)
    df_test = pd.concat([df_p, df_t], ignore_index=True)
    
    X_test = df_test.drop('Diagnosis', axis=1).values.astype('float32')
    y_test = df_test['Diagnosis'].values.astype('float32')

    # --- 3. ÎNCĂRCARE MODEL (.keras) ---
    model_path = os.path.join(PATH_MODELS, "optimized_model.keras")
    if not os.path.exists(model_path):
        print(f"❌ Modelul optimizat nu a fost găsit la {model_path}")
        return

    model = tf.keras.models.load_model(model_path)

    # --- 4. CALCUL METRICI ---
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    acc = np.mean(y_pred == y_test)
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"\n# Performanță Model Optimizat (Detectată din: {PATH_MODELS})")
    print(f"# Test Accuracy: {acc:.4f}")
    print(f"# Test F1-score (macro): {f1:.4f}")

    # --- 5. SALVARE YAML (Configurația Modelului) ---
    config_data = {
        "model_metadata": {
            "name": "TBC_Pneumo_Balanced_MLP",
            "experiment": "Exp3_Balanced",
            "version": "1.0_Etapa6"
        },
        "architecture": {
            "input_features": 20,
            "hidden_layers": [64, 64, 64],
            "activation": "relu",
            "output_activation": "sigmoid",
            "dropout": 0.2
        },
        "semantic_boosting_rules": {
            "tbc_anchor_boost": 1.55,
            "general_tbc_boost": 1.25,
            "pneu_decay": 0.98,
            "anchor_indices": [14, 15]  # Q15, Q16
        },
        "inference_settings": {
            "threshold": 0.5,
            "scaler_type": "StandardScaler",
            "scaler_file": "scaler_optimized.pkl"
        }
    }

    yaml_path = os.path.join(PATH_CONFIG, "optimized_config.yaml")
    with open(yaml_path, "w") as y_file:
        yaml.dump(config_data, y_file, default_flow_style=False, sort_keys=False)
    print(f"# ✓ Configuration saved to {yaml_path}")

    # --- 6. SALVARE REZULTATE VIZUALE ȘI JSON ---
    metrics = {
        "test_accuracy": round(float(acc), 4),
        "f1_macro": round(float(f1), 4),
        "status": "Evaluare Etapa 6 Completă"
    }
    json_path = os.path.join(PATH_SAVE_FINAL, "final_test_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', 
                xticklabels=['Pneumonie', 'TBC'], 
                yticklabels=['Pneumonie', 'TBC'])
    plt.title('Confusion Matrix - Model Optimizat (Etapa 6)')
    plt.xlabel('Predicție AI')
    plt.ylabel('Realitate (Ground Truth)')
    
    cm_path = os.path.join(PATH_SAVE_FINAL, "final_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    
    print(f"# ✓ Final metrics and plots saved to {PATH_SAVE_FINAL}")

if __name__ == "__main__":
    run_evaluation()
