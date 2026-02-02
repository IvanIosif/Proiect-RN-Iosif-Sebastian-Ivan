import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
import os
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go

# --- 0. CONFIGURARE CĂI RELATIVE (AUTOMATIZARE) ---

current_dir = os.path.dirname(os.path.abspath(__file__))

# Urcăm două niveluri pentru a ajunge la rădăcina proiectului (RN)
PATH_BASE = os.path.abspath(os.path.join(current_dir, "../../"))

# Definim căile către resurse relativ la rădăcină
PATH_MODELS = os.path.join(PATH_BASE, "models")
PATH_CONFIG = os.path.join(PATH_BASE, "config")
PATH_STORAGE = os.path.join(PATH_BASE, "src", "stocare_date")

# --- 1. ÎNCĂRCARE RESURSE (Cu Căi Dinamice) ---
@st.cache_resource
def load_resources():
    model_path = os.path.join(PATH_MODELS, "optimized_model.keras")
    config_path = os.path.join(PATH_CONFIG, "scaler_optimized.skl")
    
    try:
        if os.path.exists(model_path) and os.path.exists(config_path):
            model = tf.keras.models.load_model(model_path)
            scaler_cfg = joblib.load(config_path)
            return model, scaler_cfg, None
        else:
            return None, None, f"Resursele nu au fost găsite la: {model_path}"
    except Exception as e:
        return None, None, f"Eroare critică la încărcarea resurselor: {str(e)}"

# --- 2. LOGICĂ STOCARE DATE (Cale Relativă) ---
def log_diagnostic_to_csv(user_name, prediction, confidence, raw_values):
    try:
        # Creăm folderul de stocare dacă nu există
        if not os.path.exists(PATH_STORAGE):
            os.makedirs(PATH_STORAGE, exist_ok=True)
        
        file_path = os.path.join(PATH_STORAGE, "istoric_triaj.csv")
        diagnostic = "TBC" if prediction >= 0.5 else "Pneumonie"
        
        new_entry = {
            "Data_Ora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Utilizator": user_name,
            "Diagnostic": diagnostic,
            "Incredere": f"{confidence*100:.2f}%",
            "Scor_Brut": round(float(prediction), 4)
        }
        for i, val in enumerate(raw_values):
            new_entry[f"Q{i+1}"] = val

        df = pd.DataFrame([new_entry])
        if not os.path.exists(file_path):
            df.to_csv(file_path, index=False)
        else:
            df.to_csv(file_path, mode='a', header=False, index=False)
        return True, None
    except Exception as e:
        return False, f"Eroare la scrierea datelor în CSV: {str(e)}"

# --- 3. CONFIGURARE PAGINĂ ---
st.set_page_config(page_title="SIA Diagnostic Expert", page_icon="🔬", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stSelectbox label { font-weight: bold; color: #1e3a8a; }
    .stButton button { background-color: #1e3a8a; color: white; border-radius: 8px; height: 3em; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# Încercăm încărcarea resurselor
model, scaler_cfg, error_msg = load_resources()

if error_msg:
    st.error(f"## 🚩 ERROR_STATE: Resurse Inaccesibile")
    st.warning(f"Sistemul nu poate porni. Detalii: {error_msg}")
    st.stop()

# --- 4. DEFINIRE ÎNTREBĂRI ---
questions = [
    {"id": "Q1", "text": "Cât de ridicată este febra?", "options": ["Normală", "Ușoară", "Moderată", "Ridicată", "Foarte ridicată"]},
    {"id": "Q2", "text": "Cât de mult te afecteaza conditia ta să întreprinzi activitățile zilnice?", "options": ["Foarte puțin", "Puțin", "Moderat", "Mult", "Foarte mult"]},
    {"id": "Q3", "text": "Cât de dificil este pentru tine să respiri?", "options": ["Deloc", "Foarte puțin", "Moderat", "Semnificativ", "Foarte greu"]},
    {"id": "Q4", "text": "Cât de mult durează un episod de tuse?", "options": ["Sub 10 secunde", "10-30 sec", "30 sec-2 min", "2-4 min", ">4 min"]},
    {"id": "Q5", "text": "Cât de frecvent îți vine să tușești?", "options": ["Foarte rar", "Ocazional", "Moderat", "Frecvent", "Foarte frecvent"]},
    {"id": "Q6", "text": "Cât de puternic simți durerea în piept?", "options": ["Deloc", "Ușor", "Moderat", "Intens", "Foarte intens"]},
    {"id": "Q7", "text": "Cât de productivă este tusea ta?", "options": ["Deloc", "Foarte puțin", "Moderată", "Multă", "Foarte multă"]},
    {"id": "Q8", "text": "Cum resimți frisoanele?", "options": ["Deloc", "Ușor", "Moderat", "Puternic", "Foarte puternic"]},
    {"id": "Q9", "text": "Cât de des ai dureri de cap?", "options": ["Niciodată", "Rareori", "Uneori", "Des", "Foarte des"]},
    {"id": "Q10", "text": "Cât de intensă este durerea ta musculară?", "options": ["Deloc", "Ușoară", "Moderată", "Puternică", "Foarte puternică"]},
    {"id": "Q11", "text": "Cât de des transpiri în timpul nopții?", "options": ["Niciodată", "Foarte rar", "Ocazional", "Frecvent", "Permanent"]},
    {"id": "Q12", "text": "Cât de mult te incomodează să respiri întins pe spate?", "options": ["Deloc", "Foarte puțin", "Moderat", "Mult", "Foarte Mult"]},
    {"id": "Q13", "text": "Cât de des ai greață și/sau dureri abdominale?", "options": ["Niciodată", "Rareori", "Ocazional", "Frecvent", "Foarte frecvent"]},
    {"id": "Q14", "text": "Cât de pronunțată este pierderea gustului/mirosului?", "options": ["Deloc", "Foarte ușoară", "Moderată", "Pronunțată", "Foarte pronunțată"]},
    {"id": "Q15", "text": "Câte kg ai pierdut în ultimele 3 luni?", "options": ["Niciun kg", "1–2 kg", "3–5 kg", "6–10 kg", ">10 kg"]},
    {"id": "Q16", "text": "Câte episoade de tuse au fost cu sânge?", "options": ["Niciunul", "Foarte puține", "Puține", "Multe", "Foarte multe"]},
    {"id": "Q17", "text": "Cât de mult efort depui la respirație?", "options": ["Deloc", "Foarte puțin", "Moderat", "Mult", "Foarte mult"]},
    {"id": "Q18", "text": "Cât de des ai avut ganglionii gâtului inflamați?", "options": ["Niciodată", "Foarte rar", "Ocazional", "Frecvent", "Permanent"]},
    {"id": "Q19", "text": "Cât de mult ți s-a redus pofta de mâncare?", "options": ["Deloc", "Foarte puțin", "Moderată", "Foarte mult", "Nu mai mănânc"]},
    {"id": "Q20", "text": "Cât de des ai avut febră intermitentă?", "options": ["Niciodată", "Foarte rar", "Ocazional", "Des", "Foarte Des"]}
]

# --- 5. INTERFAȚĂ UTILIZATOR ---
st.markdown("<h1 style='text-align: center; color: #1e3a8a;'>🔬 Sistem Expert de Diagnostic SIA</h1>", unsafe_allow_html=True)

with st.form("main_form"):
    st.markdown("### 👤 Identificare Utilizator")
    user_name = st.text_input("Nume complet utilizator / cod pacient", placeholder="Ex: Popescu Ion")
    
    st.divider()
    st.markdown("### 📋 Chestionar Simptomatologie")
    
    col1, col2 = st.columns(2, gap="large")
    raw_indices = []
    for i, q in enumerate(questions):
        with (col1 if i < 10 else col2):
            choice = st.selectbox(f"**{i+1}. {q['text']}**", q['options'], key=q['id'])
            raw_indices.append(q['options'].index(choice))
    
    submit = st.form_submit_button("🚀 ANALIZEAZĂ CAZUL", use_container_width=True)

# --- 6. PROCESARE ȘI REZULTATE ---
if submit:
    if not user_name.strip():
        st.error("## ⚠️ ERROR_STATE: Date de intrare incomplete")
        st.info("Sistemul necesită identificarea utilizatorului pentru a salva diagnosticul.")
    else:
        try:
            # Procesare matematică
            input_numeric = np.array(raw_indices).astype(float) / 4.0
            input_numeric_reshaped = input_numeric.reshape(1, -1)
            
            # Predicție Model
            prediction = model.predict(input_numeric_reshaped, verbose=0)[0][0]
            confidence = prediction if prediction >= 0.5 else (1 - prediction)
            
            # Salvare Date
            success_log, log_err = log_diagnostic_to_csv(user_name, prediction, confidence, raw_indices)
            
            # Afișare Rezultate
            st.divider()
            res_c1, res_c2 = st.columns([2, 1])
            
            with res_c1:
                if prediction >= 0.5:
                    st.error(f"## 🚩 DIAGNOSTIC SUGERAT: TUBERCULOZĂ")
                    st.warning(f"**Utilizator:** {user_name}\n\n⚠️ **RECOMANDARE:** Izolare imediată și transfer la spital specializat.")
                else:
                    st.success(f"## 🟦 DIAGNOSTIC SUGERAT: PNEUMONIE")
                    st.info(f"**Utilizator:** {user_name}\n\n✅ **RECOMANDARE:** Tratament local sub supraveghere medicală.")
                
                if success_log:
                    st.caption(f"✅ Datele au fost salvate în siguranță.")
                else:
                    st.error(f"⚠️ LOG_ERROR: Diagnosticul a fost afișat, dar salvarea a eșuat: {log_err}")

            with res_c2:
                st.metric("Nivel de Încredere", f"{confidence*100:.2f}%")
                st.progress(float(confidence))

            # Analiza Vizuală
            weights = model.layers[0].get_weights()[0]
            influence = np.mean(weights, axis=1) * input_numeric
            with st.expander("📊 Vezi analiza impactului simptomelor"):
                fig = go.Figure(go.Bar(x=influence, y=[q['text'] for q in questions], orientation='h', marker_color='#1e3a8a'))
                fig.update_layout(height=600, margin=dict(l=250))
                st.plotly_chart(fig, use_container_width=True)

        except Exception as ex:
            st.error(f"## 🚩 ERROR_STATE: Eroare neașteptată în timpul procesării")
            st.info(f"Detalii tehnice: {str(ex)}")
