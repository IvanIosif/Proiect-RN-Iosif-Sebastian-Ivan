import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import os

# --- 1. ÎNCĂRCARE RESURSE ---
@st.cache_resource
def load_resources():
    model_path = r"D:\Facultate\RN\models\untrain_model.keras"
    config_path = r"D:\Facultate\RN\config\scaler.pkl"
    if os.path.exists(model_path) and os.path.exists(config_path):
        model = tf.keras.models.load_model(model_path)
        with open(config_path, 'rb') as f:
            scaler_cfg = pickle.load(f)
        return model, scaler_cfg
    return None, None

# --- 2. CONFIGURARE PAGINĂ ---
st.set_page_config(page_title="SIA Diagnostic", page_icon="🔬", layout="wide")

# CSS pentru Titlu vizibil și Box-uri înguste
st.markdown("""
    <style>
    /* Resetăm padding-ul pentru a vedea titlul sus */
    .block-container {padding-top: 2rem; padding-bottom: 1rem;}
    
    /* Limităm lățimea box-urilor de selecție la 300px */
    div[data-baseweb="select"] {
        max-width: 300px !important;
    }
    
    /* Design compact pentru întrebări */
    label {
        font-size: 0.9rem !important; 
        font-weight: 500 !important;
        margin-bottom: 2px !important;
    }
    
    .stSelectbox {
        margin-bottom: -10px;
    }

    /* Centrarea titlului */
    .title-text {
        text-align: center;
        padding-bottom: 20px;
        color: #2E4053;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 class='title-text'>🔬 Sistem Expert Diagnostic</h1>", unsafe_allow_html=True)

model, scaler_cfg = load_resources()

if not model:
    st.error("⚠️ Resursele (model/scaler) nu au fost găsite în locațiile specificate.")
    st.stop()

# --- 3. LISTA TA DE ÎNTREBĂRI ---
questions = [
    {"id": "Q1", "text": "Cât de ridicată este febra?", "options": ["Normală", "Ușoară", "Moderată", "Ridicată", "Foarte ridicată"]},
    {"id": "Q2", "text": "Cât de afectata iti simți întreprinderea de activități normale?", "options": ["Foarte puțin", "Puțin", "Moderat", "Mult", "Foarte mult"]},
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
    {"id": "Q20", "text": "Cât de des ai avut febră intermitentă?", "options": ["Niciodată", "Rareori", "Ocazional", "Des", "Foarte Des"]}
]

# --- 4. FORMULAR ---
with st.form("diagnostic_form"):
    raw_inputs = []
    c1, c2 = st.columns(2)
    
    for i, q in enumerate(questions):
        with (c1 if i < 10 else c2):
            choice = st.selectbox(q['text'], q['options'], key=q['id'])
            raw_inputs.append(q['options'].index(choice) + 1)
    
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    # Buton centrat și mai îngust
    _, btn_col, _ = st.columns([1.5, 1, 1.5])
    submit = btn_col.form_submit_button("ANALIZEAZĂ", use_container_width=True)

# --- 5. LOGICA DE PREDICȚIE ---
if submit:
    # Normalizare manuală (x-1)/4
    input_norm = (np.array(raw_inputs).astype(float) - 1) / 4.0
    prediction = model.predict(input_norm.reshape(1, -1), verbose=0)[0][0]
    
    st.divider()
    res_c1, res_c2 = st.columns(2)
    
    if prediction >= 0.5:
        res_c1.error("### DIAGNOSTIC: TUBERCULOZĂ")
        siguranta = prediction
    else:
        res_c1.success("### DIAGNOSTIC: PNEUMONIE")
        siguranta = 1 - prediction

    res_c2.metric("Nivel de Încredere", f"{siguranta*100:.2f}%")
    res_c2.progress(float(siguranta))