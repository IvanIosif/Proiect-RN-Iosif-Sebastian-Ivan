# Sistem cu inteligență artificială de Diagnosticare si Triaj Medical

**Student:** Ivan Iosif-Sebastian  
**Grupa:** 633AB
**Facultatea:** Ingineria Industrială și Robotică (FIIR) - UPB  
**Disciplina:** Rețele Neuronale

---

## 📌 Descrierea Proiectului (Overview)

Acest proiect propune o soluție software avansată de asistență medicală (SIA) destinată triajului rapid între două patologii respiratorii severe: Tuberculoza (TBC) și Pneumonia.

Spre deosebire de triajul clasic bazat pe observație umană, care poate fi subiectiv, acest sistem utilizează o Rețea Neuronală Artificială (Deep Neural Network) antrenată pe un set complex de date
simptomatice pentru a identifica pattern-uri și a oferi un diagnostic preliminar de mare precizie.

### 🎯 Obiectiv Principal: Suport medical in triaj medical
Scopul central este reducerea timpului de diagnostic și eliminarea erorilor de clasificare, prin strategii de optimizare a rețelei neuronale:

Semantic Boosting: Ponderarea manuală a simptomelor critice (ex. scăderea în greutate etc.) pentru ca modelul sa inteleaga gravitatea ancorelor medicale.

Prevenirea Fals-Negativelor: Ajustarea ponderilor claselor (Class Weights) pentru a prioritiza detectarea TBC, minimizând riscul de a rata un pacient critic.

Stabilitate prin Optimizare: Utilizarea tehnicilor de Batch Normalization și Dropout pentru a asigura un diagnostic stabil indiferent de zgomotul din răspunsurile subiective ale pacienților.
---

## ⚙️ Arhitectura Sistemului
`
Sistemul este modularizat în 3 componente interconectate:

1.  **Modulul de Procesare Date & Normalizare (`src/data_acquisition`):**
    * Maparea răspunsurilor subiective în intervalul unitar [0, 1] pentru eliminarea diferențelor de scară.
    * Gestionarea datelor sintetice și echilibrarea seturilor de antrenament.
    * [Detalii complete aici](./src/data_acquisition/README.md)

2.  **Modulul de Inteligență Artificială (`src/neural_network`):**
    * **Tehnologie:** TensorFlow / Keras.
    * **Arhitectură:** MLP (Multi-Layer Perceptron) cu 3 straturi ascunse, activări ReLU și strat de ieșire Sigmoid.
    * **Loss Function: Binary Crossentropy (pentru clasificare binară de înaltă precizie).
    * [Detalii complete aici](./src/neural_network/README.md)

3.  **Interfața Grafică - Virtual Cockpit (`src/app`):**
    * Chestionar ce cuprinde 20 de indicatori clinici.
    * Afișează diagnosticul și probabilitatea .
    * [Detalii complete aici](./src/app/README.md)

---

## 📂 Structura și Progresul Proiectului

Proiectul a fost dezvoltat incremental, fiecare etapă fiind documentată separat:

| Etapa | Descriere | Documentație                                                                                                                                                                          |
| :--- | :--- |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Etapa 3** | Analiza datelor, generarea fizică și preprocesarea. | [Vezi README Etapa 3]   "./docs/Etapa 3 -Analiza si Pregatirea Setului de Date pentru Retele Neuronale.md"                                                                            |
| **Etapa 4** | Definirea arhitecturii software și a Diagramelor de Stare. | [Vezi README Etapa 4] "./docs/README_Etapa4_Arhitectura_SIA functionala.md"                                                                                                           |
| **Etapa 5** | Antrenarea modelului Keras, optimizare și validare finală. | [Vezi README Etapa 5](./docs/README_Etapa5_Antrenare_RN - Accuracy-0.8156, , F1=0.815.md")                                                                                            |
| **Etapa 6** | Analiza performanței, optimizare finală și concluzii. | [Vezi README Etapa 6](./docs/README_ Etapa 6 completă – Accuracy=82.73%, F1=.82.53% (optimizat).md")                                                                                  |

---

## 🚀 Cum se rulează proiectul (Quick Start)

### 1. Cerințe de sistem
* Python 3.8+
* Dependențe: Vezi `requirements.txt`

### 2. Instalare
```bash
# Clonare repository
git clone https://github.com/IvanIosif/Proiect-RN

# Instalare librării
pip install -r requirements.txt

# Lansare interfață Streamlit (Dashboard)
streamlit run src/app/main.py
```
