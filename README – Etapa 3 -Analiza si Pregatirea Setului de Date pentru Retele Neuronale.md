# 📘 README – Etapa 3: Analiza și Pregătirea Setului de Date pentru Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Ivan Iosif Sebastian    
**Data:** 15 Ianuarie 2026 

---

## Introducere

Acest document descrie activitățile realizate în Etapa 3, concentrate pe generarea,
curățarea și normalizarea setului de date pentru diagnosticul diferențial între Pneumonie și Tuberculoză.
Specificul acestei etape a fost crearea unui set de date care să simuleze ambiguitatea medicală reală,
asigurând în același timp o structură corectă pentru antrenarea rețelei neuronale.
---

##  1. Structura Repository-ului Github (versiunea Etapei 3)

```
project-name/
├── README.md
├── docs/
│   └── datasets/          # descriere seturi de date, surse, diagrame
├── data/
│   ├── raw/               # date brute
│   ├── processed/         # date curățate și transformate
│   ├── train/             # set de instruire
│   ├── validation/        # set de validare
│   └── test/              # set de testare
├── src/
│   ├── preprocessing/     # funcții pentru preprocesare
│   ├── data_acquisition/  # generare / achiziție date (dacă există)
│   └── neural_network/    # implementarea RN (în etapa următoare)
├── config/                # fișiere de configurare
└── requirements.txt       # dependențe Python (dacă aplicabil)
```

---

##  2. Descrierea Setului de Date

### 2.1 Sursa datelor

Origine: Generare programatică bazată pe un profil clinic predefinit.

Modul de achiziție: Generare programatică prin script Python (numpy.random).

Caracteristică specială: Datele au fost generate pentru a fi neconcludente/haotice în zonele de suprapunere a simptomelor,
pentru a testa capacitatea de generalizare a RN.

### 2.2 Caracteristicile dataset-ului

Număr total de observații: 8,000 (4,000 Pneumonie / 4,000 Tuberculoză).

Număr de caracteristici (features): 20 de întrebări clinice.

Tipuri de date: Numerice (Scara Likert 1–5).

Format fișiere: CSV.

### 2.3 Descrierea fiecărei caracteristici
Fiecare observație conține 20 de input-uri (Q1-Q20) evaluate de la 1 la 5.
Grup Simptome                                    Întrebări Cheie                         Corelație Clasă
Neutre                                      (N)Opțiunea 1 la toate                    Nicio boală / Sănătos
Pneumonie                                 (P)Q1, Q6, Q7, Q10, Q13, Q14, Q18     Scrutin mare (4-5) indică Pneumonie
Tuberculoză (T)Q2, Q3, Q4, Q5, Q8, Q9, Q11, Q12, Q15, Q16, Q17, Q19, Q20       Scrutin mare (4-5) indică Tuberculoză
**Fișier recomandat:**  `data/README.md`

---

##  3. Analiza Exploratorie a Datelor (EDA) – Sintetic

### 3.1 Statistici descriptive aplicate

Domeniu valori: Toate intrările sunt constrânse în intervalul [1, 5].

Distribuție: S-a utilizat o distribuție randomizată cu bias controlat pentru a simula pacienți cu simptome atipice.

### 3.2 Analiza calității datelor
Lipsa valorilor nule: Dataset-ul este complet (0% missing values).

Zgomot intenționat: S-a introdus zgomot în datele RAW prin atribuirea de valori medii (2-3) simptomelor care nu aparțin clasei respective, făcând separarea liniară imposibilă.

### 3.3 Probleme identificate

Suprapunerea caracteristicilor (Feature Overlap): Multe simptome (caracteristici) au distribuții statistice similare pentru ambele clase (Pneumonie și Tuberculoză).
De exemplu, valorile medii (3) apar frecvent în ambele patologii, creând ambiguitate pentru model.
---

##  4. Preprocesarea Datelor

### 4.1 Curățarea datelor

Min-Max Scaling: Datele brute (1–5) au fost transformate în valori reale în intervalul [0, 1].
    1 -> 0.0 | 3 -> 0.5 | 5 -> 1.0
Salvare Scaler: Parametrii normalizării au fost salvați în config/scaler.pkl pentru a asigura consistența datelor introduse de utilizator în interfața Streamlit.

### 4.2 Transformarea caracteristicilor

În cadrul acestei etape, datele brute au fost supuse unui proces de transformare matematică pentru a asigura compatibilitatea 
cu arhitectura rețelei neuronale implementate.Normalizare Min-Max: Toate cele 20 de caracteristici (simptomele) 
au fost scalate din intervalul original $[1, 5]$ în intervalul unitar $[0, 1]$.
Importanță pentru Sigmoid: Deoarece clasa NeuralNetworkAbsoluteZero utilizează funcția de activare Sigmoid, este critic ca valorile de intrare să fie mici.
Dacă am introduce valorile brute (până la 5), neuronii s-ar "satura" rapid, ducând la derivate foarte mici și la blocarea procesului de învățare (gradient vanishing).
Formula aplicată: x_new ={x - 1}\{5 - 1}
Codificarea Etichetelor (Label Encoding): Variabila țintă "Diagnosis" a fost mapată binar:
0 pentru Pneumonie.1 pentru Tuberculoză.Acest lucru corespunde pragului de decizie (pred = 1 if o >= 0.5 else 0) definit în funcția main().

### 4.3 Structurarea seturilor de date

**Împărțire recomandată:**
* 70–80% – train
* 10–15% – validation
* 10–15% – test

**Principii respectate:**
* Stratificare pentru clasificare
* Fără scurgere de informație (data leakage)
* Statistici calculate DOAR pe train și aplicate pe celelalte seturi

### 4.4 Salvarea rezultatelor preprocesării

* Date preprocesate în `data/processed/`
* Seturi train/val/test în foldere dedicate
* Parametrii de preprocesare în `config/preprocessing_config.*` (opțional)

---

##  5. Fișiere Generate în Această Etapă

* `data/raw/` – date brute
* `data/processed/` – date curățate & transformate
* `data/train/`, `data/validation/`, `data/test/` – seturi finale
* `src/preprocessing/` – codul de preprocesare
* `data/README.md` – descrierea dataset-ului

---

##  6. Stare Etapă (de completat de student)

- [X] Dataset analizat (EDA realizată)
- [X] Date preprocesate
- [X] Seturi train/val/test generate
- [X] Documentație actualizată în README + `data/README.md`

---
