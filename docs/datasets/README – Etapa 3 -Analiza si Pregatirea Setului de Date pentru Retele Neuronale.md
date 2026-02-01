# 📘 README – Etapa 3: Analiza și Pregătirea Setului de Date pentru Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Ivan Iosif Sebastian  
https://github.com/IvanIosif/Proiect-RN-Iosif-Sebastian-Ivan
**Data:** 15 Ianuarie 2026 

---

## Introducere

Acest document descrie activitățile realizate în Etapa 3, concentrate pe generarea, 
curățarea și normalizarea setului de date pentru diagnosticul diferențial între Pneumonie și Tuberculoză.
Specificul acestei etape a fost extinderea volumului de date la 30.000 de instanțe pentru a asigura o robustețe ridicată a antrenării.
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

Origine: Generare programatică bazată pe un profil clinic predefinit. Din cauza restricțiilor de confidențialitate (GDPR), s-a optat pentru simularea algoritmică a datelor.

Mecanismul de profilare: Generarea nu a fost pur aleatorie, ci bazată pe matrice de probabilitate. Pentru fiecare boală, s-au definit simptome "ancoră" (ex: scădere în greutate pentru Tuberculoză) care au o probabilitate statistică mai mare (~75%) de a primi valori de 4 sau 5 pe scara Likert.

Simularea Ambiguității: S-a introdus zgomot statistic controlat pentru a simula pacienți atipici, forțând modelul să identifice pattern-uri complexe, nu doar reguli simple.

### 2.2 Caracteristicile dataset-ului

Volum Total: 30,000 de observații (15,000 Pneumonie / 15,000 Tuberculoză).

Echilibrarea claselor: Dataset perfect echilibrat (50/50), eliminând riscul de bias în favoarea unei patologii.

Număr de caracteristici (features): 20 de variabile clinice (Q1-Q20).

Tipuri de date: Numerice (Scara Likert 1–5).

Format: CSV.

### 2.3 Descrierea fiecărei caracteristici
ID                 Întrebare                                                                            Opțiuni (Scara 1-5)
Q1,Cât de ridicată este febra?,                                                  "Normală(T)", "Ușoară (T)", "Moderată (T)", "Ridicată (P)", "Foarte ridicată (P)
Q2,Cât de mult te afecteaza conditia ta să întreprinzi activitățile zilnice?,    "Foarte puțin(P)", "Puțin (P)", "Moderat (P)", "Mult (T)", "Foarte mult (T)"
Q3,Cât de dificil este pentru tine să respiri?,                                  "Deloc(T)", "Foarte puțin (P)", "Moderat (P)", "Semnificativ (T)", "Foarte greu (T)"
Q4,Cât de mult durează un episod de tuse?,                                       "Sub 10 secunde(P)", "10-30 sec(P)", "30 sec-2 min(P)", "2-4 min (T)", ">4 min (T)"
Q5,Cât de frecvent îți vine să tușești?,                                         "Foarte rar(P)", "Ocazional (P)", "Moderat (P)", "Frecvent (T)", "Foarte frecvent (T)"
Q6,Cât de puternic simți durerea în piept?,                                      "Deloc(T)", "Ușor (T)", "Moderat (P)", "Intens (P)", "Foarte intens (P)"
Q7,"Cât de productivă este tusea ta?,                                            "Deloc(T)", "Foarte puțin (P)", "Moderată (P)", "Multă (P)", "Foarte multă (P)"
Q8,Cum resimți frisoanele?,                                                      "Deloc(T)", "Ușor (P)", "Moderat (P)", "Puternic (P)", "Foarte puternic (T)"
Q9,Cât de des ai dureri de cap?,                                                 "Niciodată(T)", "Rareori (P)", "Uneori (P)", "Des (P)", "Foarte des (T)"
Q10,Cât de intensă este durerea ta musculară?,                                   "Deloc(P)", "Ușoară (P)", "Moderată (P)", "Puternică (P)", "Foarte puternică (P)"
Q11,Cât de des transpiri în timpul nopții?,                                      "Niciodată(T)", "Foarte rar (P)", "Ocazional (P)", "Frecvent (T)", "Permanent (T)"
Q12,Cât de mult te incomodează să respiri întins pe spate?,                      "Deloc(P)", "Foarte puțin (P)", "Moderat (P)", "Mult (T)", "Foarte Mult (T)"
Q13,Cât de des ai greață și/sau dureri abdominale?,                              "Niciodată(T)", "Rareori (P)", "Ocazional (P)", "Frecvent (P)", "Foarte frecvent (P)"
Q14,Cât de pronunțată este pierderea gustului/mirosului?,                        "Deloc(T)", "Foarte ușoară (P)", "Moderată (P)", "Pronunțată (P)", "Foarte pronunțată (P)"
Q15,Câte kg ai pierdut în ultimele 3 luni?,                                      "Niciun kg(P)", "1–2 kg (P)", "3–5 kg (T)", "6–10 kg (T)", ">10 kg (T)"
Q16,Câte episoade de tuse au fost cu sânge?,                                     "Niciunul(P)", "Foarte puține (P)", "Puține (T)", "Multe (T)", "Foarte multe (T)"
Q17,Cât de mult efort depui la respirație?,                                      "Deloc(P)", "Foarte puțin (P)", "Moderat (P)", "Mult (T)", "Foarte mult (T)"
Q18,Cât de des ai avut ganglionii gâtului inflamaț                               "Niciodată(T)", "Foarte rar (P)", "Ocazional (P)", "Frecvent (P)", "Permanent (P)"
Q19,Cât de mult ți s-a redus pofta de mâncare?                                   "Deloc(P)", "Foarte puțin (P)", "Moderat(P)", "Foarte mult (T)", "Nu mai mănânc (T)"
Q20,Cât de des ai avut febră intermitentă?,                                      "Niciodată", "Foarte rar (P)", "Ocazional (P)", "Des (T)", "Foarte Des (T)"
**Fișier recomandat:**  `data/README.md`

---

##  3. Analiza Exploratorie a Datelor (EDA) – Sintetic

### 3.1 Statistici descriptive aplicate

Domeniu: Toate intrările sunt constrânse în intervalul [1, 5].
Distribuție: S-a utilizat un bias controlat pentru a asigura că modelul primește suficiente exemple din ambele patologii.

### 3.2 Analiza calității datelor
Lipsa valorilor nule: Dataset-ul este complet (0% missing values).

Feature Overlap: Multe caracteristici au distribuții similare (ex: febra sau tusea pot fi moderate în ambele cazuri), 
creând ambiguitate și făcând separarea liniară imposibilă.

### 3.3 Probleme identificate

În urma analizei statistice a celor 30,000 de instanțe, au fost identificate următoarele provocări tehnice care vor influența procesul de antrenare:

      A. Suprapunerea caracteristicilor (Feature Overlap)Cea mai mare provocare o reprezintă simptomele comune.
De exemplu, Febra (Q1) și Tusea (Q5) apar în ambele patologii. Efect: Dacă un pacient are valori de "3" (Moderat) la majoritatea simptomelor, modelul intră într-o zonă de incertitudine clinică. 
Această suprapunere face imposibilă atingerea unei acuratețe de 100% cu un model liniar simplu.
      B. Separabilitatea Non-LiniarăDin cauza "zgomotului" introdus programatic (pacienți cu Pneumonie care prezintă simptome izolate de Tuberculoză),
clasele nu pot fi separate printr-o simplă linie dreaptă în spațiul n-dimensional al caracteristicilor.
Soluție necesară: Această problemă justifică utilizarea straturilor ascunse (Hidden Layers) în rețeaua neuronală,
care vor crea "suprafețe de decizie" complexe pentru a izola corect cele două boli.
      C. Riscul de saturare a funcției SigmoidValorile brute de 1–5 sunt problematice pentru funcția de activare Sigmoid.
Problema: Valorile mari (precum 4 sau 5) ar împinge neuronii în zona de platou a funcției Sigmoid, unde derivata este aproape zero.
Consecință: Procesul de învățare s-ar bloca (Gradient Vanishing). 
Această problemă a fost rezolvată prin Min-Max Scaling, aducând toate intrările în intervalul $[0, 1]$.
      D. Ambiguitatea "Cazurilor Neutre"Cazurile în care toate răspunsurile sunt "1" (Normal) reprezintă un punct critic.
Deși teoretic indică un subiect sănătos, în acest sistem de clasificare binară, modelul trebuie forțat să ia o decizie (0 sau 1).
Observație: Aceste instanțe pot introduce erori în statistica finală dacă nu sunt tratate ca o clasă separată sau eliminate din setul de antrenare pentru diagnostic diferențial.
---

##  4. Preprocesarea Datelor

### 4.1 Curățarea datelor

Deși setul de date este generat programatic, ceea ce elimină erorile umane tipice (cum ar fi typos sau date lipsă), a fost efectuat un proces de audit al datelor pentru a asigura stabilitatea procesului de antrenare:
Verificarea Constrângerilor de Domeniu: S-a validat faptul că toate cele 30,000 de intrări se încadrează strict în intervalul discret [1, 5].

Orice valoare în afara acestui interval ar fi invalidat modelul probabilistic clinic.

Auditarea Valorilor Nule: S-a confirmat absența valorilor de tip NaN (Not a Number) sau Null.
Într-o rețea neuronală, o singură valoare nulă poate propaga erori în întregul proces de backpropagation, ducând la eșecul antrenării.

Analiza Redundanței (De-duplication): S-a verificat dacă există rânduri identice generate accidental. 
Deși în 30,000 de instanțe bazate pe 20 de variabile cu 5 opțiuni pot apărea combinații identice, s-a asigurat că acestea nu sunt rezultatul unei erori de loop în scriptul de generare.

Pregătirea Tipului de Date (Type Casting): Toate valorile au fost forțate la tipul de date float32.
Rețelele neuronale (precum cele din TensorFlow/Keras) funcționează optim cu numere cu virgulă mobilă, nu cu întregi, pentru a permite calculele gradienților.

### 4.2 Transformarea caracteristicilor
Min-Max Scaling: Deoarece datele brute sunt colectate pe o scară (1–5), acestea au fost transformate în valori reale în intervalul unitar [0, 1].
Această etapă asigură că toate caracteristicile au aceeași pondere matematică în procesul de învățare.


### 4.3 Structurarea seturilor de date

Pentru a asigura o evaluare obiectivă a performanței modelului și o capacitate ridicată de generalizare, setul de 30.000 de instanțe a fost divizat în trei subseturi distincte, utilizând o strategie de tip Hold-Out:

Set de Antrenare (70% – 21,000 instanțe): Reprezintă baza de cunoștințe pe care rețeaua neuronală o utilizează pentru a-și ajusta ponderile prin procesul de backpropagation.

Set de Validare (15% – 4,500 instanțe): Utilizat pentru monitorizarea performanței în timpul antrenării. Rolul său este critic în reglarea hiperparametrilor (fine-tuning) și în aplicarea tehnicii de Early Stopping pentru a preveni overfitting-ul.

Set de Testare (15% – 4,500 instanțe): Acest set este complet izolat de procesul de antrenare. Este utilizat o singură dată, la final, pentru a raporta acuratețea reală a sistemului pe date complet noi.

Principii tehnice respectate:
Stratificarea pentru clasificare: S-a utilizat un mecanism de stratified splitting pentru a ne asigura că distribuția claselor (50% Pneumonie / 50% Tuberculoză) se menține identică în toate cele trei seturi. Acest lucru previne apariția unui model părtinitor (biased).

Prevenirea scurgerii de informații (Data Leakage): Parametrii de scalare (Min și Max) au fost calculați exclusiv pe setul de antrenare și ulterior aplicați pe seturile de validare și testare. Aceasta simulează un scenariu real în care modelul întâlnește date noi despre care nu are informații statistice prealabile.

Independența observațiilor: Amestecarea datelor (shuffling) a fost realizată înainte de split, garantând că ordinea generării programatice nu introduce corelații artificiale în procesul de învățare.

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
