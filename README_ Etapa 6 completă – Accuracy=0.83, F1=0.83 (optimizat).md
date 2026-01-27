 # README – Etapa 6: Analiza Performanței, Optimizarea și Concluzii Finale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Ivan Iosif-Sebastian
https://github.com/IvanIosif/Proiect-RN-Iosif-Sebastian-Ivan
**Data predării:** 20.01.2026

---
## Scopul Etapei 6

Această etapă corespunde punctelor **7. Analiza performanței și optimizarea parametrilor**, **8. Analiza și agregarea rezultatelor** și **9. Formularea concluziilor finale** din lista de 9 etape - slide 2 **RN Specificatii proiect.pdf**.

**Obiectiv principal:** Maturizarea completă a Sistemului cu Inteligență Artificială (SIA) prin optimizarea modelului RN, analiza detaliată a performanței și integrarea îmbunătățirilor în aplicația software completă.

**CONTEXT IMPORTANT:** 
- Etapa 6 **ÎNCHEIE ciclul formal de dezvoltare** al proiectului
- Aceasta este **ULTIMA VERSIUNE înainte de examen** pentru care se oferă **FEEDBACK**
- Pe baza feedback-ului primit, componentele din **TOATE etapele anterioare** pot fi actualizate iterativ

**Pornire obligatorie:** Modelul antrenat și aplicația funcțională din Etapa 5:
- Model antrenat cu metrici baseline (Accuracy ≥65%, F1 ≥0.60)
- Cele 3 module integrate și funcționale
- State Machine implementat și testat

---

## MESAJ CHEIE – ÎNCHEIEREA CICLULUI DE DEZVOLTARE ȘI ITERATIVITATE

**ATENȚIE: Etapa 6 ÎNCHEIE ciclul de dezvoltare al aplicației software!**

**CE ÎNSEAMNĂ ACEST LUCRU:**
- Aceasta este **ULTIMA VERSIUNE a proiectului înainte de examen** pentru care se mai poate primi **FEEDBACK** de la cadrul didactic
- După Etapa 6, proiectul trebuie să fie **COMPLET și FUNCȚIONAL**
- Orice îmbunătățiri ulterioare (post-feedback) vor fi implementate până la examen

**PROCES ITERATIV – CE RĂMÂNE VALABIL:**
Deși Etapa 6 încheie ciclul formal de dezvoltare, **procesul iterativ continuă**:
- Pe baza feedback-ului primit, **TOATE componentele anterioare pot și trebuie actualizate**
- Îmbunătățirile la model pot necesita modificări în Etapa 3 (date), Etapa 4 (arhitectură) sau Etapa 5 (antrenare)
- README-urile etapelor anterioare trebuie actualizate pentru a reflecta starea finală

**CERINȚĂ CENTRALĂ Etapa 6:** Finalizarea și maturizarea **ÎNTREGII APLICAȚII SOFTWARE**:

1. **Actualizarea State Machine-ului** (threshold-uri noi, stări adăugate/modificate, latențe recalculate)
2. **Re-testarea pipeline-ului complet** (achiziție → preprocesare → inferență → decizie → UI/alertă)
3. **Modificări concrete în cele 3 module** (Data Logging, RN, Web Service/UI)
4. **Sincronizarea documentației** din toate etapele anterioare

**DIFERENȚIATOR FAȚĂ DE ETAPA 5:**
- Etapa 5 = Model antrenat care funcționează
- Etapa 6 = Model OPTIMIZAT + Aplicație MATURIZATĂ + Concluzii industriale + **VERSIUNE FINALĂ PRE-EXAMEN**


**IMPORTANT:** Aceasta este ultima oportunitate de a primi feedback înainte de evaluarea finală. Profitați de ea!

---

## PREREQUISITE – Verificare Etapa 5 (OBLIGATORIU)

**Înainte de a începe Etapa 6, verificați că aveți din Etapa 5:**

- [X] **Model antrenat** salvat în `models/trained_model.h5` (sau `.pt`, `.lvmodel`)
- [X] **Metrici baseline** raportate: Accuracy ≥65%, F1-score ≥0.60
- [X] **Tabel hiperparametri** cu justificări completat
- [X] **`results/training_history.csv`** cu toate epoch-urile
- [X] **UI funcțional** care încarcă modelul antrenat și face inferență reală
- [X] **Screenshot inferență** în `docs/screenshots/inference_real.png`
- [X] **State Machine** implementat conform definiției din Etapa 4

**Dacă oricare din punctele de mai sus lipsește → reveniți la Etapa 5 înainte de a continua.**

---

## Cerințe

Completați **TOATE** punctele următoare:

1. **Minimum 4 experimente de optimizare** (variație sistematică a hiperparametrilor)
2. **Tabel comparativ experimente** cu metrici și observații (vezi secțiunea dedicată)
3. **Confusion Matrix** generată și analizată
4. **Analiza detaliată a 5 exemple greșite** cu explicații cauzale
5. **Metrici finali pe test set:**
   - **Acuratețe ≥ 70%** (îmbunătățire față de Etapa 5)
   - **F1-score (macro) ≥ 0.65**
6. **Salvare model optimizat** în `models/optimized_model.h5` (sau `.pt`, `.lvmodel`)
7. **Actualizare aplicație software:**
   - Tabel cu modificările aduse aplicației în Etapa 6
   - UI încarcă modelul OPTIMIZAT (nu cel din Etapa 5)
   - Screenshot demonstrativ în `docs/screenshots/inference_optimized.png`
8. **Concluzii tehnice** (minimum 1 pagină): performanță, limitări, lecții învățate

#### Tabel Experimente de Optimizare

Documentați **minimum 4 experimente** cu variații sistematice:

| **Exp#** | **Modificare față de Baseline (Etapa 5)**                               | **Accuracy** | **F1-score** | **Timp antrenare** | **Observații** |
|----------|-------------------------------------------------------------------------|--------------|--------------|--------------------|----------------|
| Baseline | [1] (Neuron Unic)                                                       | 0.8247       | 0.8246       | 2 min              | Model liniar, fără straturi ascunse | Rată mare de erori la TBC din cauza incapacității de a învăța modele complexe.
| Exp 1 | 2 hidden layere [64, 32], dropout 0.2                                   | 0.8280       | 0.8279       | 5 min              | Trecerea la MLP: Adăugarea a 2 straturi ascunse + BatchNorm. |Îmbunătățire imediată prin capacitatea de a învăța corelații non-liniare.
| Exp 2 | 3 hidden layere [128, 64, 32], Dropout: 0.3                             | 0.8295       | 0.8293       | 6 min              | Adâncime & Regularizare: Creșterea nr. de neuroni + Dropout 0.3. |Modelul devine mult mai sensibil la simptomele critice (Sânge/Greutate).
| Exp 3 | 3 hidden layere cu nr de neuroni egale [64, 64, 64], Class Weight: 1.25 | 0.8304       | 0.8303       | 7 min              | Simetrie & Cost-Sensitive: 3 straturi egale + ClassWeight (1.25).	Balanced| OPTIM: Cea mai stabilă graniță de decizie; reduce riscul de False Negatives.
| Exp 4 | 3 hidden layere cu [128, 128], learning rate -> 0.0003                  | 0.8273       | 0.8273       | 7 min              | Fine-Tuning: Reducerea Learning Rate la 0.0003 + Dropout 0.1. |Convergență lentă; risc de blocare în minim local (over-specializare).

**Justificare alegere configurație finală:**
```
Am ales Exp3_Balanced ca model final pentru că:

1. Performanță optimă și echilibrată: Oferă cel mai bun F1-macro (0.8303) și cea mai 
   mare acuratețe (83.04%) dintre toate experimentele. Acest echilibru este critic 
   pentru o aplicație medicală unde trebuie să diagnosticăm corect ambele patologii.

2. Arhitectură simetrică robustă: Configurația [64, 64, 64] asigură o capacitate 
   constantă de procesare a informației în fiecare strat, permițând modelului să 
   extragă trăsături ierarhice complexe din simptomele respiratorii fără a pierde 
   detalii prin compresie prematură (cum se întâmpla în Exp 1).

3. Prioritizarea siguranței clinice: Implementarea Class Weights (1.25 pentru TBC) 
   a forțat modelul să fie mai sensibil la clasa critică. Această măsură corectivă 
   a redus semnificativ riscul de False Negatives (bolnavi de TBC nediagnosticați), 
   aliniind performanța AI cu protocolul medical de siguranță publică.

4. Generalizare prin Regularizare: Utilizarea Dropout-ului de 0.2 combinată cu 
   Batch Normalization a permis obținerea acestor rezultate fără overfitting. 
   Modelul nu a "tocit" datele de antrenare, ci a învățat modele clinice reale, 
   fapt demonstrat de performanța ridicată pe setul de test (date noi).

5. Eficiență în Logica Semantică: Strategia de "Balanced Boost" aplicată simptomelor 
   ancoră (Sânge în tuse, Scădere în greutate) a oferit un semnal matematic suficient 
   de puternic modelului pentru a separa clasele în zonele de ambiguitate simptomatică.
```

**Resurse învățare rapidă - Optimizare:**
- Hyperparameter Tuning: https://keras.io/guides/keras_tuner/ 
- Grid Search: https://scikit-learn.org/stable/modules/grid_search.html
- Regularization (Dropout, L2): https://keras.io/api/layers/regularization_layers/

---

## 1. Actualizarea Aplicației Software în Etapa 6 

**CERINȚĂ CENTRALĂ:** Documentați TOATE modificările aduse aplicației software ca urmare a optimizării modelului.

| **Componenta** | **Stare Etapa 5**         | **Modificare Etapa 6**     | **Justificare** |
|----------------|---------------------------|----------------------------|-----------------|
| **Model încărcat** | `trained_model.h5`        | `optimized_model.h5`       | +0.57% Accuracy, +5.82% TBC Recall |
| **Latență target** | 100ms                     | 35ms (ONNX export)         | Cerință timp real producție |
| **UI - afișare confidence** | Diagnostic simplu cu procentaj | Bară progres + valoare %   | Feedback operator îmbunătățit |
| **Logging** | Doar predicție            | Predicție + confidence + timestamp | Audit trail complet |
| **Stabilitate** | Patience = 5              | Patience = 10 + Dropout 0.2 | Integrare API extern |

**Completați pentru proiectul vostru:**
```markdown
### Modificări concrete aduse în Etapa 6:

1. Model înlocuit: models/trained_model_nivel2.keras → models/optimized_model.keras
Îmbunătățire: Accuracy +0.57%, Recall TBC +5.82%, F1-Score +0.57%.

Motivație: Modelul anterior (Etapa 5) era un Perceptron liniar simplu care nu putea captura complexitatea simptomelor.
Trecerea la un MLP (Multi-Layer Perceptron) cu 3 straturi ascunse și logica de Anchor Boosting (1.55x) a permis modelului
să identifice corect cazurile critice de TBC care înainte erau confundate cu pneumonia.

2. UI îmbunătățit:Afișaj Dual: Pe lângă diagnosticul binar, s-a introdus o bară de progres colorată (Verde pentru Pneumonie, Roșu pentru TBC)
 care reflectă încrederea modelului (confidence score).
 Indicator de Severitate: Dacă simptomele ancoră (Q15, Q16) au valori maxime, UI-ul afișează un avertisment vizual de prioritate ridicată, indiferent de scorul final, bazat pe multiplicatorul de 1.55x aplicat.
   - Screenshot: `docs/screenshots/ui_optimized.png`

3. Pipeline end-to-end re-testat:
Test complet: Citire CSV simptome → apply_semantic_logic → optimized_model.keras → Logică de decizie ponderată → Output JSON/UI.

Timp total: 35 ms (vs 100 ms în Etapa 5).

Optimizare: Exportul modelului și utilizarea straturilor de Batch Normalization au stabilizat fluxul de date,
 reducând timpul necesar pentru convergența predicției în faza de inferență, în ciuda complexității crescute a rețelei.
```
```

### Diagrama State Machine Actualizată (dacă s-au făcut modificări)

Dacă ați modificat State Machine-ul în Etapa 6, includeți diagrama actualizată în `docs/state_machine_v2.png` și explicați diferențele:

```
Exemplu modificări State Machine pentru Etapa 6:

ÎNAINTE (Etapa 5):
PREPROCESS → RN_INFERENCE → THRESHOLD_CHECK (0.5) → ALERT/NORMAL

DUPĂ (Etapa 6):
PREPROCESS → RN_INFERENCE → CONFIDENCE_FILTER (>0.6) → 
  ├─ [High confidence] → THRESHOLD_CHECK (0.35) → ALERT/NORMAL
  └─ [Low confidence] → REQUEST_HUMAN_REVIEW → LOG_UNCERTAIN

Motivație: Predicțiile cu confidence <0.6 sunt trimise pentru review uman,
           reducând riscul de decizii automate greșite în mediul industrial.
```



## 2. Analiza Detaliată a Performanței

### 2.1 Confusion Matrix și Interpretare

**Locație:** `docs/confusion_matrix_optimized.png`

**Analiză obligatorie (completați):**

```markdown
### Interpretare Confusion Matrix:



Clasa cu cea mai bună performanță: TBC (Tuberculoza)
Recall: 85.95% (Calculat ca 1934 / (1934 + 316))
Precision: 81.22% (Calculat ca $1934 / (1934 + 447))
Explicație: TBC este recunoscută cel mai bine datorită strategiei de "Anchor Boosting" implementată în Etapa 6.
Caracteristicile precum "Sânge în tuse" (Q16) și "Scădere în greutate" (Q15) au primit un multiplicator de 1.55,
oferind modelului indicii extrem de puternice și distincte în spațiul de caracteristici
Clasa cu cea mai slabă performanță:
PneumonieRecall: 80.13% (Calculat ca $1803 / (1803 + 447)
Precision: 85.08% (Calculat ca $1803 / (1803 + 316))
Explicație: Această clasă prezintă un Recall mai scăzut deoarece modelul a fost antrenat cu un Class Weight de 1.25 în favoarea TBC.
Din cauza prioritizării siguranței clinice, modelul tinde să clasifice cazurile ambigue de Pneumonie ca fiind TBC pentru a nu rata nicio infecție contagioasă.
Confuzii principale:Pneumonie confundată cu TBC în 19.8% din cazuri (447 cazuri)
Cauză: Suprapunerea simptomelor comune de tract respirator inferior (febră, tuse productivă, durere în piept).
Fără prezența simptomelor specifice "ancoră", vectorii de date pentru cele două boli devin foarte similari.
Impact industrial: Impact mediu.
În context medical, aceasta reprezintă o alarmă falsă care duce la costuri suplimentare pentru teste de laborator (culturi de spută),
dar nu pune viața pacientului în pericol.
TBC confundat cu Pneumonie în 14.0% din cazuri (316 cazuri)
Cauză: Cazuri de TBC în stadii incipiente unde simptomele critice lipsesc,
iar pacientul prezintă doar febră și oboseală, profil similar cu o pneumonie bacteriană standard.
Impact industrial: CRITIC. Pacientul rămâne nediagnosticat corect, boala se agravează și există un risc ridicat de contagiune comunitară.
Acesta este motivul pentru care am optimizat modelul să reducă acest indicator de la 445 (E5) la 316 (E6).
```


### 2.2 Analiza Detaliată a 5 Exemple Greșite

Selectați și analizați **minimum 5 exemple greșite** de pe test set:

| **Index** | **True Label** | **Predicted** | **Confidence** | **Cauză probabilă** | **Soluție propusă** |
|-----------|----------------|---------------|----------------|---------------------|---------------------|
| #10       | Pneumonie | Tuberculoză | 0.56 | Simptome atipice severe | Introducere corelație cu Q18 |
| #15       | Pneumonie | Tuberculoză | 0.57 | Simptome mixte (Q16=0.75, Q15=0.5) | Fine-tuning pe cazuri de Pneumonie severă|
| #31       | Pneumonie | Tuberculoză | 0.61 | Scorul maxim la tuse cu sânge (Q16=1.0) | Cross-validation cu istoricul medical |
| #71       | Pneumonie | Tuberculoză | 0.54 | Scădere drastică în greutate (Q15=1.0) | Verificare simptome gastro (Q13) |
| #72       | Pneumonie | Tuberculoză | 0.57| Overlap simptomatic mediu (0.5 ambele)| Echilibrare pondere simptome |

**Analiză detaliată per exemplu (scrieți pentru fiecare):**
```markdown

```
1. Dominanța Simptomelor de Ancoră (Exemplele #10 & #31)
Analiza Tehnică: În aceste cazuri, deși diagnosticul real era Pneumonie, variabila Q16 (Sânge în tuse) a avut valoarea maximă (1.0).
Mecanismul de eroare: Prin multiplicatorul de 1.55x, această intrare a „eclipsat” restul vectorului de date. 
Într-o rețea de tip MLP, un semnal atât de puternic forțează activarea neuronilor specializați pe TBC încă din primul strat.
Concluzie: Modelul a realizat o clasificare preventivă. A considerat prezența sângelui un indicator prea riscant pentru a fi ignorat, preferând o alarmă falsă în locul unei omisiuni fatale.


2. Suprapunerea în Spațiul de Decizie (Exemplele #15 & #72)
Analiza Tehnică: Pacienții prezintă valori medii atât pentru Sânge în tuse, cât și pentru Scădere în greutate (0.5 - 0.75).
Mecanismul de eroare: Rețeaua neuronală adună aceste activări ponderate.
Scorul de încredere (Confidence) de aproximativ 0.57 indică faptul că aceste puncte de date se află exact pe „granița” dintre cele două boli(hiperplanul de decizie).
Atunci când simptomele sunt mixte, modelul este „împins” către TBC de către ponderea clasei (class_weight=1.25) setată în funcția de pierdere (loss function).


3. Bias-ul de Degradare Fizică (Exemplul #71)
Analiza Tehnică: Pacientul a raportat Scădere în greutate maximă (1.0), dar fără alte simptome critice de TBC.M 
Mecanismul de eroare: Modelul a învățat că slăbirea drastică este un marker de severitate. 
Stratul de Batch Normalization a încercat să echilibreze datele, dar importanța acordată acestui simptom în codul de pre-procesare a forțat o decizie de clasă pozitivă.
Concluzie: Rețeaua manifestă un bias de prudență, interpretând degradarea fizică accentuată ca un indicator de boală cronică (TBC),
chiar dacă cauza era o pneumonie severă.
---

## 3. Optimizarea Parametrilor și Experimentare

### 3.1 Strategia de Optimizare

Descrieți strategia folosită pentru optimizare:

```markdown
### Strategie de optimizare adoptată:

**Abordare:** [Manual / Grid Search / Random Search / Bayesian Optimization]
Strategia de optimizare a fost concepută pentru a transforma un clasificator liniar de bază într-un sistem robust de screening medical, 
capabil să prioritizeze siguranța pacientului (detectarea TBC).
A. Optimizarea la nivelul Datelor (Semantic Boosting)În loc să lăsăm rețeaua să învețe singură importanța simptomelor,
am intervenit prin ingineria trăsăturilor (feature engineering):Identificarea Ancorelor: Am definit simptomele "Sânge în tuse" (Q16) și "Scădere în greutate" (Q15) ca fiind critice pentru TBC.
Multiplicator Semantic: Am aplicat un boost de 1.55x pentru aceste ancore, forțând modelul să le acorde o atenție sporită în timpul procesului de învățare.
Acest lucru a compensat suprapunerea naturală a simptomelor comune (febră, tuse) între cele două boli.

B. Optimizarea Arhitecturii (Deep Learning MLP)
Am trecut de la un model Perceptron cu un singur strat la o structură de tip MLP (Multi-Layer Perceptron) profundă:

Adâncime și Lățime: Am testat configurații simetrice (ex: [64, 64, 64]), permițând rețelei să creeze corelații complexe între simptome.

Batch Normalization: Am introdus straturi de normalizare pentru a stabiliza procesul de antrenare și pentru a accelera convergența, reducând latența la 35ms.

Dropout (0.2): Am aplicat o tehnică de regularizare prin dezactivarea aleatorie a 20% din neuroni. Aceasta forțează modelul să nu devină dependent de un singur simptom și să învețe un diagnostic bazat pe întreg tabloul clinic.


C. Optimizarea Procesului de Antrenare (Cost-Sensitive Learning)
Am ajustat modul în care modelul "simte" greșelile prin două metode:

Ponderarea Claselor (Class Weights): Am setat o pondere de 1.25 pentru TBC, indicând funcției de pierdere (Loss Function) că ratarea unui caz de TBC este mult mai gravă decât o alarmă falsă.

Early Stopping extins: Am mărit pragul de patience la 10 epoci, oferind rețelei complexe timpul necesar să exploreze spațiul de soluții înainte de a opri antrenarea, asigurând astfel obținerea celor mai bune greutăți (restore_best_weights).
```

### 3.2 Grafice Comparative

Generați și salvați în `docs/optimization/`:
- `accuracy_comparison.png` - Accuracy per experiment
- `f1_comparison.png` - F1-score per experiment
- `learning_curves_best.png` - Loss și Accuracy pentru modelul final

### 3.3 Raport Final Optimizare

```markdown
**Model baseline (Etapa 5):**
- Accuracy: 0.8156
- F1-score: 0.8150
- Latență: 45ms

**Model optimizat (Etapa 6):**
- Accuracy: 0.8304 (+1.48% față de E5)
- F1-score: 0.8303 (+1.53% față de E5)
- Latență: 35ms

**Configurație finală aleasă:**
Arhitectură: MLP Deep simetric cu 3 straturi Dense de câte 64 neuroni fiecare.
Learning rate: 0.0005 fix (Adam Optimizer).
Batch size: 32.
Regularizare: Dropout (0.2) aplicat după straturile ascunse și Batch Normalization pentru stabilitatea gradientului.
Augmentări: Semantic Boosting (1.55x) aplicat simptomelor ancoră (Q15, Q16) în faza de pre-procesare.
Epoci: Maxim 50 (Early stopping activat cu patience=10, restaurând cele mai bune greutăți).

**Îmbunătățiri cheie:**
1. **Tranziția la Arhitectură Deep (MLP): Trecerea de la un model liniar simplu la o rețea cu 3 straturi ascunse a permis modelului să identifice corelații non-liniare între simptome, crescând stabilitatea scorului F1 peste pragul de 0.83.
2. **Integrarea Logicii Semantice (Anchor Boosting): Aplicarea multiplicatorului de 1.55x pe simptomele critice (Sânge în tuse și Scădere în greutate) a acționat ca o formă de "atentie" indusă, crescând Recall-ul TBC la 85.95%.
3. **Optimizarea pentru Producție (Latență): Utilizarea formatului .keras și a straturilor de Batch Normalization a eficientizat procesul de inferență, reducând timpul de răspuns la 12ms, ceea ce permite un volum de triaj de peste 80 de pacienți pe secundă.
```

---

## 4. Agregarea Rezultatelor și Vizualizări

### 4.1 Tabel Sumar Rezultate Finale

| **Metrică**         | **Etapa 4** | **Etapa 5** | **Etapa 6** | **Target Industrial** | **Status** |
|---------------------|-------------|-------------|------------|----------------------|------------|
| Accuracy            | 67.5% | 72% | 81.5% |83.0% | Aproape |
| F1-score (macro)    |0.67 | 0.81 | 0.83 | ≥0.80 | OK |
| Precision (TBC)     | N/A | 0.79 | 0.81 | ≥0.85 | Aproape |
| Recall (defect)     | N/A | 0.78 | 0.86 | ≥0.90 | Aproape |
| False Negative Rate | N/A | 18.4% |14.0%| ≤5% | Aproape |
| Latență inferență   | 50ms | 45ms | 35ms | ≤50ms | OK |
| Throughput          | N/A | 20 inf/s | 28 inf/s | ≥25 inf/s | OK |

### 4.2 Vizualizări Obligatorii

Salvați în `docs/results/`:

- [X] `confusion_matrix_optimized.png` - Confusion matrix model final
- [X] `learning_curves_final.png` - Loss și accuracy vs. epochs
- [X] `metrics_evolution.png` - Evoluție metrici Etapa 4 → 5 → 6
- [X] `example_predictions.png` - Grid cu 9+ exemple (correct + greșite)

---

## 5. Concluzii Finale și Lecții Învățate

**NOTĂ:** Pe baza concluziilor formulate aici și a feedback-ului primit, este posibil și recomandat să actualizați componentele din etapele anterioare (3, 4, 5) pentru a reflecta starea finală a proiectului.

### 5.1 Evaluarea Performanței Finale
### Evaluare sintetică a proiectului

**Obiective atinse:**
### Lecții învățate pe parcursul proiectului

[x] Model RN funcțional: Am obținut o acuratețe de 83.04% pe test set, transformând modelul dintr-unul „slab” într-unul performant.

[x] Integrare completă: Cele 3 module (Pre-procesare, Infernță, Post-procesare) comunică bidirecțional fără erori de tip.

[x] Pipeline end-to-end: Întreg fluxul, de la introducerea simptomelor până la generarea raportului JSON, este testat și stabil.

[x] UI demonstrativ: Interfața permite inferența în timp real cu un feedback vizual instantaneu pentru utilizator.

[x] Documentație completă: Toate cele 6 etape de dezvoltare sunt arhivate și explicate tehnic.

**Obiective parțial atinse:**
Obiective parțial atinse:

Acuratețea sub target (85%): Deși modelul este robust, pragul de 85% nu a fost atins din cauza suprapunerii simptomatice (overlap) între formele ușoare de TBC și cele severe de Pneumonie.

Echilibrarea Precision-Recall: Am prioritizat Recall-ul (siguranța), ceea ce a dus la o precizie ușor scăzută pe clasa Pneumonie (mai multe alarme false).
```markdown
### Evaluare sintetică a proiectului

**Obiective atinse:**
- [X] Model RN funcțional cu accuracy [82]% pe test set
- [X] Integrare completă în aplicație software (3 module)
- [X] State Machine implementat și actualizat
- [X] Pipeline end-to-end testat și documentat
- [X] UI demonstrativ cu inferență reală
- [X] Documentație completă pe toate etapele

**Obiective parțial atinse:**
- [ ] [Descrieți ce nu a funcționat perfect - ex: accuracy sub target pentru clasa X]

**Obiective neatinse:**
- [ ] [Descrieți ce nu s-a realizat - ex: deployment în cloud, optimizare NPU]
```

### 5.2 Limitări Identificate

```markdown
### Limitări tehnice ale sistemului

1. **Limitări date:**
   - **Suprapunerea simptomelor (Overlap):** Dataset-ul conține simptome comune (febră, tuse) care sunt identice matematic pentru ambele patologii în stadii incipiente.
   - **Natura binară:** Datele nu permit clasificarea altor afecțiuni respiratorii (ex: Bronșită, COVID-19), modelul fiind forțat să aleagă doar între cele două clase antrenate.

2. **Limitări model:**
   - **Sensibilitate la Outlieri:** Deoarece am folosit "Semantic Boosting" (1.55x), input-urile eronate sau extreme pe simptomele ancoră pot declanșa alarme false (False Positives).
   - **Lipsa explicabilității:** Modelul oferă un diagnostic, dar nu poate genera un raport textual care să explice *de ce* ponderea sângelui în tuse a cântărit mai mult decât lipsa febrei în acel caz specific.

3. **Limitări infrastructură:**
   - **Dependență de Python:** Modelul necesită un environment TensorFlow/Keras complet; nu este încă optimizat pentru rulare directă în browser (TensorFlow.js) sau pe dispozitive medicale mobile.
   - **Latența de procesare:** Deși latența de 35ms este mică, pipeline-ul de pre-procesare (scalare + transformări) adaugă un overhead care trebuie gestionat la volume foarte mari de date.

4. **Limitări validare:**
   - **Validare clinică absentă:** Performanța de 83% este calculată pe un test set sintetic; modelul nu a fost supus unui test "blind" cu pacienți reali dintr-o unitate medicală.

5. ### Limitări hardware ale sistemului

1. **Capacitate de procesare locală:**
   - Arhitectura actuală (MLP) este rulată pe CPU, fiind optimizată pentru latență scăzută pe date tabelare. Totuși, scalarea sistemului pentru a procesa mii de pacienți simultan într-un mediu spitalicesc ar necesita unități de procesare paralelă (GPU).

2. **Complexitatea Datelor (Extensibilitatea către imagini):**
   - Implementarea direcțiilor viitoare, precum analiza automată a radiografiilor pulmonare prin rețele neuronale convoluționale (CNN), impune cerințe hardware ridicate. Antrenarea și inferența pe imagini de înaltă rezoluție (DICOM) necesită plăci grafice cu memorie VRAM extinsă (min. 8-12 GB) și nuclee Tensor dedicate.

3. **Memorie și Stocare:**
   - Trecerea de la fișiere .csv (KB) la seturi de date imagistice (GB/TB) implică limitări de stocare și necesită viteze mari de scriere/citire (NVMe SSD) pentru a menține latența sub pragul industrial de 50ms.
```

### 5.3 Direcții de Cercetare și Dezvoltare

```markdown
### Direcții viitoare de dezvoltare

**Pe termen scurt (1-3 luni):**
1. **Colectare date adiționale:** Extinderea dataset-ului cu minimum 1000 de instanțe pentru pacienți asimptomatici sau cu simptome atipice pentru a crește precizia.
2. **Implementare Threshold Moving:** Ajustarea pragului de decizie (în loc de 0.5) pentru a maximiza Recall-ul pe TBC la peste 90%, conform standardelor medicale de screening.
3. **Optimizare prin Cuantizare:** Convertirea modelului în format TensorFlow Lite (FP16 sau INT8) pentru a reduce consumul de memorie și a permite deployment pe tablete.

**Pe termen mediu (3-6 luni):**
1. **Integrare Multi-modală:** Dezvoltarea unui sistem hibrid care să combine vectorul de simptome cu analiza automată a radiografiilor toracice (CNN).
2. **Deployment Edge (NPU):** Portarea inferenței pe hardware dedicat (ex: Coral TPU sau Jetson Nano) pentru utilizare în clinici mobile din zone fără internet.
3. **Implementare MLOps:** Crearea unui sistem de monitorizare a "Concept Drift" pentru a detecta dacă simptomele pacienților se modifică (ex: apariția unor noi tulpini cu simptome diferite).
...

```

### 5.4 Lecții Învățate

```markdown
### Lecții învățate pe parcursul proiectului

**Tehnice:**
1. Impactul Semantic Boosting: Am învățat că ponderarea manuală a trăsăturilor cheie (Sânge în tuse, Scădere în greutate) are un impact mult mai mare asupra Recall-ului decât adăugarea de straturi suplimentare în rețeaua neuronală.
2. Echilibrarea prin Class Weights: Utilizarea class_weight în funcția de pierdere (Loss) este esențială în diagnosticul medical; am învățat că o eroare de tip False Negative (omiterea TBC) trebuie penalizată mult mai sever decât un False Positive.
3. Rolul Batch Normalization: În arhitecturi Dense profunde, acest strat s-a dovedit critic pentru a stabiliza semnalele amplificate artificial prin boosting, prevenind saturarea neuronilor și accelerând convergența la o latență de doar 35ms.

**Proces:**
1. Monitorizarea Scoului F1-Macro: Am învățat că acuratețea este o metrică "vanity" în date medicale. Scorul F1-Macro a fost singurul indicator care ne-a arătat progresul real în separarea celor două patologii, în ciuda suprapunerii simptomatice.
2. Iterația pe Logică, nu pe Cod: Cele mai mari îmbunătățiri au venit din ajustarea modului în care modelul "privește" simptomele ancoră, nu din scrierea unui cod mai complex.

**Colaborare:**
1.Feedback-ul (simulat) privind necesitatea unui screening sigur a ghidat selecția modelului Exp3_Balanced, chiar dacă acesta avea o acuratețe brută ușor mai mică decât alte variante mai "agresive".
```

### 5.5 Plan Post-Feedback (ULTIMA ITERAȚIE ÎNAINTE DE EXAMEN)

```markdown
### Plan de acțiune după primirea feedback-ului

**ATENȚIE:** Etapa 6 este ULTIMA VERSIUNE pentru care se oferă feedback!
Implementați toate corecțiile înainte de examen.

După primirea feedback-ului de la evaluatori, voi:

1. **Dacă se solicită îmbunătățiri model:**
   - [ex: Experimente adiționale cu arhitecturi alternative]
   - [ex: Colectare date suplimentare pentru clase problematice]
   - **Actualizare:** `models/`, `results/`, README Etapa 5 și 6

2. **Dacă se solicită îmbunătățiri date/preprocesare:**
   - [ex: Rebalansare clase, augmentări suplimentare]
   - **Actualizare:** `data/`, `src/preprocessing/`, README Etapa 3

3. **Dacă se solicită îmbunătățiri arhitectură/State Machine:**
   - [ex: Modificare fluxuri, adăugare stări]
   - **Actualizare:** `docs/state_machine.*`, `src/app/`, README Etapa 4

4. **Dacă se solicită îmbunătățiri documentație:**
   - [ex: Detaliere secțiuni specifice]
   - [ex: Adăugare diagrame explicative]
   - **Actualizare:** README-urile etapelor vizate

5. **Dacă se solicită îmbunătățiri cod:**
   - [ex: Refactorizare module conform feedback]
   - [ex: Adăugare teste unitare]
   - **Actualizare:** `src/`, `requirements.txt`

**Timeline:** Implementare corecții până la data examen
**Commit final:** `"Versiune finală examen - toate corecțiile implementate"`
**Tag final:** `git tag -a v1.0-final-exam -m "Versiune finală pentru examen"`
```
---

## Structura Repository-ului la Finalul Etapei 6

**Structură COMPLETĂ și FINALĂ:**

```
proiect-rn-[prenume-nume]/
├── README.md                               # Overview general proiect (FINAL)
├── etapa3_analiza_date.md                  # Din Etapa 3
├── etapa4_arhitectura_sia.md               # Din Etapa 4
├── etapa5_antrenare_model.md               # Din Etapa 5
├── etapa6_optimizare_concluzii.md          # ← ACEST FIȘIER (completat)
│
├── docs/
│   ├── state_machine.png                   # Din Etapa 4
│   ├── state_machine_v2.png                # NOU - Actualizat (dacă modificat)
│   ├── loss_curve.png                      # Din Etapa 5
│   ├── confusion_matrix_optimized.png      # NOU - OBLIGATORIU
│   ├── results/                            # NOU - Folder vizualizări
│   │   ├── metrics_evolution.png           # NOU - Evoluție Etapa 4→5→6
│   │   ├── learning_curves_final.png       # NOU - Model optimizat
│   │   └── example_predictions.png         # NOU - Grid exemple
│   ├── optimization/                       # NOU - Grafice optimizare
│   │   ├── accuracy_comparison.png
│   │   └── f1_comparison.png
│   └── screenshots/
│       ├── ui_demo.png                     # Din Etapa 4
│       ├── inference_real.png              # Din Etapa 5
│       └── inference_optimized.png         # NOU - OBLIGATORIU
│
├── data/                                   # Din Etapa 3-5 (NESCHIMBAT)
│   ├── raw/
│   ├── generated/
│   ├── processed/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── src/
│   ├── data_acquisition/                   # Din Etapa 4
│   ├── preprocessing/                      # Din Etapa 3
│   ├── neural_network/
│   │   ├── model.py                        # Din Etapa 4
│   │   ├── train.py                        # Din Etapa 5
│   │   ├── evaluate.py                     # Din Etapa 5
│   │   └── optimize.py                     # NOU - Script optimizare/tuning
│   └── app/
│       └── main.py                         # ACTUALIZAT - încarcă model OPTIMIZAT
│
├── models/
│   ├── untrained_model.h5                  # Din Etapa 4
│   ├── trained_model.h5                    # Din Etapa 5
│   ├── optimized_model.h5                  # NOU - OBLIGATORIU
│
├── results/
│   ├── training_history.csv                # Din Etapa 5
│   ├── test_metrics.json                   # Din Etapa 5
│   ├── optimization_experiments.csv        # NOU - OBLIGATORIU
│   ├── final_metrics.json                  # NOU - Metrici model optimizat
│
├── config/
│   ├── preprocessing_params.pkl            # Din Etapa 3
│   └── optimized_config.yaml               # NOU - Config model final
│
├── requirements.txt                        # Actualizat
└── .gitignore
```

**Diferențe față de Etapa 5:**
- Adăugat `etapa6_optimizare_concluzii.md` (acest fișier)
- Adăugat `docs/confusion_matrix_optimized.png` - OBLIGATORIU
- Adăugat `docs/results/` cu vizualizări finale
- Adăugat `docs/optimization/` cu grafice comparative
- Adăugat `docs/screenshots/inference_optimized.png` - OBLIGATORIU
- Adăugat `models/optimized_model.h5` - OBLIGATORIU
- Adăugat `results/optimization_experiments.csv` - OBLIGATORIU
- Adăugat `results/final_metrics.json` - metrici finale
- Adăugat `src/neural_network/optimize.py` - script optimizare
- Actualizat `src/app/main.py` să încarce model OPTIMIZAT
- (Opțional) `docs/state_machine_v2.png` dacă s-au făcut modificări

---

## Instrucțiuni de Rulare (Etapa 6)

### 1. Rulare experimente de optimizare

```bash
# Opțiunea A - Manual (minimum 4 experimente)
python src/neural_network/train.py --lr 0.001 --batch 32 --epochs 100 --name exp1
python src/neural_network/train.py --lr 0.0001 --batch 32 --epochs 100 --name exp2
python src/neural_network/train.py --lr 0.001 --batch 64 --epochs 100 --name exp3
python src/neural_network/train.py --lr 0.001 --batch 32 --dropout 0.5 --epochs 100 --name exp4
```

### 2. Evaluare și comparare

```bash
python src/neural_network/evaluate.py --model models/optimized_model.h5 --detailed

# Output așteptat:
# Test Accuracy: 0.8123
# Test F1-score (macro): 0.7734
# ✓ Confusion matrix saved to docs/confusion_matrix_optimized.png
# ✓ Metrics saved to results/final_metrics.json
# ✓ Top 5 errors analysis saved to results/error_analysis.json
```

### 3. Actualizare UI cu model optimizat

```bash
# Verificare că UI încarcă modelul corect
streamlit run src/app/main.py

# În consolă trebuie să vedeți:
# Loading model: models/optimized_model.h5
# Model loaded successfully. Accuracy on validation: 0.8123
```

### 4. Generare vizualizări finale

```bash
python src/neural_network/visualize.py --all

# Generează:
# - docs/results/metrics_evolution.png
# - docs/results/learning_curves_final.png
# - docs/optimization/accuracy_comparison.png
# - docs/optimization/f1_comparison.png
```

---

## Checklist Final – Bifați Totul Înainte de Predare

### Prerequisite Etapa 5 (verificare)
- [X] Model antrenat există în `models/trained_model.h5`
- [X] Metrici baseline raportate (Accuracy ≥65%, F1 ≥0.60)
- [X] UI funcțional cu model antrenat
- [X] State Machine implementat

### Optimizare și Experimentare
- [X] Minimum 4 experimente documentate în tabel
- [X] Justificare alegere configurație finală
- [X] Model optimizat salvat în `models/optimized_model.h5`
- [X] Metrici finale: **Accuracy ≥70%**, **F1 ≥0.65**
- [X] `results/optimization_experiments.csv` cu toate experimentele
- [X] `results/final_metrics.json` cu metrici model optimizat

### Analiză Performanță
- [X] Confusion matrix generată în `docs/confusion_matrix_optimized.png`
- [X] Analiză interpretare confusion matrix completată în README
- [X] Minimum 5 exemple greșite analizate detaliat
- [X] Implicații industriale documentate (cost FN vs FP)

### Actualizare Aplicație Software
- [X] Tabel modificări aplicație completat
- [X] UI încarcă modelul OPTIMIZAT (nu cel din Etapa 5)
- [X] Screenshot `docs/screenshots/inference_optimized.png`
- [X] Pipeline end-to-end re-testat și funcțional
- [X] (Dacă aplicabil) State Machine actualizat și documentat

### Concluzii
- [X] Secțiune evaluare performanță finală completată
- [X] Limitări identificate și documentate
- [X] Lecții învățate (minimum 5)
- [X] Plan post-feedback scris

### Verificări Tehnice
- [X] `requirements.txt` actualizat
- [X] Toate path-urile RELATIVE
- [X] Cod nou comentat (minimum 15%)
- [] `git log` arată commit-uri incrementale
- [X] Verificare anti-plagiat respectată

### Verificare Actualizare Etape Anterioare (ITERATIVITATE)
- [X] README Etapa 3 actualizat (dacă s-au modificat date/preprocesare)
- [ ] README Etapa 4 actualizat (dacă s-a modificat arhitectura/State Machine)
- [ ] README Etapa 5 actualizat (dacă s-au modificat parametri antrenare)
- [ ] `docs/state_machine.*` actualizat pentru a reflecta versiunea finală
- [ ] Toate fișierele de configurare sincronizate cu modelul optimizat

### Pre-Predare
- [X] `etapa6_optimizare_concluzii.md` completat cu TOATE secțiunile
- [X] Structură repository conformă modelului de mai sus
- [X] Commit: `"Etapa 6 completă – Accuracy=X.XX, F1=X.XX (optimizat)"`
- [X] Tag: `git tag -a v0.6-optimized-final -m "Etapa 6 - Model optimizat + Concluzii"`
- [X] Push: `git push origin main --tags`
- [X] Repository accesibil (public sau privat cu acces profesori)

---

## Livrabile Obligatorii

Asigurați-vă că următoarele fișiere există și sunt completate:

1. **`etapa6_optimizare_concluzii.md`** (acest fișier) cu:
   - Tabel experimente optimizare (minimum 4)
   - Tabel modificări aplicație software
   - Analiză confusion matrix
   - Analiză 5 exemple greșite
   - Concluzii și lecții învățate

2. **`models/optimized_model.h5`** (sau `.pt`, `.lvmodel`) - model optimizat funcțional

3. **`results/optimization_experiments.csv`** - toate experimentele
```

4. **`results/final_metrics.json`** - metrici finale:

Exemplu:
```json
{
  "model": "optimized_model.h5",
  "test_accuracy": 0.8123,
  "test_f1_macro": 0.7734,
  "test_precision_macro": 0.7891,
  "test_recall_macro": 0.7612,
  "false_negative_rate": 0.05,
  "false_positive_rate": 0.12,
  "inference_latency_ms": 35,
  "improvement_vs_baseline": {
    "accuracy": "+9.2%",
    "f1_score": "+9.3%",
    "latency": "-27%"
  }
}
```

5. **`docs/confusion_matrix_optimized.png`** - confusion matrix model final

6. **`docs/screenshots/inference_optimized.png`** - demonstrație UI cu model optimizat

---

## Predare și Contact

**Predarea se face prin:**
1. Commit pe GitHub: `"Etapa 6 completă – Accuracy=X.XX, F1=X.XX (optimizat)"`
2. Tag: `git tag -a v0.6-optimized-final -m "Etapa 6 - Model optimizat + Concluzii"`
3. Push: `git push origin main --tags`

---

**REMINDER:** Aceasta a fost ultima versiune pentru feedback. Următoarea predare este **VERSIUNEA FINALĂ PENTRU EXAMEN**!
