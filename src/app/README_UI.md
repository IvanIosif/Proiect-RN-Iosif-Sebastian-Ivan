### 📂 README pentru `src/app/` (Interfața Grafică)
**Fișier:** `src/app/README.md`

# 🖥️ Modulul 3: Interfața Grafică

Acest modul implementează partea de interactiune dintre sistemul SIA si utilizatori, oferind medicilor sau asistenților o interfață web interactivă pentru triajul rapid al pacienților.
## 📋 Descriere
Aplicația este construită folosind framework-ul Streamlit și funcționează ca un punct central de integrare pentru proiect.
1.  **Încarcă modelul optimizat (.keras) și efectuează predicții în timp real bazate pe 20 de indicatori clinici.
2.  **Sistemul de Stocare: Salvează automat fiecare diagnostic într-un istoric local (istoric_triaj.csv) pentru audit medical.
3.  **Logica de Interpretare: Traduce scorul numeric al rețelei (0.0 - 1.0) în recomandări clinice clare.

## 🎮 Funcționalități UI
* **Identificare Pacient: Câmp dedicat pentru trasabilitatea diagnosticului.
* **Chestionar Dinamic: 20 de întrebări cu selecție multipla, mapate automat pe intervalul de intrare [0, 1] al rețelei.
* **Vizualizare Rezultate:

Diagnostic Sugerat: Alertă vizuală colorată (Roșu pentru TBC, Albastru pentru Pneumonie).

Nivel de Încredere: Indicator grafic (Metric & Progress Bar) care arată siguranța predicției AI.

Analiza Impactului: Grafic interactiv (Plotly) care explică utilizatorului care simptome au influențat cel mai mult decizia modelului (Explainable AI).


## 🚀 Rulare
Din folderul rădăcină al proiectului:
```bash
streamlit run src/app/main.py
```

streamlit (Interfața web)

tensorflow (Încărcarea modelului)

plotly (Grafice interactive de impact)

pandas & numpy (Procesare date)

joblib (Încărcarea configurărilor de scalare)
