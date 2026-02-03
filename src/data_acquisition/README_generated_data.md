### 📂 README pentru `src/data_acquisition/`
**Fișier:** `src/data_acquisition/README.md`

# 📡 Modulul 1: Achiziție Date

Acest modul gestionează întregul flux de date al sistemului, transformând logica medicală în seturi de date structurate,
normalizate și gata pentru antrenarea rețelei neuronale.
## 🧪 Metodologie: Simulare vs. Date Reale
Spre deosebire de o generare pur aleatorie, sistemul utilizează un Logic Map (hartă de probabilități) pentru a simula profiluri reale de pacienți:
### Scriptul `generate.py`
1.  **Ancore Medicale (Heavy Features): Întrebările critice precum Q15 (Greutate) și Q16 (Sânge) au o probabilitate de 85% de a urma diagnosticul corect, în timp ce restul simptomelor au o probabilitate de 65%.
2.  **Zgomot Clinic (Chaotic Mode): Am introdus o rată de 35% de date haotice pentru a simula pacienții care oferă răspunsuri contradictorii sau simptome atipice, forțând astfel rețeaua să învețe generalizarea, nu doar memorarea.
3.  **Volum: Generăm un set echilibrat de 30.000 de cazuri (15.000 per clasă).

## 2. Curățare și Normalizare

Pentru a optimiza antrenamentul, datele brute (scară 1-5) sunt convertite în Interval Unitar [0, 1].Formula: $x_{norm} = \frac{x - 1}{4}$Scop: Eliminarea diferențelor de scară și prevenirea saturării funcțiilor de activare ale modelului.

## 3. Distribuție și Stratificare

Datele sunt împărțite într-un flux de tip 70-15-15, asigurând o distribuție echilibrată prin Stratified Splitting:

Train (70%): Utilizat pentru ajustarea ponderilor modelului.

Validation (15%): Utilizat pentru reglarea hiperparametrilor și prevenirea overfitting-ului.

Test (15%): Set „blind” folosit exclusiv pentru raportul final de performanță (Etapa 6).

## ⚙️ Execuție
```bash
# 1. Generare date brute (raw)
python src/data_acquisition/generate.py

# 2. Normalizare simptome (0.0 - 1.0)
python src/data_acquisition/data_cleaner.py

# 3. Distribuție în folderele de antrenament
python src/data_acquisition/data_splitter.py

Execuția va popula automat folderul data/ cu următoarea ierarhie:

raw/ -> Fișierele inițiale cu valorile 1-5.

processed/ -> Fișierele normalizate.

train/, validation/, test/ -> Seturile finale, separate pe clase (Pneumonie/TBC).
