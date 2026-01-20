SISTEM INTELIGENT HIBRID PENTRU DIAGNOSTICUL AFECȚIUNILOR RESPIRATORII
Student: Ivan Iosif-Sebastian
Grupa: 633AB | Facultatea: Ingineria Industrială și Robotică (FIIR) - UPB
Disciplina: Rețele Neuronale

📌 Descrierea Proiectului (Overview)
Acest proiect propune o soluție software avansată (SIA - Sistem de Inteligență Artificială) destinată triajului medical rapid și precis.
Spre deosebire de sistemele de diagnostic clasice bazate pe arbori de decizie statici, acest sistem utilizează o Rețea Neuronală Artificială (Perceptron)
antrenată pe date clinice structurate pentru a "înțelege" corelația dintre 20 de simptome și patologia maferentă.                                

🎯 Obiectiv Principal: Eficiența Triajului Medical
Scopul central este reducerea timpului de diagnosticare și eliminarea erorilor umane în mediile aglomerate (Urități de Primiri Urgențe), prin strategii adaptive:

Diferențiere Acut/Cronic: Detectează profilul de Pneumonie (febra înaltă, tuse productivă) și îl separă de cel de TBC (transpirații nocturne, scădere în greutate).

Explainable AI (XAI): Nu oferă doar un diagnostic, ci explică ponderea matematică (weights) a simptomelor care au dus la acea decizie.

Robustete Clinică: Diferențiază corect simptomele comune (tuse, durere în piept) pe baza intensității și a simptomelor asociate.

⚙️ Arhitectura Sistemului
Sistemul este modularizat în 3 componente interconectate, dezvoltate pe parcursul etapelor 3, 4 și 5:

1. Modulul de Achiziție Date & Preprocesare (Etapa 3)
Generează un set de date sintetic de 30000 de cazuri bazat pe protocoale medicale reale.

Implementează Nivelul 1 (Expert System) pentru etichetarea logică a datelor.

Include normalizare Min-Max și augmentare tip Jittering (zgomot de senzori) pentru a simula imprecizia raportării simptomelor de către pacienți.

2. Modulul de Inteligență Artificială (Etapa 4 & 5)
Tehnologie: TensorFlow / Keras.

Arhitectură: Perceptron cu funcție de activare Sigmoid pentru clasificare binară.

Performanță: Acuratețe de 81.55% pe setul de testare, cu o convergență stabilă a funcției de Loss.

3. Interfața Expert (Virtual Clinic - Etapa 5)
Dashboard Digital: Realizat în Streamlit, optimizat pentru interacțiune rapidă.

Analiză în timp real: Afișează probabilitatea diagnosticului și grafice interactive (Plotly) cu influența fiecărui simptom.

Logica Hibridă: Combină predicția rețelei cu vizualizarea ponderilor (Nivel 2).

Etapa	Descriere	Documentație
Etapa 3	Analiza datelor, generarea logică (Nivel 1) și preprocesarea.	 https://github.com/IvanIosif/Proiect-RN-Iosif-Sebastian-Ivan/blob/main/README%20%E2%80%93%20Etapa%203%20-Analiza%20si%20Pregatirea%20Setului%20de%20Date%20pentru%20Retele%20Neuronale.md
Etapa 4	Proiectarea arhitecturii modelului (Un-trained) și diagramele de flux.	https://github.com/IvanIosif/Proiect-RN-Iosif-Sebastian-Ivan/blob/main/README_Etapa4_Arhitectura_SIA%20functionala.md


🚀 Cum se rulează proiectul (Quick Start)
1. Cerințe de sistem
Python 3.9+

Librării: tensorflow, pandas, numpy, scikit-learn, streamlit, plotly, joblib

2. Instalare
Bash

3. Rulare Aplicație (Interfața Finală)
Bash
streamlit run src/app/main.py
pip install tensorflow pandas numpy scikit-learn streamlit plotly joblib

4. Re-antrenare Model
Bash
python train.py
