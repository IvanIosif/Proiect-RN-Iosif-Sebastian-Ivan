import pandas as pd
import numpy as np
import os
import random

def generate_raw_data_with_options_logic():
    base_path = r"D:\Facultate\RN\data\raw"
    paths = {0: os.path.join(base_path, "pneumonie"), 1: os.path.join(base_path, "tuberculoza")}
    for p in paths.values(): os.makedirs(p, exist_ok=True)

    # Maparea opțiunilor: N=Normal/Neutru, P=Pneumonie, T=Tuberculoza
    # Fiecare listă are 5 elemente corespunzătoare opțiunilor 1, 2, 3, 4, 5
    logic_map = {
        'Q1':  ['N', 'P', 'P', 'P', 'P'], 'Q2':  ['N', 'P', 'P', 'T', 'T'],
        'Q3':  ['N', 'P', 'P', 'T', 'T'], 'Q4':  ['N', 'P', 'P', 'T', 'T'],
        'Q5':  ['N', 'P', 'P', 'T', 'T'], 'Q6':  ['N', 'P', 'P', 'P', 'P'],
        'Q7':  ['N', 'P', 'P', 'P', 'P'], 'Q8':  ['N', 'P', 'P', 'P', 'T'],
        'Q9':  ['N', 'P', 'P', 'P', 'T'], 'Q10': ['N', 'P', 'P', 'P', 'P'],
        'Q11': ['N', 'P', 'P', 'T', 'T'], 'Q12': ['N', 'P', 'P', 'T', 'T'],
        'Q13': ['N', 'P', 'P', 'P', 'P'], 'Q14': ['N', 'P', 'P', 'P', 'P'],
        'Q15': ['N', 'P', 'P', 'T', 'T'], 'Q16': ['N', 'P', 'P', 'T', 'T'],
        'Q17': ['N', 'P', 'P', 'T', 'T'], 'Q18': ['N', 'P', 'P', 'P', 'P'],
        'Q19': ['N', 'P', 'P', 'T', 'T'], 'Q20': ['N', 'P', 'P', 'P', 'T']
    }

    def create_samples(label, count):
        samples = []
        for _ in range(count):
            row = {}
            # Decidem dacă acest pacient este "clar" sau "ambiguu" pentru a forța acuratețea de 55-59%
            is_ambiguous = random.random() < 0.60 
            
            for qid, options in logic_map.items():
                weights = []
                for opt_type in options:
                    if is_ambiguous:
                        # Dacă e ambiguu, probabilitățile sunt aproape egale (zgomot mare)
                        weights.append(0.20)
                    else:
                        # Dacă e clar, ponderăm spre diagnosticul corect
                        target_char = 'P' if label == 0 else 'T'
                        if opt_type == target_char: weights.append(0.50)
                        elif opt_type == 'N': weights.append(0.30)
                        else: weights.append(0.10)
                
                # Normalizăm greutățile și alegem opțiunea
                weights = np.array(weights) / sum(weights)
                row[qid] = np.random.choice([1, 2, 3, 4, 5], p=weights)
            
            ordered_row = [row[f'Q{i}'] for i in range(1, 21)]
            samples.append(ordered_row + [label])
        return samples

    cols = [f"Q{i}" for i in range(1, 21)] + ["Diagnosis"]
    pd.DataFrame(create_samples(0, 15000), columns=cols).to_csv(os.path.join(paths[0], "cases.csv"), index=False)
    pd.DataFrame(create_samples(1, 15000), columns=cols).to_csv(os.path.join(paths[1], "cases.csv"), index=False)
    print("✅ Etapa 1: Date RAW generate conform logicii P/T per opțiune.")

generate_raw_data_with_options_logic()