import pandas as pd
import numpy as np
import os
import random

def generate_raw_data_refined():
    base_path = r"D:\Facultate\RN\data\raw"
    paths = {0: os.path.join(base_path, "pneumonie"), 1: os.path.join(base_path, "tuberculoza")}
    for p in paths.values(): os.makedirs(p, exist_ok=True)

    # LOGICA ACTUALIZATĂ CONFORM NOILOR TALE TAG-URI
    # N=Neutru, P=Pneumonie, T=Tuberculoza
    # Fiecare listă corespunde opțiunilor [1, 2, 3, 4, 5]
    logic_map = {
        'Q1':  ['T', 'P', 'P', 'P', 'P'], # 1 e T, restul P
        'Q2':  ['P', 'P', 'P', 'T', 'T'], # 1,2,3 sunt P
        'Q3':  ['T', 'P', 'P', 'T', 'T'], # 1 e T
        'Q4':  ['P', 'P', 'P', 'T', 'T'], 
        'Q5':  ['P', 'P', 'P', 'T', 'T'],
        'Q6':  ['P', 'P', 'P', 'P', 'P'],
        'Q7':  ['T', 'P', 'P', 'P', 'P'],
        'Q8':  ['N', 'P', 'P', 'P', 'T'],
        'Q9':  ['T', 'P', 'P', 'P', 'T'],
        'Q10': ['P', 'P', 'P', 'P', 'P'],
        'Q11': ['N', 'P', 'P', 'T', 'T'],
        'Q12': ['P', 'P', 'P', 'T', 'T'],
        'Q13': ['T', 'P', 'P', 'P', 'P'],
        'Q14': ['T', 'P', 'P', 'P', 'P'],
        'Q15': ['P', 'P', 'P', 'T', 'T'],
        'Q16': ['P', 'P', 'P', 'T', 'T'],
        'Q17': ['P', 'P', 'P', 'T', 'T'],
        'Q18': ['T', 'P', 'P', 'P', 'P'],
        'Q19': ['P', 'P', 'P', 'T', 'T'],
        'Q20': ['P', 'P', 'P', 'P', 'T']
    }

    def create_samples(label, count):
        samples = []
        for _ in range(count):
            row = {}
            # Păstrăm un grad de ambiguitate pentru a simula un caz real (nu 100% acuratețe)
            is_ambiguous = random.random() < 0.35 
            
            for qid, options in logic_map.items():
                weights = []
                target_char = 'P' if label == 0 else 'T'
                
                for opt_type in options:
                    if is_ambiguous:
                        weights.append(0.20) # Zgomot
                    else:
                        if opt_type == target_char:
                            weights.append(0.60) # Probabilitate mare pentru tag-ul corect
                        elif opt_type == 'N':
                            weights.append(0.25) # Neutru
                        else:
                            weights.append(0.05) # Probabilitate mică pentru tag-ul greșit
                
                weights = np.array(weights) / sum(weights)
                row[qid] = np.random.choice([1, 2, 3, 4, 5], p=weights)
            
            ordered_row = [row[f'Q{i}'] for i in range(1, 21)]
            samples.append(ordered_row + [label])
        return samples

    cols = [f"Q{i}" for i in range(1, 21)] + ["Diagnosis"]
    pd.DataFrame(create_samples(0, 15000), columns=cols).to_csv(os.path.join(paths[0], "cases.csv"), index=False)
    pd.DataFrame(create_samples(1, 15000), columns=cols).to_csv(os.path.join(paths[1], "cases.csv"), index=False)
    print("✅ Date noi generate! Modelul va învăța acum importanța fiecărui tag (P/T).")

generate_raw_data_refined()
