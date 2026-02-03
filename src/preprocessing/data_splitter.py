import pandas as pd
import os
from sklearn.model_selection import train_test_split

current_dir = os.path.dirname(os.path.abspath(__file__))
PATH_BASE = os.path.abspath(os.path.join(current_dir, "..", ".."))

def final_split_and_distribute():
    proc_base = os.path.join(PATH_BASE, "data", "processed")
    final_base = os.path.join(PATH_BASE, "data")
    
    p_file = os.path.join(proc_base, "pneumonie", "processed.csv")
    t_file = os.path.join(proc_base, "tuberculoza", "processed.csv")
    
    if not os.path.exists(p_file) or not os.path.exists(t_file):
        print(f"❌ Eroare: Nu am găsit fișierele procesate în {proc_base}")
        return

    # Încărcare date procesate
    df_p = pd.read_csv(p_path)
    df_t = pd.read_csv(t_path)
    
    # Unire pentru un shuffle global (esențial ca modelul să nu vadă pattern-uri de ordine)
    df_full = pd.concat([df_p, df_t]).sample(frac=1, random_state=42).reset_index(drop=True)

    train_df, temp_df = train_test_split(
        df_full, 
        test_size=0.30, 
        stratify=df_full['Diagnosis'], 
        random_state=42
    )
    
    # Split 30% rest în 15% Validation și 15% Test
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.50, 
        stratify=temp_df['Diagnosis'], 
        random_state=42
    )

    sets = {
        'train': train_df,
        'validation': val_df,
        'test': test_df
    }

    print("⏳ Distribuire fișiere în foldere...")

    for mode, data in sets.items():
        for label, name in [(0, "pneumonie"), (1, "tuberculoza")]:
            subset = data[data['Diagnosis'] == label]
            
            folder_path = os.path.join(final_base, mode, name)
            os.makedirs(folder_path, exist_ok=True)
            
            # SALVARE conform cerințelor load_dataset (ex: pneumonie_train.csv)
            file_name = f"{name}_{mode}.csv"
            save_path = os.path.join(folder_path, file_name)
            subset.to_csv(save_path, index=False)
            
    print(f"✅ Succes! Datele au fost împărțite și salvate în {final_base}")
    print(f"📊 Statistici: Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

if __name__ == "__main__":
    final_split_and_distribute()
