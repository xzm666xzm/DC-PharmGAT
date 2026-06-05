import os
import argparse
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

def get_canonical_smiles_and_inchikey(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    can_smi = Chem.MolToSmiles(mol, canonical=True)
    try:
        inchi_key = Chem.inchi.MolToInchiKey(mol)
    except:
        inchi_str = Chem.MolToInchi(mol)
        inchi_key = Chem.InchiToInchiKey(inchi_str) if inchi_str else None
    return can_smi, inchi_key

def load_ism(file_path, label=None):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                smiles, id_ = parts[0], parts[1]
                data.append({"SMILES": smiles, "id": id_, "label": label})
            elif len(parts) == 1:
                data.append({"SMILES": parts[0], "id": "", "label": label})
    return pd.DataFrame(data)

def process_dataset(df, desc="Processing"):
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        can_smi, inchi_key = get_canonical_smiles_and_inchikey(row['SMILES'])
        if can_smi:
            results.append({
                "SMILES": row['SMILES'],
                "canonical_smiles": can_smi,
                "inchi_key": inchi_key,
                "id": row['id'],
                "label": row['label']
            })
    return pd.DataFrame(results)

def parse_args():
    parser = argparse.ArgumentParser(description="Build a de-overlapped CDK2 ChEMBL external validation set.")
    parser.add_argument("--external-active", default="outputs/cdk2_chembl_external/cdk2_chembl_external_active.ism")
    parser.add_argument("--external-inactive", default="outputs/cdk2_chembl_external/cdk2_chembl_external_inactive.ism")
    parser.add_argument("--train-active", required=True, help="Training active .ism file used for overlap removal.")
    parser.add_argument("--train-inactive", required=True, help="Training inactive/decoy .ism file used for overlap removal.")
    parser.add_argument("--output", default="outputs/cdk2_chembl_external/cdk2_chembl_external_dataset.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    print("load details validation set (ChEMBL-CDK2)...")
    ext_act_df = load_ism(args.external_active, label=1)
    ext_inact_df = load_ism(args.external_inactive, label=0)
    
    ext_df = pd.concat([ext_act_df, ext_inact_df], ignore_index=True)
    ext_processed = process_dataset(ext_df, desc="process details validation set")
    print(f"details count: {len(ext_processed)}")
    
    print("\nload training set (ZINC-CDK2)...")
    zinc_act_df = load_ism(args.train_active)
    zinc_inact_df = load_ism(args.train_inactive)
    zinc_df = pd.concat([zinc_act_df, zinc_inact_df], ignore_index=True)
    
    zinc_processed = process_dataset(zinc_df, desc="process ZINC dataset")
    
    zinc_smiles = set(zinc_processed['canonical_smiles'].dropna())
    zinc_inchikeys = set(zinc_processed['inchi_key'].dropna())
    
    print(f"\nExtracted {len(zinc_smiles)} canonical SMILES and {len(zinc_inchikeys)} InChIKeys from the ZINC dataset.")
    
    # Identify molecules overlapping with the ZINC-ESR1 training/test set
    overlap_mask = ext_processed['canonical_smiles'].isin(zinc_smiles) | ext_processed['inchi_key'].isin(zinc_inchikeys)
    overlaps = ext_processed[overlap_mask]
    
    print(f"\ndetails {len(overlaps)} details molecule. ")
    
    # Remove overlapping molecules
    final_ext_df = ext_processed[~overlap_mask].copy()
    print(f"Validation set size after removing overlaps: {len(final_ext_df)}")
    
    # Report active/inactive label distribution
    print("\nLabel distribution:")
    print(final_ext_df['label'].value_counts())
    
    # save results
    # Format output as SMILES, chembl_id, label
    final_output = final_ext_df[['SMILES', 'id', 'label']].rename(columns={'id': 'chembl_id'})
    final_output.to_csv(args.output, index=False)
    print(f"\nFinal external validation set saved to {args.output}")

if __name__ == "__main__":
    main()
