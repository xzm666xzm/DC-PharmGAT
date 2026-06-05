#!/usr/bin/env python3
"""
will actives.smi details as and actives_final.ism details of format
format: SMILES ID CHEMBL_ID (text UNKNOWN)
"""

import sys

def convert_smi_to_ism(input_file, output_file):
    """details.smi file as.ism format"""
    
    print(f"read: {input_file}")
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    print(f"total {len(lines)} information")
    
    output_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        if len(parts) >= 2:
            smiles = parts[0]
            mol_id = parts[1]
            
            # result details have No. three text, save details
            if len(parts) >= 3:
                chembl_id = parts[2]
            else:
                # no CHEMBL ID, details
                chembl_id = f"MOL_{mol_id}"
            
            output_lines.append(f"{smiles} {mol_id} {chembl_id}")
        elif len(parts) == 1:
            # has SMILES, generate ID
            smiles = parts[0]
            mol_id = str(len(output_lines) + 1)
            chembl_id = f"MOL_{mol_id}"
            output_lines.append(f"{smiles} {mol_id} {chembl_id}")
    
    # write information file
    with open(output_file, 'w') as f:
        f.write('\n'.join(output_lines))
        f.write('\n')
    
    print(f"details: {output_file}")
    print(f"information complete! total {len(output_lines)} information")

if __name__ == "__main__":
    input_file = "actives.smi"
    output_file = "actives_final.ism"
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    convert_smi_to_ism(input_file, output_file)
