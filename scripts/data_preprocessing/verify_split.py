"""validationCB1data split details successfully"""
import pandas as pd

# read data
print("read data...")
a = pd.read_csv('ZINC/CB1R/CB1_actives.csv')
i = pd.read_csv('ZINC/CB1R/CB1_inactives.csv')

total = len(a) + len(i)
print(f"\nmolecule data: {total:,}")
print(f"active molecules: {len(a):,} ({len(a)/total*100:.2f}%)")
print(f"inactive molecules: {len(i):,} ({len(i)/total*100:.2f}%)")

print(f"\nactive molecules split data details: {a['score'].min():.2f} text {a['score'].max():.2f}")
print(f"inactive molecules split data details: {i['score'].min():.2f} text {i['score'].max():.2f}")

# validation details
act_worst = a['score'].max()
inact_best = i['score'].min()
print(f"\ndetails validation:")
print(f"  active molecules split data: {act_worst:.2f}")
print(f"  inactive molecules split data: {inact_best:.2f}")
print(f"  details successfully (active details <= inactive details): {act_worst <= inact_best}")

# validation details
print(f"\ndetails validation:")
print(f"  active moleculeslabelas1: {(a['label'] == 1).all()}")
print(f"  inactive moleculeslabelas0: {(i['label'] == 0).all()}")
