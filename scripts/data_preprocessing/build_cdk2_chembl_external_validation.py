"""
CDK2 ChEMBL External Validation Dataset Builder
=================================================
text ChEMBL details CDK2 (CHEMBL301) of information test active data,
use details file middle details bioactivity-oriented external validation. 
dataset use information validation, use training ZINC docking-score model. 

details need point:
  - target details: CHEMBL301 (CDK2, SINGLE PROTEIN, Homo sapiens)
  - active type: IC50 / Ki / Kd, assay_type='B', confidence>=8
  - details: active (<=1000nM), inactive (>=10000nM), ambiguous (middle information)
  - details: InChIKey details, median pActivity, information test
  - negative summary CDK2 assay text test text active molecules
=================================================
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict

from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Scaffolds, inchi
from rdkit.Chem.Scaffolds import MurckoScaffold

warnings.filterwarnings("ignore")


class CDK2ChEMBLCurator:
    """CDK2 ChEMBL details validation dataset build text"""

    TARGET_CHEMBL_ID = "CHEMBL301"
    ALLOWED_STANDARD_TYPES = {"IC50", "Ki", "Kd"}
    ACTIVE_THRESHOLD_NM = 1000.0
    INACTIVE_THRESHOLD_NM = 10000.0
    ACTIVE_PCHEMBL = 6.0
    INACTIVE_PCHEMBL = 5.0

    # ChEMBL activity details middle details need of information(API details return of key)
    RAW_FIELDS = [
        "molecule_chembl_id", "canonical_smiles",
        "standard_type", "standard_value", "standard_units",
        "pchembl_value", "assay_chembl_id", "document_chembl_id",
        "data_validity_comment",
]

    def __init__(self, output_dir: str = "outputs/cdk2_chembl_external"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.activity_client = new_client.activity
        self.molecule_client = new_client.molecule
        self.target_client = new_client.target
        self.assay_client = new_client.assay

        self.raw_records = []
        self.curated_df = None

        self._log_header("CDK2 ChEMBL External Validation Builder")
        self._log(f"target: {self.TARGET_CHEMBL_ID} (CDK2)")
        self._log(f"details directory: {os.path.abspath(self.output_dir)}")
        self._log(f"Active threshold: standard_value <= {self.ACTIVE_THRESHOLD_NM} nM "
                   f"or pChEMBL >= {self.ACTIVE_PCHEMBL}")
        self._log(f"Inactive threshold: standard_value >= {self.INACTIVE_THRESHOLD_NM} nM "
                   f"or pChEMBL <= {self.INACTIVE_PCHEMBL}")

    # ─── summary ───────────────────────────────────────────────────────
    @staticmethod
    def _log(msg: str):
        print(f"  {msg}", flush=True)

    @staticmethod
    def _log_header(title: str):
        print("\n" + "=" * 72)
        print(f"  {title}")
        print("=" * 72, flush=True)

    @staticmethod
    def _log_step(step: int, title: str):
        print(f"\n{'─' * 72}")
        print(f"  step {step}: {title}")
        print(f"{'─' * 72}", flush=True)

    @staticmethod
    def _safe_get(obj, key):
        """details dict and details property of API object"""
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    # ─── Step 1: details original active details ──────────────────────────────────────
    def collect_raw_activities(self) -> pd.DataFrame:
        """text ChEMBL get CDK2 (CHEMBL301) of Binding assay active details. 
        information point details: information directory already exists have effect of raw CSV, details read details API get. 
        """
        self._log_step(1, "text ChEMBL details CDK2 original active data")

        raw_csv = os.path.join(self.output_dir, "cdk2_chembl_raw_activities.csv")

        # ── text point details: result has have effect raw CSV, details read ──
        if os.path.exists(raw_csv):
            df_existing = pd.read_csv(raw_csv)
            if len(df_existing) > 0 and "molecule_chembl_id" in df_existing.columns:
                self._log(f"information have original data: {raw_csv} ({len(df_existing)} text)")
                self._log("details API get, directly use cache data")
                return df_existing

        # ── 1a. validation target ──
        self._log(f"validation target {self.TARGET_CHEMBL_ID}...")
        tgt = self.target_client.get(self.TARGET_CHEMBL_ID)
        self._log(f"  pref_name   = {self._safe_get(tgt, 'pref_name')}")
        self._log(f"  target_type = {self._safe_get(tgt, 'target_type')}")
        self._log(f"  organism    = {self._safe_get(tgt, 'organism')}")

        # ── 1b. get active (confidence_score >= 8 text API details) ──
        self._log("get Binding assay active (assay_type='B', confidence>=8)...")
        self._log("  (details data need details ChEMBL API details, please summary...)")
        activities = self.activity_client.filter(
            target_chembl_id=self.TARGET_CHEMBL_ID,
            assay_type="B",
            confidence_score__gte=8,
)

        raw = []
        for i, act in enumerate(activities):
            record = {}
            for f in self.RAW_FIELDS:
                record[f] = self._safe_get(act, f)
            raw.append(record)
            if (i + 1) % 500 == 0:
                self._log(f"  read {i + 1} text...")
            time.sleep(0.001)

        self._log(f"API return information data (confidence>=8): {len(raw)}")

        # details: details
        if raw:
            sample = {k: v for k, v in raw[0].items() if v is not None}
            self._log(f"  summary: {sample}")

        # ── 1c. summary ──
        kept = []
        stats = {"no_type": 0, "no_unit": 0, "bad_value": 0,
                 "no_id_smi": 0, "invalid_data": 0}
        for r in raw:
            if r.get("standard_type") not in self.ALLOWED_STANDARD_TYPES:
                stats["no_type"] += 1
                continue
            if r.get("standard_units")!= "nM":
                stats["no_unit"] += 1
                continue
            try:
                sv = float(r["standard_value"])
                if sv <= 0:
                    stats["bad_value"] += 1
                    continue
            except (TypeError, ValueError, KeyError):
                stats["bad_value"] += 1
                continue
            if not r.get("molecule_chembl_id") or not r.get("canonical_smiles"):
                stats["no_id_smi"] += 1
                continue
            # data_validity
            dvc = r.get("data_validity_comment")
            if dvc and "outside" in str(dvc).lower():
                stats["invalid_data"] += 1
                continue
            kept.append(r)

        self._log(f"  details:")
        self._log(f"    standard_type information: {stats['no_type']}")
        self._log(f"    standard_units!= nM: {stats['no_unit']}")
        self._log(f"    standard_value none effect:  {stats['bad_value']}")
        self._log(f"    details ID/SMILES:       {stats['no_id_smi']}")
        self._log(f"    data_validity exception:   {stats['invalid_data']}")
        self._log(f"  ==> save information data: {len(kept)}")

        if len(kept) == 0:
            self._log("WARNING: details after none details! check API return data. ")
            # text test details confidence information new get
            self._log("text test details confidence_score information new get...")
            activities2 = self.activity_client.filter(
                target_chembl_id=self.TARGET_CHEMBL_ID,
                assay_type="B",
                standard_type__in=["IC50", "Ki", "Kd"],
                standard_units="nM",
)
            raw2 = []
            for i, act in enumerate(activities2):
                record = {}
                for f in self.RAW_FIELDS:
                    record[f] = self._safe_get(act, f)
                raw2.append(record)
                if (i + 1) % 500 == 0:
                    self._log(f"  read {i + 1} text...")
                time.sleep(0.001)
            self._log(f"  summary return: {len(raw2)} text")
            # base information
            kept = []
            for r in raw2:
                try:
                    sv = float(r.get("standard_value", 0))
                    if sv <= 0:
                        continue
                except (TypeError, ValueError):
                    continue
                if not r.get("molecule_chembl_id") or not r.get("canonical_smiles"):
                    continue
                dvc = r.get("data_validity_comment")
                if dvc and "outside" in str(dvc).lower():
                    continue
                kept.append(r)
            self._log(f"  summary after: {len(kept)} text")

        self.raw_records = kept

        df_raw = pd.DataFrame(kept)
        self._log(f"  df_raw details: {list(df_raw.columns)}")
        df_raw.to_csv(raw_csv, index=False)
        self._log(f"  saved: {raw_csv}")
        return df_raw

    # ─── Step 2: get molecule data ────────────────────────────────────────
    def fetch_molecule_metadata(self, df_raw: pd.DataFrame) -> dict:
        """use RDKit calculation molecule information; return {chembl_id: metadata_dict}"""
        self._log_step(2, "calculation molecule information (RDKit)")

        if df_raw.empty:
            self._log("ERROR: details DataFrame as text, none information. ")
            return {}

        unique_smiles = df_raw.drop_duplicates("molecule_chembl_id")[
            ["molecule_chembl_id", "canonical_smiles"]
]
        self._log(f"one molecule data: {len(unique_smiles)}")

        meta = {}
        bad = 0
        for idx, (_, row) in enumerate(unique_smiles.iterrows()):
            cid = row["molecule_chembl_id"]
            smi = row["canonical_smiles"]
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                bad += 1
                continue
            can_smi = Chem.MolToSmiles(mol, canonical=True)

            # InChIKey
            try:
                inchi_str = Chem.MolToInchi(mol)
                inchi_key = Chem.InchiToInchiKey(inchi_str) if inchi_str else ""
            except Exception:
                inchi_key = ""

            # Murcko scaffold
            try:
                scaffold = MurckoScaffold.MakeScaffoldGeneric(
                    MurckoScaffold.GetScaffoldForMol(mol)
)
                scaffold_smi = Chem.MolToSmiles(scaffold, canonical=True)
            except Exception:
                scaffold_smi = ""

            meta[cid] = {
                "canonical_smiles": can_smi,
                "inchi_key": inchi_key,
                "molecular_weight": round(Descriptors.ExactMolWt(mol), 2),
                "alogp": round(Descriptors.MolLogP(mol), 2),
                "psa": round(Descriptors.TPSA(mol), 2),
                "hba": Descriptors.NumHAcceptors(mol),
                "hbd": Descriptors.NumHDonors(mol),
                "rtb": Descriptors.NumRotatableBonds(mol),
                "murcko_scaffold": scaffold_smi,
            }
            if (idx + 1) % 500 == 0:
                self._log(f"  process {idx + 1} / {len(unique_smiles)} molecule...")

        self._log(f"success details: {len(meta)} molecule, failed: {bad}")
        return meta

    # ─── Step 3: details, details, details process ──────────────────────────────────
    def curate_records(self, df_raw: pd.DataFrame, mol_meta: dict) -> pd.DataFrame:
        """molecule details, text note active/inactive/ambiguous/conflict"""
        self._log_step(3, "molecule information & information note")

        if df_raw.empty or not mol_meta:
            self._log("ERROR: none data can details. ")
            return pd.DataFrame()

        # 3a. molecule_chembl_id details
        groups = defaultdict(list)
        for _, row in df_raw.iterrows():
            cid = row["molecule_chembl_id"]
            if cid not in mol_meta:
                continue
            groups[cid].append(row)

        self._log(f"details molecule data: {len(groups)}")

        records = []
        for cid, rows in groups.items():
            m = mol_meta[cid]
            vals = []
            for r in rows:
                try:
                    vals.append(float(r["standard_value"]))
                except (TypeError, ValueError):
                    pass
            if not vals:
                continue

            pvals = []
            for r in rows:
                try:
                    pv = float(r["pchembl_value"])
                    if not np.isnan(pv):
                        pvals.append(pv)
                except (TypeError, ValueError):
                    pass

            median_val = float(np.median(vals))
            median_pact = float(np.median(pvals)) if pvals else np.nan

            # summary
            is_active = False
            is_inactive = False
            if not np.isnan(median_pact) and median_pact >= self.ACTIVE_PCHEMBL:
                is_active = True
            if median_val <= self.ACTIVE_THRESHOLD_NM:
                is_active = True
            if not np.isnan(median_pact) and median_pact <= self.INACTIVE_PCHEMBL:
                is_inactive = True
            if median_val >= self.INACTIVE_THRESHOLD_NM:
                is_inactive = True

            if is_active and is_inactive:
                label = "conflict"
            elif is_active:
                label = "active"
            elif is_inactive:
                label = "inactive"
            else:
                label = "ambiguous"

            st_types = sorted(set(str(r["standard_type"]) for r in rows
                                  if r.get("standard_type")))
            assay_ids = sorted(set(str(r["assay_chembl_id"]) for r in rows
                                   if r.get("assay_chembl_id")))
            doc_ids = sorted(set(str(r["document_chembl_id"]) for r in rows
                                 if r.get("document_chembl_id")))

            records.append({
                "chembl_id": cid,
                "canonical_smiles": m["canonical_smiles"],
                "inchi_key": m["inchi_key"],
                "median_pactivity": round(median_pact, 3) if not np.isnan(median_pact) else None,
                "median_value_nM": round(median_val, 2),
                "activity_label": label,
                "n_records": len(rows),
                "standard_types": ";".join(st_types),
                "assay_ids": ";".join(assay_ids),
                "document_ids": ";".join(doc_ids),
                "molecular_weight": m["molecular_weight"],
                "alogp": m["alogp"],
                "psa": m["psa"],
                "hba": m["hba"],
                "hbd": m["hbd"],
                "rtb": m["rtb"],
                "murcko_scaffold": m["murcko_scaffold"],
            })

        df = pd.DataFrame(records)
        if df.empty:
            self._log("WARNING: details after none details. ")
            return df

        # use inchi_key information (details chembl_id can information one result text)
        before = len(df)
        df = df.sort_values("n_records", ascending=False).drop_duplicates(
            subset="inchi_key", keep="first"
)
        after = len(df)
        if before!= after:
            self._log(f"  InChIKey details: {before} -> {after}")

        # details
        for lab in ["active", "inactive", "ambiguous", "conflict"]:
            cnt = (df["activity_label"] == lab).sum()
            self._log(f"  {lab:12s}: {cnt}")

        self.curated_df = df
        out_path = os.path.join(self.output_dir,
                                "cdk2_chembl_curated_molecule_level.csv")
        df.to_csv(out_path, index=False)
        self._log(f"  saved: {out_path}")
        return df

    # ─── Step 4: details ─────────────────────────────────────────────────
    def export_outputs(self, df: pd.DataFrame):
        """details active/inactive ISM, conflicts, ambiguous, summary JSON"""
        self._log_step(4, "summary file")

        if df.empty:
            self._log("WARNING: none data can details. ")
            return

        # active
        df_act = df[df["activity_label"] == "active"].copy()
        p = os.path.join(self.output_dir, "cdk2_chembl_external_active.ism")
        with open(p, "w", encoding="utf-8") as f:
            for _, r in df_act.iterrows():
                f.write(f"{r['canonical_smiles']} {r['chembl_id']}\n")
        self._log(f"  active   -> {p}  ({len(df_act)} molecules)")

        # inactive
        df_inact = df[df["activity_label"] == "inactive"].copy()
        p = os.path.join(self.output_dir, "cdk2_chembl_external_inactive.ism")
        with open(p, "w", encoding="utf-8") as f:
            for _, r in df_inact.iterrows():
                f.write(f"{r['canonical_smiles']} {r['chembl_id']}\n")
        self._log(f"  inactive -> {p}  ({len(df_inact)} molecules)")

        # conflict
        df_conf = df[df["activity_label"] == "conflict"].copy()
        p = os.path.join(self.output_dir, "cdk2_chembl_conflicts.csv")
        df_conf.to_csv(p, index=False)
        self._log(f"  conflict -> {p}  ({len(df_conf)} molecules)")

        # ambiguous
        df_amb = df[df["activity_label"] == "ambiguous"].copy()
        p = os.path.join(self.output_dir, "cdk2_chembl_ambiguous_midrange.csv")
        df_amb.to_csv(p, index=False)
        self._log(f"  ambiguous-> {p}  ({len(df_amb)} molecules)")

        # scaffold details
        act_scaffolds = df_act["murcko_scaffold"].nunique() if len(df_act) else 0
        inact_scaffolds = df_inact["murcko_scaffold"].nunique() if len(df_inact) else 0

        summary = {
            "target": "CDK2",
            "target_chembl_id": self.TARGET_CHEMBL_ID,
            "total_curated_molecules": int(len(df)),
            "active": int(len(df_act)),
            "inactive": int(len(df_inact)),
            "ambiguous": int(len(df_amb)),
            "conflict": int(len(df_conf)),
            "active_scaffolds": int(act_scaffolds),
            "inactive_scaffolds": int(inact_scaffolds),
            "active_threshold_nM": self.ACTIVE_THRESHOLD_NM,
            "inactive_threshold_nM": self.INACTIVE_THRESHOLD_NM,
            "active_pchembl_threshold": self.ACTIVE_PCHEMBL,
            "inactive_pchembl_threshold": self.INACTIVE_PCHEMBL,
            "active_median_pact_mean": (
                round(df_act["median_pactivity"].dropna().mean(), 3)
                if len(df_act) and df_act["median_pactivity"].dropna().any()
                else None
),
            "inactive_median_pact_mean": (
                round(df_inact["median_pactivity"].dropna().mean(), 3)
                if len(df_inact) and df_inact["median_pactivity"].dropna().any()
                else None
),
        }
        p = os.path.join(self.output_dir, "cdk2_chembl_label_summary.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        self._log(f"  summary  -> {p}")

        # summary
        self._log_header("Final Summary")
        self._log(f"Active: {len(df_act):5d} molecules, "
                   f"{act_scaffolds:4d} scaffolds")
        self._log(f"Inactive: {len(df_inact):5d} molecules, "
                   f"{inact_scaffolds:4d} scaffolds")
        self._log(f"Ambiguous: {len(df_amb):5d} molecules (not in main set)")
        self._log(f"Conflict: {len(df_conf):5d} molecules (removed)")
        if summary["active_median_pact_mean"]:
            self._log(f"Active   median pActivity mean: "
                       f"{summary['active_median_pact_mean']}")
        if summary["inactive_median_pact_mean"]:
            self._log(f"Inactive median pActivity mean: "
                       f"{summary['inactive_median_pact_mean']}")
        self._log("All inactive molecules are from CDK2 assay measured data.")

    # ─── information ────────────────────────────────────────────────────────
    def run(self):
        """run summary"""
        self._log_header("Start building CDK2 ChEMBL External Validation")
        t0 = time.time()

        df_raw = self.collect_raw_activities()
        if df_raw.empty:
            self._log("ERROR: No raw data collected. Exiting.")
            return None

        mol_meta = self.fetch_molecule_metadata(df_raw)
        if not mol_meta:
            self._log("ERROR: No molecule metadata. Exiting.")
            return None

        df_curated = self.curate_records(df_raw, mol_meta)
        if not df_curated.empty:
            self.export_outputs(df_curated)

        elapsed = time.time() - t0
        self._log(f"\nTotal time: {elapsed:.1f}s")
        self._log("Done!")
        return df_curated


def main():
    curator = CDK2ChEMBLCurator(output_dir="outputs/cdk2_chembl_external")
    curator.run()


if __name__ == "__main__":
    main()
