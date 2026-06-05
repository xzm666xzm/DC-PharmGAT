#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train and evaluate one DC-PharmGAT model configuration."""

from __future__ import annotations

import argparse
import copy
import os
import pickle
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from dc_pharmgat import dockingDataset, dockingProtocol
from features import num_atom_features


HP_RESULTS_HEADER = (
    "model number,oversampled size,batch size,learning rate,dropout rate,gfe threshold,"
    "fingerprint length,validation auc,validation prauc,validation precision,"
    "validation recall,validation f1,validation hits,tr,test auc,test prauc,"
    "test precision,test recall,test f1,test hits,avg gfe,t1enrichment,"
    "t5enrichment,t10enrichment,t50enrichment,t100enrichment\n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a single DC-PharmGAT model.")
    parser.add_argument("-dropout", "--df", required=True, type=float)
    parser.add_argument("-learn_rate", "--lr", required=True, type=float)
    parser.add_argument("-os", "--os", required=True, type=int)
    parser.add_argument("-protein", "--pro", required=True)
    parser.add_argument("-bs", "--batch_size", required=True, type=int)
    parser.add_argument("-fplen", "--fplength", required=True, type=int)
    parser.add_argument("-mnum", "--model_number", required=True)
    parser.add_argument("-wd", "--weight_decay", required=True, type=float)
    parser.add_argument("-dataset", "--d", required=True, choices=["dude", "lit-pcba", "scaffold", "normal"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_tpp_bias", action="store_true", default=False)
    parser.add_argument("--hard_concat_tpp", action="store_true", default=False)
    parser.add_argument("--fixed_bias", action="store_true", default=False)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_ism(path: Path, label: int | None = None) -> list[dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for index, line in enumerate(handle, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            smiles = parts[0]
            molecule_id = parts[2] if len(parts) >= 3 else (parts[1] if len(parts) >= 2 else str(index))
            parsed_label = label
            if parsed_label is None:
                raw_label = parts[3] if len(parts) >= 4 else parts[-1]
                parsed_label = 1 if str(raw_label).lower() in {"1", "active", "actives"} else 0
            rows.append({"id": str(molecule_id), "smile": smiles, "label": int(parsed_label), "score": int(parsed_label)})
    return rows


def split_rows(rows: list[dict[str, object]], seed: int) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    df = pd.DataFrame(rows)
    stratify = df["label"] if df["label"].nunique() > 1 else None
    train_df, holdout_df = train_test_split_df(df, test_size=0.30, seed=seed, stratify=stratify)
    holdout_stratify = holdout_df["label"] if holdout_df["label"].nunique() > 1 else None
    valid_df, test_df = train_test_split_df(holdout_df, test_size=0.50, seed=seed, stratify=holdout_stratify)
    return (
        train_df.to_dict("records"),
        valid_df.to_dict("records"),
        test_df.to_dict("records"),
    )


def train_test_split_df(df: pd.DataFrame, test_size: float, seed: int, stratify=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    return train_test_split(df, test_size=test_size, random_state=seed, shuffle=True, stratify=stratify)


def load_dude_dataset(run_dir: Path, protein: str, seed: int):
    protein_dir = run_dir.parent / "dude" / protein
    rows = parse_ism(protein_dir / "actives_final.ism", 1)
    rows.extend(parse_ism(protein_dir / "decoys_final.ism", 0))
    return split_rows(rows, seed)


def load_litpcba_dataset(run_dir: Path, protein: str, seed: int):
    protein_dir = run_dir.parent / "lit-pcba" / protein.upper()
    rows = parse_ism(protein_dir / "actives.smi", 1)
    rows.extend(parse_ism(protein_dir / "inactives.smi", 0))
    return split_rows(rows, seed)


def load_scaffold_dataset(run_dir: Path, protein: str, seed: int, max_decoy_ratio: int = 50):
    protein_dir = run_dir.parent / "dude" / protein
    train_rows = parse_ism(protein_dir / "train.ism")
    valid_rows = parse_ism(protein_dir / "valid.ism")
    test_rows = parse_ism(protein_dir / "test.ism")

    active_rows = [row for row in train_rows if row["label"] == 1]
    decoy_rows = [row for row in train_rows if row["label"] == 0]
    max_decoys = min(len(decoy_rows), max_decoy_ratio * max(1, len(active_rows)))
    rng = random.Random(seed)
    if len(decoy_rows) > max_decoys:
        decoy_rows = rng.sample(decoy_rows, max_decoys)
    train_rows = active_rows + decoy_rows
    rng.shuffle(train_rows)
    return train_rows, valid_rows, test_rows


def rows_to_model_input(rows: list[dict[str, object]]) -> tuple[list[list[str]], list[int], pd.DataFrame]:
    features = [[str(row["id"]), str(row["smile"])] for row in rows]
    labels = [int(row["label"]) for row in rows]
    frame = pd.DataFrame(rows)[["id", "score", "label"]].rename(columns={"score": "labels"})
    return features, labels, frame


def find_tpp_path(run_dir: Path, protein: str) -> str | None:
    candidates = [
        run_dir / f"{protein}_tpp.pt",
        run_dir.parent / protein / f"{protein}_tpp.pt",
        Path.cwd() / f"{protein}_tpp.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def build_model_params(args: argparse.Namespace) -> dict[str, object]:
    hidden_features = [64] * 4
    layers = [num_atom_features()] + hidden_features
    return {
        "fpl": args.fplength,
        "batchsize": args.batch_size,
        "use_pharmacophore_gat": True,
        "use_dual_channel": True,
        "num_heads": 4,
        "aggregation_type": "attention",
        "enable_spcl": True,
        "spcl_weight": 0.01,
        "spcl_proj_dim": 64,
        "spcl_temperature": 0.2,
        "disable_tpp_bias": args.disable_tpp_bias,
        "hard_concat_tpp": args.hard_concat_tpp,
        "fixed_bias": args.fixed_bias,
        "conv": {"layers": layers, "activations": False},
        "ann": {
            "layers": layers,
            "ba": [args.fplength, args.fplength // 4, args.fplength // 8, 1],
            "dropout": args.df,
        },
    }


def make_loader(rows, labels, batch_size, name, shuffle):
    dataset = dockingDataset(train=rows, labels=labels, name=name)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def class_pos_weight(labels: list[int], device: str) -> torch.Tensor:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return torch.tensor([1.0], device=device)
    return torch.tensor([negatives / positives], device=device)


def predict(model, loader, device: str) -> tuple[list[int], list[float], list[str]]:
    labels: list[int] = []
    scores: list[float] = []
    ids: list[str] = []
    model.eval()
    with torch.no_grad():
        for atoms, bonds, edges, (batch_labels, batch_ids) in loader:
            logits = model((atoms.to(device), bonds.to(device), edges.to(device)))
            probabilities = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1).tolist()
            labels.extend([int(x) for x in batch_labels.numpy().reshape(-1).tolist()])
            scores.extend([float(x) for x in probabilities])
            ids.extend([str(x) for x in batch_ids])
    return labels, scores, ids


def best_threshold(labels: list[int], scores: list[float]) -> float:
    thresholds = np.arange(0.0, 1.0, 0.001)
    if len(set(labels)) < 2:
        return 0.5
    values = [fbeta_score(labels, (np.array(scores) >= t).astype(int), beta=1.75, zero_division=0) for t in thresholds]
    return float(thresholds[int(np.argmax(values))])


def binary_metrics(labels: list[int], scores: list[float], threshold: float) -> dict[str, object]:
    predictions = (np.array(scores) >= threshold).astype(int)
    if len(set(labels)) < 2:
        fpr, tpr, roc_thresholds = np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([threshold])
        precision_curve, recall_curve = np.array([sum(labels) / max(1, len(labels))]), np.array([1.0])
        roc_auc = 0.0
        pr_auc = 0.0
    else:
        fpr, tpr, roc_thresholds = roc_curve(labels, scores)
        precision_curve, recall_curve, _ = precision_recall_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        pr_auc = average_precision_score(labels, scores)

    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc),
        "prauc": float(pr_auc),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "hits": int(sum(labels)),
        "confusion": (int(tn), int(fp), int(fn), int(tp)),
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": roc_thresholds,
        "precision_curve": precision_curve,
        "recall_curve": recall_curve,
    }


def enrichment_values(labels: list[int], scores: list[float], percentages=(1, 5, 10, 50, 100)) -> list[float]:
    total = len(labels)
    total_actives = sum(labels)
    if total == 0 or total_actives == 0:
        return [0.0 for _ in percentages]
    baseline_rate = total_actives / total
    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    values = []
    for pct in percentages:
        selected_count = max(1, int(np.ceil(total * pct / 100.0)))
        selected = ranked[:selected_count]
        selected_rate = sum(label for _, label in selected) / selected_count
        values.append(float(selected_rate / baseline_rate))
    return values


def save_curves(model_dir: Path, metrics: dict[str, object]) -> None:
    plt.figure()
    plt.plot(metrics["fpr"], metrics["tpr"], label=f"ROC AUC = {metrics['auc']:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(model_dir / "rocCurve.png")
    plt.close()

    plt.figure()
    plt.plot(metrics["recall_curve"], metrics["precision_curve"], label=f"PR AUC = {metrics['prauc']:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(model_dir / "prCurve.png")
    plt.close()


def ensure_result_file(run_dir: Path) -> None:
    hp_file = run_dir / "hpResults.csv"
    if not hp_file.exists():
        hp_file.write_text(HP_RESULTS_HEADER, encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    run_dir = Path.cwd().parent
    model_dir = run_dir / "res" / f"model{args.model_number}"
    weights_dir = run_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    ensure_result_file(run_dir)

    if args.d == "dude":
        train_rows, valid_rows, test_rows = load_dude_dataset(run_dir, args.pro, args.seed)
    elif args.d == "lit-pcba":
        train_rows, valid_rows, test_rows = load_litpcba_dataset(run_dir, args.pro, args.seed)
    elif args.d == "scaffold":
        train_rows, valid_rows, test_rows = load_scaffold_dataset(run_dir, args.pro, args.seed)
    else:
        raise ValueError("The 'normal' dataset mode is not supported by this clean training entry point.")

    x_train, y_train, _ = rows_to_model_input(train_rows)
    x_valid, y_valid, _ = rows_to_model_input(valid_rows)
    x_test, y_test, test_frame = rows_to_model_input(test_rows)

    print(f"dataset: {args.d}")
    print(f"train/valid/test: {len(x_train)}/{len(x_valid)}/{len(x_test)}")
    print(f"device: {device}")
    print(f"hyperparameters: dropout={args.df}, lr={args.lr}, weight_decay={args.weight_decay}, batch={args.batch_size}")

    train_loader = make_loader(x_train, y_train, args.batch_size, "train", shuffle=True)
    valid_loader = make_loader(x_valid, y_valid, args.batch_size, "valid", shuffle=False)
    test_loader = make_loader(x_test, y_test, args.batch_size, "test", shuffle=False)

    tpp_path = find_tpp_path(run_dir, args.pro)
    if tpp_path:
        print(f"TPP path: {tpp_path}")
    else:
        print("TPP file not found; model will use default pharmacophore prior.")

    model_params = build_model_params(args)
    with (model_dir / "modelparams.pkl").open("wb") as handle:
        pickle.dump(model_params, handle)

    model = dockingProtocol(model_params, tpp_path=tpp_path).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_pos_weight(y_train, device))

    best_state = copy.deepcopy(model.state_dict())
    best_valid_loss = float("inf")
    train_losses: list[float] = []
    valid_losses: list[float] = []
    patience = 15
    stale_epochs = 0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for atoms, bonds, edges, (labels, _) in train_loader:
            labels = labels.to(device)
            logits = model((atoms.to(device), bonds.to(device), edges.to(device)))
            loss = loss_fn(logits, labels)
            if hasattr(model, "compute_contrastive_loss") and epoch >= 10:
                warmup = min(1.0, (epoch - 10) / 10)
                loss = loss + warmup * model.compute_contrastive_loss((atoms.to(device), bonds.to(device), edges.to(device)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(labels)
            seen += len(labels)

        train_loss = running_loss / max(1, seen)
        train_losses.append(train_loss)

        model.eval()
        valid_loss_sum = 0.0
        valid_seen = 0
        with torch.no_grad():
            for atoms, bonds, edges, (labels, _) in valid_loader:
                labels = labels.to(device)
                logits = model((atoms.to(device), bonds.to(device), edges.to(device)))
                valid_loss_sum += float(loss_fn(logits, labels).item()) * len(labels)
                valid_seen += len(labels)
        valid_loss = valid_loss_sum / max(1, valid_seen)
        valid_losses.append(valid_loss)
        print(f"epoch {epoch:03d}: train_loss={train_loss:.5f}, valid_loss={valid_loss:.5f}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"early stopping at epoch {epoch}")
                break

    weight_path = weights_dir / f"model{args.model_number}.pth"
    torch.save(best_state, weight_path)
    model.load_state_dict(best_state, strict=False)

    y_valid_eval, valid_scores, _ = predict(model, valid_loader, device)
    threshold = best_threshold(y_valid_eval, valid_scores)
    valid_metrics = binary_metrics(y_valid_eval, valid_scores, threshold)

    y_test_eval, test_scores, test_ids = predict(model, test_loader, device)
    test_metrics = binary_metrics(y_test_eval, test_scores, threshold)
    enrich = enrichment_values(y_test_eval, test_scores)

    print(
        "validation: "
        f"auc={valid_metrics['auc']:.4f}, prauc={valid_metrics['prauc']:.4f}, "
        f"precision={valid_metrics['precision']:.4f}, recall={valid_metrics['recall']:.4f}, f1={valid_metrics['f1']:.4f}"
    )
    print(
        "test: "
        f"auc={test_metrics['auc']:.4f}, prauc={test_metrics['prauc']:.4f}, "
        f"precision={test_metrics['precision']:.4f}, recall={test_metrics['recall']:.4f}, f1={test_metrics['f1']:.4f}"
    )

    pd.DataFrame({"train_loss": train_losses, "valid_loss": valid_losses}).to_csv(model_dir / "lossData.txt", index=False)
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
    plt.plot(range(1, len(valid_losses) + 1), valid_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(model_dir / "loss.png")
    plt.close()

    with (model_dir / "testset.txt").open("w", encoding="utf-8") as handle:
        handle.write("id,gfe,label\n")
        for row in test_frame.to_dict("records"):
            handle.write(f"{row['id']},{row['labels']},{row['label']}\n")

    with (model_dir / "test_output.txt").open("w", encoding="utf-8") as handle:
        handle.write("zid,output\n")
        for molecule_id, score in sorted(zip(test_ids, test_scores), key=lambda item: item[1], reverse=True):
            handle.write(f"{molecule_id},{score}\n")

    tn, fp, fn, tp = test_metrics["confusion"]
    with (model_dir / "miscData.txt").open("w", encoding="utf-8") as handle:
        handle.write("AUC data (thresholds, fpr, tpr)\n")
        handle.write(",".join(map(str, test_metrics["roc_thresholds"])) + "\n")
        handle.write(",".join(map(str, test_metrics["fpr"])) + "\n")
        handle.write(",".join(map(str, test_metrics["tpr"])) + "\n")
        handle.write("prAUC data (rec, prec)\n")
        handle.write(",".join(map(str, test_metrics["recall_curve"])) + "\n")
        handle.write(",".join(map(str, test_metrics["precision_curve"])) + "\n")
        handle.write("confusion matrix (tn, fp, fn, tp)\n")
        handle.write(f"{tn},{fp},{fn},{tp}\n")

    save_curves(model_dir, test_metrics)

    avg_score = float(np.mean(test_scores)) if test_scores else 0.0
    with (run_dir / "hpResults.csv").open("a", encoding="utf-8") as handle:
        handle.write(
            f"{args.model_number},{args.os},{args.batch_size},{args.lr},{args.df},0,{args.fplength},"
            f"{valid_metrics['auc']},{valid_metrics['prauc']},{valid_metrics['precision']},"
            f"{valid_metrics['recall']},{valid_metrics['f1']},{valid_metrics['hits']},{threshold},"
            f"{test_metrics['auc']},{test_metrics['prauc']},{test_metrics['precision']},"
            f"{test_metrics['recall']},{test_metrics['f1']},{test_metrics['hits']},{avg_score},"
            f"{enrich[0]},{enrich[1]},{enrich[2]},{enrich[3]},{enrich[4]}\n"
        )

    print(f"saved model to {weight_path}")
    print(f"elapsed minutes: {(time.time() - start) / 60:.1f}")


if __name__ == "__main__":
    main()
