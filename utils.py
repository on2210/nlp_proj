import pickle
import json
import torch
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from collections import defaultdict
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.backends.backend_pdf import PdfPages

SINGLE_CONT = "_neurons_cont.csv"
SINGLE_BINARY = "_neurons_bin.csv"
TOPK_CONT = "_topk_cont.csv"
TOPK_BINARY = "_topk_bin.csv"
FULL_CONT = "_full_cont.csv"
FULL_BINARY = "_full_bin.csv"

def load_my_dataset(dataset_path):
    with open(dataset_path, 'r') as f:
        return json.load(f)


def format_prompts(dataset, template_type="prompt_templates"):
    templates = dataset[template_type]
    return [template.format(sample["subject"]) for sample in dataset["samples"] for template in templates]


def load_activations(pickle_path):
    with open(pickle_path, "rb") as f:
        return pickle.load(f)  # shape: [num_prompts, seq_len, num_layers, hidden_size]


def inspect_pickle(pickle_path, num_words=5, num_layers=2):
    with open(pickle_path, "rb") as f:
        activations = pickle.load(f)

    print(f"Loaded activations from: {pickle_path}")
    print(f"Shape of tensor: {activations.shape}")  # [num_prompts, seq_len, num_layers, hidden_size]

    num_prompts, seq_len, total_layers, hidden_size = activations.shape
    print(f"\nNumber of prompts: {num_prompts}")
    print(f"Sequence length (padded): {seq_len}")
    print(f"Number of layers: {total_layers}")
    print(f"Hidden size: {hidden_size}")

    for i in range(min(num_prompts, 1)):
        print(f"\n--- Prompt #{i} ---")
        for t in range(min(seq_len, num_words)):
            print(f"Token {t}:")
            for l in range(min(total_layers, num_layers)):
                act = activations[i, t, l]
                print(f"  Layer {l}: Mean={act.mean():.4f}, Std={act.std():.4f}, Max={act.max():.4f}")


def extract_last_token_activations(activations):
    """
    Takes the activations of the last token (non-padded) for each prompt.
    """
    last_token_activations = []
    for a in activations:
        # Find last non-zero token row (assuming zero-padding)
        lengths = (a.abs().sum(dim=(1, 2)) > 0).nonzero()
        last_idx = lengths[-1].item() if lengths.numel() > 0 else -1
        last_token_activations.append(a[last_idx])
    return torch.stack(last_token_activations)  # shape: [num_prompts, num_layers, hidden_size]


def run_probe(X, y, model, binary=False, t=0, folds=10):
    if binary:
        X = (X > t).float()
    
    X = X.view(X.shape[0], -1).numpy()  # flatten to [num_prompts, num_layers * hidden_size]
    y = np.array(y)
    
    cv = StratifiedKFold(n_splits=folds, shuffle=True)
    preds = cross_val_predict(model, X, y, cv=cv)
    
    acc = accuracy_score(y, preds)
    f_1 = f1_score(y, preds, average="weighted")
    return acc, f_1


def get_labels_from_dataset(dataset_path, template_key="prompt_templates"):
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    objects_expanded = [sample["object"] for sample in dataset["samples"]]

    unique_classes = sorted(set(objects_expanded))

    label_map = {cls: idx for idx, cls in enumerate(unique_classes)}
    labels = [label_map[obj] for obj in objects_expanded]

    print(f"Detected label mapping: {label_map}")
    elements, counts = np.unique(labels, return_counts=True)
    return labels


def evaluate_single_neurons(layer_acts: torch.Tensor, labels: List[int], binary=False, classifier=LogisticRegression):
    results = []
    for i in range(layer_acts.shape[1]):
        X = layer_acts[:, i].unsqueeze(1)  # shape [num_samples, 1]
        acc, f1 = run_probe(X, labels, classifier, binary=binary)
        results.append({"neuron": i, "accuracy": acc, "f1": f1})
    return results


def evaluate_layer(layer_acts: torch.Tensor, labels: List[int], binary=False, classifier=LogisticRegression):
    acc, f1 = run_probe(layer_acts, labels, classifier, binary=binary)
    return {"accuracy": acc, "f1": f1}


def evaluate_topk_in_layer(layer_acts: torch.Tensor, labels: List[int], topk_neurons: List[int], binary=False, classifier=LogisticRegression):
    X = layer_acts[:, topk_neurons]  # shape [num_samples, k]
    acc, f1 = run_probe(X, labels, classifier, binary=binary)
    return {"accuracy": acc, "f1": f1, "topk_neurons": topk_neurons}


def save_probing_results(results, base_path, top_k, model_name):
    def flatten_dict_to_df(d, key_name="layer"):
        rows = []
        for k, lst in d.items():
            for row in lst:
                rows.append({key_name: k, **row})
        return pd.DataFrame(rows)

    # 1. Single neurons
    df_neurons_cont = flatten_dict_to_df(results["single_neuron"]["cont"])
    df_neurons_bin = flatten_dict_to_df(results["single_neuron"]["bin"])

    df_neurons_cont.sort_values(by="f1", ascending=False, inplace=True)
    df_neurons_bin.sort_values(by="f1", ascending=False, inplace=True)

    # 2. Top-k probing
    df_topk_cont = pd.DataFrame([
        {"layer": l, "accuracy": v["accuracy"], "f1": v["f1"], "topk_neurons": v["topk_neurons"]}
        for l, v in results["topk_layer"]["cont"].items()
    ])
    df_topk_bin = pd.DataFrame([
        {"layer": l, "accuracy": v["accuracy"], "f1": v["f1"], "topk_neurons": v["topk_neurons"]}
        for l, v in results["topk_layer"]["bin"].items()
    ])

    # 3. Full layer
    df_full_cont = pd.DataFrame([
        {"layer": l, **v} for l, v in results["full_layer"]["cont"].items()
    ])
    df_full_bin = pd.DataFrame([
        {"layer": l, **v} for l, v in results["full_layer"]["bin"].items()
    ])

    df_neurons_cont.to_csv(base_path + model_name + SINGLE_CONT, index=False)
    df_neurons_bin.to_csv(base_path + model_name + SINGLE_BINARY, index=False)
    df_topk_cont.to_csv(base_path + model_name + TOPK_CONT, index=False)
    df_topk_bin.to_csv(base_path + model_name + TOPK_BINARY, index=False)
    df_full_cont.to_csv(base_path + model_name + FULL_CONT, index=False)
    df_full_bin.to_csv(base_path + model_name + FULL_BINARY, index=False)


def probe_three_ways(last_token_acts, labels, save_path, model_name, sgl_clf, mtpl_clf, top_k=10):
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    num_prompts, num_layers, hidden_size = last_token_acts.shape

    results = {
        "single_neuron": {"cont": defaultdict(list), "bin": defaultdict(list)},
        "full_layer": {"cont": {}, "bin": {}},
        "topk_layer": {"cont": {}, "bin": {}}
    }

    for layer in range(num_layers):
        layer_acts = last_token_acts[:, layer, :]  # shape: [num_prompts, hidden_size]
        topk_overall = []
        for label in unique_labels:
            y_binary = (labels == label).astype(int)

            pos_acts = layer_acts[y_binary == 1]
            neg_acts = layer_acts[y_binary == 0]
            delta = np.abs(pos_acts.mean(axis=0) - neg_acts.mean(axis=0))

            topk_indices = np.argsort(delta)[-top_k:]
            topk_overall.extend(topk_indices.tolist())

            for neuron_idx in topk_indices:
                single_neuron_acts = layer_acts[:, neuron_idx].reshape(-1, 1)
                acc, f1 = run_probe(single_neuron_acts, y_binary, sgl_clf, binary=False)
                results["single_neuron"]["cont"][layer].append({
                    "neuron": neuron_idx,
                    "f1": f1,
                    "accuracy": acc,
                    "layer": layer,
                    "label": label
                })
                acc, f1 = run_probe(single_neuron_acts, y_binary, sgl_clf, binary=True)
                results["single_neuron"]["bin"][layer].append({
                    "neuron": neuron_idx,
                    "f1": f1,
                    "accuracy": acc,
                    "layer": layer,
                    "label": label
                })

        """
        # Continuous activations
        single_cont = evaluate_single_neurons(layer_acts, labels, binary=False, classifier=classifier)
        results["single_neuron"]["cont"][layer] = sorted(single_cont, key=lambda x: x["f1"], reverse=True)

        # Binary activations
        single_bin = evaluate_single_neurons(layer_acts, labels, binary=True, classifier=classifier)
        results["single_neuron"]["bin"][layer] = sorted(single_bin, key=lambda x: x["f1"], reverse=True)
        
        """

        # Whole layer probing
        results["full_layer"]["cont"][layer] = evaluate_layer(layer_acts, labels, binary=False, classifier=mtpl_clf)
        results["full_layer"]["bin"][layer] = evaluate_layer(layer_acts, labels, binary=True, classifier=mtpl_clf)

        # Top-k probing per layer

        results["topk_layer"]["cont"][layer] = evaluate_topk_in_layer(layer_acts, labels, topk_overall, binary=False,
                                                                      classifier=mtpl_clf)
        results["topk_layer"]["bin"][layer] = evaluate_topk_in_layer(layer_acts, labels, topk_overall, binary=True,
                                                                     classifier=mtpl_clf)

    # Save to disk
    if save_path:
        save_probing_results(results, save_path, top_k, model_name)

    return results


def generate_report(base_path: str, model_name: str):
    pdf_path = base_path + model_name + "_f1_acc_report.pdf"
    files = {
        "topk_cont": base_path + model_name + TOPK_CONT,
        "topk_bin": base_path + model_name + TOPK_BINARY,
        "single_cont": base_path + model_name + SINGLE_CONT,
        "single_bin": base_path + model_name + SINGLE_BINARY,
        "full_cont": base_path + model_name + FULL_CONT,
        "full_bin": base_path + model_name + FULL_BINARY,
    }

    f1_records = []
    acc_records = []

    for key, path in files.items():
        if not os.path.exists(path):
            print(f"Missing file: {path}")
            continue

        df = pd.read_csv(path)

        if "topk" in key or "single" in key:
            df_grouped_f1 = df.groupby("layer")["f1"].mean().reset_index()
            df_grouped_acc = df.groupby("layer")["accuracy"].mean().reset_index()
        else:
            df_grouped_f1 = df.copy()
            df_grouped_acc = df.copy()
            if "layer" not in df_grouped_f1.columns:
                df_grouped_f1["layer"] = df_grouped_f1.index
                df_grouped_acc["layer"] = df_grouped_acc.index
            df_grouped_f1 = df_grouped_f1[["layer", "f1"]]
            df_grouped_acc = df_grouped_acc[["layer", "accuracy"]]

        method = "Top-K" if "topk" in key else "Single" if "single" in key else "Full Layer"
        mode = "Binary" if "bin" in key else "Continuous"

        df_grouped_f1["method"] = method
        df_grouped_f1["mode"] = mode
        df_grouped_acc["method"] = method
        df_grouped_acc["mode"] = mode

        f1_records.append(df_grouped_f1)
        acc_records.append(df_grouped_acc)

    f1_df = pd.concat(f1_records)
    acc_df = pd.concat(acc_records)

    with PdfPages(pdf_path) as pdf:
        # F1 Graphs
        for method in f1_df["method"].unique():
            subset = f1_df[f1_df["method"] == method]
            plt.figure(figsize=(10, 5))
            sns.lineplot(data=subset, x="layer", y="f1", hue="mode", markers=True, style="mode")
            plt.title(f"F1 per Layer – {method} Probing")
            plt.xlabel("Layer")
            plt.ylabel("F1 Score")
            plt.grid(True)
            plt.tight_layout()
            plt.legend(title="Mode")
            pdf.savefig()
            plt.close()

        # F1 Summary
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=f1_df, x="layer", y="f1", hue="method", style="mode", markers=True)
        plt.title("Overall F1 per Layer – All Methods")
        plt.xlabel("Layer")
        plt.ylabel("F1 Score")
        plt.grid(True)
        plt.tight_layout()
        plt.legend(title="Probing Method / Mode")
        pdf.savefig()
        plt.close()

        f1_summary = f1_df.groupby(["method", "mode"]).agg(
            mean_f1=("f1", "mean"),
            max_f1=("f1", "max"),
            best_layer=("f1", lambda x: x.idxmax())
        ).reset_index()

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.axis("off")
        table = ax.table(
            cellText=f1_summary.values,
            colLabels=f1_summary.columns,
            loc="center",
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        plt.title("F1 Score Summary by Probing Method")
        pdf.savefig()
        plt.close()

        # Accuracy Graphs
        for method in acc_df["method"].unique():
            subset = acc_df[acc_df["method"] == method]
            plt.figure(figsize=(10, 5))
            sns.lineplot(data=subset, x="layer", y="accuracy", hue="mode", markers=True, style="mode")
            plt.title(f"Accuracy per Layer – {method} Probing")
            plt.xlabel("Layer")
            plt.ylabel("Accuracy")
            plt.grid(True)
            plt.tight_layout()
            plt.legend(title="Mode")
            pdf.savefig()
            plt.close()

        # Accuracy Summary
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=acc_df, x="layer", y="accuracy", hue="method", style="mode", markers=True)
        plt.title("Overall Accuracy per Layer – All Methods")
        plt.xlabel("Layer")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.tight_layout()
        plt.legend(title="Probing Method / Mode")
        pdf.savefig()
        plt.close()

        acc_summary = acc_df.groupby(["method", "mode"]).agg(
            mean_acc=("accuracy", "mean"),
            max_acc=("accuracy", "max"),
            best_layer=("accuracy", lambda x: x.idxmax())
        ).reset_index()

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.axis("off")
        table = ax.table(
            cellText=acc_summary.values,
            colLabels=acc_summary.columns,
            loc="center",
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        plt.title("Accuracy Summary by Probing Method")
        pdf.savefig()
        plt.close()

    print(f"Extended PDF report with accuracy saved to: {pdf_path}")
