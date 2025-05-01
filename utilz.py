import os
import json
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from typing import List, Dict
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold


class ActUtilz:
    @staticmethod
    def hook(tensor, hook):
        hook.ctx['activation'] = tensor.detach().cpu().to(torch.float16)
        
    @staticmethod
    def quantize_activations(activations: torch.Tensor, output_precision: int = 8, channel_axis: int = 1) -> torch.Tensor:
        """
        Transforms an activation tensor from a high-resolution floating point representation (float32) 
        to a low resolution integer representation.
        :param activations: the relevant activation tensor.
        :param output_precision: the number of bits for the low-resolution representation.
        :param channel_axis: an integer represent the channel dimension. 
        In our pipeline it corresponds to the layer dimension.
        """
        activations = activations.to(torch.float32)

        # compute per-layer min/max over the other two dims (seq_len, d_model)
        min_vals = activations.amin(dim=(0, 2))
        max_vals = activations.amax(dim=(0, 2))

        # build scale & zero_point vectors
        n_levels = 2 ** output_precision
        scale = (max_vals - min_vals) / (n_levels - 1)
        zero_points = torch.round(-min_vals / scale)

        # quantize per channel
        return torch.quantize_per_channel(
            activations,
            scale,
            zero_points,
            axis=channel_axis,
            dtype=torch.qint8
        )


class DatasetUtilz:
    @staticmethod
    def load_dataset(dataset_path: str):
        """
        Loads a dataset from a .json file.
        """
        with open(dataset_path, 'r') as f:
            return json.load(f)

    @staticmethod
    def format_prompts(dataset: str, template_type: str) -> List[str]:
        templates = dataset[template_type]
        return [template.format(sample["subject"]) for sample in dataset["samples"] for template in templates]
    
    @staticmethod
    def get_labels_from_dataset(dataset_path: str, template_key: str) -> (np.ndarray, Dict):
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)

        all_objects = [sample["object"] for sample in dataset["samples"]]
        label_index_map = {cl: idx for idx, cl in enumerate(np.unique(all_objects))}
        print(f"Detected label mapping: {label_index_map}")
        
        labels = np.array([label_index_map[obj] for obj in all_objects])
        return labels, np.unique(all_objects)


class ProbeUtilz:
    def __init__(self, bin_threshold: float, kfolds: int):
        self.bin_threshold = bin_threshold
        self.cv = StratifiedKFold(n_splits=kfolds, shuffle=True)
    
    def run_probe(self, X: np.ndarray, y: np.ndarray, probe, binary: bool = False) -> (float, float):
        """
        Runs a specific probe on data matrix X and labels vector y, 
        then calculates accuracy and F1 scores using kfold-cross validation.
        :param X: the data matrix.
        :param y: the labels vector.
        :param probe: the probe model to fit and evaluate.
        :param binary: wether to binarize the data matrix using self.threshold.
        :return acc: the mean accuracy of the probe as a classifier on the folds.
        :return f1: the mean F1 score of the probe as a classifier on the folds.
        """
        X = (X > self.bin_threshold).astype(int) if binary else X
        preds = cross_val_predict(probe, X, y, cv=self.cv, n_jobs=-1)
        acc = accuracy_score(y, preds)
        f_1 = f1_score(y, preds, average="weighted")
        return acc, f_1

    def evaluate_single_neurons(self, label, layer: int, layer_acts: torch.Tensor, labels: np.ndarray, topk_neurons: np.ndarray
                                , probe, binary: bool = False) -> List[Dict]:
        results = [None] * topk_neurons.size
        for i, neuron in enumerate(topk_neurons):
            X = layer_acts[:, neuron].reshape(-1, 1)  # [num_samples, 1]
            acc, f1 = self.run_probe(X, labels, probe, binary=binary)
            results[i] = {"label": label
                        , "layer": layer, "neuron": i
                        , "accuracy": acc, "f1": f1
                        , "mode": binary, "method": 'Single Neuron'}
        return results

    def evaluate_layer(self, label, layer: int, layer_acts: torch.Tensor, labels: np.ndarray
                       , probe, binary: bool = False) -> Dict:
        acc, f1 = self.run_probe(layer_acts, labels, probe, binary=binary)
        return {"label": label
                        , "layer": layer, "neuron": np.nan
                        , "accuracy": acc, "f1": f1
                        , "mode": binary, "method": 'Full Layer'}

    def evaluate_topk_in_layer(self, label, layer: int, layer_acts: torch.Tensor, labels: np.ndarray, topk_neurons: np.ndarray
                               , probe, binary: bool = False) -> Dict:
        X = layer_acts[:, topk_neurons]  # [num_samples, k]
        acc, f1 = self.run_probe(X, labels, probe, binary=binary)
        return {"label": label
                        , "layer": layer, "neuron": np.nan
                        , "accuracy": acc, "f1": f1
                        , "mode": binary, "method": 'Top-K Neurons'}

    def probe_three_ways(self, activations: torch.Tensor, labels: np.ndarray, label_map: Dict, save_path: str, model_name: str
                         , sgl_clf, mtpl_clf, top_k: int, probe_type: str, output_folder: str):
        unique_labels = np.unique(labels)
        num_prompts, num_layers, hidden_size = activations.shape
                
        # Preallocate list for layers, labels and binarization (True \ False)
        result_tables = []
        for layer in tqdm(range(num_layers), desc="Probing Layers"):
        
            layer_acts = activations[:, layer, :].view(activations.shape[0], -1).numpy()    
            for label in unique_labels:
                y_binary = (labels == label).astype(int)

                # Find topk neurons for label
                pos_acts = layer_acts[y_binary].mean(axis=0)
                neg_acts = layer_acts[~y_binary].mean(axis=0)
                delta = np.abs(pos_acts - neg_acts)
                topk_indices = np.argsort(delta)[-top_k:]
                
                # Probe using all difference schemes
                for binary in [True, False]:
                    single_neurons = self.evaluate_single_neurons(label, layer, layer_acts, y_binary
                                                                  , topk_indices, probe=sgl_clf, binary=binary)
                    result_tables.extend(single_neurons)
                    
                    topk_layer = self.evaluate_topk_in_layer(label, layer, layer_acts, y_binary
                                                             , topk_indices, probe=mtpl_clf, binary=binary)
                    result_tables.append(topk_layer)
                    
                    full_layer = self.evaluate_layer(label, layer, layer_acts, y_binary
                                                     , probe=mtpl_clf, binary=binary)
                    result_tables.append(full_layer)
        
        results = pd.DataFrame(result_tables)
        results['label'] = results['label'].apply(lambda label: label_map[label])
        results['mode'] = results['mode'].apply(lambda mode: 'Binary' if mode else 'Continuous')
        results.to_csv(f'{output_folder}{model_name}_{probe_type}_probing_results.csv', index=False)
        return results

class Report:
    def plot_per_method(pdf, data: pd.DataFrame, metric: str):
        for method in data["method"].unique():
            subset = data.loc[data["method"] == method]
            plt.figure(figsize=(10, 5))
            sns.lineplot(data=subset, x="layer", y=metric, hue="mode", markers=True, style="mode", errorbar="se")
            plt.title(f"{metric.title()} per Layer – {method} Probing")
            plt.xlabel("Layer")
            plt.ylabel(f"{metric.title()} [SE]")
            plt.grid(True)
            plt.tight_layout()
            plt.legend(title="Mode")
            pdf.savefig()
            plt.close()
    
    def get_summary_table(data: pd.DataFrame, metric: str) -> pd.DataFrame:
        summary = data.groupby(["method", "mode"]).agg(
            mean=(metric, "mean"),
            std=(metric, "std"),
            max=(metric, "max")
        )
        
        layer_means = data.groupby(["method", "mode", "layer"])[metric].mean().reset_index(name=f"layer_mean_{metric}")
        indices = layer_means.groupby(["method", "mode"])[f"layer_mean_{metric}"].idxmax()
        best_layers = layer_means.loc[indices, ["method", "mode", "layer"]]
        best_layers = best_layers.rename(columns={"layer": "best_layer"})
        summary = summary.merge(best_layers, on=["method", "mode"])
        
        comp_table = summary.reset_index().melt(
            id_vars=["method", "mode"],
            value_vars=["mean", "std", "max", "best_layer"],
            var_name="metric",
            value_name="score"
            ).pivot_table(
                index=["method", "metric"],
                columns="mode",
                values="score"
            ).reset_index()

        comp_table['metric'] = f'{metric}_' + comp_table['metric']
        comp_table = comp_table.round(4)        
        
        comp_table.columns = [col.title() for col in comp_table.columns]
        return comp_table
        
    def plot_summary(pdf, data: pd.DataFrame, metric: str):
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=data, x="layer", y=metric, hue="method", style="mode", markers=True, errorbar="se")
        plt.title(f"Overall {metric.title()} per Layer – All Methods")
        plt.xlabel("Layer")
        plt.ylabel(f"{metric.title()} Score [SE]")
        plt.grid(True)
        plt.tight_layout()
        plt.legend(title="Probing Method / Mode")
        pdf.savefig()
        plt.close()

        summary = Report.get_summary_table(data, metric)

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.axis("off")
        table = ax.table(
            cellText=summary.values,
            colLabels=summary.columns,
            loc="center",
            cellLoc='center'
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.1, 1.1)
        plt.title(f"{metric.title()} Score Summary by Probing Method")
        pdf.savefig()
        plt.close()
    
    def generate_report(output_folder: str, model_name: str, probe_type: str):
        pdf_path = f'{output_folder}{model_name}_{probe_type}_f1_acc_report.pdf'
        data = pd.read_csv(f'{output_folder}{model_name}_{probe_type}_probing_results.csv')
        with PdfPages(pdf_path) as pdf:
            for metric in ['f1', 'accuracy']:
                Report.plot_per_method(pdf, data, metric)
                Report.plot_summary(pdf, data, metric)

        print(f"Extended PDF report with accuracy saved to: {pdf_path}")
