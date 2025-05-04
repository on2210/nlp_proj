LLM Probing Pipeline

This repository provides a reproducible pipeline for probing neural activations of various Pythia language models using classification probes. It computes and caches model activations on a set of textual datasets, runs single-neuron and layer-wise probes (logistic regression or decision tree), and generates detailed reports with F1 and accuracy metrics.

Repository Structure

├── data/                   # Directory for JSON datasets
│   ├── landmarks_450.json
│   ├── centuries_300.json
│   ├── music.json
│   ├── europarl_lang.json
│   └── pile_data_source.json
├── landmarks/              # Example output folder after running on `landmarks`
│   ├── activations/        # Cached activations per model
│   ├── pythia-70m_lr_..._probing_results.csv
│   └── pythia-70m_lr_..._f1_acc_report.pdf
├── main.py                 # Entry point: parses args and launches Pipeline
├── pipeline.py             # Core logic: activation extraction, probing, reporting
├── utilz.py                # Utility modules: hooks, dataset loading, probing, reporting
└── requirements.txt        # Python dependencies

Prerequisites

Python 3.8+

CUDA-enabled GPU (optional; falls back to CPU)

Virtual environment (recommended)



Datasets

Place your dataset JSON files in the data/ directory. Supported names (under --dataset) and corresponding JSON files:

landmarks → landmarks_450.json

centuries → centuries_300.json

music → music.json

europarl_lang → europarl_lang.json (tokenized, global task)

pile_data_source → pile_data_source.json (tokenized, global task)

If your dataset is already tokenized, provide a <dataset>_tokens.pt file alongside the JSON.

Usage

Run the pipeline via main.py:

python main.py \
  --n_models N          # Number of Pythia models to probe (1–7)
  --dataset NAME        # One of [landmarks, centuries, music, europarl_lang, pile_data_source]
  --probe_type {lr,dt}  # `lr` for logistic regression, `dt` for decision tree
  --top_k K             # Top-K neurons to include in multi-neuron probe
  --max_iter M          # Max iterations for logistic regression (default 1000)
  --max_depth D         # Max tree depth for decision tree (default 5)
  --min_split S         # Min samples per split/leaf for decision tree (default 10)

Example — probe the landmarks dataset with a logistic-regression probe on the first model:

python main.py --n_models 1 --dataset landmarks --probe_type lr --top_k 10

Outputs

Activations: cached under <dataset>/activations/{model_name}_activations.pt.

Probing results: <dataset>/{model_name}_{probe_type}_probing_results.csv.

PDF Report: <dataset>/{model_name}_{probe_type}_f1_acc_report.pdf, containing per-method and summary plots for F1 and accuracy.
