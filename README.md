# LLM Probing Pipeline 🚀

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/) 

A reproducible pipeline for probing neural activations of Pythia language models using classification probes. Compute and cache model activations, run single-neuron and layer-wise probes, and generate comprehensive performance reports.

---

## 📁 Repository Structure

```
├── data/                   # Input JSON datasets
│   ├── landmarks_450.json
│   ├── centuries_300.json
│   ├── music.json
│   ├── europarl_lang.json
│   └── pile_data_source.json
├── <dataset>/              # Outputs per dataset (e.g., landmarks/)
│   ├── activations/        # Cached activations (.pt)
│   ├── *_probing_results.csv
│   └── *_f1_acc_report.pdf
├── main.py                 # Entry point: argument parsing & pipeline launch
├── pipeline.py             # Core logic: activation extraction, probing, reporting
├── utilz.py                # Utilities: hooks, data loaders, plotting, etc.
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 📂 Datasets

Place your JSON files in the `data/` directory. Available dataset flags:

| Name               | File                    | Description               |
| ------------------ | ----------------------- | ------------------------- |
| `landmarks`        | `landmarks_450.json`    | Landmark classification   |
| `centuries`        | `centuries_300.json`    | Century classification    |
| `music`            | `music.json`            | Genre classification      |
| `europarl_lang`    | `europarl_lang.json`    | Language ID (global task) |
| `pile_data_source` | `pile_data_source.json` | Source ID (global task)   |

> If you have pre-tokenized data, provide a matching `<dataset>_tokens.pt` next to the JSON.

---

## 🚀 Usage

Run the pipeline via:

```bash
python main.py \
  --n_models N         # Number of Pythia models to probe (1–7)
  --dataset NAME       # e.g. landmarks, centuries, music, europarl_lang, pile_data_source
  --probe_type {lr,dt} # `lr` (Logistic Regression) or `dt` (Decision Tree)
  --top_k K            # Top-K neurons for multi-neuron probe
  [--max_iter M]       # (lr) Max iterations (default=1000)
  [--max_depth D]      # (dt) Max tree depth (default=5)
  [--min_split S]      # (dt) Min samples per split (default=10)
```

Example:

```bash
python main.py --n_models 1 --dataset landmarks --probe_type lr --top_k 10
```

---

## 📊 Outputs

* **Activations**: Saved under `<dataset>/activations/{model_name}_activations.pt`
* **Results CSV**: `<dataset>/{model_name}_{probe_type}_probing_results.csv`
* **PDF Report**: `<dataset>/{model_name}_{probe_type}_f1_acc_report.pdf` (includes F1 & accuracy plots)

## 📫 Contributing & Support

Feel free to open issues or pull requests. For questions, reach out via GitHub issues or email us at danielvered@mail.tau.ac.il or omernaziri1@mail.tau.ac.il.
