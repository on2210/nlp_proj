import torch
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
from utils import (load_activations, load_my_dataset, inspect_pickle, run_probe,
                   extract_last_token_activations, get_labels_from_dataset,
                   format_prompts, probe_three_ways, generate_report)

ACTIVATIONS_PATH = "-activations.pkl"

class Pipeline:
    def __init__(self, model_names, output_folder, dataset_path, sgl_clf, mtpl_clf, template_type="prompt_templates", top_k=10):
        self.model_names = model_names
        self.output_folder = output_folder
        self.dataset_path = dataset_path
        self.template_type = template_type
        self.top_k = top_k
        self.sgl_clf = sgl_clf
        self.mtpl_clf = mtpl_clf
        self.activations = None

        for name in model_names:
            if not os.path.exists(output_path + name + ACTIVATIONS_PATH):
                self.activations = self.get_mlp_activations(name)
            else:
                self.activations = load_activations(output_path + name + ACTIVATIONS_PATH)

            self.build_and_analyze_model(name, activations=self.activations)

    def get_mlp_activations(self, name):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model from TransformerLens
        model = HookedTransformer.from_pretrained(name, device=device)

        # Load dataset and format prompts
        dataset = load_my_dataset(self.dataset_path)
        prompts = format_prompts(dataset, self.template_type)

        activations_all = []

        for prompt in tqdm(prompts, desc="Processing Prompts"):
            # Tokenize and move to device
            tokens = model.to_tokens(prompt).to(device)

            layer_outputs = []

            # Define hook function
            def make_hook(layer_outputs):
                def hook(activation, hook):
                    layer_outputs.append(activation.detach().cpu())

                return hook

            # Register hooks for MLP outputs of all layers
            act_names = [get_act_name("mlp_out", layer) for layer in range(model.cfg.n_layers)]
            hooks = [(name, make_hook(layer_outputs)) for name in act_names]

            # Run model with hooks
            _ = model.run_with_hooks(tokens, return_type=None, fwd_hooks=hooks)

            # Stack into tensor: [num_layers, seq_len, hidden_size]
            layer_outputs = torch.stack(layer_outputs)  # [num_layers, batch=1, seq_len, d_model]
            layer_outputs = layer_outputs.squeeze(1).permute(1, 0, 2)  # [seq_len, num_layers, d_model]
            activations_all.append(layer_outputs)

        # Pad all to same sequence length
        max_seq_len = max(a.shape[0] for a in activations_all)
        num_layers = activations_all[0].shape[1]
        hidden_size = activations_all[0].shape[2]

        padded_activations = torch.zeros(len(activations_all), max_seq_len, num_layers, hidden_size)

        for i, a in enumerate(activations_all):
            padded_activations[i, :a.shape[0], :, :] = a

        path = self.output_folder + name + ACTIVATIONS_PATH

        with open(path, "wb") as f:
            pickle.dump(padded_activations, f)

        print(f"Saved activations to {path}")

        return padded_activations

    def build_and_analyze_model(self, model_name, activations=None):
        save_top_neurons_path = self.output_folder
        labels = get_labels_from_dataset(self.dataset_path)

        if activations is None:
            print("fetching activations")
            pickle_path = self.output_folder + ACTIVATIONS_PATH
            activations = load_activations(pickle_path)

        last_token = extract_last_token_activations(activations)

        results = probe_three_ways(last_token, labels, top_k=self.top_k, save_path=save_top_neurons_path,
                                   sgl_clf=self.sgl_clf, mtpl_clf=self.mtpl_clf, model_name=model_name)

        generate_report(self.output_folder, model_name)

        print("Done")


model_names = ["pythia-160m"]
dataset_path = "landmarks_450.json"
output_path = "exp12/"
sgl_clf = DecisionTreeClassifier(min_samples_split=25)
mtpl_clf = DecisionTreeClassifier(min_samples_split=25)

print("Starting to Run")
p = Pipeline(model_names, output_path, dataset_path, sgl_clf=sgl_clf, mtpl_clf=mtpl_clf, top_k=20)

