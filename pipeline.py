import os
import math
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name

from utilz import ActUtilz, DatasetUtilz, ProbeUtilz, Report


ACTIVATIONS_PATH = "_activations.pt"
BIN_THRESHOLD = 0
KFOLDS = 5


class Pipeline:
    def __init__(self, model_names: list[str], output_folder: str, dataset_path: str, sgl_clf, mtpl_clf, probe_type: str
                 , is_task_global: bool, template_key="prompt_templates", top_k=10, is_tokenized: bool = False):
        """
        Initializes a Pipeline object for LLM probing.
        :param model_names: list of the model names to probe.
        :param output_folder: a path to a directory save model activations and probing outputs.
        If directory doesn't exist, creates it.
        :param dataset_path: a path to a .json file representing a dataset.
        :param sgl_clf: a classifier model used to probe single neurons.
        :param mtpl_clf: a classifier model used to probe multiple neurons (top-k or full-layer).
        :param probe_type: type of classifier models used for probing ('logistic_regression' or 'decision_tree').
        :param is_task_global: specifies wether the classification task in the dataset aims to detect global (e.g. language)
        characteristics of the input text or predict the next token (complete a sentence).
        :param template_key: the type of the template to use for probing. For tokenized datasets can be ''.
        :param top_k: the number of most distinctive neurons to probe in the 'top-k' regime. 
        :param is_tokenized: specifies wether the dataset is already tokenized or not.
        If true, there should exist a file with path '{dataset_name}_tokens.pt' with the tokens as torch.tensor.
        Otherwise, textual data is tokenized using transformer_lens functions. 
        """
        # model params
        self.model_names = model_names
                
        # output params
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        self.activations_dir = os.path.join(self.output_folder, 'activations')
        os.makedirs(self.activations_dir, exist_ok=True)
        
        # dataset params
        self.dataset_path = dataset_path
        self.template_key = template_key
        self.is_tokenized = is_tokenized
        self.is_task_global = is_task_global
        
        # probes params
        self.sgl_clf = sgl_clf
        self.mtpl_clf = mtpl_clf
        self.top_k = top_k
        self.probe_type = probe_type
        self.probe_manager = ProbeUtilz(bin_threshold=BIN_THRESHOLD, kfolds=KFOLDS)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.run_pipeline()

    # ========================================== Main ========================================== #
    
    def run_pipeline(self):
        for name in self.model_names:
            act_file_name = f"{name}{ACTIVATIONS_PATH}"
            act_save_path = os.path.join(self.activations_dir, act_file_name)
            
            if not os.path.exists(act_save_path): # No cached activations
                print(f"\nCalculating activations for {name}...")
                self.cache_mlp_activations(name, act_save_path)
            
            print(f"Loading cached activations for {name}")
            activations = self.load_activations(name, act_save_path)

            print(f"\nProbing {name}...")
            self.probe(name, activations)
            
            print("Preparing report...")
            Report.generate_report(self.output_folder, name, self.probe_type)
            print("Done")

    # ========================================== Activations I\O ========================================== #
    
    def save_activations(self, activations: torch.tensor, model_name: str, save_path: str):
        """
        Saves a list of model activations into a .pt file.
        :param activations: a torch.tensor model activations.
        :parma model_name: the name of the model.
        :param save_path: a path for saving activations.
        """
        quantized = ActUtilz.quantize_activations(activations)
        torch.save(quantized, save_path)
    
    def load_activations(self, model_name: str, save_path: str) -> list[torch.tensor]:
        """
        Loads a list of model activations from a .pt file.
        :parma model_name: the name of the model.
        :param save_path: the path for loading activations.
        """
        quantized = torch.load(save_path)
        return quantized.dequantize()
    
    # ========================================== Activations Computations ========================================== #
    
    def get_activations_by_layer(self, model, tokens: torch.tensor) -> torch.tensor:
        """
        Get activations of the model on the input tokens by layers.
        :param model: the model to get the activations of.
        :param tokens: a torch.tensor with the tokens.
        :return: a tensor of shape (num_layers, d_model) with the layer activations.
        """
        # Preallocating tensor of shape [num_layers, batch=1, seq_len, d_model]
        layer_activations = torch.zeros((model.cfg.n_layers, 1, tokens.shape[0], model.cfg.d_mlp), dtype=torch.float16)

        # define hooks to save activations from each layer
        hooks = [(f'blocks.{layer_ix}.mlp.hook_post', ActUtilz.hook) for layer_ix in range(model.cfg.n_layers)]

        # Run model with hooks
        _ = model.run_with_hooks(tokens, return_type=None, fwd_hooks=hooks)

        # Get activations from hooks
        for layer_ix, (hook_name, _) in enumerate(hooks):
            layer_activations[layer_ix, :, :, :] = model.hook_dict[hook_name].ctx['activation']
    
        layer_activations = layer_activations.squeeze(1)  # [num_layers, seq_len, d_model] 
        
        if self.is_task_global:
            return layer_activations.mean(dim=1) # Get mean activation along the sequence
        else:
            return layer_activations[:, -1, :] # Get last token activation
    
    def get_activations_by_prompts(self, model) -> torch.tensor:
        """
        Get activations of the model on a given list of textual prompts in the dataset.
        :param model: the model to get the activations of.
        :return: a tensor of shape (num_prompts, num_layers, d_model) with the layer activations for all prompts.
        """
        # Load dataset and format prompts
        print(f"Loading dataset from {self.dataset_path}")
        dataset = DatasetUtilz.load_dataset(self.dataset_path)
        prompts = DatasetUtilz.format_prompts(dataset, self.template_key)
        
        activations = torch.zeros((len(prompts), model.cfg.n_layers, model.cfg.d_mlp))
        for i, prompt in enumerate(tqdm(prompts, desc="Processing Prompts")):
            tokens = model.to_tokens(prompt).to(self.device)
            prompt_activations = self.get_activations_by_layer(model, tokens)
            activations[i, :, :] = prompt_activations
        
        return activations
    
    def get_activations_by_tokens(self, model) -> torch.tensor:
        """
        Get activations of the model on a given list of tokenized textual prompts in the dataset.
        :param model: the model to get the activations of.
        :return: a tensor of shape (num_prompts, num_layers, d_model) with the layer activations for all tokenized prompts.
        """
        # Load tokens
        tokens_path = f"{self.dataset_path.split('.')[0]}_tokens.pt"
        print(f"Loading dataset from {tokens_path}")
        all_tokens = torch.load(tokens_path)
                
        activations = torch.zeros((len(all_tokens), model.cfg.n_layers, model.cfg.d_mlp))
        for i, tokens in enumerate(tqdm(all_tokens, desc="Processing Tokens")):
            tokens = tokens.to(self.device)
            prompt_activations = self.get_activations_by_layer(model, tokens)
            activations[i, :, :] = prompt_activations
        
        return activations

    def cache_mlp_activations(self, model_name: str, save_path: str):
        """
        A wrapper function to get and cache the model activations on the dataset.
        :parma model_name: the name of the model.
        :param save_path: a path for saving activations.
        """
        # Loading model
        model = HookedTransformer.from_pretrained(model_name).to(self.device)
        
        if self.is_tokenized:
            activations = self.get_activations_by_tokens(model)
        else:
            activations = self.get_activations_by_prompts(model)

        self.save_activations(activations, model_name, save_path)
    
    # ========================================== Probing ========================================== #

    def probe(self, model_name: str, activations: torch.tensor):
        labels, label_map = DatasetUtilz.get_labels_from_dataset(self.dataset_path, self.template_key)
        return self.probe_manager.probe_three_ways(activations, labels, label_map=label_map
                                                         , top_k=self.top_k, save_path=self.output_folder,
                                                         sgl_clf=self.sgl_clf, mtpl_clf=self.mtpl_clf
                                                         , model_name=model_name, probe_type=self.probe_type
                                                         , output_folder=self.output_folder
                                                         )
    