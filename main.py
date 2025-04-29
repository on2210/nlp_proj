from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

MODELS = ["pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b", "pythia-1.4b", "pythia-2.8b", "pythia-6.9b"]

OUTPUT_PATHS = {
    "lr": "logistic_regression_probes/"
    , "dt": "decision_tree_probes/"
}

DATASET_PATHS = {
    "landmarks":  "landmarks_450.json"
    , "centuries": "centuries_300.json"
}

def parse_cmd_input(args) -> dict:
    params = {}
    
    try:
        params['models'] = MODELS[:args.n_models]
    except:
        print("Invalid number of models.")
        sys.exit(1)
    
    try:
        params['dataset_path'] = DATASET_PATHS[args.dataset]
    except:
        print("Invalid dataset name.")
        sys.exit(1)
    
    if args.probe_type == 'lr':
        params['sgl_probe'] = LogisticRegression(class_weight='balanced')
        params['mtpl_probe'] = LogisticRegression(class_weight='balanced')
        params['output_path'] = OUTPUT_PATHS["lr"]
    
    elif probe_type == 'dt':
        params['sgl_probe'] = DecisionTreeClassifier(min_samples_split=25)
        params['mtpl_probe'] = DecisionTreeClassifier(min_samples_split=25)
        params['output_path'] = OUTPUT_PATHS["dt"]
    
    else:
        print("Invalid probe type.")
        sys.exit(1)
    
    if isinstance(args.top_k, int):
        params['top_k'] = args.top_k
    else:
        print("Top K should be an integer.")
    
    return params

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--n_models', default=1, type=int,
        help='Number of models to probe.')
    parser.add_argument(
        '--dataset', default='landmarks'
        help='Name of dataset to probe.')
    parser.add_argument(
        '--probe_type', default='lr'
        help='Name of dataset to probe.')
    parser.add_argument(
        '--top_k', defalt=10, type=int,
        help='Top K neurons to probe.')
    

    args = parser.parse_args()
    params = parse_cmd_input(args)
    p = Pipeline(model_names=params['models']
                 , output_path=params['output_path']
                 , dataset_path=params['dataset_path']
                 , sgl_clf=params['sgl_clf']
                 , mtpl_clf=params['mtpl_clf']
                 , top_k=params['top_k']
                 )
