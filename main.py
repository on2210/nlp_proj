import sys
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from pipeline import Pipeline

MODELS = ["pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b", "pythia-1.4b", "pythia-2.8b", "pythia-6.9b"]

DATASET_PATHS = {
    "landmarks":  "landmarks_450.json"
    , "centuries": "centuries_300.json"
    , "music": "music.json"
    , "europarl_lang": "europarl_lang.json"
    , "pile_data_source": "pile_data_source.json"
}

IS_TOKENIZED = {
    "landmarks":  False
    , "centuries": False
    , "music": False
    , "europarl_lang": True
    , "pile_data_source": True
}

IS_TASK_GLOBAL = {
    "landmarks":  False
    , "centuries": False
    , "music": False
    , "europarl_lang": True
    , "pile_data_source": True
}


def die(msg: str):
    print(msg)
    sys.exit(1)


def parse_cmd_input(args) -> dict:
    params = {}
    
    try:
        params['models'] = MODELS[:args.n_models]
    except:
        die("Invalid number of models.")
    
    try:
        params['dataset_path'] = f"data/{DATASET_PATHS[args.dataset]}"
    except:
        die("Invalid dataset name.")
        
    if not isinstance(args.max_iter, int):
        die("Invalid maximum number of iterations.")
        
    if not isinstance(args.max_depth, int):
        die("Invalid maximal depth.")
        
    if not isinstance(args.min_split, int):
        die("Invalid minimal split sample size.")
    
    if args.probe_type == 'lr':
        params['sgl_probe'] = LogisticRegression(class_weight='balanced', max_iter=args.max_iter)
        params['mtpl_probe'] = LogisticRegression(class_weight='balanced', max_iter=args.max_iter)
        params['probe_type'] = 'logistic_regression'
    elif args.probe_type == 'dt':
        params['sgl_probe'] = DecisionTreeClassifier(max_depth=args.max_depth
                                                     , min_samples_split=args.min_split
                                                     , min_samples_leaf=args.min_split)
        params['mtpl_probe'] = DecisionTreeClassifier(max_depth=args.max_depth
                                                      , min_samples_split=args.min_split
                                                      , min_samples_leaf=args.min_split)
        params['probe_type'] = 'decision_tree'
    else:
        die("Invalid probe type.")
    
    params['output_folder'] = f"{args.dataset}/"
    
    if isinstance(args.top_k, int):
        params['top_k'] = args.top_k
    else:
        die("Top K should be an integer.")
    
    params['is_tokenized'] = IS_TOKENIZED[args.dataset]
    params['is_task_global'] = IS_TASK_GLOBAL[args.dataset]
    
    return params

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--n_models', default=1, type=int,
        help='Number of models to probe.')
    parser.add_argument(
        '--dataset', default='landmarks',
        help='Name of dataset to probe.')
    parser.add_argument(
        '--probe_type', default='lr',
        help='Name of dataset to probe.')
    parser.add_argument(
        '--top_k', default=10, type=int,
        help='Top K neurons to probe.')
    parser.add_argument(
        '--max_iter', default=1000, type=int,
        help='Maximal number of iterations for LogisticRegression probes.')
    parser.add_argument(
        '--max_depth', default=5, type=int,
        help='Maximal depth for DecisionTree probes.')
    parser.add_argument(
        '--min_split', default=10, type=int,
        help='Minimal split sample size for DecisionTree probes.')
    
    print("Parsing command line input...")
    args = parser.parse_args()
    params = parse_cmd_input(args)
    
    print("Starting the pipeline...")
    p = Pipeline(model_names=params['models']
                 , output_folder=params['output_folder']
                 , dataset_path=params['dataset_path']
                 , sgl_clf=params['sgl_probe']
                 , mtpl_clf=params['mtpl_probe']
                 , top_k=params['top_k']
                 , probe_type=params['probe_type']
                 , is_tokenized=params['is_tokenized']
                 , is_task_global=params['is_task_global']
                 )
