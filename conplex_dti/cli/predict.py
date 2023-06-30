import numpy as np
import pandas as pd
import torch

from argparse import ArgumentParser

from conplex_dti.featurizer import (
    MorganFeaturizer,
    ProtBertFeaturizer,
)
from conplex_dti.model.architectures import SimpleCoembeddingNoSigmoid
from conplex_dti.utils import set_random_seed

from torch.utils.data import DataLoader

def add_args(parser: ArgumentParser):
    parser.add_argument(
        '--data-file', 
        type=str, 
        required=True,
        default='./data/pairs.tsv', 
        help='Path to the file containing data in the form [proteinID] [moleculeID] [proteinSequence] [moleculeSmiles].',
    )
    parser.add_argument(
        '--model-path', 
        type=str, 
        required=False,
        default='./models/ConPLex_v1_BindingDB.pt', 
        help='Path to the file containing the model to use for predictions. Default: ./models/ConPLex_v1_BindingDB.pt',
    )
    parser.add_argument(
        '--results-ofile', 
        type=str, 
        required=False,
        default='results.tsv', 
        help='Filename to save the resulting dataframe. Default: results.tsv',
        # NOTE: the table is stored in the form [moleculeID] [proteinID] [prediction]
    )
    parser.add_argument(
        '--features-dir', 
        type=str, 
        required=False,
        default='.', 
        help='Directory to store the Morgan features and ProtBert .h5 files that are created within to the program. Default: .',
    )
    parser.add_argument(
        '--random-seed', 
        type=int, 
        required=False,
        default=61998, 
        help='Random seed to use for reproducibility. Default: 61998',
    )
    return parser


def main(args):
    set_random_seed(args.random_seed)

    try:
        query_df = pd.read_csv(args.data_file, sep='\t', names=["proteinID", "moleculeID", "proteinSequence", "moleculeSmiles"])
    except FileNotFoundError:
        print(f"Could not find data file: {args.data_file}")
        return


    print(f"Loading model from {args.model_path}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    target_featurizer = ProtBertFeaturizer(save_dir = args.features_dir, per_tok = False).cuda(device)
    drug_featurizer = MorganFeaturizer(save_dir = args.features_dir).cuda(device)

    drug_featurizer.preload(query_df['moleculeSmiles'].unique())
    target_featurizer.preload(query_df['proteinSequence'].unique())

    model = SimpleCoembeddingNoSigmoid(drug_featurizer.shape, target_featurizer.shape, 1024)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.eval()
    model = model.to(device)

    dt_feature_pairs = [(drug_featurizer(r['moleculeSmiles']), target_featurizer(r['proteinSequence'])) for _, r in query_df.iterrows()]
    dloader = DataLoader(dt_feature_pairs, batch_size = 128, shuffle = False)
    
    print(f"Generating predictions...")
    preds = []
    with torch.set_grad_enabled(False):
        for b in dloader:
            preds.append(model(b[0], b[1]).detach().cpu().numpy())

    preds = np.concatenate(preds)


    result_df = pd.DataFrame(query_df[['moleculeID', 'proteinID']])
    result_df['Prediction'] = preds
    
    print(f"Printing ConPLex results to {args.results_ofile}")
    result_df.to_csv(args.results_ofile, sep='\t', index=False, header=False)


    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    main(args)


