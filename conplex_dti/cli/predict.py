from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..featurizer import MorganFeaturizer, ProtBertFeaturizer
from ..model.architectures import SimpleCoembeddingNoSigmoid
from ..utils import get_logger, set_random_seed

logg = get_logger()


def add_args(parser: ArgumentParser):
    parser.add_argument(
        "--data-file",
        type=str,
        required=True,
        default="./data/pairs.tsv",
        help="Path to the file containing data in the form [proteinID] [moleculeID] [proteinSequence] [moleculeSmiles].",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=False,
        default="./models/ConPLex_v1_BindingDB.pt",
        help="Path to the file containing the model to use for predictions. Default: ./models/ConPLex_v1_BindingDB.pt",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        required=False,
        default="results.tsv",
        help="Filename to save the resulting dataframe. Default: results.tsv",
        # NOTE: the table is stored in the form [moleculeID] [proteinID] [prediction]
    )
    parser.add_argument(
        "--data-cache-dir",
        type=str,
        required=False,
        default=".",
        help="Directory to store the Morgan features and ProtBert .h5 files that are created within to the program. Default: .",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cuda:0",
        help="Device to use for predictions. Default: cuda:0",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=False,
        default=128,
        help="Batch size to use for predictions, allowing the program to be adapted to smaller or larger GPU memory. Default: 128",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        required=False,
        default=61998,
        help="Random seed to use for reproducibility. Default: 61998",
    )

    return parser


def main(args):
    logg.debug(f"Setting random state {args.random_seed}")
    set_random_seed(args.random_seed)

    try:
        query_df = pd.read_csv(
            args.data_file,
            sep="\t",
            names=["proteinID", "moleculeID", "proteinSequence", "moleculeSmiles"],
        )
    except FileNotFoundError:
        logg.error(f"Could not find data file: {args.data_file}")
        return

    # Set CUDA device
    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        device_no = args.device
        use_cuda = torch.cuda.is_available()
        device = torch.device(f"cuda:{device_no}" if use_cuda else "cpu")
    logg.info(f"Using CUDA device {device}")

    # Loading model
    logg.info(f"Loading model from {args.model_path}")
    target_featurizer = ProtBertFeaturizer(
        save_dir=args.data_cache_dir, per_tok=False
    ).to(device)
    drug_featurizer = MorganFeaturizer(save_dir=args.data_cache_dir).to(device)

    drug_featurizer.preload(query_df["moleculeSmiles"].unique())
    target_featurizer.preload(query_df["proteinSequence"].unique())

    model = SimpleCoembeddingNoSigmoid(
        drug_featurizer.shape, target_featurizer.shape, 1024
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.eval()
    model = model.to(device)

    dt_feature_pairs = [
        (drug_featurizer(r["moleculeSmiles"]), target_featurizer(r["proteinSequence"]))
        for _, r in query_df.iterrows()
    ]
    dloader = DataLoader(dt_feature_pairs, batch_size=args.batch_size, shuffle=False)

    logg.info(f"Generating predictions...")
    preds = []
    with torch.set_grad_enabled(False):
        for b in dloader:
            preds.append(model(b[0], b[1]).detach().cpu().numpy())

    preds = np.concatenate(preds)

    result_df = pd.DataFrame(query_df[["moleculeID", "proteinID"]])
    result_df["Prediction"] = preds

    logg.info(f"Printing ConPLex results to {args.outfile}")
    result_df.to_csv(args.outfile, sep="\t", index=False, header=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    main(args)
