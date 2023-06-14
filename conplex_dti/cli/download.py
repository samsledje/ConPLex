import os
import shutil
import sys
import urllib.request
from argparse import ArgumentParser
from functools import partial, wraps
from pathlib import Path

from ..dataset import get_task_dir
from ..utils import config_logger, get_logger

logg = config_logger(
    None, "%(asctime)s [%(levelname)s] %(message)s", level=2, use_stdout=True
)


def get_remote_path(bm: str):
    REMOTE_DATA_PATHS = {
        "davis": "https://raw.githubusercontent.com/samsledje/ConPLex_dev/main/dataset/DAVIS",
        "bindingdb": "https://raw.githubusercontent.com/samsledje/ConPLex_dev/main/dataset/BindingDB",
        "biosnap": "https://raw.githubusercontent.com/samsledje/ConPLex_dev/main/dataset/BIOSNAP/full_data",
        "biosnap_prot": "https://raw.githubusercontent.com/samsledje/ConPLex_dev/main/dataset/BIOSNAP/unseen_protein",
        "biosnap_mol": "https://raw.githubusercontent.com/samsledje/ConPLex_dev/main/dataset/BIOSNAP/unseen_drug",
        "dude": "http://cb.csail.mit.edu/cb/conplex/data",
        "ConPLex_v1_BindingDB": "https://cb.csail.mit.edu/cb/conplex/data/models/BindingDB_ExperimentalValidModel.pt",
    }
    return REMOTE_DATA_PATHS[bm]


def add_args(parser: ArgumentParser):
    parser.add_argument(
        "--to",
        type=str,
        required=True,
        help="Location to download the data to. If the location does not exist, it will be created.",
    )

    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        choices=[
            "davis",
            "bindingdb",
            "biosnap",
            "biosnap_prot",
            "biosnap_mol",
            "dude",
            #            "dti_dg",
        ],
        help="Benchmarks to download.",
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=["ConPLex_v1_BindingDB"],
        help="Pre-trained ConPLex models to download.",
    )

    return parser


def download_safe(
    remote_path: str, local_path: str, key: str = "file", verbose: bool = True
) -> str:
    if not os.path.exists(local_path):
        try:
            if verbose:
                logg.info(f"Downloading {key} from {remote_path} to {local_path}...")
            with urllib.request.urlopen(remote_path) as response, open(
                local_path, "wb"
            ) as out_file:
                shutil.copyfileobj(response, out_file)
        except Exception as e:
            logg.error(f"Unable to download {key} - {e}")
            sys.exit(1)
    return local_path


def main(args):
    args.to = Path(args.to).absolute()
    logg.info(f"Download Location: {args.to}")
    logg.info("")

    logg.info("[BENCHMARKS]")
    benchmarks = args.benchmarks or []
    for bm in benchmarks:
        logg.info(f"Downloading {bm}...")
        task_dir = Path(get_task_dir(bm, database_root=args.to))
        os.makedirs(task_dir, exist_ok=True)

        if bm == "dude":
            fi_list = [
                "full.tsv",
                "dude_cross_type_train_test_split.csv",
                "dude_within_type_train_test_split.csv",
            ]
        else:
            fi_list = ["train.csv", "val.csv", "test.csv"]

        for fi in fi_list:
            local_path = task_dir / fi
            remote_base = get_remote_path(bm)
            remote_path = f"{remote_base}/{fi}"
            download_safe(remote_path, local_path, key=f"{bm}/{fi}")

    logg.info("[MODELS]")
    models = args.models or []
    for mo in models:
        model_dir = args.to / "models"
        os.makedirs(model_dir, exist_ok=True)
        local_path = (model_dir / mo).with_suffix(".pt")
        remote_path = get_remote_path(mo)
        download_safe(remote_path, local_path, key=mo)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    main(args)
