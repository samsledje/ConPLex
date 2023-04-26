from argparse import ArgumentParser
from pathlib import Path

from ..utils import (
    get_logger,
    config_logger
)

logg = config_logger(None,
              "%(asctime)s [%(levelname)s] %(message)s",
              level = 2,
              use_stdout = True
              )

def add_args(parser: ArgumentParser):

    parser.add_argument("--to",
                        type = str,
                        required = True,
                        help = "Location to download the data to. If the location does not exist, it will be created."
                        )
    
    parser.add_argument("--benchmarks", 
                        type = str, 
                        nargs = "+",
                        choices = ["davis", "bindingdb", "biosnap", "biosnap_prot", "biosnap_mol", "dti_dg"],
                        help = "Benchmarks to download."
                        )
    
    parser.add_argument("--models",
                        type = str,
                        nargs = "+",
                        choices = ["ConPLex_v1_BindingDB"],
                        help = "Pre-trained ConPLex models to download."
                        )

    return parser

def main(args):
    args.to = Path(args.to).absolute()
    logg.info(f"Download Location: {args.to}")
    logg.info("")

    logg.info("[BENCHMARKS]")
    for bm in args.benchmarks:
        logg.info(f"Downloading {bm}...")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    main(args)
