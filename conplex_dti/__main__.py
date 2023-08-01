"""
ConPLex: DTI Prediction
"""

import argparse

from conplex_dti import __version__, cli


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "-v", "--version", action="version", version=f"ConPLex {__version__}"
    )
    # parser.add_argument(
    #     "-c",
    #     "--citation",
    #     action=CitationAction,
    #     nargs=0,
    #     help="show program's citation and exit",
    # )

    subparsers = parser.add_subparsers(title="ConPLex Commands", dest="cmd")
    subparsers.required = True

    modules = {
        "train": (cli.train, cli.train_parser),
        "download": (cli.download, cli.download_parser),
        # "embed": embed,
        # "evaluate": evaluate,
        "predict": (cli.predict, cli.predict_parser),
    }

    for name, (main_func, args_func) in modules.items():
        sp = subparsers.add_parser(name, description=main_func.__doc__)
        args_func(sp)
        sp.set_defaults(main_func=main_func)

    args = parser.parse_args()
    args.main_func(args)


if __name__ == "__main__":
    main()
