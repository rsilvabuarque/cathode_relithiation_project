from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hrw",
        description="Hydrothermal relithiation workflow command line interface.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "electrode-generate",
        help="Scaffold command for electrode structure generation workflow.",
    )
    subparsers.add_parser(
        "electrolyte-generate",
        help="Scaffold command for electrolyte structure generation workflow.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "electrode-generate":
        raise NotImplementedError(
            "Use `hrw-electrode-generate` for the dedicated scaffold command."
        )
    if args.command == "electrolyte-generate":
        raise NotImplementedError(
            "Use `hrw-electrolyte-generate` for the dedicated scaffold command."
        )

    parser.print_help()


if __name__ == "__main__":
    main()
