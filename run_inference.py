from __future__ import annotations

from pathlib import Path

from pipeline.active.cli import build_parser
from pipeline.active.runner import run


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    args = build_parser(PROJECT_ROOT).parse_args()
    run(args)


if __name__ == "__main__":
    main()
