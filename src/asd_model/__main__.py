"""`python -m asd_model` trains and tests one network, exactly as main.py does (see `python -m asd_model --help`)."""

import runpy


def main():
    """Run asd_model.main as a script (this is what the asd-model command does)."""
    runpy.run_module("asd_model.main", run_name="__main__")


if __name__ == "__main__":
    main()
