"""Interactive demo of the Wnt/CCL5 model (the one way to run the demo).

    streamlit run app.py

Needs the package installed once from the repository root: pip install -e .[demo]
"""

try:
    from asd_model.app import main
except ModuleNotFoundError as error:
    if error.name not in ("asd_model", "streamlit"):
        raise
    raise SystemExit(f"{error.name} is not installed. From the repository root run: pip install -e .[demo]") from error

if __name__ == "__main__":
    main()
