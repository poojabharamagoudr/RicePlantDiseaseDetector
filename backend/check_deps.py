"""Sanity-check imports and print versions for core dependencies."""
import importlib
import sys

deps = [
    ("flask", "Flask"),
    ("flask_cors", "flask_cors"),
    ("PIL", "Pillow"),
    ("numpy", "numpy"),
    ("tensorflow", "tensorflow"),
]

errors = []
for modname, pretty in deps:
    try:
        m = importlib.import_module(modname)
        ver = getattr(m, "__version__", None)
        print(f"OK: {pretty} (module '{modname}') version={ver}")
    except Exception as e:
        print(f"MISSING: {pretty} (module '{modname}'): {e}")
        errors.append((modname, str(e)))

if errors:
    print("\nSome dependencies are missing or failed to import. Try:\n  .\\setup_env.ps1\nOr install manually in your active venv:\n  pip install -r requirements.txt")
    sys.exit(2)
else:
    print("\nAll core dependencies import successfully.")
