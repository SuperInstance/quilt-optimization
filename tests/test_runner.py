"""Minimal test runner — pytest not available in sandbox."""
import sys
import traceback
import importlib
from pathlib import Path

def discover_and_run():
    test_dir = Path(__file__).parent
    test_files = sorted([f for f in test_dir.glob("test_*.py") if f.name != "test_runner.py"])

    passed = 0
    failed = 0
    skipped = 0
    errors = []

    for test_file in test_files:
        module_name = test_file.stem
        module = importlib.import_module(f"tests.{module_name}")

        # Get all functions starting with test_
        test_funcs = [
            (name, fn) for name, fn in vars(module).items()
            if name.startswith("test_") and callable(fn)
        ]

        for name, fn in test_funcs:
            try:
                fn()
                print(f"  ✓ {module_name}::{name}")
                passed += 1
            except Exception as e:
                if "pytest.skip" in str(type(e)):
                    print(f"  ⊘ {module_name}::{name} (SKIPPED)")
                    skipped += 1
                else:
                    print(f"  ✗ {module_name}::{name}")
                    print(f"      {type(e).__name__}: {e}")
                    errors.append((module_name, name, traceback.format_exc()))
                    failed += 1

    print(f"\n{'='*60}")
    print(f"  {passed} passed, {failed} failed, {skipped} skipped")
    print(f"{'='*60}")

    if errors:
        print("\n=== ERRORS ===")
        for mod, name, tb in errors:
            print(f"\n--- {mod}::{name} ---")
            print(tb)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(discover_and_run())
