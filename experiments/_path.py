"""Put the repo root on sys.path and silence the one warning we expect.

lbfgs doesn't converge in 1000 iterations on these unscaled features. The
fit is stable anyway (identical AUROC across all 30 seeds), and scaling would
change the noise calibration, so the warning goes rather than the cause.
"""
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.exceptions import ConvergenceWarning  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# joblib workers parse PYTHONWARNINGS before sklearn is importable, so use
# the UserWarning parent class. the prevalence warning runs in the parent
# and still prints
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning")


# ---------------------------------------------------------------------------
# One process per dataset
# ---------------------------------------------------------------------------
# runners kept stalling on the second dataset, never isolated why. a fresh
# interpreter per dataset is the workaround that always held
import subprocess  # noqa: E402

SINGLE_FLAG = "--_single"


def strip_datasets(argv: list[str]) -> list[str]:
    """Drop any --datasets group and the internal flag from an argv list."""
    out, skip = [], False
    for tok in argv:
        if tok == "--datasets":
            skip = True
            continue
        if skip and not tok.startswith("--"):
            continue
        skip = False
        if tok != SINGLE_FLAG:
            out.append(tok)
    return out


def run_per_dataset(script: str, datasets: list[str]) -> int:
    """Run `script` once per dataset in its own process. Report, never abort early."""
    passthrough = strip_datasets(sys.argv[1:])
    outcomes = {}
    for ds in datasets:
        cmd = [sys.executable, script, "--datasets", ds, SINGLE_FLAG, *passthrough]
        print(f"\n===== {Path(script).name}: {ds} (fresh process) =====", flush=True)
        rc = subprocess.run(cmd).returncode
        outcomes[ds] = rc
        print(f"===== {ds}: {'OK' if rc == 0 else f'FAILED (exit {rc})'} =====", flush=True)
    failed = [ds for ds, rc in outcomes.items() if rc]
    print("\nsummary: " + ", ".join(f"{ds}={'ok' if not rc else 'FAILED'}"
                                  for ds, rc in outcomes.items()))
    if failed:
        print(f"re-run only what failed:  python {Path(script).name} --datasets {' '.join(failed)}")
    return 1 if failed else 0
