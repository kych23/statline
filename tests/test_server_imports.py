import subprocess
import sys

TRAINING_ONLY = ["lightgbm", "sklearn", "nflreadpy", "polars", "scipy"]


def test_server_modules_do_not_import_training_libraries() -> None:
    code = (
        "import sys\n"
        "import statline.server.app, statline.server.data, statline.server.tools\n"
        "import statline.server.http\n"
        f"print([m for m in {TRAINING_ONLY} if m in sys.modules])\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
