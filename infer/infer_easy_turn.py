"""Run only the Easy-Turn VAD scenario through the shared inference framework."""

from common import run_cli


if __name__ == "__main__":
    run_cli(("easy_turn",), "Run Easy-Turn VAD inference.")
