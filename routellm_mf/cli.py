"""
CLI entry points.

routellm-calibrate   — train the difficulty classifier on labelled data
routellm-route       — score a prompt and print the routing decision
"""

from __future__ import annotations

import argparse
import sys


def calibrate() -> None:
    """CLI: train and save the difficulty classifier."""
    parser = argparse.ArgumentParser(
        prog="routellm-calibrate",
        description="Train the prompt difficulty classifier on labelled CSV data.",
    )
    parser.add_argument("--data", required=True, help="Path to labelled CSV (columns: prompt, label)")
    parser.add_argument("--out", required=True, help="Output path for trained classifier (.joblib)")
    parser.add_argument("--config", default=None, help="Path to router config YAML (optional)")
    parser.add_argument("--test", default=None, help="Optional held-out test CSV for evaluation")
    parser.add_argument("--cv", type=int, default=0, help="If > 1, run k-fold cross-validation")
    args = parser.parse_args()

    from routellm_mf.config import RouterConfig  # noqa: PLC0415
    from routellm_mf.calibrator import Calibrator  # noqa: PLC0415

    cfg = RouterConfig.from_yaml(args.config) if args.config else RouterConfig()
    cal = Calibrator(cfg)
    cal.load_csv(args.data)

    if args.cv and args.cv > 1:
        cal.cross_validate(cv=args.cv)

    cal.fit()
    cal.save(args.out)

    if args.test:
        cal.evaluate(test_csv=args.test)


def route() -> None:
    """CLI: score a prompt and print the routing decision."""
    parser = argparse.ArgumentParser(
        prog="routellm-route",
        description="Score a prompt and print the difficulty/routing decision.",
    )
    parser.add_argument("prompt", nargs="?", help="Prompt to score (or pass via stdin)")
    parser.add_argument("--config", default=None, help="Path to router config YAML")
    parser.add_argument("--threshold", type=float, default=None, help="Override difficulty threshold")
    args = parser.parse_args()

    from routellm_mf.config import RouterConfig  # noqa: PLC0415
    from routellm_mf.router import Router  # noqa: PLC0415

    cfg = RouterConfig.from_yaml(args.config) if args.config else RouterConfig()
    if args.threshold is not None:
        cfg.classifier.threshold = args.threshold

    prompt = args.prompt or sys.stdin.read().strip()
    if not prompt:
        parser.error("No prompt provided (use positional arg or stdin).")

    router = Router(cfg)
    result = router.score(prompt)

    try:
        from rich.console import Console  # noqa: PLC0415
        from rich.table import Table  # noqa: PLC0415
        from rich.text import Text  # noqa: PLC0415

        console = Console()
        model_label = "[bold red]STRONG[/]" if result.use_strong_model else "[bold green]weak[/]"
        console.print(f"\n[bold]Routing decision:[/] {model_label}")
        console.print(f"  Score     : [yellow]{result.score:.4f}[/]  (threshold={result.threshold})")
        console.print(f"  Heuristic : {result.heuristic_score:.4f}")
        if result.model_score is not None:
            console.print(f"  ML model  : {result.model_score:.4f}")

        tbl = Table(title="Signal Breakdown", show_lines=True)
        tbl.add_column("Signal", style="cyan")
        tbl.add_column("Contribution", justify="right")
        for k, v in sorted(result.reasoning.items(), key=lambda x: -abs(x[1])):
            sign = "+" if v > 0 else ""
            tbl.add_row(k, f"{sign}{v:.3f}")
        console.print(tbl)
    except ImportError:
        print(result)
