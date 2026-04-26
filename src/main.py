from __future__ import annotations

import argparse
from pathlib import Path

from datasets import list_builtin_datasets, load_dataset
from models import list_models
from strategies import get_strategy, list_strategies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pool-based active learning experiments on tabular data."
    )
    parser.add_argument(
        "--dataset",
        default="sklearn:breast_cancer",
        help="Dataset spec: sklearn:<name> or libsvm:/path/to/file.",
    )
    parser.add_argument(
        "--strategy",
        default="margin",
        help="Query strategy name. Use --list-strategies to see choices.",
    )
    parser.add_argument(
        "--model",
        default="RandomForest",
        choices=list_models(),
        help="Task/query model used by the sklearn-native runner.",
    )
    parser.add_argument("--seed", default=0, type=int, help="First random seed.")
    parser.add_argument("--n-trials", default=1, type=int, help="Number of seeds to run.")
    parser.add_argument(
        "--test-size",
        default=0.4,
        type=float,
        help="Fraction of the dataset reserved for testing.",
    )
    parser.add_argument(
        "--init-labels",
        default=20,
        type=int,
        help="Initial labeled pool size. Must be at least the number of classes.",
    )
    parser.add_argument(
        "--budget",
        default=50,
        type=int,
        help="Total number of active-learning queries after initialization.",
    )
    parser.add_argument("--batch-size", default=1, type=int, help="Query batch size.")
    parser.add_argument("--scale", action="store_true", help="Apply StandardScaler.")
    parser.add_argument(
        "--output-dir",
        default=None,
        type=Path,
        help="Directory where run folders are written. Defaults to repo results/runs.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run folder name. Defaults to dataset-strategy-model-seed.",
    )
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        help="Print available query strategies and exit.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="Print built-in sklearn dataset specs and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_datasets:
        for name in list_builtin_datasets():
            print(name)
        return

    if args.list_strategies:
        _print_strategies()
        return

    dataset = load_dataset(args.dataset)
    strategy = get_strategy(args.strategy)
    from experiment import ExperimentConfig, run_experiment

    output_dir = args.output_dir or Path(__file__).resolve().parents[1] / "results" / "runs"
    config = ExperimentConfig(
        dataset=dataset,
        strategy=strategy,
        model_name=args.model,
        seed=args.seed,
        n_trials=args.n_trials,
        test_size=args.test_size,
        init_labels=args.init_labels,
        budget=args.budget,
        batch_size=args.batch_size,
        scale=args.scale,
        output_dir=output_dir,
        run_name=args.run_name,
    )
    run_dir = run_experiment(config)
    print(f"Run complete: {run_dir}")
    print(f"Strategy category: {strategy.category}")


def _print_strategies() -> None:
    current_category = None
    for strategy in list_strategies():
        if strategy.category != current_category:
            current_category = strategy.category
            print(f"\n{current_category}")
        legacy = f", legacy={strategy.legacy_name}" if strategy.legacy_name else ""
        limitation = f" -- {strategy.limitation}" if strategy.limitation else ""
        print(
            f"  {strategy.name}: {strategy.display_name} "
            f"({strategy.framework}{legacy}){limitation}"
        )


if __name__ == "__main__":
    main()
