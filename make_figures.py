"""Recreate the five figures used in the Air Hockey RL README.

Run this file from anywhere with:

    python analysis/make_figures.py

Expected repository layout:

    experiments/
      first_balanced/balanced_ai_full_chaser/all_trial_results.csv
      three_reward/3rewards_full_chaser/all_trial_results.csv
    analysis/make_figures.py
    assets/figures/

The three attacking results are the preserved 100-game evaluations against
the full-speed chaser. They are stored below because the original attacking
CSV contains trial summaries rather than these individual candidate results.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIRECTORY = REPOSITORY_ROOT / "assets" / "figures"

BALANCED_CSV_LOCATIONS = (
    REPOSITORY_ROOT
    / "experiments"
    / "first_balanced"
    / "balanced_ai_full_chaser"
    / "all_trial_results.csv",
    REPOSITORY_ROOT
    / "experiments"
    / "first_balanced"
    / "balanced_ai_full_chaser"
    / "balanced_ai_full_chaser"
    / "all_trial_results.csv",
)

THREE_REWARD_CSV_LOCATIONS = (
    REPOSITORY_ROOT
    / "experiments"
    / "three_reward"
    / "3rewards_full_chaser"
    / "all_trial_results.csv",
    REPOSITORY_ROOT
    / "experiments"
    / "three_reward"
    / "3rewards_full_chaser"
    / "3rewards_full_chaser"
    / "all_trial_results.csv",
)

# Preserved candidate evaluations: 100 games of 60 seconds per agent against
# the 100%-speed chaser. These values are also explained in the README.
ATTACKING_RESULTS = (
    ("T1S3", 3.250, 5.450, -2.200),
    ("T2S1", 5.320, 6.650, -1.330),
    ("T3S1", 6.110, 8.620, -2.510),
)

ATTACKING_COLOUR = "#F28E2B"
BALANCED_COLOUR = "#8064A2"
THREE_REWARD_COLOUR = "#2878C8"
SCORED_COLOUR = "#32965D"
CONCEDED_COLOUR = "#D94B4B"


def locate_csv(candidates: tuple[Path, ...], experiment_name: str) -> Path:
    """Return the first existing CSV path or show a useful setup error."""

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    expected = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        f"Could not find the {experiment_name} results CSV.\n"
        f"Put it in one of these locations:\n{expected}"
    )


def require_columns(
    frame: pd.DataFrame,
    required: set[str],
    experiment_name: str,
) -> None:
    """Check that an input CSV is the correct file for the experiment."""

    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"The {experiment_name} CSV is missing these columns: "
            + ", ".join(missing)
        )


def add_agent_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Add labels such as T1S2 without modifying the original DataFrame."""

    labelled = frame.copy()
    labelled["agent"] = (
        "T"
        + labelled["trial"].astype(int).astype(str)
        + "S"
        + labelled["seed"].astype(int).astype(str)
    )
    return labelled


def save_figure(figure: plt.Figure, filename: str) -> Path:
    """Save a high-resolution PNG suitable for display in a GitHub README."""

    output = FIGURE_DIRECTORY / filename
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def main() -> None:
    """Load the experiment results and generate all five README figures."""

    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    balanced_path = locate_csv(
        BALANCED_CSV_LOCATIONS,
        "first-balanced",
    )
    three_reward_path = locate_csv(
        THREE_REWARD_CSV_LOCATIONS,
        "three-reward",
    )

    attacking = pd.DataFrame(
        ATTACKING_RESULTS,
        columns=(
            "agent",
            "blue_goals_per_game",
            "red_goals_per_game",
            "goal_difference",
        ),
    )
    balanced = pd.read_csv(balanced_path)
    three_reward = pd.read_csv(three_reward_path)

    common_columns = {
        "trial",
        "seed",
        "blue_goals_per_game",
        "red_goals_per_game",
        "goal_difference",
    }
    require_columns(
        balanced,
        common_columns | {"average_reward_per_game"},
        "first-balanced",
    )
    require_columns(three_reward, common_columns, "three-reward")

    balanced = add_agent_labels(balanced)
    three_reward = add_agent_labels(three_reward)

    datasets = (attacking, balanced, three_reward)
    approach_names = ("Attacking", "First balanced", "Three reward")
    colours = (
        ATTACKING_COLOUR,
        BALANCED_COLOUR,
        THREE_REWARD_COLOUR,
    )
    sample_sizes = tuple(len(frame) for frame in datasets)

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "legend.frameon": False,
        }
    )

    means = [frame["goal_difference"].mean() for frame in datasets]
    medians = [frame["goal_difference"].median() for frame in datasets]
    best_results = [frame["goal_difference"].max() for frame in datasets]
    worst_results = [frame["goal_difference"].min() for frame in datasets]
    goals_scored = [
        frame["blue_goals_per_game"].mean() for frame in datasets
    ]
    goals_conceded = [
        frame["red_goals_per_game"].mean() for frame in datasets
    ]

    created: list[Path] = []

    # Figure 1: summary statistics for goal difference.
    statistic_labels = ("Mean", "Median", "Best agent", "Worst agent")
    series = np.asarray(
        (means, medians, best_results, worst_results),
        dtype=float,
    ).T
    x_positions = np.arange(len(statistic_labels))
    bar_width = 0.24
    figure, axis = plt.subplots(figsize=(11, 6))
    for index, (name, values, colour) in enumerate(
        zip(approach_names, series, colours)
    ):
        bars = axis.bar(
            x_positions + (index - 1) * bar_width,
            values,
            bar_width,
            label=f"{name} (n={sample_sizes[index]})",
            color=colour,
        )
        axis.bar_label(bars, fmt="%+.2f", padding=2, fontsize=8)
    axis.axhline(0, color="black", linewidth=1)
    axis.set_xticks(x_positions, statistic_labels)
    axis.set_ylabel("Goal difference per game")
    axis.set_title("All three approaches: goal-difference summaries")
    axis.legend(ncol=3)
    figure.text(
        0.5,
        0.005,
        "Attacking uses three preserved full-chaser candidates; "
        "the later approaches contain all 15 agents.",
        ha="center",
        fontsize=9,
        style="italic",
    )
    created.append(save_figure(figure, "01_goal_difference_summary.png"))

    # Figure 2: average scoring and conceding rates.
    figure, axis = plt.subplots(figsize=(10, 6))
    x_positions = np.arange(len(approach_names))
    bar_width = 0.36
    scored_bars = axis.bar(
        x_positions - bar_width / 2,
        goals_scored,
        bar_width,
        color=SCORED_COLOUR,
        label="Goals scored",
    )
    conceded_bars = axis.bar(
        x_positions + bar_width / 2,
        goals_conceded,
        bar_width,
        color=CONCEDED_COLOUR,
        label="Goals conceded",
    )
    axis.bar_label(scored_bars, fmt="%.2f", padding=3)
    axis.bar_label(conceded_bars, fmt="%.2f", padding=3)
    axis.set_xticks(
        x_positions,
        [
            f"{name}\n(n={sample_size})"
            for name, sample_size in zip(approach_names, sample_sizes)
        ],
    )
    axis.set_ylabel("Average goals per game")
    axis.set_title("Average goals scored and conceded by approach")
    axis.legend()
    created.append(
        save_figure(figure, "02_goals_scored_and_conceded.png")
    )

    # Figure 3: every available agent ranked within its own approach.
    figure, axis = plt.subplots(figsize=(10, 6))
    for frame, name, colour in zip(datasets, approach_names, colours):
        ranked = np.sort(frame["goal_difference"].to_numpy())[::-1]
        axis.plot(
            np.arange(1, len(ranked) + 1),
            ranked,
            "o-",
            color=colour,
            label=f"{name} (n={len(ranked)})",
        )
    axis.axhline(0, color="black", linewidth=1)
    axis.set_xticks(range(1, 16))
    axis.set_xlabel("Agent rank within its approach")
    axis.set_ylabel("Goal difference per game")
    axis.set_title("Ranked agent performance across all three approaches")
    axis.legend()
    created.append(save_figure(figure, "03_ranked_agent_comparison.png"))

    # Figure 4: heatmap of the ranked goal differences.
    matrix = np.full((len(datasets), 15), np.nan)
    for row_index, frame in enumerate(datasets):
        values = np.sort(frame["goal_difference"].to_numpy())[::-1]
        matrix[row_index, : len(values)] = values

    colour_limit = float(np.nanmax(np.abs(matrix)))
    figure, axis = plt.subplots(figsize=(13, 4.6))
    colour_map = plt.get_cmap("RdYlGn").copy()
    colour_map.set_bad("#E5E5E5")
    image = axis.imshow(
        matrix,
        cmap=colour_map,
        vmin=-colour_limit,
        vmax=colour_limit,
        aspect="auto",
    )
    axis.set_xticks(
        range(15),
        [f"Rank {rank}" for rank in range(1, 16)],
        rotation=45,
        ha="right",
    )
    axis.set_yticks(
        range(3),
        [
            f"{name} (n={sample_size})"
            for name, sample_size in zip(approach_names, sample_sizes)
        ],
    )
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            if np.isfinite(value):
                axis.text(
                    column_index,
                    row_index,
                    f"{value:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontweight="bold",
                )
    figure.colorbar(image, ax=axis, label="Goal difference per game")
    axis.set_title("Ranked goal differences for every evaluated agent")
    created.append(save_figure(figure, "04_goal_difference_heatmap.png"))

    # Figure 5: the first-balanced approach's shaped reward versus match result.
    correlation = balanced[
        ["goal_difference", "average_reward_per_game"]
    ].corr().iloc[0, 1]
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(
        balanced["goal_difference"],
        balanced["average_reward_per_game"],
        s=80,
        color=BALANCED_COLOUR,
        alpha=0.9,
    )
    for _, row in balanced.iterrows():
        axis.annotate(
            row["agent"],
            (row["goal_difference"], row["average_reward_per_game"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axis.axvline(0, color="black", linewidth=1)
    axis.axhline(0, color="black", linewidth=1)
    axis.set_xlabel("Goal difference per game")
    axis.set_ylabel("Average evaluation reward per game")
    axis.set_title(
        "First-balanced approach: reward did not reliably track match performance"
    )
    axis.text(
        0.02,
        0.97,
        f"Pearson r = {correlation:.2f}",
        transform=axis.transAxes,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#CCCCCC"},
    )
    created.append(save_figure(figure, "05_reward_vs_goal_difference.png"))

    print("Created README figures:")
    for path in created:
        print(f"  {path.relative_to(REPOSITORY_ROOT)}")


if __name__ == "__main__":
    main()
