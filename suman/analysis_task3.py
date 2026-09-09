"""
HIT140 - Group 78 - Task 3 (Suman)
FIFA World Cup 2026 - Goalkeeper Save Percentage
(CSV version - reads goalkeeping_dataset.csv, the raw two-row-header export)

===============================================================================
ANALYTIC TASK STRUCTURE (matches assignment brief, Objective 1)
===============================================================================

1. ANALYTIC QUESTION FORMULATION
   Original brief question: "Is average goalkeeper distribution accuracy
   different from 85%?"
   Data available (goalkeeping_dataset.csv, FBref.com export) does not
   contain a distribution accuracy field. It contains Save% instead,
   defined by the source as:
       (Shots on Target Against - Goals Against) / Shots on Target Against
   This is a shot-stopping statistic, not a distribution (passing) statistic.
   Adjusted question actually answered by this script:
       On average, was a squad's goalkeeper save percentage at the 2026
       World Cup significantly different from 85%?
   Focal point: goalkeeper shot-stopping accuracy. Distinct from any task
   in this assignment that focuses on goals scored.

2. DATA WRANGLING
   Source: goalkeeping_dataset.csv - raw export with a two-row header
   (section row: Playing Time / Performance / Penalty Kicks, then the
   actual column names). Header is flattened, squad names split into a
   country code and full team name, columns renamed, then validated.
   See load_and_wrangle().

3. DATA PREPARATION AND SAMPLING
   Population: all 48 squads at the 2026 World Cup with recorded save
   percentage data.
   Sample: simple random sample, n=30, drawn without replacement, fixed
   seed for reproducibility.
   See draw_sample().

4. DESCRIPTIVE STATISTICS
   Mean, median, standard deviation, min, max, quartiles - computed for
   both the population and the drawn sample.
   See describe().

5. INFERENTIAL STATISTICS - CONFIDENCE INTERVAL
   95% confidence interval for the population mean save percentage,
   built from the sample using the t-distribution.
   See confidence_interval().

6. INFERENTIAL STATISTICS - ONE-SAMPLE T-TEST
   H0: population mean save% = 85 (the brief's benchmark)
   H1: population mean save% != 85
   See main(), stats.ttest_1samp() call.
===============================================================================

Run:  python analysis_task3_v2.py
Needs: pandas, numpy, scipy, matplotlib   (plain CSV, no lxml needed)
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
CSV_FILE = "goalkeeping_dataset.csv"   # place this file in the same folder
CLEAN_CSV = "gk_squads.csv"
BENCHMARK = 85.0                        # H0: population mean save% = 85 (from brief)
SAMPLE_SIZE = 30
CONFIDENCE = 0.95


# -----------------------------------------------------------------------
# 2. DATA WRANGLING
# -----------------------------------------------------------------------
def load_and_wrangle(csv_path):
    """Read the raw two-header CSV export, flatten headers, clean it up."""
    df = pd.read_csv(csv_path, header=[0, 1])

    # Flatten the two-row header, e.g. ('Performance','GA') -> 'Performance_GA'.
    # This source file has a duplicate column name (Save% appears once for
    # goalkeeper save rate, once under Penalty Kicks for PK save rate), so
    # duplicates are de-duplicated with a numeric suffix before use.
    flat = [
        "_".join(c).strip("_") if c[0] and "Unnamed" not in c[0] else c[1]
        for c in df.columns
    ]
    seen = {}
    deduped = []
    for name in flat:
        if name in seen:
            seen[name] += 1
            deduped.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            deduped.append(name)
    df.columns = deduped

    # Squad column is "<fbref_code> <Country Name>", e.g. "ar Argentina"
    split = df["Squad"].str.split(n=1, expand=True)
    df["code"] = split[0]
    df["team"] = split[1]

    keep = df[[
        "team", "code", "# Pl", "Playing Time_MP", "Starts", "Min", "90s",
        "Performance_GA", "GA90", "SoTA", "Saves", "Save%", "W", "D", "L",
        "CS", "CS%",
    ]].copy()
    keep.columns = [
        "team", "code", "players_used", "matches_played", "starts", "minutes",
        "nineties", "goals_against", "goals_against_per90",
        "shots_on_target_against", "saves", "save_pct", "wins", "draws",
        "losses", "clean_sheets", "clean_sheet_pct",
    ]

    # data wrangling checks
    assert keep["save_pct"].notna().all(), "missing save_pct values"
    assert keep["team"].is_unique, "duplicate squads"
    assert len(keep) == 48, f"expected 48 squads, got {len(keep)}"

    keep.to_csv(CLEAN_CSV, index=False)
    return keep


# -----------------------------------------------------------------------
# 3. DATA PREPARATION AND SAMPLING
# -----------------------------------------------------------------------
def draw_sample(population, n, seed):
    """Population = all 48 squads. Simple random sample, no replacement."""
    idx = np.random.default_rng(seed).choice(len(population), size=n, replace=False)
    return population[idx]


# -----------------------------------------------------------------------
# 4. DESCRIPTIVE STATISTICS
# -----------------------------------------------------------------------
def describe(x):
    s = pd.Series(x)
    return {
        "n": len(s), "mean": s.mean(), "median": s.median(), "std": s.std(ddof=1),
        "min": s.min(), "max": s.max(), "q1": s.quantile(.25), "q3": s.quantile(.75),
    }


# -----------------------------------------------------------------------
# 5. INFERENTIAL STATISTICS - CONFIDENCE INTERVAL
# -----------------------------------------------------------------------
def confidence_interval(sample, confidence=0.95):
    n = len(sample)
    mean = np.mean(sample)
    sem = stats.sem(sample)
    tcrit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    margin = tcrit * sem
    return mean, mean - margin, mean + margin, margin


def make_distribution_plot(df, out="fig1_distribution.png"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["save_pct"], bins=10, color="#1f4e79", edgecolor="white")
    ax.axvline(df["save_pct"].mean(), color="black", linewidth=1.5,
               label=f"Population mean ({df['save_pct'].mean():.2f}%)")
    ax.axvline(BENCHMARK, color="crimson", linestyle="--", linewidth=1.5,
               label=f"Benchmark ({BENCHMARK:.0f}%)")
    ax.set_xlabel("Goalkeeper save percentage (%)")
    ax.set_ylabel("Number of squads")
    ax.set_title("Distribution of squad save percentage\nFIFA World Cup 2026 (N=48)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)


def make_benchmark_plot(mean, margin, out="fig2_benchmark_gap.png"):
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.bar(["Sample mean\n(save%)"], [mean], yerr=[margin], capsize=8,
           color="#1f4e79", width=0.5)
    ax.axhline(BENCHMARK, color="crimson", linestyle="--", linewidth=1.5,
               label=f"Benchmark ({BENCHMARK:.0f}%)")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Save percentage (%)")
    ax.set_title(f"Sample mean (95% CI) vs. {BENCHMARK:.0f}% benchmark")
    ax.legend(fontsize=8, loc="upper left")
    ax.text(0, mean + 3, f"{mean:.2f}%", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)


def main():
    # ---- 1. ANALYTIC QUESTION FORMULATION ----
    print("=" * 70)
    print("1. ANALYTIC QUESTION FORMULATION")
    print("=" * 70)
    print("Brief question : Is average goalkeeper distribution accuracy")
    print("                  different from 85%?")
    print("Data available  : goalkeeping_dataset.csv has no distribution")
    print("                  accuracy field; it has Save% instead.")
    print("Question tested : Is average squad save percentage different")
    print("                  from 85%?\n")

    # ---- 2. DATA WRANGLING ----
    print("=" * 70)
    print("2. DATA WRANGLING")
    print("=" * 70)
    df = load_and_wrangle(CSV_FILE)
    print(f"Loaded and cleaned {len(df)} squads from {CSV_FILE}")
    print(f"Saved cleaned file: {CLEAN_CSV}\n")

    # ---- 3. DATA PREPARATION AND SAMPLING ----
    print("=" * 70)
    print("3. DATA PREPARATION AND SAMPLING")
    print("=" * 70)
    population = df["save_pct"].values
    sample = draw_sample(population, SAMPLE_SIZE, SEED)
    print(f"Population: N = {len(population)} squads (all World Cup 2026 teams)")
    print(f"Sample    : n = {len(sample)} squads (simple random sample, seed={SEED})\n")

    # ---- 4. DESCRIPTIVE STATISTICS ----
    print("=" * 70)
    print("4. DESCRIPTIVE STATISTICS")
    print("=" * 70)
    pop_stats = describe(population)
    sample_stats = describe(sample)
    print(f"Population (N={pop_stats['n']}): mean={pop_stats['mean']:.2f}%  "
          f"median={pop_stats['median']:.2f}%  sd={pop_stats['std']:.2f}pts  "
          f"range=[{pop_stats['min']:.1f}%, {pop_stats['max']:.1f}%]")
    print(f"Sample     (n={sample_stats['n']}): mean={sample_stats['mean']:.2f}%  "
          f"median={sample_stats['median']:.2f}%  sd={sample_stats['std']:.2f}pts  "
          f"range=[{sample_stats['min']:.1f}%, {sample_stats['max']:.1f}%]\n")

    # ---- 5. INFERENTIAL STATISTICS - CONFIDENCE INTERVAL ----
    print("=" * 70)
    print("5. INFERENTIAL STATISTICS - CONFIDENCE INTERVAL")
    print("=" * 70)
    mean, ci_lo, ci_hi, margin = confidence_interval(sample, CONFIDENCE)
    print(f"95% CI for population mean save%: ({ci_lo:.2f}%, {ci_hi:.2f}%)\n")

    # ---- 6. INFERENTIAL STATISTICS - ONE-SAMPLE T-TEST ----
    print("=" * 70)
    print("6. INFERENTIAL STATISTICS - ONE-SAMPLE T-TEST")
    print("=" * 70)
    print(f"H0: mu = {BENCHMARK:.0f}%   H1: mu != {BENCHMARK:.0f}%")
    t_stat, p_val = stats.ttest_1samp(sample, popmean=BENCHMARK)
    print(f"t({SAMPLE_SIZE-1}) = {t_stat:.3f}, p-value = {p_val:.4g}")
    decision = "Reject H0" if p_val < 0.05 else "Fail to reject H0"
    print(f"Decision at alpha=0.05: {decision}")

    make_distribution_plot(df)
    make_benchmark_plot(mean, margin)

    pd.DataFrame([{
        "population_n": pop_stats["n"], "population_mean_pct": pop_stats["mean"],
        "sample_n": sample_stats["n"], "sample_mean_pct": sample_stats["mean"],
        "ci_95_low_pct": ci_lo, "ci_95_high_pct": ci_hi, "benchmark_pct": BENCHMARK,
        "t_stat": t_stat, "p_value": p_val, "decision": decision,
    }]).to_csv("task3_results_summary.csv", index=False)

    print("\nSaved: gk_squads.csv, fig1_distribution.png, fig2_benchmark_gap.png, "
          "task3_results_summary.csv")


if __name__ == "__main__":
    main()
