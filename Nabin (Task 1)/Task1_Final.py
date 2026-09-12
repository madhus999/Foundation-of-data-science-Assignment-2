"""
Task 1: Shooting Efficiency — Finishing Performance Relative to Expected Goals (xG)

Analytic Question: Among World Cup 2026 players who registered a non-zero Expected
Goals (xG) value, is the average Goals-to-xG ratio significantly different from 1.0
-- the value that would indicate finishing exactly in line with shot quality?
As a secondary investigation, does this ratio differ between forwards and midfielders?

Data source: FIFA official World Cup 2026 statistics hub, Attacking category.
Data was copied manually from the website's table and parsed in Python.

"""

import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))

np.random.seed(42)


# =====================================================================
# 1. DATA WRANGLING -- Manual Data Acquisition
# =====================================================================
# The Attacking stats table was manually selected and copied from FIFA's
# website, then parsed in Python into a structured table.

df = pd.read_csv("fifa_attacking_stats_real.csv")
print("Loaded! Shape:", df.shape)
print(df.head(10))


# =====================================================================
# 2. BUILDING THE ANALYSIS VARIABLE AND APPLYING EXCLUSIONS
# =====================================================================
# The "xG Efficiency" field is published as text (e.g. "1.53x"), so it is
# cleaned into a numeric Goals-to-xG ratio. Players with an xG value of 0
# are excluded, since the ratio is undefined without any registered shot
# quality.

df['goal_xg_ratio'] = df['xG Efficiency'].str.replace('x', '', regex=False).astype(float)
df_clean_xg = df[df['xG'] > 0].copy()
df_clean_xg.to_csv("fifa_attacking_stats_cleaned.csv", index=False)
print("\nPopulation size:", len(df_clean_xg))
print("Population mean Goals/xG ratio:", round(df_clean_xg['goal_xg_ratio'].mean(), 3))


# =====================================================================
# 3. DATA PREPARATION AND SAMPLING
# =====================================================================
# A simple random sample of n = 30 is drawn from this population using a
# fixed random seed for reproducibility. This sample size satisfies the
# general Central Limit Theorem threshold for approximating the sampling
# distribution of the mean as normal.

sample_xg = df_clean_xg.sample(n=30, random_state=42)
print("\nSample size:", len(sample_xg))
print(sample_xg[['player_name', 'position', 'xG', 'goal_xg_ratio']].head(10))


# =====================================================================
# 4. DESCRIPTIVE STATISTICS, CONFIDENCE INTERVAL, AND ONE-SAMPLE T-TEST
# =====================================================================
# H0: population mean Goals/xG ratio = 1.0
# H1: population mean Goals/xG ratio != 1.0
#
# Note: the one-sample t-test reuses the same sample drawn for the
# confidence interval above, since a one-sample test only requires a
# single sample of data.

ratio = sample_xg['goal_xg_ratio']

print("\n=== DESCRIPTIVE STATISTICS ===")
print(ratio.describe())
print("Skewness:", round(ratio.skew(), 3))

print("\n=== 95% CONFIDENCE INTERVAL ===")
mean_xg = ratio.mean()
sem_xg = stats.sem(ratio)
ci_xg = stats.t.interval(confidence=0.95, df=29, loc=mean_xg, scale=sem_xg)
print(f"Sample mean: {mean_xg:.3f}   95% CI: ({ci_xg[0]:.3f}, {ci_xg[1]:.3f})")

print("\n=== ONE-SAMPLE T-TEST (vs 1.0) ===")
t_stat_xg, p_value_xg = stats.ttest_1samp(ratio, popmean=1.0)
print(f"t = {t_stat_xg:.4f}, p = {p_value_xg:.4f}")
print("Decision:", "Reject H0" if p_value_xg < 0.05 else "Fail to reject H0")


# =====================================================================
# 5. TWO-SAMPLE T-TEST -- FORWARDS VS. MIDFIELDERS
# =====================================================================
# H0: mean ratio (FW) = mean ratio (MF)
# H1: mean ratio (FW) != mean ratio (MF)
#
# Unlike the one-sample test above, a two-sample t-test requires two
# genuinely separate samples, since it compares two distinct groups
# rather than a single sample against a fixed value -- so independent
# samples are drawn from the forward and midfielder populations below.
# Levene's test checks the equal-variance assumption first.

fw_pop_xg = df_clean_xg[df_clean_xg['position'] == 'FW']['goal_xg_ratio']
mf_pop_xg = df_clean_xg[df_clean_xg['position'] == 'MF']['goal_xg_ratio']
print("\nFW population:", len(fw_pop_xg), " MF population:", len(mf_pop_xg))

n_fw2, n_mf2 = min(20, len(fw_pop_xg)), min(15, len(mf_pop_xg))
fw_sample_xg = fw_pop_xg.sample(n=n_fw2, random_state=42)
mf_sample_xg = mf_pop_xg.sample(n=n_mf2, random_state=42)
print(f"FW sample (n={n_fw2}) mean: {fw_sample_xg.mean():.3f}")
print(f"MF sample (n={n_mf2}) mean: {mf_sample_xg.mean():.3f}")

levene_stat2, levene_p2 = stats.levene(fw_sample_xg, mf_sample_xg)
equal_var2 = levene_p2 > 0.05
print(f"Levene p={levene_p2:.4f} -> equal_var={equal_var2}")

t_stat3, p_value3 = stats.ttest_ind(fw_sample_xg, mf_sample_xg, equal_var=equal_var2)
print(f"t = {t_stat3:.4f}, p = {p_value3:.4f}")
print("Decision:", "Reject H0" if p_value3 < 0.05 else "Fail to reject H0")


# =====================================================================
# 6. VISUALISING THE DISTRIBUTION
# =====================================================================
# A histogram of the sample's Goals/xG ratio, with the xG-neutral
# benchmark (1.0) and sample mean marked.

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(sample_xg['goal_xg_ratio'], bins=10, color='#1F3864', edgecolor='white')
ax.axvline(1.0, color='#C00000', linestyle='--', linewidth=2, label='xG-neutral (ratio = 1.0)')
ax.axvline(sample_xg['goal_xg_ratio'].mean(), color='#548235', linestyle='-', linewidth=2,
           label=f"Sample mean = {sample_xg['goal_xg_ratio'].mean():.2f}")
ax.set_xlabel('Goals ÷ Expected Goals ratio')
ax.set_ylabel('Number of players')
ax.set_title('Distribution of Goals-to-Expected-Goals Ratio (Sample, n=30)')
ax.legend()
plt.tight_layout()
plt.savefig('xg_ratio_distribution.png', dpi=150)
plt.show()


# =====================================================================
# CONCLUSION
# =====================================================================
# - One-sample t-test (vs 1.0): t = 1.721, p = 0.096 -> fail to reject H0.
#   No significant evidence that players over- or under-performed their
#   expected goals as a group.
# - Two-sample t-test (FW vs MF): t = -0.280, p = 0.782 -> fail to reject H0.
#   No significant difference between forwards' and midfielders'
#   finishing performance relative to expected goals.
# - The sample distribution is strongly right-skewed: most players
#   convert few or none of their high-quality chances, while a small
#   number of standout finishers substantially outperform expectation.
#
# Limitation: the population reflects FIFA's published Attacking
# leaderboard (players with >=1 assist), not every player in the
# tournament who registered an xG value.

print("\n=== DONE ===")