# Task 1: Shooting Efficiency — Finishing Performance Relative to Expected Goals (xG)

 Nabin Thapa
**Assessment:** HIT140 Foundations of Data Science — Group Project, Objective 1

---

## Analytic Question

Among World Cup 2026 players who registered a non-zero Expected Goals (xG) value, is the average Goals-to-xG ratio significantly different from 1.0 — the value that would indicate finishing exactly in line with shot quality? As a secondary investigation, does this ratio differ between forwards and midfielders?

---

## Data Source

FIFA official World Cup 2026 statistics hub, Attacking category:
`fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/statistics/player-statistics`

Data was manually selected and copied from the website's table, then parsed and cleaned in Python.

---


## Method Summary

1. **Data Wrangling** — Manually copied data cleaned and parsed into a structured table.
2. **Variable Construction & Exclusions** — Built a Goals ÷ xG ratio from FIFA's raw text field; excluded players with an xG of 0. Population: 88 players.
3. **Sampling** — Simple random sample of n = 30, fixed seed for reproducibility.
4. **Descriptive Statistics & Confidence Interval** — Summary statistics and a 95% CI for the population mean ratio.
5. **One-Sample t-Test** — Tests whether the sample mean differs from 1.0 (finishing exactly as expected).
6. **Two-Sample t-Test** — Compares forwards vs. midfielders on the same ratio.

---

## Results Summary

| Test | Result | Decision |
|---|---|---|
| One-sample t-test (vs. 1.0) | t = 1.721, p = 0.096 | Fail to reject H₀ |
| Two-sample t-test (FW vs. MF) | t = -0.280, p = 0.782 | Fail to reject H₀ |

**Conclusion:** No statistically significant evidence that players over- or under-performed their expected goals, either as a group or when compared by position. The sample distribution is strongly right-skewed — most players convert few or none of their high-quality chances, while a small number of standout finishers substantially outperform expectation.

---

## Limitations

- The population reflects FIFA's published Attacking leaderboard (players with ≥1 assist), not every player in the tournament who registered an xG value.
- Data reflects a single extraction at one point in time.
