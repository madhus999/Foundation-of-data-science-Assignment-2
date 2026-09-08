import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from pathlib import Path

# Find the CSV beside this Python file so the script works from any folder.
data_file = Path(__file__).resolve().parent / "fifa_wc2026_task4.csv"
df = pd.read_csv(data_file)

# Display a quick overview of the dataset and the positions it contains.
print(df.head())
print(df.shape)
print(df["Position"].value_counts())

# Check every column for empty or missing values.
print("\nMissing values:")
print(df.isnull().sum())

# Create separate dataframes containing only defenders or midfielders.
defenders = df[df["Position"] == "DF"]
midfielders = df[df["Position"] == "MF"]

print("\nTotal defenders:", len(defenders))
print("Total midfielders:", len(midfielders))

# Select 30 random players from each group. The random_state makes the
# selection repeatable, so the same players are chosen each time.
sample_defenders = defenders.sample(
    n=30,
    random_state=140
)

sample_midfielders = midfielders.sample(
    n=30,
    random_state=140
)

sample = pd.concat(
    [sample_defenders, sample_midfielders]
)

# Confirm that the combined sample contains 30 players from each position.
print("\nSample size:")
print(sample["Position"].value_counts())

# Save the 60 selected players in the same folder as this script.
sample.to_csv(
    Path(__file__).resolve().parent / "task4_sample_60_players.csv",
    index=False
)

# Compare the passing-accuracy distributions for the two positions.

plt.boxplot(
    [
        sample_defenders["Passing_Accuracy_Percent"],
        sample_midfielders["Passing_Accuracy_Percent"]
    ],
    tick_labels=[
        "Defenders",
        "Midfielders"
    ]
)

plt.title("Passing Accuracy: Defenders vs Midfielders")

plt.ylabel("Passing Accuracy (%)")

plt.xlabel("Player Position")

plt.show()

def confidence_interval(data):
    """Return the 95% confidence interval for a sample mean."""
    # Calculate the sample mean and its standard error.
    mean = np.mean(data)
    standard_error = stats.sem(data)

    # Use the t-distribution because the population standard deviation is unknown.
    margin_of_error = stats.t.ppf(
        0.975,
        df=len(data) - 1
    ) * standard_error

    lower = mean - margin_of_error
    upper = mean + margin_of_error

    return lower, upper


# Calculate a confidence interval for each position group.
defender_ci = confidence_interval(
    sample_defenders["Passing_Accuracy_Percent"]
)

midfielder_ci = confidence_interval(
    sample_midfielders["Passing_Accuracy_Percent"]
)


# Display the range in which the true average passing accuracy is estimated
# to fall, using a 95% confidence level.
print("\n95% Confidence Intervals")

print(
    "Defenders:",
    round(defender_ci[0], 2),
    "to",
    round(defender_ci[1], 2)
)

print(
    "Midfielders:",
    round(midfielder_ci[0], 2),
    "to",
    round(midfielder_ci[1], 2)
)

# Welch Two-Sample t-Test

t_test = stats.ttest_ind(
    sample_defenders["Passing_Accuracy_Percent"],
    sample_midfielders["Passing_Accuracy_Percent"],
    equal_var=False,
    alternative="greater"
)

print("\nWelch Two-Sample t-Test")

print(
    "t-statistic:",
    round(t_test.statistic, 3)
)

print(
    "p-value:",
    round(t_test.pvalue, 4)
)

# Decision

alpha = 0.05

if t_test.pvalue < alpha:

    print("Decision: Reject H0")

    print(
        "Conclusion: There is statistically significant "
        "evidence that defenders have higher average "
        "passing accuracy than midfielders."
    )

else:

    print("Decision: Fail to reject H0")

    print(
        "Conclusion: There is not enough statistical "
        "evidence to conclude that defenders have higher "
        "average passing accuracy than midfielders."
    )