import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

df = pd.read_csv('wc2026_player_misc.csv')
df['PrimaryPos'] = df['Pos'].apply(lambda p: str(p).split(',')[0].strip())
df2 = df[df['PrimaryPos'].isin(['MF','DF'])].copy()

mf = df2.loc[df2['PrimaryPos']=='MF', 'Fouls_per90'].dropna()
dfn = df2.loc[df2['PrimaryPos']=='DF', 'Fouls_per90'].dropna()

print('MF n=', len(mf), 'DF n=', len(dfn))
print(mf.describe())
print(dfn.describe())

print('Shapiro MF:', stats.shapiro(mf))
print('Shapiro DF:', stats.shapiro(dfn))
levene = stats.levene(mf, dfn)
print('Levene:', levene)
equal_var = levene.pvalue > 0.05

def ci95(sample):
    m = sample.mean()
    se = stats.sem(sample)
    return m, stats.t.interval(0.95, len(sample)-1, loc=m, scale=se)
print('MF CI:', ci95(mf))
print('DF CI:', ci95(dfn))

t_stat, p_two = stats.ttest_ind(mf, dfn, equal_var=equal_var)
p_one = p_two/2 if t_stat > 0 else 1 - p_two/2
print('t-stat:', t_stat, 'one-tailed p:', p_one)

pooled_std = np.sqrt(((len(mf)-1)*mf.std()**2 + (len(dfn)-1)*dfn.std()**2) / (len(mf)+len(dfn)-2))
d = (mf.mean()-dfn.mean())/pooled_std
print('Cohens d:', d)

plt.figure()
plt.boxplot([mf, dfn], tick_labels=['MF','DF'])
plt.ylabel('Fouls per 90')
plt.savefig('boxplot_fouls.png', dpi=150)
plt.close()

plt.figure()
plt.hist(mf, alpha=0.6, label='MF', bins=20)
plt.hist(dfn, alpha=0.6, label='DF', bins=20)
plt.legend()
plt.xlabel('Fouls per 90')
plt.savefig('hist_fouls.png', dpi=150)
plt.close()

print('Charts saved.')
