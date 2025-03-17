import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm

os.chdir('..')
df1 = pd.read_csv('valid_direction_labels_up.csv')
df2 = pd.read_csv('valid_direction_labels_down.csv')

TFs = df1['TF'].unique()
TGs = df1['TG'].unique()

data_combined = {TF: [0] * len(TGs) for TF in TFs}
data_combined = pd.DataFrame(data_combined, index=TGs)

for j in range(len(df1)):
    if df1['Correct'][j]:
        TF = df1['TF'][j]
        TG = df1['TG'][j]
        data_combined.loc[TG, TF] = 1
    elif not df1['Correct'][j]:
        TF = df1['TF'][j]
        TG = df1['TG'][j]
        data_combined.loc[TG, TF] = -1

for j in range(len(df2)):
    if df2['Correct'][j]:
        TF = df2['TF'][j]
        TG = df2['TG'][j]
        if data_combined.loc[TG, TF] == 1:
            data_combined.loc[TG, TF] = 1
        else:
            data_combined.loc[TG, TF] = 2
    elif not df2['Correct'][j]:
        TF = df2['TF'][j]
        TG = df2['TG'][j]
        if data_combined.loc[TG, TF] == -1:
            data_combined.loc[TG, TF] = -1
        else:
            data_combined.loc[TG, TF] = 2

cmap = ListedColormap(["blue", "white", "red", "gray"])
norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5], cmap.N)

plt.figure(figsize=(12, 12))
plt.rcParams['font.family'] = 'Arial'
data_subset = data_combined.iloc[:50, :50]
sns.heatmap(data_subset, cmap=cmap, xticklabels=True, yticklabels=True, norm=norm,
            cbar_kws={'ticks': [-1, 0, 1, 2], 'shrink': 0.5}, linewidths=0.1, linecolor='black')

cbar = plt.gca().collections[0].colorbar
cbar.outline.set_edgecolor("black")
cbar.outline.set_linewidth(1.5)
cbar.set_ticks([-1, 0, 1, 2])
cbar.set_ticklabels(['Incorrect',  'No labels',  'Correct', 'Other'])
cbar.ax.tick_params(labelsize=18)

plt.grid(True, which='both', axis='both', color='black', linestyle='-', linewidth=0.1)
plt.title('TG response simulation', fontsize=30)
plt.xlabel('TF', fontsize=28)
plt.ylabel('TG', fontsize=28)
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout(pad=2)
plt.margins(0, 0)
plt.savefig('heatmap_comparison.svg', bbox_inches='tight')
plt.show()
