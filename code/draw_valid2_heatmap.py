import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import pandas
import csv
import os

if __name__ == '__main__':
    os.chdir('..')
    df = pd.read_csv('merged_valid_ko_label_results.csv')
    TFs = df['TF'].unique()
    TGs = df['TG'].unique()
    data = {TF: [0] * len(TGs) for TF in TFs}
    data = pd.DataFrame(data, index=TGs)
    for j in range(len(df)):
        if df['Correct'][j]:
            TF = df['TF'][j]
            TG = df['TG'][j]
            data.loc[TG, TF] = 1
            # data.loc[TG, TF] = df['fail rate'][j]/100
        elif not df['Correct'][j]:
            TF = df['TF'][j]
            TG = df['TG'][j]
            data.loc[TG, TF] = -1
            # data.loc[TG, TF] = -df['fail rate'][j]/100
    plt.figure(figsize=(12, 12))
    plt.rcParams['font.family'] = 'Arial'
    data_subset = data.iloc[:50, :50]
    cmap = ListedColormap(["blue", "white", "red"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    ax = sns.heatmap(data_subset, cmap=cmap, xticklabels=True, yticklabels=True, norm=norm, cbar_kws={'ticks': [-1, 0, 1], 'shrink': 0.5}, linewidths=0.1, linecolor='black')
    cbar = ax.collections[0].colorbar
    cbar.outline.set_edgecolor("black")
    cbar.outline.set_linewidth(1.5)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(['Incorrect', 'No labels', 'Correct'])
    plt.title('TF knockout simulation', fontsize=30)
    plt.xlabel('TF', fontsize=28)
    plt.ylabel('TG', fontsize=28)
    plt.rcParams['font.family'] = 'Arial'
    cbar.ax.tick_params(labelsize=18)
    plt.grid(True, which='both', axis='both', color='black', linestyle='-', linewidth=0.1)
    plt.tight_layout(pad=2)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.margins(0, 0)
    plt.savefig('heatmap2.svg', bbox_inches='tight')
    plt.show()
