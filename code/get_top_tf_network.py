import pandas as pd
import os
import ast


os.chdir('..')
df = pd.read_csv('shap_with_analysis.csv', encoding='gbk')
results = []
df['key_TFs'] = df['key_TFs'].apply(ast.literal_eval)
for index, row in df.iterrows():
    tg = row['TG']
    key_TFs = row['key_TFs']
    sorted_TFs = list(key_TFs.keys())
    top_TFs = sorted_TFs[:1] if len(sorted_TFs) >= 1 else sorted_TFs
    for tf in top_TFs:
        results.append([tf, tg, 1])
output_df = pd.DataFrame(results, columns=['TF', 'TG', 'num'])
output_df.to_csv('top_1_TFs_for_TGs.csv', index=False)

