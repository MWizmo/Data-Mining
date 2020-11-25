from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd


seeds_df = pd.read_csv("http://qps.ru/jNZUT")
varieties = list(seeds_df.pop('grain_variety'))
varieties = [v.split()[0] for v in varieties]
samples = seeds_df.values
methods = [['single', 'Single linkage'], ['complete', 'Complete linkage'], ['average', 'Group average'],
           ['ward', 'Расстояние Уорда']]
for method in methods:
    mergings = linkage(samples, method=method[0])
    dendrogram(mergings, labels=varieties, leaf_rotation=90, leaf_font_size=6)
    plt.title(method[1])
    plt.show()