import scipy.stats as stats
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_spearman_ranks(df1, df2):
    """
    Calculate Spearman's rank correlation coefficient between two dataframes.

    Parameters:
    df1 (pandas.DataFrame): First dataframe.
    df2 (pandas.DataFrame): Second dataframe.

    Returns:
    list: List of Spearman's rank correlation coefficients.
    """
    filtered_df1 = df1.loc[df1.index.intersection(df2.index)]
    filtered_df2 = df2.loc[df2.index.intersection(df1.index)]
    spearman_rs = []
    for idx, row in filtered_df1.iterrows():
        ranks1 = row.values
        ranks2 = filtered_df2.loc[idx].values
        spearman_r = stats.spearmanr(ranks1, ranks2)
        spearman_rs.append(spearman_r.statistic)
    return spearman_rs
def calculate_percent_overlap(df1, df2, top_n=5):
    """
    Calculate the percentage overlap between the top_n items in two DataFrames.

    Parameters:
    df1 (DataFrame): The first DataFrame.
    df2 (DataFrame): The second DataFrame.
    top_n (int): The number of top items to consider for overlap calculation. Default is 5.

    Returns:
    list: A list of percentage overlaps for each row in df1.
    """
    filtered_df1 = df1.loc[df1.index.intersection(df2.index)]
    filtered_df2 = df2.loc[df2.index.intersection(df1.index)]
    percent_overlaps = []
    for idx, row in filtered_df1.iterrows():
        items1 = set(row.values[:top_n])
        items2 = set(filtered_df2.loc[idx].values[:top_n])
        overlap = len(items1.intersection(items2))
        percent_overlap = (overlap / top_n) * 100
        percent_overlaps.append(percent_overlap)
    return percent_overlaps

rank_arrs={}
for rank_path in glob.glob('results/*rank_df.csv'):
    rank_df = pd.read_csv(rank_path,index_col=0)
    rating_var = rank_path.split('/')[1].split('_')[1]
    rank_arrs[rating_var] = rank_df
# fear - hp
spearman_rs_dict = {}
spearman_rs_dict['fear-hp']=calculate_spearman_ranks(rank_arrs['fear'], rank_arrs['hp'])
spearman_rs_dict['fear-scr']=calculate_spearman_ranks(rank_arrs['fear'], rank_arrs['scr'])
spearman_rs_dict['hp-scr']=calculate_spearman_ranks(rank_arrs['hp'], rank_arrs['scr'])

percent_overlap_dict = {}
percent_overlap_dict['fear-hp']=calculate_percent_overlap(rank_arrs['fear'], rank_arrs['hp'])
percent_overlap_dict['fear-scr']=calculate_percent_overlap(rank_arrs['fear'], rank_arrs['scr'])
percent_overlap_dict['hp-scr']=calculate_percent_overlap(rank_arrs['hp'], rank_arrs['scr'])

sns.swarmplot(data=[spearman_rs_dict['fear-hp'], spearman_rs_dict['fear-scr'], spearman_rs_dict['hp-scr']])
sns.barplot(data=[spearman_rs_dict['fear-hp'], spearman_rs_dict['fear-scr'], spearman_rs_dict['hp-scr']],alpha=0.33)
plt.xticks([0,1,2],['fear-hp','fear-scr','hp-scr'])
plt.xlabel('Comparison')
plt.ylabel('Spearman R')
plt.savefig('figures/spearman_r.png')
plt.show()


sns.swarmplot(data=[percent_overlap_dict['fear-hp'], percent_overlap_dict['fear-scr'], percent_overlap_dict['hp-scr']])
sns.barplot(data=[percent_overlap_dict['fear-hp'], percent_overlap_dict['fear-scr'], percent_overlap_dict['hp-scr']],alpha=0.33)
plt.xticks([0,1,2],['fear-hp','fear-scr','hp-scr'])
plt.xlabel('Comparison')
plt.ylabel('Percent Overlap')
plt.savefig('figures/percent_overlap.png')
plt.show()