# %%
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

def calculate_spearman_ranks_participant_pairs(df1, df2):
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
        for idx_2, row_2 in filtered_df2.iterrows():
            if idx == idx_2:
                continue
            ranks1 = row.values
            ranks2 = row_2.values
            spearman_r = stats.spearmanr(ranks1, ranks2)
            spearman_rs.append(spearman_r.statistic)
    return spearman_rs
def calculate_percent_overlap_participant_pairs(df1, df2, top_n=5):
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
        for idx_2, row_2 in filtered_df2.iterrows():
            if idx == idx_2:
                continue
            items1 = set(row.values[:top_n])
            items2 = set(row_2.values[:top_n])
            overlap = len(items1.intersection(items2))
            percent_overlap = (overlap / top_n) * 100
            percent_overlaps.append(percent_overlap)
    return percent_overlaps

rank_arrs={}
for rank_path in glob.glob('results/*rank_df.csv'):
    rank_df = pd.read_csv(rank_path,index_col=0)
    rating_var = rank_path.split('/')[1].split('_')[1]
    rank_arrs[rating_var] = rank_df

# %%
# fear - hp
spearman_rs_dict = {}
spearman_rs_dict['fear']=calculate_spearman_ranks_participant_pairs(rank_arrs['fear'], rank_arrs['fear'])
spearman_rs_dict['scr']=calculate_spearman_ranks_participant_pairs(rank_arrs['scr'], rank_arrs['scr'])
spearman_rs_dict['hp']=calculate_spearman_ranks_participant_pairs(rank_arrs['hp'], rank_arrs['hp'])

percent_overlap_dict = {}
percent_overlap_dict['fear']=calculate_percent_overlap_participant_pairs(rank_arrs['fear'], rank_arrs['fear'])
percent_overlap_dict['scr']=calculate_percent_overlap_participant_pairs(rank_arrs['scr'], rank_arrs['scr'])
percent_overlap_dict['hp']=calculate_percent_overlap_participant_pairs(rank_arrs['hp'], rank_arrs['hp'])

# %%
sns.violinplot(data=[spearman_rs_dict['fear'], spearman_rs_dict['scr'], spearman_rs_dict['hp']])
plt.xticks([0,1,2],['fear','scr','hp'])
plt.xlabel('Comparison')
plt.ylabel('Spearman R')
plt.savefig('figures/participants_spearman_r.png')
plt.show()


sns.violinplot(data=[percent_overlap_dict['fear'], percent_overlap_dict['scr'], percent_overlap_dict['hp']])
plt.xticks([0,1,2],['fea','scr','hp'])
plt.xlabel('Comparison')
plt.ylabel('Percent Overlap')
plt.savefig('figures/participants_percent_overlap.png')
plt.show()
# %%
