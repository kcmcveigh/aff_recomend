# %%
import pandas as pd
import numpy as np
import glob

# %%
embeddings =[]
for factor_csv in glob.glob('*_video_factors.csv'):
   df = pd.read_csv(factor_csv, index_col=0)
   embeddings.append(df.values.reshape(1,-1)[0])
print(np.corrcoef(np.vstack(embeddings)))
# %%
par_factor_df = []
for factor_csv in glob.glob('*_participant_factors.csv'):
   df = pd.read_csv(factor_csv, index_col=0)
   par_factor_df.append(df)
par_factor_df = pd.concat(par_factor_df,axis=1)
par_factor_df.dropna(inplace=True)
par_fact_1 = par_factor_df.values[:,:2].reshape(1,-1)[0]
par_fact_2 = par_factor_df.values[:,2:4].reshape(1,-1)[0]
par_fact_3 = par_factor_df.values[:,4:].reshape(1,-1)[0]
print(np.corrcoef(np.vstack([par_fact_1,par_fact_2,par_fact_3])))

# %%
