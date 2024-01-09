import pandas as pd
import glob
from helpers import create_val_loss_list_across_models


def main():
    mean_model = pd.read_csv('results/mean_model_mse.csv',index_col=0)
    for target_var in ['resp_fear','video_hp','video_scr']:
        mean_mse = mean_model.loc[target_var].values[0]
        valid_loss_vals, n_factors = create_val_loss_list_across_models(target_var)
        valid_loss_vals.insert(0,mean_mse), n_factors.insert(0,'mean')
        df = pd.DataFrame({'n_factors':n_factors,'valid_loss':valid_loss_vals})
        df.to_csv(f'results/{target_var}_losses.csv')

if __name__ == '__main__':
    main()
