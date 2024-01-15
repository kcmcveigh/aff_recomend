# %%
from helpers import *
# %%
def main():
    # Set up parameters
    item_name = 'video'
    user_name = 'participant_num'
    n_epochs = 20
    file_path = 'full_logfile_scr_0.csv'
    rating_names = ['resp_fear','video_scr','video_hp']
    model_path = '{rating_name}_{epochs}_{n_factors}_model'

    for rating_name in rating_names:
        
        df = load_csv(file_path, rating_name)
        
        dls = create_data_loader(df, item_name, user_name, rating_name)

        n_users, n_items = get_n_users(dls, user_name),get_n_items(dls, item_name)

        y_range = get_rating_range(df, rating_name)

        n_factors = get_n_factors_for_min_loss(rating_name)

        model = create_cf_model(n_users, n_items, n_factors, y_range=y_range)

        model_path_formatted = model_path.format(rating_name=rating_name,epochs=n_epochs,n_factors=n_factors)

        trained_model = load_saved_model_weights(model, dls, model_path_formatted)

        pred_df = reconstruct_matrix(trained_model, y_range)
        
        rank_series = pred_df.apply(lambda row: np.argsort(-row.values), axis=1)
        rank_df = pd.DataFrame(np.vstack(rank_series.values))
        rank_df.to_csv(f'results/{rating_name}_rank_df.csv')

    
#%%
if __name__ == '__main__':
    main()

# %%
