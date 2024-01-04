from helpers import *

def convert_dense_matrix_to_df(dense_matrix, item_name, user_name, rating_name):
    sim_df = pd.DataFrame(dense_matrix)
    melted_df = sim_df.reset_index().melt(id_vars='index', var_name=item_name, value_name=rating_name)
    melted_df.columns = [user_name, item_name, rating_name]
    return melted_df

def main():
    # Set up parameters
    item_name = 'video'
    user_name = 'participant'
    rating_name = 'sim_rating'

    n_epochs = 20
    learning_rate = 5e-3 
    weight_decay = 0.1

    n_latent_factors = 3
    n_items_gen = 30
    n_people_gen = 100

    set_seed_wrapper(42)

    dense_matrix,_,_,_,_= simulate_collaborative_data(n_latent_factors,n_latent_factors,n_items_gen,n_people_gen)
    
    df = convert_dense_matrix_to_df(dense_matrix, item_name, user_name, rating_name)
    
    dls = create_data_loader(df, item_name, user_name, rating_name)

    n_users, n_items = get_n_users(dls, user_name),get_n_items(dls, item_name)

    for n_factors in range(1,7):

        cf_model = create_cf_model(n_users, n_items, n_factors)

        learn = train_model(cf_model, dls, n_epochs=n_epochs, lr=learning_rate, wd=weight_decay)

        save_model_and_stats(learn,n_epochs,n_factors,rating_name)


if __name__ == '__main__':
    main()