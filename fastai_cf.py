from helpers import *

def main():
    # Set up parameters
    item_name = 'video'
    user_name = 'participant_num'
    n_epochs = 20
    learning_rate = 5e-3 
    weight_decay = 0.1
    file_path = 'full_logfile_scr_0.csv'
    rating_names = ['resp_fear','video_scr','video_hp']

    for rating_name in rating_names:

        df = load_csv(file_path, rating_name)
        
        dls = create_data_loader(df, item_name, user_name, rating_name)

        n_users, n_items = get_n_users(dls, user_name),get_n_items(dls, item_name)

        y_range = get_rating_range(df, rating_name)

        for n_factors in range(1,6):

            cf_model = create_cf_model(n_users, n_items, n_factors, y_range=y_range)

            learn = train_model(cf_model, dls, n_epochs=n_epochs, lr=learning_rate, wd=weight_decay)

            save_model_and_stats(learn,n_epochs,n_factors,rating_name)


if __name__ == '__main__':
    main()
