import pandas as pd
import numpy as np
import glob
from fastai.collab import *
from fastai.tabular.all import *
from fastai.test_utils import *

def set_seed_wrapper(seed):
    """
    Sets the seed for numpy and torch.
    
    Parameters:
    - seed (int): The seed to set.
    """
    set_seed(seed, reproducible=True)
    
def simulate_collaborative_data(n_person_factors, n_item_factors, n_items, n_persons):
    """
    Simulates collaborative data by generating latent factors for persons and items,
    as well as intercepts for persons and items. It then creates a dense matrix using
    the dot product of the latent factors and adds the intercepts.
    
    Parameters:
    - n_person_factors (int): Number of factors for each person.
    - n_item_factors (int): Number of factors for each item.
    - n_items (int): Number of items.
    - n_persons (int): Number of persons.
    
    Returns:
    - dense_matrix (ndarray): Dense matrix representing the collaborative data.
    - person_factors (ndarray): Latent factors for each person.
    - item_factors (ndarray): Latent factors for each item.
    - person_intercepts (ndarray): Intercepts for each person.
    - item_intercepts (ndarray): Intercepts for each item.
    """
    # Simulate latent factors for persons and items
    person_factors = np.random.normal(size=(n_persons, n_person_factors))
    item_factors = np.random.normal(size=(n_items, n_item_factors))
    
    # Simulate intercepts for persons and items
    person_intercepts = np.random.normal(size=(n_persons,))
    item_intercepts = np.random.normal(size=(n_items,))
    
    # Create dense matrix using dot product of latent factors and add intercepts
    dense_matrix = np.dot(person_factors, item_factors.T) + person_intercepts[:, np.newaxis] + item_intercepts[np.newaxis, :]
    
    return dense_matrix, person_factors, item_factors, person_intercepts, item_intercepts

def load_csv(
        file_name,
        rating_name = 'resp_fear'
    ):
    '''
    Read a CSV file and drop rows with missing values in the specified rating column.

    Parameters:
        file_name (str): The path to the CSV file.
        item_name (str, optional): The name of the item column. Defaults to 'video'.
        user_name (str, optional): The name of the user column. Defaults to 'participant_num'.
        rating_name (str, optional): The name of the rating column. Defaults to 'resp_fear'.

    Returns:
        pandas.DataFrame: The cleaned DataFrame.
    '''
    df = pd.read_csv(file_name)
    df = df.dropna(subset=[rating_name])
    return df

def create_data_loader(df, item_name, user_name, rating_name, bs=8):
    '''
    Create data loaders for collaborative filtering.

    Parameters:
        df (pandas.DataFrame): The input DataFrame.
        item_name (str): The name of the item column.
        user_name (str): The name of the user column.
        rating_name (str): The name of the rating column.
        bs (int, optional): The batch size. Defaults to 8.

    Returns:
        fastai.collab.CollabDataLoaders: The data loaders.
    '''

    dls = CollabDataLoaders.from_df(
        df,
        item_name=item_name,
        user_name=user_name,
        rating_name=rating_name,
        bs=bs
    )
    return dls

def get_n_users(dls, user_name):
    '''
    Get the number of users in the DataFrame.

    Parameters:
        dls (fastai.data.core.DataLoaders): The input DataLoaders object.
        user_name (str): The name of the user column.

    Returns:
        int: The number of users.
    '''
    return len(dls.classes[user_name])

def get_n_items(dls, item_name):
    '''
    Get the number of items in the DataFrame.

    Parameters:
        dls (fastai.data.core.DataLoaders): The input DataLoaders object.
        item_name (str): The name of the item column.

    Returns:
        int: The number of items.
    '''
    return len(dls.classes[item_name])

def get_rating_range(df, rating_name):
    '''
    Get the range of ratings in the DataFrame.

    Parameters:
        df (pandas.DataFrame): The input DataFrame.
        rating_name (str): The name of the rating column.

    Returns:
        tuple: The range of ratings.
    '''
    return (df[rating_name].min(), df[rating_name].max())

def create_cf_model(n_users, n_items, n_factors, y_range=None):
    '''
    Create a collaborative filtering model.

    Parameters:
        n_users (int): The number of users.
        n_items (int): The number of items.
        n_factors (int): The number of factors for the embedding.
        y_range (tuple, optional): The range of the predicted ratings.

    Returns:
        EmbeddingDotBias: The collaborative filtering model.
    '''
    return EmbeddingDotBias(
        n_factors=n_factors,
        n_items=n_items,
        n_users=n_users,
        y_range=y_range
    )

def train_model(model, dls, n_epochs=5, lr=5e-3, wd=0.1):
    '''
    Train a collaborative filtering model.

    Parameters:
        model (EmbeddingDotBias): The collaborative filtering model.
        dls (fastai.data.core.DataLoaders): The data loaders.
        n_epochs (int, optional): The number of epochs. Defaults to 5.
        lr (float, optional): The learning rate. Defaults to 5e-3.
        wd (float, optional): The weight decay. Defaults to 0.1.

    Returns:
        fastai.learner.Learner: The trained learner.
    '''
    learn = Learner(dls, model, loss_func=MSELossFlat(),cbs=CSVLogger())
    learn.fit_one_cycle(n_epochs, lr, wd=wd)
    return learn

def save_model_and_stats(learn,n_epochs,n_factors,rating_name):
    '''
    Save the model and stats.

    Parameters:
        learn (fastai.learner.Learner): The trained learner.
        n_epochs (int): The number of epochs.
        n_factors (int): The number of factors for the embedding.
        rating_name (str): The name of the rating column.
    '''
    learn.save(f'{rating_name}_{n_epochs}_{n_factors}_model')
    log = learn.csv_logger.read_log()
    log.to_csv(f'results/{rating_name}_{n_epochs}_{n_factors}_stats.csv')

def get_n_factors_for_min_loss(target_var):
    """
    Returns the number of factors that result in the minimum loss for a given target variable.
    
    Parameters:
    target_var (str): The target variable for which the number of factors is determined.
    
    Returns:
    int: The number of factors that result in the minimum loss.
    """
    valid_loss_vals = []
    for idx,stats_path in enumerate(glob.glob(f'results/{target_var}*stats.csv')):
        df = pd.read_csv(stats_path)
        valid_loss_vals.append(df['valid_loss'].values[-1])
    return np.argmin(valid_loss_vals)+1

def load_saved_model_weights(model, dls, path):
    """
    Load the saved model weights from the specified path.

    Args:
        model (nn.Module): The model to load the weights into.
        dls (DataLoaders): The data loaders used for training the model.
        path (str): The path to the saved model weights.

    Returns:
        nn.Module: The model with the loaded weights.
    """
    learn = Learner(dls, model, loss_func=MSELossFlat(), cbs=CSVLogger())
    learn.load(path)
    return learn.model

def reconstruct_matrix(model, y_range):
    '''
    Reconstructs the matrix using the trained model.

    Args:
        model: The trained model object.
        y_range: The range of values for the sigmoid function.

    Returns:
        A pandas DataFrame containing the reconstructed matrix.
    '''
    par_weights = model.u_weight.weight.detach().numpy()
    par_bias = model.u_bias.weight.detach().numpy()
    item_weights = model.i_weight.weight.detach().numpy()
    item_bias = model.i_bias.weight.detach().numpy()
    model_preds = sigmoid_range(torch.tensor(np.dot(par_weights, item_weights.T) + par_bias + item_bias.T), *y_range)
    return pd.DataFrame(model_preds)
