import pandas as pd

def preprocess_data(data, target_col):
    """
    Preprocess the data by filling NaN values, transforming the target column, and creating a pivot table.
    """
    data = data.dropna()
    data[target_col] = data[target_col] - data.groupby('participant_num')[target_col].transform('mean')
    vid_idx, vid_name = pd.factorize(data.video)
    vid_name_idx_dict = dict(zip(vid_name, vid_idx))
    data['video'] = data['video'].map(vid_name_idx_dict)
    par_video_matrix = data.pivot(index='participant_num', columns='video', values=target_col)
    return par_video_matrix