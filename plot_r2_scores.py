import pickle
from GVAE import plot_r2_scores
dataset_name = 'seqfish'
for exp in ['exp2', 'exp3', 'r2_neighbors', 'r2_prediction_mode', 'r2_thresholds']:
    try:
        with open(exp+".pkl", 'rb') as f:
            r2_dict = pickle.load(f)
        plot_r2_scores(r2_dict, f'{exp}', dataset_name+exp)
    except:
        print(f'{exp} not found...')
