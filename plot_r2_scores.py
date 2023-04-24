import pickle
from GVAE import plot_r2_scores
dataset_name = 'seqfish'
for exp in ['exp2', 'exp3', 'exp4', 'exp5', 'exp6']:
    with open(exp+".pkl", 'rb') as f:
        r2_dict = pickle.load(f)
    plot_r2_scores(r2_dict, 'core_model', dataset_name)
