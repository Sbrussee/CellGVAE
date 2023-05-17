import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

files = ['error_per_gene_GVAE_exp2_mouse_merfish_full__variational_adversarial.pkl',
'error_per_gene_GVAE_exp2_mouse_seqfish__non-variational_adversarial.pkl',
'error_per_gene_GVAE_exp2_mouse_seqfish_non-variational_non-adverserial.pkl',
'error_per_gene_GVAE_exp2_mouse_seqfish__variational_adversarial.pkl',
'error_per_gene_GVAE_exp2_mouse_seqfish_variational_non-adverserial.pkl',
'error_per_gene_GVAE_exp3_mouse_seqfish_GAT.pkl',
'error_per_gene_GVAE_exp3_mouse_seqfish_GCN.pkl',
'error_per_gene_GVAE_exp3_mouse_seqfish_SAGE_max.pkl',
'error_per_gene_GVAE_exp3_mouse_seqfish_SAGE_mean.pkl']
for file in files:
    with open(file, 'rb') as f:
        error_per_gene = pickle.load(f)
    print(error_per_gene)
    # Plot the 10 genes with the highest relative error
    error_gene_df = pd.DataFrame.from_dict(error_per_gene, orient='index',
                                           columns=['total', 'average', 'relative'])
    error_gene_df['relative_total'] = np.sum(error_gene_df['relative']))  # Convert 'relative' column to numeric
    error_gene_df = error_gene_df.sort_values(by=['relative_total'], ascending=False)

    print(error_gene_df)
    top10 = error_gene_df.iloc[:10]
    print(top10)
    print(top10.reset_index())

    sns.barplot(data=top10.reset_index(), x='relative', y='index', label='Relative prediction error', orient='h')
    plt.xlabel('Relative prediction error')
    plt.ylabel('Gene')
    plt.legend()
    plt.savefig(f'figures/gene_error_{name}.png', dpi=300)
    plt.close()
