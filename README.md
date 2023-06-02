# CellGirAffE
Framework for testing (Adversarial) Graph (Variational) AutoEncoder techniques for analyzing spatial transcriptomics data. Built for a small-scale research project at the VIB-IRC in Ghent. Currently, two datasets can be analyzed using the framework. A MERFISH Preoptic brain region dataset and a seqFISH mouse embryo dataset. Both of these are available from the Squidpy package.

![alt text](https://github.com/Sbrussee/CellGirAffE/edit/main/CellGirAffE_overview.png)

## Setup
A requirements file was provided for easy install of the required python packages required to run the framework. One can use
`pip install -r requirements.txt`
To install the necessary packages.

## Running an analysis
To run a single analysis one can use 
`python3 CellGirAffE_seqfish.py`
for the seqFISH dataset and
`python3 CellGirAffE_merfish.py`
for the MERFISH dataset.

To specify hyperparameters one can give them as command line arguments to the command line statement above:
The following arguments are available:
- '-v', "--variational", action='store_true', help="Whether to use a variational AE model", default=False
- '-a', "--adversarial", action="store_true", help="Whether to use a adversarial AE model", default=False
- '-d', "--dataset", help="Which dataset to use", required=False
- '-e', "--epochs", type=int, help="How many training epochs to use", default=1
- '-c', "--cells", type=int, default=-1,  help="How many cells to sample per epoch."
- '-t', '--type', type=str, choices=['GCN', 'GAT', 'SAGE'], help="Model type to use (GCN, GAT, SAGE)", default='GCN'
- '-pm', "--prediction_mode", type=str, choices=['full', 'spatial', 'expression'], default='expression', help="Prediction mode to use, full uses all information, spatial uses spatial information only, expression uses expression +spatial information only"
- '-w', '--weight', action='store_true', help="Whether to use distance-weighted edges")=
-'-n', '--normalization', choices=["Laplacian", "Normal", "None"], default="None", help="Adjanceny matrix normalization strategy (Laplacian, Normal, None)"
- '-aggr', '--aggregation_method', choices=['max', 'mean'], help='Which aggregation method to use for GraphSAGE'
- '-th', '--threshold', type=float, help='Distance threshold to use when constructing graph. If neighbors is specified, threshold is ignored.', default=-1
- '-ng', '--neighbors', type=int, help='Number of neighbors per cell to select when constructing graph. If threshold is specified, neighbors are ignored.', default=-1
- '-ls', '--latent', type=int, help='Size of the latent space to use', default=4
- '-hid', '--hidden', type=str, help='Specify hidden layers', default='64,32'
- '-gs', '--graph_summary', action='store_true', help='Whether to calculate a graph summary', default=True
- '-ex', '--experiments', type=list, help='Which experiments to run', default=[1,2,3,4,5,6]
- '-f', '--filter', action='store_true', help='Whether to filter out non-LR genes', default=False
- '-ipd', '--innerproduct', action='store_true', help="Whether to add the IPD loss to the model", default=False

## Running experiments
Instead of running a single analysis, one can also run all experiments as conducted in the project report, to do this one can simply run
`python3 experiments.py --experiments 12345678`
for the seqFISH dataset and
`python3 run_on_merfish.py --experiments 12345678`
for the MERFISH dataset. Note that the experiments argument specifies which experiments to conduct, one can also only run a subsection of those experiments.

## HPO
To optimize the hyperparameters as done in the report, one can run:
`python3 optimize.py -d [dataset]`
Where [dataset] should be the name of the dataset that one wishes to use.
