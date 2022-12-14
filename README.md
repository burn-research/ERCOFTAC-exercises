# ERCOFTAC Exercises

A set of exercises on machine-learning techniques.

The folder includes exercises on:
- Principal Component Analysis
- Proper Orthogonal Decomposition
- Dynamic Mode Decomposition

The libraries required to perform the exercises are:
- Numpy (scientific computing)
- Matplotlib (plotting)
- [OpenMORe](https://github.com/gdalessi/OpenMORe) (collection of Python modules for Model-Order-Reduction, clustering and classification)

The datasets are available on MareNostrum at /gpfs/projects/nct00/nct00021/Data

To connect to the cluster:

'''console
ssh -Y username@mn3.bsc.es
'''

To reserve an interactive session:

'''console
salloc -p interactive --x11
'''

To load the numpy module

module load miniconda3
source activate spyder-notebook
ulimit -Ss 10240 

