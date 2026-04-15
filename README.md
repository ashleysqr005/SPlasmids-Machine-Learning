# SPlasmids-Machine-Learning

## Set up
### Connect to GPU
srun -p youlab-gpu --nodes=1 --gres=gpu:1 --pty bash -i

### Virtual Environment:
source /hpc/group/youlab/jlei912/taylor_law/kegg/ml_env/bin/activate

### Embeddings:
/hpc/group/youlab/jlei912/taylor_law/kegg/output/ko_embeddings_650m
- /mean
- /cls
- /attention