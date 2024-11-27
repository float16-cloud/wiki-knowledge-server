from cuvs.neighbors import ivf_pq
from numba import cuda
import numpy as np
import pandas as pd

n_lists = 1024 * 4
pq_dim = 512
pq_bits = 8
n_probes = 256
internal_distance_dtype = np.float32
lut_dtype = np.float32

df = pd.read_parquet('data/vector/wiki-vector.parquet')
corpus_embeddings_list = df['vector'].tolist()
corpus_embeddings_np = np.array(corpus_embeddings_list).astype(np.float32)

corpus_embeddings = cuda.to_device(corpus_embeddings_np)
print('start build index')
print(f'embedding shape :{corpus_embeddings.shape}')
params = ivf_pq.IndexParams(n_lists=n_lists,pq_dim=pq_dim,pq_bits=pq_bits,kmeans_n_iters=50)
ivf_pq_index = ivf_pq.build(params, corpus_embeddings)
ivf_pq.save('data/index/wiki-index', ivf_pq_index, include_dataset=False)
print('finish build index')