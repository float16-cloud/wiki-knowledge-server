from cuvs.neighbors import ivf_pq
from utils import get_embedding, get_tokenizer_model
import numpy as np
import cupy as cp
import torch
import pandas as pd
import time
import pylibraft
pylibraft.config.set_output_as(lambda device_ndarray: device_ndarray.copy_to_host())

df = pd.read_parquet('data/chunk/train_embedding.parquet')
corpus_embeddings_list = df['vector'].tolist()
context_list = df['content'].tolist()

corpus_embeddings_np = np.array(corpus_embeddings_list).astype(np.float32)

embedding_model = get_embedding().half().to('cuda:0')
tokenizer = get_tokenizer_model()

n_probes = 4096
internal_distance_dtype = np.float32
lut_dtype = np.float32

search_params = ivf_pq.SearchParams(n_probes=n_probes,internal_distance_dtype=internal_distance_dtype,lut_dtype=lut_dtype)
ivf_pq_index = ivf_pq.load('./data/index/ivf_pq_index') # load index from disk
def search_cuvs_ivf_pq(query,ivf_pq_index,top_k = 30):
    hits = ivf_pq.search(search_params, ivf_pq_index, query, top_k)
    print("Input question:", query)
    for k in range(top_k):
        print('-'*50)
        match_index = int(np.array(hits)[1][0, k])
        print(k,np.array(hits)[0][0, k],context_list[match_index].split('\n')[0:5])
        


def embedding(content):
    global embedding_model
    global tokenizer
    content_length_array = []
    for sentence in content : 
        content_tokenized = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
        content_length = content_tokenized['input_ids'].shape[1]
        content_length_array.append(content_length)

    model_input = tokenizer(content, padding=True, truncation=True, return_tensors='pt').to('cuda:0')
    
    with torch.no_grad():
        model_output = embedding_model(**model_input)
        sentence_embeddings = model_output[0][:, 0]
    # print('sentence_embeddings',sentence_embeddings.shape)
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    sentence_list = sentence_embeddings.tolist()
    prepare_output = []

    for idx,sentence in enumerate(sentence_list) : 
        prepare_output.append({
            'token' : content_length_array[idx],
            'content' : content[idx],
            'vector' : sentence,
        })
    return prepare_output[0]

text = 'แนะนำสถานที่ท่องเที่ยวเชิงอนุรักษ์จังหวัดเชียงใหม่ 1 วันหน่อย'
top_k = 10
text_vector = embedding([text])['vector']
print('text_vector',np.array(text_vector).shape)

text_vector_cp = cp.array([text_vector],dtype=cp.float32)
# print('text_vector_cp',text_vector_cp.shape)
start_time = time.time()
search_cuvs_ivf_pq(text_vector_cp,ivf_pq_index,top_k)
print(f'Search time: {time.time()-start_time:.2f} seconds')