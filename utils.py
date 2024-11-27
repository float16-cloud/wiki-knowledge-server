from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForSequenceClassification
import torch
from numba import cuda
import numpy as np
import time
from rmm.allocators.torch import rmm_torch_allocator
import torch
torch.cuda.memory.change_current_allocator(rmm_torch_allocator)

def get_reranker_model():
    model_name_or_path = "Alibaba-NLP/gte-multilingual-reranker-base"
    revision = "4e88bd5dec38b6b9a7e623755029fc124c319d67"
    return AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        revision=revision,
        cache_dir="model/")

def get_reranker_tokenizer_model():
    model_name_or_path = "Alibaba-NLP/gte-multilingual-reranker-base"
    revision = "4e88bd5dec38b6b9a7e623755029fc124c319d67"
    return AutoTokenizer.from_pretrained(
        model_name_or_path, 
        trust_remote_code=True,
        revision=revision,
        cache_dir="model/")

def get_embedding_tokenizer_model():
    return AutoTokenizer.from_pretrained('BAAI/bge-m3',cache_dir="model/")

def get_embedding():
    return AutoModel.from_pretrained('BAAI/bge-m3',
                                     cache_dir="model/"
                                     )

embedding_model = get_embedding().half().to('cuda:0')
embedding_tokenizer = get_embedding_tokenizer_model()
reranker_model = get_reranker_model().half().to('cuda:0')
reranker_tokenizer = get_reranker_tokenizer_model()


def batch_rerank(content_list,search_result,query_list,MAX_TOP_K):
    global reranker_model
    global reranker_tokenizer

    pair_content = []
    for idx,query in enumerate(query_list):
        for i in range(MAX_TOP_K):
            match_index = search_result[1][idx][i]
            pair_content.append([query,content_list[match_index]])

    with torch.no_grad():
        tokenizer_time = time.time()
        inputs = reranker_tokenizer(pair_content, padding=True, truncation=True, return_tensors='pt', max_length=2048).to('cuda:0')
        reranker_time = time.time()
        scores = reranker_model(**inputs, return_dict=True).logits.view(-1, ).float()
        # print(f'Reranker time: {time.time()-reranker_time} seconds')
        

    
    top_k_sorted = torch.tensor([],dtype=torch.uint8).to('cuda:0')
    for idx in range(0,len(query_list)):
        top_k_indices = torch.argsort(scores[idx:idx+MAX_TOP_K], descending=True)
        top_k_sorted = torch.cat((top_k_sorted,top_k_indices))
    # print(f'top_k_sorted: {top_k_sorted.shape}')

    move_back_time = time.time()
    # top_k_sorted = cuda.as_cuda_array(top_k_sorted)
    # top_k_sorted = top_k_sorted.copy_to_host()
    top_k_sorted = top_k_sorted.tolist()
    # print(f'    Batch Move back time to cpu : {time.time()-move_back_time} seconds')
    reshaped_time = time.time()
    top_k_sorted = np.array(top_k_sorted).reshape(-1,MAX_TOP_K)
    # print(f'    Reshape time : {time.time()-reshaped_time} seconds')
    # print(f'    top_k_sorted: {top_k_sorted.shape}')
    # print(f'    top_k_sorted: {top_k_sorted}')
    return top_k_sorted


def rerank(content,query, top_k):
    global reranker_model
    global reranker_tokenizer

    reformat_time = time.time()

    pair_content = []
    for sentence in content:
        pair_content.append([query,sentence])
    print(f'Reformat time: {time.time()-reformat_time} seconds')
    with torch.no_grad():
        tokenizer_time = time.time()
        inputs = reranker_tokenizer(pair_content, padding=True, truncation=True, return_tensors='pt', max_length=2048)
        print(f'Tokenizer time: {time.time()-tokenizer_time} seconds')
        move_time = time.time()
        inputs = {k: v.to(reranker_model.device) for k, v in inputs.items()}
        print(f'Move time: {time.time()-move_time} seconds')
        reranker_time = time.time()
        scores = reranker_model(**inputs, return_dict=True).logits.view(-1, ).float()
        print(f'Reranker time: {time.time()-reranker_time} seconds')
        

    
    # Get the indices of the top_k scores
    start_sort_time = time.time()
    top_k_indices = torch.argsort(scores, descending=True)[:top_k]
    print(f'Sort time: {time.time()-start_sort_time} seconds')

    # Convert tensor to list of integers
    move_back_time = time.time()
    top_k_indices = top_k_indices.cpu()
    print(f'Move back time: {time.time()-move_back_time} seconds')
    return top_k_indices

def embedding(content):
    global embedding_model
    global embedding_tokenizer
    content_length_array = []
    for sentence in content : 
        content_tokenized = embedding_tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
        content_length = content_tokenized['input_ids'].shape[1]
        content_length_array.append(content_length)

    model_input = embedding_tokenizer(content, padding=True, truncation=True, return_tensors='pt').to('cuda:0')
    
    with torch.no_grad():
        model_output = embedding_model(**model_input)
        sentence_embeddings = model_output[0][:, 0]
    # print('sentence_embeddings',sentence_embeddings.shape)
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    sentence_list = sentence_embeddings.tolist()
    prepare_output = []
    vector_list = []
    for idx,sentence in enumerate(sentence_list) : 
        prepare_output.append({
            'token' : content_length_array[idx],
            'content' : content[idx],
            'vector' : sentence,
        })
        vector_list.append(np.array(sentence,dtype=np.float32))
    return prepare_output,vector_list
