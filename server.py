import time
import traceback
from typing import Optional 
start_time = time.time()
import numpy as np
from cuvs.common import Resources
from numba import cuda
import asyncio
from cuvs.neighbors import ivf_pq
from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import uuid
import pandas as pd
from utils import embedding,batch_rerank
from fastapi.middleware.cors import CORSMiddleware
import rmm
from rmm.allocators.numba import RMMNumbaManager
import pylibraft
pylibraft.config.set_output_as(lambda device_ndarray: device_ndarray.copy_to_host())
pool = rmm.mr.PoolMemoryResource(
    rmm.mr.CudaMemoryResource(),
    initial_pool_size= 5*2**30
)
rmm.mr.set_current_device_resource(pool)
cuda.set_memory_manager(RMMNumbaManager)

resources = Resources()
df = pd.read_parquet('data/cleaned/wiki.parquet')
content_list = df['text'].tolist()
topic_list = df['topic'].tolist()
url_list = df['url'].tolist()

print(f'import module time : {time.time() - start_time}')
n_probes = 1024 * 4
MAX_TOP_K = 100
internal_distance_dtype = np.float32
lut_dtype = np.float32
app = FastAPI()
load_index_time = time.time()
ivf_pq_index = ivf_pq.load('data/index/wiki-index',resources=resources) # load index from disk
print(f"Time taken to load index: {time.time() - load_index_time} seconds")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

search_params = ivf_pq.SearchParams(n_probes=n_probes,internal_distance_dtype=internal_distance_dtype,lut_dtype=lut_dtype)
def search_cuvs_ivf_pq(query):
    global MAX_TOP_K
    hits =  ivf_pq.search(search_params, ivf_pq_index, query, MAX_TOP_K)
    return hits

class Query(BaseModel):
    query: str
    top_k : Optional[int] = 5

class BatchProcessor:
    def __init__(self):
        self.batch = []
        self.vector_ids = []
        self.lock = asyncio.Lock()
        self.event = asyncio.Event()
        self.results = []
        self.results_ids = []
        self.top_k = []
        self.query = []
        self.top_k_indices = []
        
    async def add_to_batch(self, vectors, id, query,top_k=5):
        async with self.lock:
            for v in vectors:
                self.batch.append(v)
                self.vector_ids.append(id)
                self.top_k.append(top_k)
                self.query.append(query)

    async def process_batch(self):
        while True:
            await asyncio.sleep(0.01)  # Wait for 0.01 second
            async with self.lock:
                current_batch = self.batch.copy()
                current_batch_ids = self.vector_ids.copy()
                current_batch_top_k = self.top_k.copy()
                current_query = self.query.copy()
                self.vector_ids.clear()
                self.batch.clear()
                self.top_k.clear()
                self.query.clear()

            if current_batch:
                # print(f'Processing batch of size {len(current_batch)}, {len(current_batch_ids)}')
                embedding_time = time.time()
                _,current_batch = embedding(current_batch)
                print(f'Embedding time: {time.time()-embedding_time} seconds')
                copy_to_device_time = time.time()
                current_batch = cuda.to_device(current_batch)
                print(f'Copy to device time: {time.time()-copy_to_device_time} seconds')
                search_time = time.time()
                search_result = search_cuvs_ivf_pq(current_batch)
                print(f'Search time: {time.time()-search_time} seconds')
                top_k_indices = batch_rerank(content_list,search_result, current_query, MAX_TOP_K)
                self.results = search_result
                self.results_ids = current_batch_ids
                self.results_top_k = current_batch_top_k
                self.top_k_indices = top_k_indices
                self.event.set()
                self.event.clear()

    async def get_result(self, query_id):
        try:
            result_idx = [idx for idx, id in enumerate(self.results_ids) if id == query_id]
            req_idx = result_idx[0]
            new_result = self.results
            hits = new_result[1]
            # print(f'top_k per query: {len(hits[req_idx])}')
            top_k = self.results_top_k[req_idx]
            result_text_list = []
            result_url_list = []
            result_topic_list = []
            for k in range(MAX_TOP_K):
                match_index = int(hits[req_idx][k])
                result_text_list.append(content_list[match_index])
                result_url_list.append(url_list[match_index])
                result_topic_list.append(topic_list[match_index])

            rerank_time = time.time()
            top_k_indices = self.top_k_indices[req_idx]
            # print(f"Time taken to rerank: {time.time() - rerank_time} seconds")
            filter_time = time.time()
            filtered_result_text_list = [result_text_list[i] for i in top_k_indices]
            filtered_result_url_list = [result_url_list[i] for i in top_k_indices]
            filtered_result_topic_list = [result_topic_list[i] for i in top_k_indices]
            filtered_result_text_list = filtered_result_text_list[0:top_k]
            filtered_result_url_list = filtered_result_url_list[0:top_k]
            filtered_result_topic_list = filtered_result_topic_list[0:top_k]
            # print(f"Time taken to filter: {time.time() - filter_time} seconds")
            return filtered_result_text_list, filtered_result_url_list, filtered_result_topic_list
            # return [], [], []
        except Exception as e:
            print(f"An error occurred: {e}")
            print(traceback.format_exc())
            return [], [], []
        
main_batch = BatchProcessor()

@app.post("/search")
async def search(Query: Query):
    embedding_time = time.time()
    query_id = uuid.uuid4()
    top_k = Query.top_k
    if top_k > 20 : 
        top_k = 20
    await main_batch.add_to_batch([Query.query], query_id, Query.query, top_k)
    await main_batch.event.wait()
    result_text_list, result_url_list,result_topic_list = await main_batch.get_result(query_id)
    print(f'END SEARCH TIME : {time.time()-embedding_time} seconds')
    return {
            "message": "Search completed successfully", 
            "content": result_text_list, 
            "url": result_url_list, 
            "topics": result_topic_list
            }

async def main():
    asyncio.create_task(main_batch.process_batch())
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, loop="asyncio")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())