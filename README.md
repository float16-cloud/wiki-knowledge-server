# Wiki Knowledge Server

## Highlights

Wiki knowledge server is perform pipeline to high performance search wiki data.

Pipeline include:
- Text -> Embedding (Text-to-Vector, GPU) -> Search (Vector Search, GPU) -> Retrieve (Text, URL, Topic) -> Reranking (Re-ranker, GPU) -> Response

## Features

- End-to-End Search 1M records less than 1 sec.
- Thai and English language supported.
- Queue system supported.
- Multi concurrent request supported.

## Limitations

- Knowledge cut off Nov-2023.
- Contain only thai wiki.

## Specification

- Python 3.10
- Storage more than 50GB
- NVIDIA GPU

## You should know

FIRST TIME run server, store and search will be slow.

Because the server need to load the model into storage.

Everytime you initialize the server, The server need to load the model from storage to GPU memory.

This process will take time.

CUVS installation:

```
pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu12==24.10.*" "cuvs-cu12==24.10.*"
```

## How it works

### Prepare wiki data
1. Download wiki data (parquet format) from [huggingface/wiki](https://huggingface.co/datasets/wikimedia/wikipedia)
2. Process data if it longer than 512 tokens with Pythainlp. (```prepare_wiki.py```)
    - if the data is longer than 512 tokens, split the data into multiple parts and overlap 128 tokens.
    - Store data in parquet format at ```data/chunk/wiki-chunk.parquet```
3. Process data with BGE-M3 to get vector representation. (```embedding_wiki.py```)
    - Load the model from storage.
    - Convert data to vector.
    - Store vector in parquet format at ```data/vector/wiki-vector.parquet```
4. Indexing vector with cuvs ```indexing_wiki.py```
    - Load vector from storage.
    - Index vector with cuvs.
    - Store index in storage at ```data/index/wiki-index```

### Offline Search wiki
1. ```quick_search_wiki.py``` will perform search with cuvs.
    - Load the model from storage.
    - Load the index from storage.
    - Perform search with cuvs.

### Benchmark accuracy
1. ```create_ground_truth.py``` will perform search with brute force.
    - Store ground truth at ```data/vector/ground-truth.parquet```
2. ```quick_eval_wiki.py``` will perform search with cuvs.
    - Compare the result with ground truth.
    - Calculate accuracy.

### Online (Server) search
1. Refer to ```server.py``` and Endpoint section.

## Endpoints

- POST /search
    Request: 
    ```json
    {
        "query": "string query",
        "top_k": "number of result"
    }
    ```
    **query**: Search query.

    **top_k**: Number of result to return.

    Response:
    ```json
    {
        "message": "Search completed successfully",
        "content": ["result_text_1", "result_text_2", ...],
        "url": ["result_url_1", "result_url_2", ...],
        "topics": ["result_topic_1", "result_topic_2", ...]
    }
    ```

## Change logs
- 2024-11-27: Wiki-Thai have accuracy 95%.
