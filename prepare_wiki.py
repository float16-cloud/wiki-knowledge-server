import pandas as pd
from utils import get_tokenizer_model
from pythainlp.tokenize import word_tokenize
from tqdm import tqdm

files_path = [
    "data/raw/train-00000-of-00003.parquet",
    "data/raw/train-00001-of-00003.parquet",
    "data/raw/train-00002-of-00003.parquet"
]

total_sm_text = 0
total_text = 0

new_df = pd.DataFrame({
    'id': [],
    'url': [],
    'topic': [],
    'text': [],
    'part_index': [],
    'total_part': []
})

def add_new_row(text, part_index, total_part, topic, wiki_id, wiki_url):
    global new_df
    new_row = pd.DataFrame({'id': [wiki_id], 
                            'url': [wiki_url], 
                            'topic' : [topic], 
                            'text': [text], 
                            'part_index': [part_index], 
                            'total_part': [total_part]
                            })
    new_df = pd.concat([new_df, new_row], ignore_index=True)
    return new_df

def sliding_window(text, window_size=512, overlap=128):
    words = word_tokenize(text,keep_whitespace=False)
    windows = []
    start = 0
    
    while start < len(words):
        window = ''.join(words[start:start + window_size])        
        windows.append(window)
        start += window_size - overlap
        
    return windows

for i in files_path:
    dfs = pd.read_parquet(i)
    tokenizer = get_tokenizer_model()
    for df in tqdm(dfs.itertuples(), total=len(dfs), desc=f"Processing {i}"):
        text = df.text
        title = df.title
        wiki_id = df.id
        wiki_url = df.url

        total_text += 1
        tokenized_text = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        token_size = len(tokenized_text['input_ids'][0])
        if token_size > 512:
            windows = sliding_window(text)
            for idx, window in enumerate(windows):
                part_index = idx + 1
                total_part = len(windows)
                add_new_row(window, part_index, total_part, title, wiki_id, wiki_url)
        else:
            add_new_row(text, 1, 1, title, wiki_id, wiki_url)
            total_sm_text += 1
        
        if total_text % 1000 == 0:  # Print progress every 1000 texts
            print(f"Total text: {total_text}, Total small text: {total_sm_text}")

new_df.to_parquet("data/cleaned/wiki-cleaned.parquet")
print(f"Final count - Total text: {total_text}, Total small text: {total_sm_text}")