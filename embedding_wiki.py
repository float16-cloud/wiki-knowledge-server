from utils import embedding
import pandas as pd
from tqdm import tqdm

def process_dataframe_in_batches(df, batch_size=1):
    results = []
    total_batches = (len(df) + batch_size - 1) // batch_size  # Calculate total number of batches

    for start_idx in tqdm(range(0, len(df), batch_size), total=total_batches, desc="Processing batches"):
        batch = df.iloc[start_idx:start_idx + batch_size]
        content_list = batch['text'].tolist()  # Replace 'text' with the actual column name
        batch_results,_ = embedding(content_list)
        results.extend(batch_results)
    return results


df = pd.read_parquet('data/cleaned/wiki-cleaned.parquet')
processed_results = process_dataframe_in_batches(df, batch_size=4)

save_path = 'data/vector/wiki-vector.parquet'
processed_results_df = pd.DataFrame(processed_results)
processed_results_df.to_parquet(save_path, index=False)