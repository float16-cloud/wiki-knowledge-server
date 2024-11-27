from utils import get_embedding, get_tokenizer_model
import numpy as np
import cupy as cp
import torch
import pandas as pd
import time
from tqdm import tqdm

thai_topics = [
    # เทคโนโลยี (Technology)
    "อินเทอร์เน็ตในประเทศไทย",
    "5G ในประเทศไทย",
    "สตาร์ทอัพไทย",
    "เทคโนโลยีชีวภาพในไทย",
    "ปัญญาประดิษฐ์ในประเทศไทย",
    "เทคโนโลยีการเกษตรของไทย",
    "การพัฒนาซอฟต์แวร์ในไทย",
    "เทคโนโลยีพลังงานทดแทนในไทย",
    "อุตสาหกรรม 4.0 ในประเทศไทย",
    "เทคโนโลยีการศึกษาในไทย",
    "FinTech ในประเทศไทย",
    "การพัฒนาแอปพลิเคชันมือถือในไทย",
    "เทคโนโลยีการแพทย์ในประเทศไทย",
    "ความมั่นคงทางไซเบอร์ในไทย",
    "เทคโนโลยีสะอาดในประเทศไทย",

    # ประวัติศาสตร์ (History)
    "อาณาจักรสุโขทัย",
    "อาณาจักรอยุธยา",
    "การปฏิวัติสยาม พ.ศ. 2475",
    "สงครามโลกครั้งที่ 2 ในประเทศไทย",
    "ประวัติศาสตร์กรุงรัตนโกสินทร์",
    "การสถาปนากรุงเทพมหานคร",
    "วิกฤตการณ์ทางการเงินเอเชีย 2540",
    "การเลิกทาสในสยาม",
    "สนธิสัญญาเบาว์ริง",
    "การปฏิรูปการปกครองสมัยรัชกาลที่ 5",
    "เหตุการณ์ 14 ตุลาคม 2516",
    "เหตุการณ์ 6 ตุลาคม 2519",
    "การรัฐประหารในประเทศไทย",
    "ประวัติศาสตร์ชาติไทย",
    "สงครามเก้าทัพ",

    # การท่องเที่ยว (Tourist)
    "วัดพระศรีรัตนศาสดาราม",
    "เกาะพีพี",
    "อุทยานประวัติศาสตร์สุโขทัย",
    "เชียงใหม่ไนท์ซาฟารี",
    "หาดพัทยา",
    "อุทยานแห่งชาติเขาใหญ่",
    "ตลาดน้ำดำเนินสะดวก",
    "พระธาตุดอยสุเทพ",
    "เกาะสมุย",
    "อุทยานประวัติศาสตร์พระนครศรีอยุธยา",
    "สะพานข้ามแม่น้ำแคว",
    "เกาะเต่า",
    "อุทยานแห่งชาติเขาสก",
    "วัดร่องขุ่น",
    "หมู่เกาะสิมิลัน",

    # วัฒนธรรม (Culture)
    "ประเพณีสงกรานต์",
    "ประเพณีลอยกระทง",
    "มวยไทย",
    "อาหารไทย",
    "ศิลปะการแสดงโขน",
    "ภาษาไทย",
    "พุทธศาสนาในประเทศไทย",
    "การแต่งกายประจำชาติไทย",
    "ดนตรีไทย",
    "นาฏศิลป์ไทย",
    "ประเพณีบวชนาค",
    "การละเล่นพื้นบ้านของไทย",
    "ศิลปะการแกะสลักผลไม้",
    "วรรณกรรมไทย",
    "ประเพณีแห่เทียนพรรษา",

    # การคมนาคม (Transportation)
    "รถไฟฟ้าบีทีเอส",
    "รถไฟฟ้ามหานคร",
    "ท่าอากาศยานสุวรรณภูมิ",
    "การรถไฟแห่งประเทศไทย",
    "ทางด่วนในกรุงเทพมหานคร",
    "ท่าเรือแหลมฉบัง",
    "สนามบินดอนเมือง",
    "รถไฟความเร็วสูงในประเทศไทย",
    "ขนส่งมวลชนกรุงเทพ",
    "ท่าอากาศยานภูเก็ต",
    "เส้นทางรถไฟในประเทศไทย",
    "การขนส่งทางน้ำในประเทศไทย",
    "โครงการรถไฟฟ้าสายสีต่างๆ ในกรุงเทพฯ",
    "ท่าอากาศยานเชียงใหม่",
    "ระบบขนส่งมวลชนในต่างจังหวัด",

    # เศรษฐกิจ (Economic)
    "ตลาดหลักทรัพย์แห่งประเทศไทย",
    "นโยบายประชารัฐ",
    "เขตเศรษฐกิจพิเศษในประเทศไทย",
    "อุตสาหกรรมการท่องเที่ยวไทย",
    "การส่งออกของประเทศไทย",
    "เศรษฐกิจพอเพียง",
    "อุตสาหกรรมยานยนต์ไทย",
    "ธนาคารแห่งประเทศไทย",
    "การลงทุนโดยตรงจากต่างประเทศในไทย",
    "นโยบายไทยแลนด์ 4.0",
    "อุตสาหกรรมอาหารของไทย",
    "ตลาดค้าปลีกในประเทศไทย",
    "การเกษตรในประเทศไทย",
    "อุตสาหกรรมสิ่งทอไทย",
    "ระบบภาษีของประเทศไทย",

    # บุคคล (People)
    "พระบาทสมเด็จพระบรมชนกาธิเบศร มหาภูมิพลอดุลยเดชมหาราช บรมนาถบพิตร",
    "สมเด็จพระนางเจ้าสิริกิติ์ พระบรมราชินีนาถ พระบรมราชชนนีพันปีหลวง",
    "พระบาทสมเด็จพระปรเมนทรรามาธิบดีศรีสินทรมหาวชิราลงกรณ พระวชิรเกล้าเจ้าอยู่หัว",
    "สมเด็จพระนางเจ้าสุทิดา พัชรสุธาพิมลลักษณ พระบรมราชินี",
    "ปรีดี พนมยงค์",
    "ศาสตราจารย์ ดร.ป๋วย อึ๊งภากรณ์",
    "หม่อมราชวงศ์คึกฤทธิ์ ปราโมช",
    "พุทธทาสภิกขุ",
    "สมเด็จพระสังฆราช (เจริญ สุวฑฺฒโน)",
    "ศรีบูรพา",
    "เสก โลโซ",
    "ทูลกระหม่อมหญิงอุบลรัตนราชกัญญา สิริวัฒนาพรรณวดี",
    "ชาติชาย ชุณหะวัณ",
    "ดร.ถาวร พรประภา",
    "สืบ นาคะเสถียร",

    # กฎหมาย (Legal)
    "รัฐธรรมนูญแห่งราชอาณาจักรไทย",
    "ประมวลกฎหมายอาญา",
    "ประมวลกฎหมายแพ่งและพาณิชย์",
    "พระราชบัญญัติคุ้มครองผู้บริโภค",
    "กฎหมายลิขสิทธิ์ไทย",
    "พระราชบัญญัติการศึกษาแห่งชาติ",
    "กฎหมายแรงงานไทย",
    "พระราชบัญญัติป่าไม้",
    "กฎหมายจราจรทางบก",
    "พระราชบัญญัติยาเสพติดให้โทษ",
    "กฎหมายภาษีอากรของไทย",
    "พระราชบัญญัติว่าด้วยการกระทำความผิดเกี่ยวกับคอมพิวเตอร์",
    "กฎหมายสิ่งแวดล้อมของไทย",
    "พระราชบัญญัติการเลือกตั้ง",
    "กฎหมายทรัพย์สินทางปัญญาของไทย",

    # เพิ่มเติม (Additional topics)
    "การแพทย์แผนไทย",
    "ระบบการศึกษาไทย",
    "กีฬาซีเกมส์",
    "วิทยาศาสตร์และเทคโนโลยีในประเทศไทย",
    "การเปลี่ยนแปลงสภาพภูมิอากาศในประเทศไทย",
    "ความหลากหลายทางชีวภาพในไทย",
    "อุตสาหกรรมภาพยนตร์ไทย",
    "การประมงในประเทศไทย",
    "พลังงานทดแทนในประเทศไทย",
    "การท่องเที่ยวเชิงนิเวศในไทย",
    "นโยบายต่างประเทศของไทย",
    "ศิลปะร่วมสมัยไทย",
    "การพัฒนาชนบทในประเทศไทย",
    "ระบบสาธารณสุขไทย",
    "การอนุรักษ์สัตว์ป่าในประเทศไทย",
    "วิสาหกิจขนาดกลางและขนาดย่อมในไทย",
    "การจัดการขยะในประเทศไทย",
    "ภาษาถิ่นในประเทศไทย",
    "การพัฒนาเมืองอัจฉริยะในไทย",
    "อุตสาหกรรมเกมในประเทศไทย",
    "การแพทย์ฉุกเฉินในประเทศไทย",
    "ระบบประกันสังคมไทย",
    "การอนุรักษ์พลังงานในประเทศไทย",
    "อุตสาหกรรมแฟชั่นไทย",
    "การจัดการภัยพิบัติในประเทศไทย",
    "ระบบธนาคารในประเทศไทย",
    "การพัฒนาทรัพยากรมนุษย์ในไทย",
    "อุตสาหกรรมการบินของไทย",
    "การพัฒนาเด็กและเยาวชนในประเทศไทย",
    "ระบบนิเวศป่าชายเลนในไทย",
    "การค้าชายแดนของประเทศไทย",
    "อุตสาหกรรมเครื่องสำอางไทย",
    "การอนุรักษ์มรดกทางวัฒนธรรมในไทย",
    "ระบบการเมืองไทย",
    "การพัฒนาโครงสร้างพื้นฐานในประเทศไทย",
    "อุตสาหกรรมการแพทย์ครบวงจรของไทย",
    "การจัดการทรัพยากรน้ำในประเทศไทย",
    "ระบบการขนส่งสาธารณะในกรุงเทพฯ",
    "การพัฒนาพลังงานสะอาดในไทย",
    "อุตสาหกรรมการบริการของไทย"
]

df = pd.read_parquet('data/chunk/train_embedding.parquet')
corpus_embeddings_list = df['vector'].tolist()
context_list = df['content'].tolist()

embedding_model = get_embedding().half().to('cuda:0')
tokenizer = get_tokenizer_model()

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
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    sentence_list = sentence_embeddings.tolist()
    prepare_output = []

    for idx,sentence in enumerate(sentence_list) : 
        prepare_output.append({
            'token' : content_length_array[idx],
            'content' : content[idx],
            'vector' : sentence,
        })
    return prepare_output

def cosine_similarity(a, b):
    return cp.dot(a, b) / (cp.linalg.norm(a) * cp.linalg.norm(b))

def similarity_search(query_vector, corpus_embeddings, top_k=50):
    query_vector = cp.array(query_vector)
    similarities = cp.zeros(len(corpus_embeddings), dtype=cp.float32)
    
    # Calculate similarities in batches
    batch_size = 1000  # Adjust this based on your GPU memory
    for i in tqdm(range(0, len(corpus_embeddings), batch_size), desc="Calculating similarities"):
        batch = cp.array(corpus_embeddings[i:i+batch_size])
        similarities[i:i+batch_size] = cp.dot(batch, query_vector) / (cp.linalg.norm(batch, axis=1) * cp.linalg.norm(query_vector))
    
    # Get top k indices
    top_k_indices = cp.argsort(similarities)[-top_k:][::-1]
    
    # Get the top k results
    top_k_similarities = similarities[top_k_indices]
    
    # Convert back to CPU and return as a list of tuples
    return [(int(idx), float(sim)) for idx, sim in zip(top_k_indices.get(), top_k_similarities.get())]

# Convert corpus embeddings to CuPy arrays for faster computation
corpus_embeddings_cp = [cp.array(embedding) for embedding in corpus_embeddings_list]
# thai_topics = ["อุตสาหกรรมการบริการของไทย"]
text_vectors = embedding(thai_topics)
df = pd.DataFrame({'query': [], 'similar_doc': [], 'idx': [], 'similarity': []})
for v in text_vectors:
    query_vector = v['vector']
    top_100 = similarity_search(query_vector, corpus_embeddings_cp, top_k=100)
    
    print(f"Top 100 similar documents for query: '{v['content']}'")
    for idx, similarity in tqdm(top_100, desc="Saving similar documents", unit="doc"):
        query = v['content']
        similar_doc = context_list[idx]
        new_row = pd.DataFrame({'query': [query], 'similar_doc': [similar_doc], 'idx': [idx], 'similarity': [similarity]})
        df = pd.concat([df, new_row], ignore_index=True)

    df.to_parquet('data/vector_ground_truth/ground_truth.parquet', index=False)