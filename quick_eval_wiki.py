from cuvs.neighbors import ivf_pq
from utils import get_embedding, get_tokenizer_model
import numpy as np
import cupy as cp
import torch
import pandas as pd
import time
import pylibraft
pylibraft.config.set_output_as(lambda device_ndarray: device_ndarray.copy_to_host())

ground_truth = pd.read_parquet('data/vector_ground_truth/ground_truth.parquet')
# new_row = pd.DataFrame({'query': [query], 'similar_doc': [similar_doc], 'idx': [idx], 'similarity': [similarity]})



embedding_model = get_embedding().half().to('cuda:0')
tokenizer = get_tokenizer_model()

n_probes = 1024 * 4
internal_distance_dtype = np.float32
lut_dtype = np.float32

search_params = ivf_pq.SearchParams(n_probes=n_probes,internal_distance_dtype=internal_distance_dtype,lut_dtype=lut_dtype)
ivf_pq_index = ivf_pq.load('./data/index/ivf_pq_index') # load index from disk
total_recall = []


def search_cuvs_ivf_pq(text,query,ivf_pq_index,top_k = 30):
    max_hit = 100
    start_time = time.time()
    hits = ivf_pq.search(search_params, ivf_pq_index, query, top_k)
    # print(f'Search time: {time.time()-start_time:.2f} seconds')
    acc = 0
    match_text = ground_truth.where(ground_truth['query'] == text).dropna()
    for k in range(top_k):
        match_index = int(np.array(hits)[1][0, k])
        #get ground truth only match text with query
        match_content = match_text.where(match_text['idx'] == match_index).dropna()
        if match_content.empty == False:
            acc += 1
    # print(f'Accuracy: {acc}/{max_hit} = {acc/max_hit*100:.2f}%')
    total_recall.append(acc/max_hit*100)
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

total_time = time.time()
for text in thai_topics :
    top_k = 100
    text_vector = embedding([text])['vector']

    text_vector_cp = cp.array([text_vector],dtype=cp.float32)
    # print('text_vector_cp',text_vector_cp.shape)
    search_cuvs_ivf_pq(text,text_vector_cp,ivf_pq_index,top_k)

time_usage = time.time() - total_time
# print(total_recall)
print(f'Total time usage: {time_usage:.2f} seconds, accuracy: {sum(total_recall)/len(total_recall):.2f}%')