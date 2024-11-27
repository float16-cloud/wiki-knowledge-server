import requests

API_URL = "http://127.0.0.1:8000/search"

def send_search_request(text):
    try:
        prepare_data = {
            "query": text,  
            "top_k": 10
        }  
        response = requests.post(API_URL, json=prepare_data)
        return response.json()
    except Exception as e :# as e:
        print(f"Error sending request: {e}")

text = 'จะเริ่มต้นเรียนรู้เรื่องการเขียนโปรแกรมด้วยภาษาไพธอน'

res = send_search_request(text)
print(len(res['content']))