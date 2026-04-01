import requests
import json


def intent_classifier(text):
    url = 'http://127.0.0.1:6001/service/api/bert_intent_recognize'
    data = {"text": text}
    headers = {'Content-Type':'application/json; charset=utf8'}
    reponse = requests.post(url, data=json.dumps(data), headers=headers)
    # print(f'reponse--》{reponse}')
    if reponse.status_code == 200:
        # print(f'reponse.text-->{reponse.text}')
        reponse = json.loads(reponse.text)
        return reponse['result']
    else:
        return -1
if __name__ == '__main__':
    result = intent_classifier(text="不同类型的肌无力症状表现有什么不同？")
    print(f'result--》{result}')