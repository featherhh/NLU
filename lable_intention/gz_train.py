import time
import torch
import torch.optim as optim
from model import *
from intent_config import *
from flask import Flask, request, jsonify
app = Flask(__name__)
conf = Config()

# 获取所有标签类别
label_list = [line.strip() for line in open('./data/label.txt', 'r', encoding='utf-8')]
print(f'label_list--》{label_list}')
# 获取id2label的字典
id2label = {idx: label for idx, label in enumerate(label_list)}
print(f'id2label=---》{id2label}')
# 加载bert_tokenizer
bert_tokenizer = BertTokenizer.from_pretrained(conf.bert_path)
# 加载训练好的模型
model = MyModel(bert_path=conf.bert_path, bert_hidden=768, tag_size=13)
model.load_state_dict(torch.load('./save_model/epoch_10.pth'))
model = model.to(conf.device)

def model2predict(sample, model):
    # 对数据进行处理
    inputs = tokenizer.encode_plus(sample,
                                   padding='max_length',
                                   truncation=True,
                                   max_length=60,
                                   return_tensors='pt')
    input_ids = inputs["input_ids"].to(conf.device)
    attention_mask = inputs["attention_mask"].to(conf.device)
    token_type_ids = inputs["token_type_ids"].to(conf.device)
    # 将数据送入模型
    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_mask, token_type_ids)
    # print(f'logits--》{logits}')
    logits = torch.softmax(logits, dim=-1)
    # print(f'logits--》{logits}')
    out = torch.argmax(logits, dim=-1).item()
    # print(f'out--》{out}')
    value, index = torch.topk(logits, k=1)
    # print(f'value--》{value}')
    # print(f'index--》{index}')
    return {"name": id2label[out], "confidence": round(float(value.item()), 3)}

@app.route("/service/api/bert_intent_recognize", methods=["GET","POST"])
def bert_intent_recognize():
    data = {"sucess": 0}
    result = None
    param = request.get_json()
    print(f'param-->{param}')
    text = param["text"]
    try:
        result = model2predict(text, model)
        data["result"] = result
        data["sucess"]  = 1
    except:
        print(f'模型调用有误')
    return jsonify(data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6001)