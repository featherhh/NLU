#  coding:utf-8
import pickle
import os
import numpy as np

class CLF_Model():
    def __init__(self, model_save_path):
        # 加载训练好的文件
        self.id2label = pickle.load(open(os.path.join(model_save_path, 'id2label.pkl'), 'rb'))
        self.vec = pickle.load(open(os.path.join(model_save_path, 'vec.pkl'), 'rb'))
        self.LR = pickle.load(open(os.path.join(model_save_path, 'LR.pkl'), 'rb'))
        self.gbdt = pickle.load(open(os.path.join(model_save_path, 'gbdt.pkl'), 'rb'))

    def predict(self, text):
        text = ' '.join(list(text.lower()))
        x = self.vec.transform([text])
        pred1 = self.LR.predict_proba(x)
        pred2 = self.gbdt.predict_proba(x)
        label = np.argmax((pred1+pred2)/2, axis=1)
        # print(f'label---》{label}')
        return self.id2label[label[0]]

if __name__ == '__main__':
    clf = CLF_Model(model_save_path='./model_file')
    text = "你好吗"
    result = clf.predict(text)
    print(result)