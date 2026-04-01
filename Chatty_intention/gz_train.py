import os
import pickle
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

seed = 222
random.seed(seed)
np.random.seed(seed)

def load_data(data_path):
    X,y = [],[]
    with open(data_path,'r',encoding='utf8') as f:
        for line in f.readlines():
            # print(f'line-->{line}')
            text, label = line.strip().split(',')
            text = ' '.join(list(text.lower()))
            X.append(text)
            y.append(label)
    index = np.arange(len(X))
    np.random.shuffle(index)
    X = [X[i] for i in index]
    y = [y[i] for i in index]
    return X, y

def run(data_path, model_save_path):
    X,y = load_data(data_path)
    # print(f'X---》{X}')
    # print(f'y--->{y}')
    # 获取当前意图识别的所有类型标签
    label_set = sorted(list(set(y)))
    # print(f'label_set--》{label_set}')
    # 获取label2id这个字典
    label2id = {value: index for index, value in enumerate(label_set)}
    # print(f'label2id--》{label2id}')
    # 获取id2label这个字典
    id2label = {index: value for index, value in enumerate(label_set)}
    # print(f'id2label--》{id2label}')
    # 将真实的label转换为对应的id
    y = [label2id[i] for i in y]
    # print(f'y--->{y}')
    label_names = sorted(label2id.items(), key=lambda x: x[1])
    target_names = [i[0] for i in label_names]
    labels = [i[1] for i in label_names]
    # print(f'target_names-->{target_names}')
    # print(f'labels-->{labels}')

    train_X, text_X, train_y, text_y = train_test_split(X, y, test_size=0.15, random_state=42)
    # min_df=0.0：表示词汇表中最小文档频率阈值。设为 0.0 表示不排除任何词，即使它只在一个文档中出现过。
    # max_df：表示词汇表中最大文档频率阈值。设为 0.9 表示排除在超过 90% 文档中出现的词。
    vec = TfidfVectorizer(ngram_range=(1, 3), min_df=0.0, max_df=0.9, analyzer='char')
    train_X = vec.fit_transform(train_X)
    text_X = vec.transform(text_X)
    # -------------LR--------------
    # C=8：该参数是正则化强度的倒数。较小的值表示更强的正则化。值为 8 表示适度的正则化，有助于防止过拟合，通过惩罚较大的系数来实现
    #n_jobs=4：指定用于并行处理的CPU核数
    # max_iter=400：模型训练的最大迭代次数
    # multi_class='ovr'：指定多分类问题的处理方式。ovr（one-vs-rest）表示对每个类分别训练一个二分类器，
    LR = LogisticRegression(C=8, n_jobs=4, max_iter=400, multi_class='ovr',random_state=122)
    LR.fit(train_X, train_y)
    pred = LR.predict(text_X)
    print(classification_report(text_y, pred, target_names=target_names))
    print(confusion_matrix(text_y, pred, labels=labels))
    # -------------GBDT--------------
    # 450代表450棵树；max_depth代表树深度
    gbdt = GradientBoostingClassifier(n_estimators=450, learning_rate=0.01, max_depth=8, random_state=24)
    gbdt.fit(train_X, train_y)
    pred = gbdt.predict(text_X)
    # print(f'gbdt-->{pred}')
    print(classification_report(text_y, pred, target_names=target_names))
    print(confusion_matrix(text_y, pred, labels=labels))
    # -------------融合--------------
    pred_prob1 = LR.predict_proba(text_X)
    # print(f'pred_prob1--》{pred_prob1.shape}')
    pred_prob2 = gbdt.predict_proba(text_X)
    # print(f'pred_prob2-->{pred_prob2.shape}')
    pred = np.argmax((pred_prob1 + pred_prob2) / 2, axis=1)
    # print(f'pred-->{pred}')
    print(classification_report(text_y, pred,target_names=target_names))
    print(confusion_matrix(text_y, pred, labels=labels))
    # print(f'id2label--》{id2label}')
    # 保存数据
    pickle.dump(id2label, open(os.path.join(model_save_path, 'id2label.pkl'), 'wb'))
    pickle.dump(vec, open(os.path.join(model_save_path, 'vec.pkl'), 'wb'))
    pickle.dump(LR, open(os.path.join(model_save_path, 'LR.pkl'), 'wb'))
    pickle.dump(gbdt, open(os.path.join(model_save_path, 'gbdt.pkl'), 'wb'))

if __name__ == '__main__':
    run(data_path='./data/train.txt', model_save_path='./model_file')