import torch

class Config():
    def __init__(self):
        # self.device = "mps" if torch.cuda.is_available() else "cpu"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.train_path = r'F:\ai\第八阶段-nlp2\完整代码\MedicalKB\NLU\Medical_intention\data\train.csv'
        self.test_path = r'F:\ai\第八阶段-nlp2\完整代码\MedicalKB\NLU\Medical_intention\data\test.csv'
        self.label_path = r'F:\ai\第八阶段-nlp2\完整代码\MedicalKB\NLU\Medical_intention\data\label.txt'
        self.epochs = 10
        self.lr = 2e-5
        self.batch_size = 16
        self.max_len = 60
        self.num_class = 13
        self.bert_path = r'F:\ai\第八阶段-nlp2\完整代码\MedicalKB\bert-base-chinese'
