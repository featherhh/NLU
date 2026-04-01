# NLU

1.用GBDT/lr做初始闲聊意图识别

2.用Bilstm+CRF做具体行为意图识别

3.语意槽

用户输入
    ↓
【第一层】闲聊识别 (Chatty_intention)  LR/GBDT+词袋模型
    ↓
    ├─ 是闲聊 → 闲聊回复 → 结束
    ↓
    └─ 不是闲聊 → 进入第二层
              ↓
        【第二层】intension意图识别 (intention) BERT+线性层
              ↓
              ├─ 医疗咨询 → 进入第三层
              ↓
              └─ 其他意图 → 相应处理
                        ↓
                  【第三层】语义槽填充 (Slot_filling)  BILSTM+CRF
                        ↓
                  提取：症状、疾病、药物、科室等
                        ↓
                  查询知识库 → 生成回复
