import datetime
import operator

import numpy
from pytorch_pretrained_bert import BertTokenizer
import math
import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import copy
import os
import pandas as pd
import re
import numpy as np

tokenizer = BertTokenizer.from_pretrained('../Bert/bert-base-uncased-vocab.txt')
vocab_size=len(tokenizer.vocab)
maxlen = 128
word_maxlen=16
batch_size =24
max_pred = 5  # 最大被maksed然后预测的个数 max tokens of prediction
n_layers = 1  # encoder的层数
n_heads = 1  # 多头注意力机制头数
d_model = 64  # 中间层维度
d_ff = 64 * 4  # 全连接层的维度 4*d_model, FeedForward dimension
d_k = d_v = 32  # QKV的维度 dimension of K(=Q), V
n_segments = 2  # 一个Batch里面有几个日志语句


torch.manual_seed(0)
torch.cuda.manual_seed(0)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class format_log:    # this part of code is from LogPai https://github.com/LogPai

    def __init__(self, log_format, indir='./'):
        self.path = indir
        self.logName = None
        self.df_log = None
        self.log_format = log_format

    def format(self, logName):


        self.logName=logName

        self.load_data()

        return self.df_log





    def generate_logformat_regex(self, logformat):
        """ Function to generate regular expression to split log messages
        """
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(' +', '\\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex
    def log_to_dataframe(self, log_file, regex, headers, logformat):
        """ Function to transform log file to dataframe
        """
        log_messages = []
        linecount = 0
        with open(log_file, 'r', encoding='UTF-8') as fin:
            for line in fin.readlines():
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
                    if linecount==1000000:
                        break
                except Exception as e:
                    pass
        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf.insert(0, 'LineId', None)
        logdf['LineId'] = [i + 1 for i in range(linecount)]
        return logdf


    def load_data(self):
        headers, regex = self.generate_logformat_regex(self.log_format)
        self.df_log = self.log_to_dataframe(os.path.join(self.path, self.logName), regex, headers, self.log_format)


class MyDataSet(Data.Dataset):
    def __init__(self, input_ids, masked_tokens, masked_pos):
        self.input_ids = input_ids
        self.masked_tokens = masked_tokens
        self.masked_pos = masked_pos
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.masked_tokens[idx], self.masked_pos[idx]


def self_model(stage, vocab_size, weight):


    def get_attn_pad_mask(seq_q, seq_k):
        batch_size, seq_len = seq_q.size()
        # eq(zero) is PAD token
        pad_attn_mask = seq_q.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]
        return pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]

    def gelu(x):
        """
          Implementation of the gelu activation function.
          For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
          0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
          Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    class Embedding(nn.Module):
        def __init__(self):
            super(Embedding, self).__init__()
            self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
            self.pos_embed = nn.Embedding(maxlen*word_maxlen, d_model)  # position embedding
            # segment(token type) embedding
            self.norm = nn.LayerNorm(d_model)

        def forward(self, x):
            seq_len = x.size(1)
            pos = torch.arange(seq_len, dtype=torch.long)
            pos = pos.unsqueeze(0).expand_as(x).to(device)  # [seq_len] -> [batch_size, seq_len]
            embedding = self.tok_embed(x.to(device)) + self.pos_embed(pos.to(device))
            return self.norm(embedding)

    class ScaledDotProductAttention(nn.Module):
        def __init__(self):
            super(ScaledDotProductAttention, self).__init__()

        def forward(self, Q, K, V, attn_mask):
            scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
                d_k)  # scores : [batch_size, n_heads, seq_len, seq_len]
            scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
            attn = nn.Softmax(dim=-1)(scores)
            at = attn.squeeze(dim=0).squeeze(dim=0)
            if stage == "parse":
                return at.detach().cpu().numpy()
            context = torch.matmul(attn, V)
            return context

    class MultiHeadAttention(nn.Module):
        def __init__(self):
            super(MultiHeadAttention, self).__init__()
            self.W_Q = nn.Linear(d_model, d_k * n_heads)
            self.W_K = nn.Linear(d_model, d_k * n_heads)
            self.W_V = nn.Linear(d_model, d_v * n_heads)
            self.fc=nn.Linear(n_heads * d_v, d_model)

        def forward(self, Q, K, V, attn_mask):
            # q: [batch_size, seq_len, d_model], k: [batch_size, seq_len, d_model], v: [batch_size, seq_len, d_model]
            residual, batch_size = Q, Q.size(0)
            # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
            q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2).to(
                device)  # q_s: [batch_size, n_heads, seq_len, d_k]
            k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2).to(
                device)  # k_s: [batch_size, n_heads, seq_len, d_k]
            v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2).to(
                device)  # v_s: [batch_size, n_heads, seq_len, d_v]

            attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1,
                                                      1).to(
                device)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

            # context: [batch_size, n_heads, seq_len, d_v], attn: [batch_size, n_heads, seq_len, seq_len]
            context = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
            if stage == "parse":
                return context
            context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                                n_heads * d_v).to(
                device)  # context: [batch_size, seq_len, n_heads, d_v]
            output = self.fc(context).to(device)
            # return nn.LayerNorm(d_model).to(device)(output.to(device)).to(device)
            if stage=='check':
                return context
            output_o = nn.LayerNorm(d_model).to(device)(output.to(device) + weight * residual.to(device)).to(
                device)
            return output_o  # output: [batch_size, seq_len, d_model]

    class PoswiseFeedForwardNet(nn.Module):
        def __init__(self):
            super(PoswiseFeedForwardNet, self).__init__()
            self.fc1 = nn.Linear(d_model, d_ff)
            self.fc2 = nn.Linear(d_ff, d_model)

        def forward(self, x):
            # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
            return self.fc2(gelu(self.fc1(x)))

    class EncoderLayer(nn.Module):
        def __init__(self):
            super(EncoderLayer, self).__init__()
            self.enc_self_attn = MultiHeadAttention()
            self.pos_ffn = PoswiseFeedForwardNet()

        def forward(self, enc_inputs, enc_self_attn_mask):
            enc_outputs = self.enc_self_attn(enc_inputs.to(device), enc_inputs.to(device), enc_inputs.to(device),
                                             enc_self_attn_mask.to(device))  # enc_inputs to same Q,K,V
            if stage == "check":
                return  enc_outputs
            enc_outputs = self.pos_ffn(enc_outputs).to(device)  # enc_outputs: [batch_size, seq_len, d_model]
            return enc_outputs

    class BERT(nn.Module):
        def __init__(self):
            super(BERT, self).__init__()
            self.embedding = Embedding()
            self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
            self.fc = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Dropout(0.5),
                nn.Tanh(),
            )
            self.classifier = nn.Linear(d_model, 2)
            self.linear = nn.Linear(d_model, d_model)
            self.activ2 = gelu
            # fc2 is shared with embedding layer

            self.fc2 = nn.Linear(d_model, 1, bias=False)

        def forward(self, input_ids, masked_pos):
            output = self.embedding(input_ids.to(device)).to(device)  # [bach_size, seq_len, d_model]
            enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids).to(device)  # [batch_size, maxlen, maxlen]
            for layer in self.layers:
                # output: [batch_size, max_len, d_model]
                output = layer(output, enc_self_attn_mask)
                if stage == "check":
                    return output
            # it will be decided by first token(CLS)
            ###   h_pooled = self.fc(output[:, 0])  # [batch_size, d_model]
            ####  logits_clsf = self.classifier(h_pooled)  # [batch_size, 2] predict isNext

            #masked_pos = masked_pos[:, :, None].expand(-1, -1, d_model)  # [batch_size, max_pred, d_model]
            #h_masked = torch.gather(output.to(device), 1,
                                   # masked_pos.to(
                                        #device))  # masking position [batch_size, max_pred, d_model]  位置对齐，将masked的和原本的token对齐
            h_masked = self.linear(output.to(device))  # [batch_size, max_pred, d_model]
            a=self.fc2(h_masked)
            logits_lm = torch.sigmoid(self.fc2(h_masked))  # [batch_size, max_pred, vocab_size]     #
            return logits_lm
    return BERT()




def sentence_process(sentences2,stage,regx,regx_use):   #delimeter让空格转变成[PAD]
    sentences = sentences2  #################存储删去源文件索引的日志语句 # filter '.', ',', '?', '!'  re.sub 正则表达式处理数据
    token_list1 = list()
    log_sentence=list()
    delimeters=[","]
    count=0
    log_tokens=list()
    if stage=="classfy":
        for s in sentences:
            count += 1
            # if count >=100000:
            # break
            for delimeter in delimeters:
                s = re.sub(delimeter, delimeter + ' ', s)
            if regx_use == True:
                for ree in regx:
                    s = re.sub(ree, '<*>', s)
            s = re.split(' +', s)
            log_tokens.append(s)
            if len(s) == 0:
                continue
            indexed_tokens = []
            for word in s:
                tokenized_text = tokenizer.tokenize(word)
                index_new = tokenizer.convert_tokens_to_ids(tokenized_text)
                if len(index_new) > word_maxlen:
                    index_new = index_new[:word_maxlen]
                else:
                    index_new = index_new + (word_maxlen - len(index_new)) * [0]
                indexed_tokens = indexed_tokens + index_new
            tokens_tensor = torch.tensor([indexed_tokens])
            k = np.squeeze(tokens_tensor.numpy()).tolist()
            token_list1.append(k)
            log_sentence.append(s)


        else:
            token_list=token_list1

        '''
            tokenized_text = tokenizer.tokenize(s)
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            tokens_tensor = torch.tensor([indexed_tokens])
            k = np.squeeze(tokens_tensor.numpy()).tolist()
            token_list.append(k)
            '''

        batch = []
        count_index = 0
        while count_index < len(token_list):
            tokens_value = token_list[count_index]
            input_ids = tokens_value
            lenth = len(log_tokens[count_index])
            if lenth >= maxlen:
                input_ids = input_ids[:word_maxlen * maxlen]
            cand_maked_pos = [i for i, token in enumerate(input_ids)
                              if token != 0]  # 排除分隔的CLS和SEP candidate masked position
            random.shuffle(cand_maked_pos)

            if len(cand_maked_pos) > int(1.0 * lenth):
                cand_maked_pos = cand_maked_pos[:int(1.0 * lenth)]
            count_index += 1
            for pos in cand_maked_pos[
                       :len(
                           cand_maked_pos)]:  #########################################for pos in cand_maked_pos[:n_pred]:  # 随机打乱后取前n_pred个token做mask
                masked_tokens, masked_pos = [], []
                pos = (pos // word_maxlen) * word_maxlen
                masked_tokens = [0] * word_maxlen * maxlen

                if len(input_ids) >= maxlen * word_maxlen:
                    input_ids = input_ids[0:maxlen * word_maxlen]
                n_pad = maxlen * word_maxlen - len(input_ids)
                input_ids.extend([0] * n_pad)  # 填充操作，每个句子的长度是30 余下部分补000000
                if stage == 'classfy':
                    batch.append([input_ids, masked_tokens, masked_pos])
                    break
    else:
        with open('../Train_data/Training_data_BGL.csv',
                  'w') as f:
            for s in sentences:
                count += 1
                # if count >=100000:
                # break
                for delimeter in delimeters:
                    s = re.sub(delimeter, delimeter + ' ', s)
                s = re.split(' +', s)
                log_tokens.append(s)
                if len(s) == 0:
                    continue
                indexed_tokens = []
                for word in s:
                    tokenized_text = tokenizer.tokenize(word)
                    index_new = tokenizer.convert_tokens_to_ids(tokenized_text)
                    if len(index_new) > word_maxlen:
                        index_new = index_new[:word_maxlen]
                    else:
                        index_new = index_new + (word_maxlen - len(index_new)) * [0]
                    indexed_tokens = indexed_tokens + index_new
                tokens_tensor = torch.tensor([indexed_tokens])
                k = np.squeeze(tokens_tensor.numpy()).tolist()
                token_list1.append(k)
                log_sentence.append(s)

            if stage == 'train':
                token_list = []
                for i in token_list1:
                    if i not in token_list:
                        token_list.append(i)
            else:
                token_list = token_list1

            '''
                tokenized_text = tokenizer.tokenize(s)
                indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
                tokens_tensor = torch.tensor([indexed_tokens])
                k = np.squeeze(tokens_tensor.numpy()).tolist()
                token_list.append(k)
                '''

            batch = []
            count_index = 0
            while count_index < len(token_list):
                tokens_value = token_list[count_index]
                input_ids = tokens_value
                lenth = len(log_tokens[count_index])
                if lenth >= maxlen:
                    input_ids = input_ids[:word_maxlen * maxlen]
                cand_maked_pos = [i for i, token in enumerate(input_ids)
                                  if token != 0]  # 排除分隔的CLS和SEP candidate masked position
                random.shuffle(cand_maked_pos)

                if len(cand_maked_pos) > int(1.0 * lenth):
                    cand_maked_pos = cand_maked_pos[:int(1.0 * lenth)]
                count_index += 1
                for pos in cand_maked_pos[
                           :len(
                               cand_maked_pos)]:  #########################################for pos in cand_maked_pos[:n_pred]:  # 随机打乱后取前n_pred个token做mask
                    masked_tokens, masked_pos = [], []
                    pos = (pos // word_maxlen) * word_maxlen
                    masked_tokens = [0] * word_maxlen * maxlen
                    if len(input_ids) >= word_maxlen:
                        label_index = word_maxlen
                    else:
                        label_index = len(input_ids)
                    for i in range(label_index):
                        masked_tokens[pos + i] = 1
                        masked_pos.append(pos + i)
                        if stage == 'train':
                            input_ids[pos + i] = random.randint(1, vocab_size - 1)  # random ，随机替换所选中的单词，训练模型去还原。
                            #####masked_tokens.append(input_ids[pos])
                    ####if random() < 0.8:  # 80%                    #按Bert论文概率选取mask的方式
                    if len(input_ids) >= maxlen * word_maxlen:
                        input_ids = input_ids[0:maxlen * word_maxlen]
                    n_pad = maxlen * word_maxlen - len(input_ids)
                    input_ids.extend([0] * n_pad)  # 填充操作，每个句子的长度是30 余下部分补000000
                    if stage == 'classfy':
                        batch.append([input_ids, masked_tokens, masked_pos])
                        break
                    for i in input_ids:
                        f.write(str(i))
                        f.write(',')
                    f.write('\n')
                    for i in masked_tokens:
                        f.write(str(i))
                        f.write(',')
                    f.write('\n')
                    for i in masked_pos:
                        f.write(str(i))
                        f.write(',')
                    f.write('\n')
    return batch,log_sentence


def train(data,epoch_n,output,weight,TTT):
    f = open(data, encoding='UTF-8')
    batch = []
    lines = []
    save_count = 0
    count_check = 0
    for line in f.readlines():
        line = line.strip().split(',')
        line.pop()
        line = [int(x) for x in line]
        lines.append(line)
        save_count += 1
        count_check += 1
        print(str(count_check))
        if save_count == 3:
            batch.append(lines)
            save_count = 0
            lines = []
    input_ids, masked_tokens, masked_pos, = zip(*batch)
    input_ids, masked_tokens, masked_pos, = \
        torch.LongTensor(input_ids), torch.LongTensor(masked_tokens), \
        torch.LongTensor(masked_pos),
    loader = Data.DataLoader(MyDataSet(input_ids, masked_tokens, masked_pos), batch_size, True)
    model = self_model("train",vocab_size,weight).to(device)
    if TTT==True:
        model.load_state_dict(torch.load('model' + output))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=0.05)
    d=0
    for epoch in range(epoch_n):
        d+=1
        print('=================now you are in =================epoch'+ str(d))
        for input_ids, masked_tokens, masked_pos in loader:
            input_ids, masked_tokens, masked_pos = input_ids.to(device), masked_tokens.to(device), masked_pos.to(device)
            logits_lm = model(input_ids, masked_pos).to(device)
            a = logits_lm.view(-1)
            b = masked_tokens.view(-1)
            loss_lm = criterion(logits_lm.view(-1).to(device),
                                    masked_tokens.view(-1).float()).to(
            device)  # for masked LM  Tensor.View元素不变，Tensor形状重构，当某一维为-1时，这一维的大小将自动计算。
            loss_lm = (loss_lm.float()).mean().to(device)
            loss = loss_lm.to(device)
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #if epoch % 1==0:
          #torch.save(model.state_dict(), 'model' + str(output)+ str(epoch)+'weight='+str(weight))
    torch.save(model.state_dict(), 'model'+ str(output))

    # Predict mask tokens ans isNext
    print('============== Train finished==================')

def get_PA(group,ground_list,groundtruth,sum):
    correct = 0
    for key in group.keys():
        if key == '1':
            continue
        predict=group[key]
        predict_group_num=len(predict)
        groundtruth_num=ground_list.count(groundtruth[predict[0]])
        if predict_group_num==groundtruth_num:
            correct+=predict_group_num
    PA=correct/sum
    print('Parsing accuracy'+str(correct/sum))
    return PA
def classfywords(data,output,weight,modelpath,log_sentence, threshold,template_group,logid,stage):
    with torch.no_grad():
       # sum_len = 0
        #for log in log_sentence:
         #   sum_len+=len(log)
        #average=sum_len/len(log_sentence)
        #print(str(average))

        batch = data

        input_ids, masked_tokens, masked_pos, = zip(*batch)
        input_ids, masked_tokens, masked_pos, = \
            torch.LongTensor(input_ids), torch.LongTensor(masked_tokens), \
            torch.LongTensor(masked_pos),

        loader = Data.DataLoader(MyDataSet(input_ids, masked_tokens, masked_pos), 128, False)
        model = self_model("train", vocab_size, weight).to(device)
        model.load_state_dict(torch.load(modelpath))
        template_group= template_group
        new_template_group={}
        template_group.setdefault('1',[]).append(-1)
        log_id = 0
        predict_label=list()
        beyond=0
        if stage=='one':
            log_id=logid
        for input_ids, masked_tokens, masked_pos in loader:
            input_ids, masked_tokens, masked_pos = input_ids.to(device), masked_tokens.to(device), masked_pos.to(device)
            logits_lm = model(input_ids, masked_pos).to(device)

            predict_e = logits_lm.view(logits_lm.shape[0],-1)
            for number in range(logits_lm.shape[0]):
                log = log_sentence[log_id]
                input_ids_this = input_ids[number]
                input_ids_this = input_ids_this.cpu().detach().numpy().tolist()
                true_length = len(log)
                #threshold=0.8*true_length
                predict=predict_e[number]
                predict_distrbution = predict.cpu().detach().numpy().tolist()
                predict_for_words = list()
                sum_pre=0
                for count in range(true_length):
                    count = count * word_maxlen
                    input_ids_this_wrod = input_ids_this[count:count + word_maxlen]
                    if 0 in input_ids_this_wrod:
                        zero_index=input_ids_this_wrod.index(0)
                    else:
                        zero_index=word_maxlen
                    if zero_index == 0:
                        predict_for_words.append(0)
                        continue
                    probablity = predict_distrbution[count:count + word_maxlen]
                    representative = max(probablity[0:zero_index])
                    sum_pre+=representative
                    predict_for_words.append(representative)
                average=sum_pre/true_length
                numpy_predict=numpy.array(predict_for_words)
                std=numpy_predict.std()
                predict_label.append(predict_for_words)
                sorted_predict = np.argsort(predict_for_words)
                sorted_log = list()
                threshold_count = 0
                path = list()
                for index in sorted_predict:
                    sorted_log.append(log[index])

                for index in sorted_predict:
                    if not (bool(re.search(r"[\d]", log[index]))):  # if not log[index].isdigit():
                        # my_re = re.compile(r'[A-Za-z]', re.S)
                        # res = re.findall(my_re, log[index])
                        # if len(res):
                        if "<*>" not in log[index]:  # for Log3T with tokenization filters
                            path.append(log[index])
                            threshold_count += 1
                    if threshold_count >= threshold:
                        break
                    if predict_for_words[index]>average+3*std:
                        break


                matched = ''
                mark = 0
                for key in template_group.keys():
                    mark = 0
                    key1 = re.split(' ', key)
                    for word in key1:

                        if word not in path:
                            mark = 1
                            break

                    if mark == 0:
                        matched = key
                        break
                if mark == 1:
                    path = ' '.join(path)
                    template_group.setdefault(path, []).append(log_id)
                else:
                    template_group.setdefault(matched, []).append(log_id)

                # path=' '.join(path)
                # template_group.setdefault(path, []).append(log_id)
                log_id += 1
                # sorted_log = ' '.join(sorted_log)
                # print(sorted_log)

        for key in template_group.keys():
            group_list=template_group[key]
            if group_list[0]== -1:
                continue
            len_save=list()
            for id in group_list:
                lenth=len(log_sentence[id])
                len_save.append(lenth)
            max_lenth=max(len_save, key=len_save.count)
            filter={}

            for id in group_list:
                group_member=log_sentence[id]
                if len(group_member) != max_lenth:
                    continue
                else:
                    ind=0
                    for word in group_member:
                        filter.setdefault(ind,[]).append(word)
                        ind+=1
            templat=list()
            for key in filter.keys():
                all_words=filter[key]
                all_words=list(set(all_words))
                if len(all_words)==1:
                    templat.append(all_words[0])
                else:
                    templat.append('<*>')
            templat=' '.join(templat)
            for id in group_list:
                new_template_group.setdefault(templat, []).append(id)

        return template_group,new_template_group,predict_label





