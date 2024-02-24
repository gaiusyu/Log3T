
from pytorch_pretrained_bert import BertTokenizer
import torch
import random
import torch.nn as nn
import torch.utils.data as Data
import copy
import re
import numpy as np
import csv
from Log3T import preprocess

tokenizer = BertTokenizer.from_pretrained('Vocab/Vocab.txt')
vocab_size=len(tokenizer.vocab)
maxlen = 128 # max number of words in a log
word_maxlen=16 # max sub tokens in a word
batch_size =24 # batch size
n_layers = 1  # encoder layer number
n_heads = 1  # head number
d_model = 64  # model dimension
d_ff = 64 * 4  # FeedForward dimension
d_k = d_v = 32  # dimension QKV dimension of K(=Q), V


torch.manual_seed(0)
torch.cuda.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyDataSet(Data.Dataset):
    def __init__(self, input_ids, masked_tokens):
        self.input_ids = input_ids
        self.masked_tokens = masked_tokens

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.masked_tokens[idx]


def read_csv_to_list(file_path):
    data_list = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data_list.append(row[0])
    return data_list

def data_format(vocab_index_list,log_split_words):

    '''
    Function to convert the log to vector using WordPiece vocabulary (BERT_base)

    Parameters
    ----------
    vocab_index_list
    log_split_words

    Returns
    -------

    '''

    train_data = []
    count_index = 0
    while count_index < len(vocab_index_list):
        tokens_value = vocab_index_list[count_index]
        input_ids = tokens_value
        lenth = len(log_split_words[count_index])
        if lenth >= maxlen:
            input_ids = input_ids[:word_maxlen * maxlen]
        count_index += 1
        variable_tokens = [0] * word_maxlen * maxlen
        if len(input_ids) >= maxlen * word_maxlen:
            input_ids = input_ids[0:maxlen * word_maxlen]
        n_pad = maxlen * word_maxlen - len(input_ids)
        input_ids.extend([0] * n_pad)
        train_data.append([input_ids, variable_tokens])
    return train_data

def tokenize_words(log):
    '''
    Function to tokenize each word in the log

    Parameters
    ----------
    log: list

    Returns

    tokens_index: list
    -------

    '''
    indexed_tokens = []
    for word in log:  # tokenize each word
        tokenized_text = tokenizer.tokenize(word)
        index_new = tokenizer.convert_tokens_to_ids(tokenized_text)
        if len(index_new) > word_maxlen:
            index_new = index_new[:word_maxlen]
        else:
            index_new = index_new + (word_maxlen - len(index_new)) * [0]
        indexed_tokens = indexed_tokens + index_new
    tokens_id_tensor = torch.tensor([indexed_tokens])
    tokens_index = np.squeeze(tokens_id_tensor.numpy()).tolist()
    return tokens_index

def log_annotation(vocab_index_list,log_split_words,stage,variablelist):
    '''
    Function to assign logs with annotations (variable 1, constant 0) based on historical logs and imitated logs
    '''
    train_data = []
    count_index = 0
    vocab_size = len(variablelist)
    while count_index < len(vocab_index_list):
        tokens_value = vocab_index_list[count_index]
        input_ids = tokens_value
        lenth = len(log_split_words[count_index])
        log = log_split_words[count_index]
        if lenth >= maxlen:
            input_ids = input_ids[:word_maxlen * maxlen]
        cand_imitated_vpos = [i for i, token in enumerate(log)
                              if i <= maxlen]  # select candidate imitated variable position
        common_variable = [i for i, token in enumerate(log)
                           if token in variablelist]   # collect historical variable
        cand_imitated_vpos = [x for x in cand_imitated_vpos if x not in common_variable]
        random.shuffle(cand_imitated_vpos)
        count_index += 1
        imitated_variable_num = 0
        max_num = 3   # The number of imitated variable generated in each log is up to 'max_num'
        for i in range(max_num):
            input_ids_clone = input_ids.copy()
            variable_tokens = [0] * word_maxlen * maxlen
            for ind in common_variable:
                for c in range(word_maxlen):
                    if input_ids_clone[ind * word_maxlen + c] != 0:
                        variable_tokens[ind * word_maxlen + c] = 1
            if stage == 'train_without_imitation':
                n_pad = maxlen * word_maxlen - len(input_ids_clone)
                input_ids_clone.extend([0] * n_pad)
                train_data.append([input_ids_clone, variable_tokens])
                continue
            imitated_variable_count = 0
            last_pos = []
            if len(cand_imitated_vpos) == 0:
                break
            for id_pos in cand_imitated_vpos[
                       :len(
                           cand_imitated_vpos)]:
                random.shuffle(cand_imitated_vpos)
                if id_pos in last_pos:
                    continue
                word_pos = id_pos
                id_pos = word_pos * word_maxlen
                last_pos.append(word_pos)
                replace_variable = variablelist[random.randint(0, vocab_size - 1)]
                tokenized_text = tokenizer.tokenize(replace_variable)
                variable_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
                if word_maxlen >= len(variable_ids):
                    looptimes = len(variable_ids)
                else:
                    looptimes = word_maxlen
                for i in range(looptimes):
                    if variable_ids[i] != 0:
                        variable_tokens[id_pos + i] = 1
                        input_ids_clone[id_pos + i] = variable_ids[i]
                    else:
                        break
                if imitated_variable_count >= imitated_variable_num:
                    break
                imitated_variable_count += 1
            n_pad = maxlen * word_maxlen - len(input_ids_clone)
            input_ids_clone.extend([0] * n_pad)
            train_data.append([input_ids_clone, variable_tokens])
    return train_data

def log_to_model(rawlog,stage,regx,regx_use,dataset,variablelist):
    '''
    Function make log ready to feed into model, and generate log list with split words

    Parameters
    ----------
    rawlog
    stage: 'train' : (log data, label), 'parser': (log data, _)
    regx: all our experiment will not use regex to pre-parse variable but this is optional
    regx_use: boolean
    dataset: dataset name
    variablelist: variables in historical logs (first 100 logs in chronological order)

    Returns
    -------

    '''

    vocab_index_list = list()
    log_sentence=list()
    log_split_words=list()
    if stage=="parse":
        for log in rawlog:
            log=preprocess.wordsplit(log,dataset,regx,regx_use)  # split words using delimiters
            log_split_words.append(log)
            if len(log) == 0:
                continue
            tokens_index = tokenize_words(log)
            vocab_index_list.append(tokens_index)
            log_sentence.append(log)
        train_data = data_format(vocab_index_list,log_split_words)
    else:
            for log in rawlog:
                log = preprocess.wordsplit(log,dataset,regx,regx_use)
                log_split_words.append(log)
                if len(log) == 0:
                    continue
                tokens_index = tokenize_words(log)
                vocab_index_list.append(tokens_index)
                log_sentence.append(log)
            train_data=log_annotation(vocab_index_list,log_split_words,stage,variablelist)

    return train_data, log_sentence


def train(log_data,epoch_n,output,model):
    '''
        Function to implement offline training

        Parameters
        ==========
        log_data: log data ready to be fed into model
        epoch_n: how many epoch will train
        model: model structure
    '''
    model.to(device)
    train_data=log_data
    input_ids, masked_tokens, masked_pos, = zip(*train_data)
    input_ids, masked_tokens, = \
        torch.LongTensor(input_ids), torch.LongTensor(masked_tokens),
    loader = Data.DataLoader(MyDataSet(input_ids, masked_tokens), batch_size, True)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    d=0
    for epoch in range(epoch_n):
        d+=1
        print('=================now you are in =================epoch'+ str(d))
        for input_ids, masked_tokens in loader:
            input_ids, masked_tokens = input_ids.to(device), masked_tokens.to(device)
            logits_lm = model(input_ids, []).to(device)
            a = logits_lm.squeeze(-1)
            b= input_ids.clone()
            c = torch.where(b != 0, 1, 0)
            d=a*c
            mask = d > 0
            x_selected = d[mask]
            y_selected = masked_tokens[mask]
            loss_lm = criterion(x_selected.to(device),
                                    y_selected.float()).to(device)
            loss_lm = (loss_lm.float()).mean().to(device)
            loss = loss_lm.to(device)
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), 'torch_model/model'+ str(output))
    print('============== Train finished==================')

def online_parsing_withTTT(train_data,parse_data,log_sentence, threshold, ground_truth_list,modelpath,model,model2,model3):
    '''

    Parameters
    ----------
    train_data：labelled imitated log vector, label
    parse_data: raw logs vector
    log_sentence: raw logs
    threshold: how many words will be used in log partition
    ground_truth_list: ground truth
    modelpath: where the model parameter trained with all logs is
    model: to load model parameters which will be updated using TTT
    model2: to load model parameters trained with first batch
    model3: to load model parameters trained with all logs

    Returns
    -------
    Group accuracy
    '''
    torch.cuda.empty_cache()
    input_ids, masked_tokens = zip(*parse_data)
    input_ids, masked_tokens = \
        torch.LongTensor(input_ids), torch.LongTensor(masked_tokens)
    input_ids_train, masked_tokens_train = zip(*train_data)
    input_ids_train, masked_tokens_train = \
        torch.LongTensor(input_ids_train), torch.LongTensor(masked_tokens_train)
    loader = Data.DataLoader(MyDataSet(input_ids, masked_tokens), 10, False)  # batchsize
    loader_train = Data.DataLoader(MyDataSet(input_ids_train, masked_tokens_train), 30,
                                   False)  # this batchsize is num✖ times  by the last batchsize
    model = model.to(device)
    model2 = model2.to(device)
    model3 = model3.to(device)
    model3.load_state_dict(torch.load(modelpath))
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    log_group = {}
    log_group_witoutTTT = {}
    log_group_origin = {}
    log_group.setdefault('initialkeywords#@!', []).append(-1)
    log_group_witoutTTT.setdefault('initialkeywords#@!', []).append(-1)
    log_group_origin.setdefault('initialkeywords#@!', []).append(-1)
    log_id = 0
    log_id_withoutTTT = 0
    log_id_origin = 0
    predict_label = list()
    predict_label_withoutTTT = list()
    predict_1_withoutTTT = []
    predict_label_origin = list()
    predict_1_origin = []
    predict_1 = []
    GA_of_batches = []
    GA_of_batches_withoutTTT = []
    GA_of_batches_origin = []
    first_round = 0
    for (input_ids, masked_tokens), (input_ids_train, masked_tokens_train) in zip(loader, loader_train):

        input_ids, masked_tokens = input_ids.to(device), masked_tokens.to(device)
        input_ids_train, masked_tokens_train = input_ids_train.to(device), masked_tokens_train.to(device)
        logits_lm = model(input_ids_train, stage='train').to(device)
        a = logits_lm.squeeze(-1)
        b = input_ids_train.clone()
        c = torch.where(b != 0, 1, 0)
        d = a * c
        mask = d > 0  #
        x_selected = d[mask]  #
        y_selected = masked_tokens_train[mask]
        loss_lm = criterion(x_selected.to(device),
                            y_selected.float()).to(
            device)  # f
        loss_lm = (loss_lm.float()).mean().to(device)
        loss = loss_lm.to(device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log_group, log_id, predict_1, predict_label = batch_parse(input_ids, masked_tokens, model,
                                                                         log_sentence, threshold, log_id,
                                                                         predict_label, predict_1, log_group,
                                                                         stage='train')

        group_with_template = template_update(log_group, log_sentence)

        GA = get_GA(group_with_template, ground_truth_list[0:log_id], sum=log_id)
        GA_of_batches.append(GA)

        if first_round == 0:
            first_round_parameter = model.state_dict()
            model2.load_state_dict(first_round_parameter)
            log_group_witoutTTT, log_id_withoutTTT, predict_1_withoutTTT, predict_label_withoutTTT = batch_parse(
                input_ids, masked_tokens, model2,
                log_sentence, threshold, log_id_withoutTTT,
                predict_label_withoutTTT, predict_1_withoutTTT,
                log_group_witoutTTT, stage='train')
            group_with_template_withoutTTT = template_update(log_group_witoutTTT, log_sentence)

            GA = get_GA(group_with_template_withoutTTT, ground_truth_list[0:log_id_withoutTTT], sum=log_id_withoutTTT)
            GA_of_batches_withoutTTT.append(GA)
        else:

            log_group_witoutTTT, log_id_withoutTTT, predict_1_withoutTTT, predict_label_withoutTTT = batch_parse(
                input_ids, masked_tokens, model2,
                log_sentence, threshold, log_id_withoutTTT,
                predict_label_withoutTTT, predict_1_withoutTTT,
                log_group_witoutTTT, stage='train')

            group_with_template_withoutTTT = template_update(log_group_witoutTTT, log_sentence)

            GA = get_GA(group_with_template_withoutTTT, ground_truth_list[0:log_id_withoutTTT], sum=log_id_withoutTTT)
            GA_of_batches_withoutTTT.append(GA)

        log_group_origin, log_id_origin, predict_1_origin, predict_label_origin = batch_parse(
            input_ids, masked_tokens, model3,
            log_sentence, threshold, log_id_origin,
            predict_label_origin, predict_1_origin,
            log_group_origin, stage='train')
        group_with_template_origin = template_update(log_group_origin, log_sentence)
        GA = get_GA(group_with_template_origin, ground_truth_list[0:log_id_origin],
                    sum=log_id_origin)
        GA_of_batches_origin.append(GA)
        first_round += 1
    constant, variable = average_label_generate(ground_truth_list=ground_truth_list, log_sentence=log_sentence,
                                                predict_label=predict_label)
    return GA_of_batches, constant, variable, GA_of_batches_withoutTTT, GA_of_batches_origin


def online_parsing_withstandardTTT(train_data, train_data2,data,log_sentence, threshold,ground_truth_list,
                           TTT_application, modelpath,model):
    torch.cuda.empty_cache()
    parse_data = data
    input_ids, masked_tokens, masked_pos, = zip(*parse_data)
    input_ids, masked_tokens, masked_pos, = \
        torch.LongTensor(input_ids), torch.LongTensor(masked_tokens), \
        torch.LongTensor(masked_pos),
    train_batch = train_data
    input_ids_train, masked_tokens_train, masked_pos_train, = zip(*train_batch)
    input_ids_train, masked_tokens_train, masked_pos_train, = \
        torch.LongTensor(input_ids_train), torch.LongTensor(masked_tokens_train), \
        torch.LongTensor(masked_pos_train),

    train_batch2 = train_data2
    input_ids_train2, masked_tokens_train2, masked_pos_train2, = zip(*train_batch2)
    input_ids_train2, masked_tokens_train2, masked_pos_train2, = \
        torch.LongTensor(input_ids_train2), torch.LongTensor(masked_tokens_train2), \
        torch.LongTensor(masked_pos_train2),

    loader = Data.DataLoader(MyDataSet(input_ids, masked_tokens), 10, False)  # batchsize
    loader_train = Data.DataLoader(MyDataSet(input_ids_train, masked_tokens_train), 30,
                                   False)  # this batchsize is num✖ times  by the last batchsize
    loader_train2 = Data.DataLoader(MyDataSet(input_ids_train2, masked_tokens_train2), 30,
                                   False)

    if TTT_application == False:
        model.load_state_dict(torch.load(modelpath))
    criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.linear.requires_grad = False
    optimizer2 = torch.optim.Adam(model.parameters(), lr=0.001)
    d = 0
    log_group = {}
    template_group_witoutTTT = {}
    template_group_origin = {}
    log_group.setdefault('initialkeywords#@!', []).append(-1)
    template_group_witoutTTT.setdefault('initialkeywords#@!', []).append(-1)
    template_group_origin.setdefault('initialkeywords#@!', []).append(-1)
    log_id = 0
    predict_label = list()
    predict_1 = []
    PA_of_batches = []
    first_round = 0

    for (input_ids, masked_tokens), (input_ids_train, masked_tokens_train),(input_ids_train2, masked_tokens_train2) in zip(loader, loader_train, loader_train2):

        input_ids, masked_tokens = input_ids.to(device), masked_tokens.to(device)
        input_ids_train, masked_tokens_train = input_ids_train.to(device), masked_tokens_train.to(device)
        input_ids_train2, masked_tokens_train2 = input_ids_train2.to(device), masked_tokens_train2.to(device)

        logits_lm2, logits_lm3 = model(input_ids_train2, [])

        logits_lm, logits_lm1 = model(input_ids_train, [])
        a = logits_lm.squeeze(-1)
        b = input_ids_train.clone()
        c = torch.where(b != 0, 1, 0)
        d = a * c
        mask = d > 0
        x_selected = d[mask]
        y_selected = masked_tokens_train[mask]

        a1 = logits_lm3.squeeze(-1)
        d1 = a1 * c
        mask1 = d1 > 0
        x_selected1 = d1[mask1]
        y_selected1 = masked_tokens_train[mask1]

        if TTT_application == True:
            loss_lm = criterion(x_selected.to(device),
                                y_selected.float()).to(
                device)
            loss_lm1 = criterion(x_selected1.to(device),
                                 y_selected1.float()).to(
                device)
            loss_lm1 = (loss_lm1.float()).mean().to(device)
            loss_lm = (loss_lm.float()).mean().to(device)

            if first_round == 0:
                loss = (loss_lm + loss_lm1).to(device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                loss = loss_lm1.to(device)
                optimizer2.zero_grad()
                loss.backward()
                optimizer2.step()

        log_group, log_id, predict_1, predict_label = batch_parse(input_ids, masked_tokens, model,
                                                                         log_sentence, threshold, log_id,
                                                                         predict_label, predict_1, log_group,stage='train')

        new_template_group = template_update(log_group, log_sentence)

        GA = get_GA(new_template_group, ground_truth_list[0:log_id], sum=log_id)
        PA_of_batches.append(GA)

        first_round += 1
    constant, variable = average_label_generate(ground_truth_list=ground_truth_list, log_sentence=log_sentence,
                                                      predict_label=predict_label)
    return PA_of_batches, constant, variable


def get_GA(group,ground_list,sum):
    correct = 0
    count=0

    for key in group.keys():
        tag = 0
        if key == '1':
            continue
        predict=group[key]
        predict_group_num=len(predict)
        count+=predict_group_num
        groundtruth_num=ground_list.count(ground_list[predict[0]])
        if predict_group_num==groundtruth_num:
            for i in range(len(predict)):
                if ground_list[predict[i]] != ground_list[predict[0]]:
                    tag=1
            if tag==1:
                continue
            else:
                correct+=predict_group_num
    GA=correct/sum
    return GA

def parse(log_data,modelpath,log_sentence, threshold,log_group,logid,model):
    '''
        parse log

        Parameters
        ==========
        modelpath: where the trained model is
        log_sentence: raw log list
        threshold: how many constant word will be used in log partition
        log_group: to store the parsing result
        model: model structure

        Returns
        =======
        log_group:
        group_with_template:
        predict_label:
    '''
    with torch.no_grad():
        input_ids, variable_tokens, = zip(*log_data)
        input_ids, variable_tokens, = \
            torch.LongTensor(input_ids), torch.LongTensor(variable_tokens)
        loader = Data.DataLoader(MyDataSet(input_ids, variable_tokens), 64, False)
        model.load_state_dict(torch.load(modelpath))
        log_group.setdefault('initialkeywords#@!',[]).append(-1)
        log_id = 0
        predict_label=list()
        log_id=logid
        predict_1=[]
        for input_ids, variable_tokens in loader:
            log_group,log_id,predict_1,predict_label=batch_parse(input_ids,variable_tokens,model.to(device),log_sentence,threshold,log_id,predict_label,predict_1,log_group,stage='train')
        # with open("log_messages.csv", "w", newline='') as f:
        #     writer = csv.writer(f)
        #     writer.writerows(log_sentence)
        # with open("label.csv", "w", newline='') as f:
        #     writer = csv.writer(f)
        #     writer.writerows(predict_1)
        group_with_template=template_update(log_group,log_sentence)
        return log_group,group_with_template,predict_label


def batch_parse(input_ids,variable_tokens,model,log_sentence,threshold,log_id,predict_label,predict_1,log_group,stage):
    '''
        parse this batch of log
    '''
    input_ids,_ = input_ids.to(device), variable_tokens.to(device)
    logits_lm = model(input_ids,stage)
    predict_e = logits_lm.view(logits_lm.shape[0], -1)
    for number in range(logits_lm.shape[0]):
        log = log_sentence[log_id]
        input_ids_this = input_ids[number].cpu().detach().numpy().tolist()
        log = label_same_words(log)
        true_length = len(log)
        predict = predict_e[number]
        predict_distrbution = predict.cpu().detach().numpy().tolist()
        predict_for_words = list()
        sum_pre = 0
        for count in range(true_length):
            count = count * word_maxlen
            input_ids_this_wrod = input_ids_this[count:count + word_maxlen]
            if 0 in input_ids_this_wrod:
                zero_index = input_ids_this_wrod.index(0)
            else:
                zero_index = word_maxlen
            if zero_index == 0:
                predict_for_words.append(0)
                continue
            probablity = predict_distrbution[count:count + word_maxlen]
            representative = max(probablity[0:zero_index])
            sum_pre += representative
            predict_for_words.append(representative)
        predict_1.append(predict_for_words)
        predict_label.append(predict_for_words)
        sorted_predict = np.argsort(predict_for_words)
        sorted_log = list()
        index_u = list()
        threshold_count = 0
        partial_constant = list()
        for index in sorted_predict:
            sorted_log.append(log[index])
        for index in sorted_predict:
            if threshold_count >= threshold:
                break
            if not exclude_digits(log[index]):
                new_index = index
                for pi in range(index):
                    if exclude_digits(log[pi]):
                        new_index = new_index - 1
                partial_constant.append(log[index])
                index_u.append(index)
                threshold_count += 1
        if len(partial_constant) == 0:
            partial_constant.append(log[0])
        log_group = Log_partition(log_group, partial_constant, log, log_id)
        log_id += 1
    return log_group,log_id,predict_1,predict_label

def Log_partition(log_group,partial_constant,log,log_id):
    '''
    Function to group log belonging to the same event together
    '''
    canditate=[]
    for key in log_group.keys():
        template=list(key)
        if word_comparison(partial_constant,template):
            matched = template
            if not order_comparison(template,partial_constant,log):
                continue
            if length_comparison(matched, log, partial_constant):
                canditate.append(matched)  # template_group.setdefault(matched, []).append(log_id)
            else:
                if consecutive_variable_detection(log, matched):
                    canditate.append(matched)
                else:
                    continue
    if len(canditate) >=1:
        sim_list=[]
        for cand in canditate:
            sim_list.append(sim(cand,log))
        maxsim=sim_list.index(max(sim_list))
        group_update(log_group, canditate[maxsim], log_id, log, flag=0)

    else:
        partial_constant = log
        group_update(log_group,partial_constant,log_id,log,flag=1)
    return log_group

def sim(seq1, seq2):

    if len(seq1) > len(seq2):
        long=copy.copy(seq1)
        short=copy.copy(seq2)
    else:
        long=copy.copy(seq2)
        short=copy.copy(seq1)

    simTokens = 0

    for token in short:
        if token in long:
            simTokens+=1
    retVal=simTokens / len(long)
    return retVal

#
# def Log_partition(log_group,partial_constant,log,log_id):
#     '''
#     Function to group log belonging to the same event together
#     '''
#     for key in log_group.keys():
#         template=list(key)
#         mark=word_comparison(partial_constant,template)
#         if mark == True:
#             matched = template
#             if not order_comparison(template,partial_constant,log):
#                 continue
#             if length_comparison(matched, log, partial_constant):
#                 group_update(log_group, matched, log_id, log, flag=0,
#                             )  # template_group.setdefault(matched, []).append(log_id)
#             else:
#                 if consecutive_variable_detection(log, matched):
#                     group_update(log_group, matched, log_id, matched, flag=0)
#                 else:
#                     mark=False
#                     continue
#             break
#     if mark == False :
#         partial_constant=log
#         group_update(log_group,partial_constant,log_id,log,flag=1)
#     return log_group


def template_update(template_group,log_sentence):
    '''
        Function to generate the final template of each log group
    '''
    new_template_group={}
    for key in template_group.keys():
        group_list = template_group[key]
        if group_list[0] == -1:
            continue
        len_save = list()
        for id in group_list:
            lenth = len(log_sentence[id])
            len_save.append(lenth)
        max_lenth = max(len_save, key=len_save.count)
        filter = {}
        for id in group_list:
            group_member = log_sentence[id]
            if len(group_member) != max_lenth:
                continue
            else:
                ind = 0
                for word in group_member:
                    filter.setdefault(ind, []).append(word)
                    ind += 1
        templat = list()
        for key in filter.keys():
            all_words = filter[key]
            all_words = list(set(all_words))
            if len(all_words) == 1:
                if not exclude_digits(all_words[0]):
                    templat.append(all_words[0])
                else:
                    templat.append('<*>')
            else:
                templat.append('<*>')
        templat = ' '.join(templat)
        for id in group_list:
            new_template_group.setdefault(templat, []).append(id)
    return new_template_group

def group_update(log_group,template,appendvalue,log,flag):  #log_group,partial_constant,log_id,
    '''
    Function to assign new log to exist log group and update log template
    '''

    if flag==0:
        if len(template) == len(log):
            new_key=copy.copy(template)
            for index,word in enumerate(template):
                if word != log[index]:
                    new_key[index]='<*>'
            new_key=tuple(new_key)
            if new_key in log_group and new_key != tuple(template):
                log_group[new_key] = log_group.pop(tuple(template))+log_group.pop(tuple(new_key))
                log_group[new_key].append(appendvalue)
            else:
                log_group[new_key]=log_group.pop(tuple(template))
                log_group[new_key].append(appendvalue)
        else:
            log_group[tuple(template)].append(appendvalue)
    else:
        template=tuple(template)
        log_group[template]=[appendvalue]

def word_comparison(partial_constant,template):
    '''
    Function to implement 'order comparison'
    '''
    mark = 0
    for word in partial_constant:
        if word not in template:
            mark = 1
            break
    if mark==0:
        return True
    else:
        return False

def order_comparison(template,partial_constant,log):
    '''
    Function to implement 'order comparison'
    '''
    poslist=[]
    for index,word in enumerate(log):
        if word in partial_constant:
            poslist.append(word)
        else:
            continue
    return check_order(template,poslist)

def check_order(big_string, strings):
    '''
    Function to implement 'order comparison'
    '''
    indices = [big_string.index(s) for s in strings if s in big_string]
    return indices == sorted(indices)

def length_comparison(template,log,partial_constant):
    '''
    Function to implement 'length comparison'
    '''
    if not len(template) == len(log):
        return False
    else:
        for word in partial_constant:
            if log.index(word) != template.index(word):
                return False
    return True

def consecutive_variable_detection(log,key):
    '''
    Function to implement 'consecutive variable detection'
    '''
    if len(log) > len(key):
        long=copy.copy(log)
        short=copy.copy(key)
    else:
        long=copy.copy(key)
        short=copy.copy(log)
    save=[]
    for word in long:
        if word not in short:
           save.append(long.index(word))
    consecutive=find_continuous_subsequences(save)
    for i in range(len(consecutive)):
                start=consecutive[i][0]
                end=consecutive[i][len(consecutive[i])-1]
                comppare=0
                for word in long[start:end+1]:
                    if comppare == 0:
                        regex = convert_to_regex_pattern(word)
                        comppare = 1
                    else:
                        if not re.findall(regex, word):
                            return False
                long[start:end + 1] = ['<*>']
    if len(long)==len(short):
        return True
    else:
        return False

    #
    # start=0
    # for i,w in enumerate(key):
    #     if w != '<*>':
    #         if w not in log:
    #             return False
    #         end=log.index(w)
    #         if end-start <= 2:
    #             start=end
    #             continue
    #         else:
    #             comppare=0
    #             for word in log[start+1:end]:
    #                 if comppare == 0:
    #                     regex = convert_to_regex_pattern(word)
    #                     comppare = 1
    #                 else:
    #                     if not re.findall(regex, word):
    #                         return False
    #             start=end
    # comppare = 0
    # lenth = len(key)-1
    # while key[lenth] == '<*>':
    #     lenth = lenth - 1
    #     if lenth == -1:
    #         break
    # start=log.index(key[lenth])
    # end=len(log)
    # for word in log[start+1:end+1]:
    #     if comppare == 0:
    #         regex = convert_to_regex_pattern(word)
    #         comppare = 1
    #     else:
    #         if not re.findall(regex, word):
    #             return False
    # return True

def find_continuous_subsequences(nums):
    subsequences = []
    current_subsequence = []

    for i in range(len(nums)):
        if i > 0 and nums[i] != nums[i - 1] + 1:
            if len(current_subsequence) > 1:
                subsequences.append(current_subsequence)
            current_subsequence = []

        current_subsequence.append(nums[i])

    if len(current_subsequence) > 1:
        subsequences.append(current_subsequence)

    return subsequences
def label_same_words(log):
    '''
    Function to assign different labels to the identical words in a log
    '''
    for word in log:
        if log.count(word) > 1:
            for i in range(log.count(word)):
                log[log.index(word)]=word+str(i)
    return log


def exclude_digits(string):
    '''
    exclude the digits-domain words from partial constant
    '''
    pattern = r'\d'
    digits = re.findall(pattern, string)
    if len(digits)==0:
        return False
    return len(digits)/len(string) >= 0.3

def convert_to_regex_pattern(string):

    '''
        Function used in 'consecutive variable detection', to check if these consecutive words belong to the same pattern
    '''
    regex_pattern = ''
    for char in string:
        if char.isalpha():
            regex_pattern += r'[a-zA-Z]'  # 匹配任意数量的字母（包括大小写）
        elif char.isdigit():
            regex_pattern += r'-?\d*'  # 匹配任意数量的数字（包括可选的负号）
        else:
            if char != '-':
                regex_pattern += re.escape(char)  # 使用re.escape转义特殊字符

    return regex_pattern


def random_replace(string):
    '''
        variable → regular expression → imitated variable
    '''
    letters = re.findall('[a-zA-Z]', string)
    numbers = re.findall('[0-9]', string)

    replaced_letters = random.choices(letters, k=len(letters))
    replaced_numbers = random.choices(numbers, k=len(numbers))

    replaced_string = re.sub('[a-zA-Z]', lambda _: replaced_letters.pop(0), string)
    replaced_string = re.sub('[0-9]', lambda _: replaced_numbers.pop(0), replaced_string)

    return replaced_string

def generate_imitated_variable(dataset):
    '''
    Function to generate imitated variable
    '''
    variablelist=read_csv_to_list('Variableset/variablelist1'+dataset+'.csv')
    new_list=[]
    for index,value in enumerate(variablelist):
        new_list.append(value)
        if value.isdigit():
            continue
        else:
            for i in range(5):
                value=random_replace(value)
                new_list.append(value)
    new_list=list(set(new_list))
    writefile(filename='Variableset/variablelist1'+dataset+'.csv', content=new_list)



def average_label_generate(log_sentence,predict_label,ground_truth_list):
    '''
    Function to calculate the average label of variable assigned by the model
    '''
    constant_count = 0
    constant_labelsum = 0
    variable_count = 0
    variable_labelsum = 0
    for index,log in enumerate(log_sentence):
        log_label=predict_label[index]        # label analysis need to delete [0]
        template_str=ground_truth_list[index]
        for word_index,word in enumerate(log):
            if len(log)<=1 or '<*>' not in template_str:
                continue
            word_label=log_label[word_index]
            if word in template_str:
                constant_count+=1
                constant_labelsum+=word_label
            else:
                variable_count+=1
                variable_labelsum+=word_label
    return str(constant_labelsum/constant_count),str(variable_labelsum/variable_count)



def variable_wordlist_generate(lastlist,log_sentences,ground_truth):
    '''
    Function to simulate historical labeled logs.

    The first 100 logs of 2k dataset in chronological order will be regarded as historical logs.
    '''
    #random.shuffle(log_sentences)
    for index,log in enumerate(log_sentences):
        for word in log:
            if word not in ground_truth[index] and word not in lastlist:
                lastlist.append(word)
        if index >= 100:
            break

    return lastlist

def pure_number_variable(lastlist,log_sentences,ground_truth):
    '''
    Function to calculate the ratio of pure number of all variable
    '''
    #random.shuffle(log_sentences)
    count=0
    for index,log in enumerate(log_sentences):
        for word in log:
            if word not in ground_truth[index]:
                lastlist.append(word)
    for word in lastlist:
        if word.isdigit():
            count+=1

    return count / len(lastlist)

def writefile(filename,content):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for item in content:
            writer.writerow([item])
