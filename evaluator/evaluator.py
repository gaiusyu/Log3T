"""
Description : This file implements the function to evaluation accuracy of log parsing
PS: Methods 'evaluate' and 'get_accuracy' are from LogPAI team. LogPAI team calculate group accuracy based
    on EventID comparison. Since there are some minor errors in EventID annotations, we code a method "get_GA" to
    calculate group accuracy based on EventTemplate comparison. Method 'get_editdistance' is based on Nulog github
     repository.

"""

import pandas as pd
import scipy.misc
from nltk.metrics.distance import edit_distance
import numpy as np


def evaluate(groundtruth, parsedresult):
    """ Evaluation function to benchmark log parsing accuracy
    
    Arguments
    ---------
        groundtruth : str
            file path of groundtruth structured csv file 
        parsedresult : str
            file path of parsed structured csv file

    Returns
    -------
        f_measure : float
        accuracy : float
    """ 
    df_groundtruth = pd.read_csv(groundtruth)
    df_parsedlog = pd.read_csv(parsedresult)
    # Remove invalid groundtruth event Ids
    non_empty_log_ids = df_groundtruth[~df_groundtruth['EventId'].isnull()].index
    df_groundtruth = df_groundtruth.loc[non_empty_log_ids]
    df_parsedlog = df_parsedlog.loc[non_empty_log_ids]
    (precision, recall, f_measure, accuracy) = get_accuracy(df_groundtruth['EventId'], df_parsedlog['EventId'])
    print('Precision: %.4f, Recall: %.4f, F1_measure: %.4f, Parsing_Accuracy: %.4f'%(precision, recall, f_measure, accuracy))
    return f_measure, accuracy

def get_accuracy(series_groundtruth, series_parsedlog, debug=False):
    """ Compute accuracy metrics between log parsing results and ground truth
    
    Arguments
    ---------
        series_groundtruth : pandas.Series
            A sequence of groundtruth event Ids
        series_parsedlog : pandas.Series
            A sequence of parsed event Ids
        debug : bool, default False
            print error log messages when set to True

    Returns
    -------
        precision : float
        recall : float
        f_measure : float
        accuracy : float
    """
    series_groundtruth_valuecounts = series_groundtruth.value_counts()
    real_pairs = 0
    for count in series_groundtruth_valuecounts:
        if count > 1:
            real_pairs += scipy.misc.comb(count, 2)

    series_parsedlog_valuecounts = series_parsedlog.value_counts()
    parsed_pairs = 0
    for count in series_parsedlog_valuecounts:
        if count > 1:
            parsed_pairs += scipy.misc.comb(count, 2)

    accurate_pairs = 0
    accurate_events = 0 # determine how many lines are correctly parsed
    for parsed_eventId in series_parsedlog_valuecounts.index:
        logIds = series_parsedlog[series_parsedlog == parsed_eventId].index
        series_groundtruth_logId_valuecounts = series_groundtruth[logIds].value_counts()
        error_eventIds = (parsed_eventId, series_groundtruth_logId_valuecounts.index.tolist())
        error = True
        if series_groundtruth_logId_valuecounts.size == 1:
            groundtruth_eventId = series_groundtruth_logId_valuecounts.index[0]
            if logIds.size == series_groundtruth[series_groundtruth == groundtruth_eventId].size:
                accurate_events += logIds.size
                error = False
        if error and debug:
            print('(parsed_eventId, groundtruth_eventId) =', error_eventIds, 'failed', logIds.size, 'messages')
        for count in series_groundtruth_logId_valuecounts:
            if count > 1:
                accurate_pairs += scipy.misc.comb(count, 2)

    precision = float(accurate_pairs) / parsed_pairs
    recall = float(accurate_pairs) / real_pairs
    f_measure = 2 * precision * recall / (precision + recall)
    accuracy = float(accurate_events) / series_groundtruth.size
    return precision, recall, f_measure, accuracy


def get_editdistance(groundtruth,parsedlog):
    """ Compute Edit distance between log parsing results and ground truth

    Arguments
    ---------
        groundtruth : pandas.Series
            A sequence of groundtruth event Ids
        parsedlog : pandas.Series
            A sequence of parsed event Ids

    Returns
    -------
        Edit distance mean : float
        Edit distance std : float

    """
    edit_distance_result = []
    for i, j in zip(np.array(groundtruth.EventTemplate.values, dtype='str'),
                    np.array(parsedlog.template.values, dtype='str')):
        edit_distance_result.append(edit_distance(i, j))

    edit_distance_result_mean = np.mean(edit_distance_result)
    edit_distance_result_std = np.std(edit_distance_result)
    return edit_distance_result_mean,edit_distance_result_std




def get_GA(group,ground_list,groundtruth,sum):
    """ Evaluation function to benchmark log parsing group accuracy

    Arguments
    ---------
        groundtruth : str
            file path of groundtruth structured csv file
        parsedresult : str
            file path of parsed structured csv file

    Returns
    -------
        f_measure : float
        accuracy : float
    """
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
    PA=correct/sum
    return PA

