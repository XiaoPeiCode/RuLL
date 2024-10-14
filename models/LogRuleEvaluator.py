import json
import os
from collections import defaultdict
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import pandas as pd
import ast

def read_hdfs_dataset(train_ratio, random_state, file_path ='../dataset/HDFS_v1/preprocessed/Event_traces.csv'):
    """

    :return:
    """
    def parse_string_to_list(string):
        return string.strip('[]').split(',')

    df = pd.read_csv(file_path)

    # Apply the custom parsing function to the 'Features' and 'TimeInterval' columns
    df['Features'] = df['Features'].apply(parse_string_to_list)
    df['TimeInterval'] = df['TimeInterval'].apply(parse_string_to_list)
    df["Label"] = df["Label"].apply(lambda x:int(x != 'Success'))
    all_events = set(event for seq in df['Features'] for event in seq)

    df = df.iloc[:20000,:]

    train_data, test_data = train_test_split(list(zip(df['Features'], df['Label'])), train_size=train_ratio, random_state=random_state)
    train_count_label_1 = sum(1 for feature, label in train_data if label == 1)
    test_count_label_1 = sum(1 for feature, label in test_data if label == 1)

    print(f"len train_data:{len(train_data)}")
    print(f"abnormal datat:{train_count_label_1}")

    print(f"len test_data:{len(test_data)}")
    print(f"abnormal datat:{test_count_label_1}")

    return train_data,test_data,all_events
class LogRuleEvaluator:
    def __init__(self, config):
        self.config = config
        # 所有事件序列的保存[num],
        # all_seq（seq,label), seq_set=[E1,E2，E3
        # self.log_seq_dict, self.log_seq_len_dict = get_log_seq_dict(config)
        if config["dataset"]["dataset_name"] == "HDFS":
            self.train_dataset,self.test_dataset,self.all_events = read_hdfs_dataset(train_ratio=config["dataset"]["train_ratio"],random_state=config["global"]["random_seed"])
        else: # 默认为HDFS
            self.train_dataset,self.test_dataset,self.all_events = read_hdfs_dataset(train_ratio=config["dataset"]["train_ratio"],random_state=config["global"]["random_seed"])


    # 定义统一的evaluate接口
    def evaluate(self,rule,target,mode="train"):
        if mode == "train":
            precision, recall, f1_score,true_positive = self.calculate_precision_recall_f1(rule, target,self.train_dataset)
        else:
            precision, recall, f1_score,true_positive = self.calculate_precision_recall_f1(rule, target,self.test_dataset)

        # print(f"Precision: {precision:.2f}")
        # print(f"Recall: {recall:.2f}")
        # print(f"F1-score: {f1_score:.2f}")

        return f1_score

    def evaluate_pr(self,rule,target,mode="train"):
        if mode == "train":
            precision, recall, f1_score,true_positive = self.calculate_precision_recall_f1(rule, target,self.train_dataset)
        else:
            precision, recall, f1_score,true_positive = self.calculate_precision_recall_f1(rule, target,self.test_dataset)

        # print(f"Precision: {precision:.2f}")
        # print(f"Recall: {recall:.2f}")
        # print(f"F1-score: {f1_score:.2f}")

        return precision


    def matches_rule(self, seq, rule_elements):
        # Check if all elements of the rule exist in sequence (in order or any order)
        # Check for in-order existence
        if len(seq) < len(rule_elements):
            return False
        # # elem都存在
        # if all(elem in seq for elem in rule_elements):
        #     return True

        def is_subsequence(s, seq): # s是否依次在seq中存在
            it = iter(seq)
            return all(item in it for item in s)

        result = is_subsequence(rule_elements, seq)
        return result

    def calculate_precision_recall_f1(self,rule_elements,expected_label,dataset):
        # rule: tuple of events and expected label, e.g., (E1, E2, E3, 'abnormal')
        # dataset: list of tuples (seq, label), where seq is a list of events and label is 'normal' or 'abnormal'

        # rule_elements, expected_label = rule[:-1], rule[-1]

        # Initialize counts
        true_positive = 0
        false_positive = 0
        false_negative = 0
        total_abnormal = 0

        for seq, label in dataset:
            if expected_label == 1:
                total_abnormal += 1

            # Check if the rule matches the sequence
            if self.matches_rule(seq, rule_elements):
                if label == expected_label:
                    true_positive += 1
                else:
                    false_positive += 1
            elif label == expected_label:
                false_negative += 1

        # Calculate Precision
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0

        # Calculate Recall
        recall = true_positive / (total_abnormal) if total_abnormal > 0 else 0

        # Calculate F1-score
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return precision, recall, f1_score,true_positive






if __name__ == '__main__':
    # 取top10个test rule进行合并
    path = './config/LogRule_bgl_gpt.yaml'
    # path = './config/LogRule_bgl_gpt.yaml'
    with open(path, "r", encoding='utf-8-sig') as f:
        config = yaml.safe_load(f)
    print(json.dumps(config, indent=4))
    logRuleEvaluator = LogRuleEvaluator(config)
    logRuleEvaluator.get_all_rule()
    # Sample dataset
    dataset = [
        (['E1', 'E2', 'E3', 'E4'], 'abnormal'),
        (['E1', 'E2', 'E5', 'E3'], 'normal'),
        (['E1', 'E2', 'E3', 'E6'], 'abnormal'),
        (['E1', 'E4', 'E3', 'E2'], 'normal'),
        (['E1', 'E2', 'E3', 'E7'], 'abnormal'),
        (['E3', 'E2', 'E1', 'E6'], 'normal'),
    ]

    # Define the rule
    rule = ('E1', 'E2', 'E3', 'abnormal')

    # Calculate precision, recall, and F1-score
    precision, recall, f1_score = logRuleEvaluator.calculate_precision_recall_f1(rule, dataset)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-score: {f1_score:.2f}")
