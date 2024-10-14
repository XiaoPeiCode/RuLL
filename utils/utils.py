import ast
import os
import time

import numpy as np
import random

import openai
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils import cloudgpt_aoai

# import cloudgpt_aoai
# from utils import cl
temperature = 0  # 0 0.7
top_p = 1  # 0 0.95
frequency_penalty = 0
presence_penalty = 0
MAX_TOKENS=500


def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def is_subsequence(s, seq):  # s是否依次在seq中存在
    it = iter(seq)
    return all(item in it for item in s)


def ensure_file_path_exists(file_path):
    # 获取文件所在的文件夹路径
    dir_name = os.path.dirname(file_path)

    # 如果文件夹路径不存在，则创建该文件夹
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
def save_to_txt(file_path, str):
    # 获取文件所在的文件夹路径
    dir_name = os.path.dirname(file_path)

    # 如果文件夹路径不存在，则创建该文件夹
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(file_path, 'w') as file:
        file.write(str)
def join_strings_with_tabs_and_newline(strings, n):
    new_string = []
    for str in strings:
        if not str.endswith('\n'):
            str += '\n'
        new_string.append(str)
    strings = new_string
    if not strings or n < 0:
        return ""

    tab = "\t" * n
    result = tab.join(strings) + "\n"
    return result



def parse_tuple_string(s):
    return ast.literal_eval(s)

def eval_result(pred_labels, gt_labels):
    # gt_labels = [x["gt_label"] for x in test_data]
    acc = sum([1 for x, y in zip(gt_labels, pred_labels) if x == y]) / len(gt_labels)
    sum_gt_1 = sum([1 for x in gt_labels if x == 1])
    if sum_gt_1:
        recall = sum([1 for x, y in zip(gt_labels, pred_labels) if x == 1 and y == 1]) / sum_gt_1
    else:
        recall = 0
    sum_pred_1 = sum([1 for y in pred_labels if y == 1])
    if sum_pred_1:
        precision = sum([1 for x, y in zip(gt_labels, pred_labels) if x == 1 and y == 1]) / sum_pred_1
    else:
        precision = 0

    f1 = 2 * recall * precision / (recall + precision + 1e-8)
    # f1 = f1_score(truth_labels, pred_labels, average='binary')
    # recall = recall_score(truth_labels, pred_labels, average='binary')
    # precision = precision_score(truth_labels, pred_labels, average='binary')
    # acc = accuracy_score(truth_labels, pred_labels)
    # return {'f1': f1, 'recall': recall, 'precision': precision, 'acc': acc}
    acc, f1, precision, recall = round(acc,4),round(f1,4),round(precision,4),round(recall,4)
    print(f"Acc:{acc},F1:{f1},Pr:{precision},Re:{recall}")
    return acc,f1,precision,recall


def get_chat_completion_with_retry(engine, messages, MAX_TOKENS=300, max_retries=10, sleep_duration=5):
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = cloudgpt_aoai.get_chat_completion(
                engine=engine,
                messages=messages,
                temperature=temperature,  # 0 0.7
                max_tokens=MAX_TOKENS,
                top_p=top_p,  # 0 0.95
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=None
            )

            return response
        except openai.RateLimitError:
            print(f"达到速率限制，等待{sleep_duration}秒后重试...")
            time.sleep(sleep_duration)
            retry_count += 1


class LogDataset(Dataset):
    def __init__(self, features, domain_labels, labels):
        super().__init__()
        self.features = features
        self.domain_labels = domain_labels
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return (self.features[idx], self.domain_labels[idx], self.labels[idx])

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def get_dist(ts, center):
    ts = ts.cpu().detach().numpy()
    center = center.cpu().numpy()
    temp = []
    for i in ts:
        temp.append(np.linalg.norm(i-center))
    return temp

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_center(emb, label = None):
    if label == None:
        return torch.mean(emb, 0)
    else:
        return 'Not defined'

def get_iter(X, y_d, y, batch_size = 1024, shuffle = True):
    dataset = LogDataset(X,y_d, y)
    if shuffle:
        iter = DataLoader(dataset, batch_size, shuffle=True, worker_init_fn=np.random.seed(42))
    else:
        iter = DataLoader(dataset, batch_size)
    return iter

def get_train_eval_iter(train_normal_s, train_normal_t, window_size=20, emb_dim=300):
    X = list(train_normal_s.Embedding.values)
    X.extend(list(train_normal_t.Embedding.values))
    X_new = []
    for i in tqdm(X):
        temp = []
        for j in i:
            temp.extend(j)
        X_new.append(np.array(temp).reshape(window_size, emb_dim))
    y_d = list(train_normal_s.target.values)
    y_d.extend(list(train_normal_t.target.values))
    y = list(train_normal_s.Label.values)
    y.extend(list(train_normal_t.Label.values))
    X_train, X_eval, y_d_train, y_d_eval, y_train, y_eval = train_test_split(X_new, y_d, y, test_size=0.2,
                                                                             random_state=42)
    X_train = torch.tensor(X_train, requires_grad=False)
    X_eval = torch.tensor(X_eval, requires_grad=False)
    y_d_train = torch.tensor(y_d_train).reshape(-1, 1).long()
    y_d_eval = torch.tensor(y_d_eval).reshape(-1, 1).long()
    y_train = torch.tensor(y_train).reshape(-1, 1).long()
    y_eval = torch.tensor(y_eval).reshape(-1, 1).long()
    train_iter = get_iter(X_train, y_d_train, y_train)
    eval_iter = get_iter(X_eval, y_d_eval, y_eval)
    return train_iter, eval_iter

def dist2label(lst_dist, R):
    y = []
    for i in lst_dist:
        if i <= R:
            y.append(0)
        else:
            y.append(1)
    return y
