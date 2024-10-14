from collections import defaultdict

import pandas as pd
import yaml
import random
from models.monte_carlo_tree_search_model import MCTS, Node
import random
import json

from models.LogRuleEvaluator import LogRuleEvaluator


# to:


class LogRuleNode(Node):
    def __init__(self, rule, target, ruleEvaluator, actions_candidates,max_rule_len,Early_termination_reward):
        self.rule = rule  # rule ,such as  ['r1', 'r2']
        self.target = target  # 目标关系
        self.ruleEvaluator = ruleEvaluator
        self.E_T_R = Early_termination_reward
        self.max_rule_len = max_rule_len
        self.actions_candidates = actions_candidates

        self.terminal = None
        if not self.terminal:
            self.terminal = self.is_terminal()


    def find_children(self):
        """
        扩展全部可能子节点
        :return:
        """
        if self.terminal:  # 是叶节点不需要expand了
            return set()
        children = set()

        for feature in self.actions_candidates:
            # if feature not in self.actions_candidates:
            new_rule = self.rule + [feature]
            _actions_candidates = self.actions_candidates - set(feature)
            child = LogRuleNode(rule=new_rule, target=self.target, ruleEvaluator=self.ruleEvaluator,
                                actions_candidates=_actions_candidates,  max_rule_len=self.max_rule_len,Early_termination_reward=self.E_T_R)
            children.add(child)

        return children

    def find_random_child(self):
        if self.terminal:
            return None
        feature = random.choice(list(set(self.actions_candidates)))
        new_rule = self.rule + [feature]
        _actions_candidates = self.actions_candidates - set(feature)
        return LogRuleNode(rule=new_rule, target=self.target, ruleEvaluator=self.ruleEvaluator,
                           actions_candidates=_actions_candidates,  max_rule_len=self.max_rule_len,Early_termination_reward=self.E_T_R)

    def is_terminal(self):
        # 终止条件：rule长度或其他逻辑
        # todo:is_terminal只需要计算一次
        # 如果候选集无合适关系也终止

        if self.terminal == None:
            if len(self.rule) >= self.max_rule_len:
                self.terminal = True
            # elif self.ruleEvaluator.evaluate_f1(self.rule, self.target,mcts_mode) >= Early_termination_reward:
            elif self.ruleEvaluator.evaluate(self.rule, self.target,reward_mode) >= self.E_T_R:
                self.terminal = True
            else:
                self.terminal = False

        return self.terminal

    def reward(self):
        # 调用评估函数计算奖励
        # while True:
        # precision = self.ruleEvaluator.evaluate(self.rule, self.target)
        # if total_prediction > 0:
        #     break
        return self.ruleEvaluator.evaluate(self.rule, self.target,reward_mode)

    def __hash__(self):
        return hash(tuple(self.rule))

    def __eq__(self, other):
        return tuple(self.rule) == tuple(other.rule)


def extract_logic_rule(target, initial_actions_candidates, ruleEvaluator,max_rule_len,EPOCH,Early_termination_reward):
    root = LogRuleNode(rule=[], target=target, ruleEvaluator=ruleEvaluator,
                       actions_candidates=initial_actions_candidates, max_rule_len=max_rule_len,Early_termination_reward=Early_termination_reward)

    tree = MCTS(exploration_weight=0.7)  # 树记录全局信息
    board = root
    while True:
        if board.is_terminal():
            break

        for _ in range(EPOCH):
            tree.do_rollout(board)
            print('-----------------' + str(_) + '-------------------')
            # if _ > 10:
            #     exit()
        print('choose_board rule:', board.rule)
        board = tree.choose(board)
        # print(board.to_pretty_string())
        if board.is_terminal():
            break
    print("Extracted rule:", " , ".join(board.rule), "->", target)

    # 获取排序后的规则及其最后一次的reward
    sorted_rules = tree.get_all_rules_last_reward()
    return sorted_rules

def save_rules(rules,target,file_prefix):
    # sorted_rules = sorted(target_logic_rules.items(), key=lambda x: x[1][2], reverse=True)
    file_path = f'./result/{file_prefix}.txt'
    json_path = f'./result/{file_prefix}.json'
    json_data = []

    target_logic_rules = {}


    for rule in rules:
        # print(f"Rule: {' , '.join(rule)},-> ,{target},Reward: {reward}")
        path_reward_key = tuple(rule)
        f1, precision, recall = ruleEvaluator.evaluate(rule, target,mode='test')
        f1, precision, recall = round(f1, 4), round(precision, 4), round(recall, 4)
        target_logic_rules[path_reward_key] = (precision, recall, f1)

    print("*****************************")
    print("Target Sorted Rules and Rewards:")
    # 按f1排序，筛选出pr够高的
    sorted_rules = sorted(target_logic_rules.items(), key=lambda x: x[1][2], reverse=True)


    # for rule
    with open(file_path, 'w') as file:
        for rule, reward in sorted_rules:
            item_data = {}
            test_f1,test_pr, test_re  = reward
            precision, recall, f1 = ruleEvaluator.evaluate(rule, target, mode="train")
            if precision > 0.5:
                rule_0 = rule
                # rule_0 = [continuous_feature_content[i] for i in rule_0 if i in continuous_feature_content else i]
                # print(f"Rule: {' , '.join(rule[0])} -> {rule[1]}; F1:{f1},Pr:{precision},Re:{recall}")
                # print(f"Rule: {' , '.join(rule[0])} -> {rule[1]}; F1:{f1},Pr:{precision},Re:{recall}")
                print(f"Pr:{precision},Re:{recall},F1:{f1}; Rule: {' , '.join(rule_0)} -> isUPA={target}\n")
                file.write(
                    f"train(Pr:{precision:.4f},Re:{recall:.4f},F1:{f1:.4f}) test(:{test_pr:.4f},Re:{test_re:.4f},F1:{test_f1:.4f}) # Rule: {' , '.join(rule_0)} -> IsUPA={target}\n")
                item_data["train_eval"] = f"train(Pr:{precision:.4f},Re:{recall:.4f},F1:{f1:.4f})"
                item_data["test_eval"] = f"test(Pr:{test_pr:.4f},Re:{test_re:.4f},F1:{test_f1:.4f}"
                item_data["rule"] = rule_0
                # item_data["matching_indices"] = list(ruleEvaluator.get_matching_indices(rule[0]))
                # item_data["target"] = f"IsUPA={target}"
                # json_data[rule_0]["train_eval"] = f"train(Pr:{precision:.4f},Re:{recall:.4f},F1:{f1:.4f})"
                # json_data[rule_0]["test_eval"] = f"test(:{test_pr:.4f},Re:{test_re:.4f},F1:{test_f1:.4f}"
                json_data.append(item_data)
    with open(json_path, "w") as file:
        json.dump(json_data, file, indent=4)

def construct_rules_base(config):
    ruleEvaluator = LogRuleEvaluator(config)
    initial_actions_candidates = ruleEvaluator.all_events
    dataset_name = config["dataset"]["dataset_name"]
    TOTAL_ = config["MCTS"]["TOTAL_"]  # 计算reward时，加在分母上，避免小样本抽取的rule

    max_rule_len = config["MCTS"]["max_rule_len"]
    max_rule_lens = [max_rule_len]

    target = 1
    Early_termination_reward = config["MCTS"]["Early_termination_reward"]

    file_prefix = f"rules_target_{target}_EarlyR_{Early_termination_reward}_e_{EPOCH}_max_rule_len_{max_rule_len}"
    RULE_BASE_PATH = f'../result/rule_base/test_{dataset_name}_{file_prefix}.json'

    print(f"len of initial_actions_candidates:{len(initial_actions_candidates)}")
    print(f"initial_actions_candidates:{initial_actions_candidates}")
    rules_bases = []
    rules_bases.append(config)

    path_reward = extract_logic_rule(target, initial_actions_candidates, ruleEvaluator, max_rule_len,EPOCH,Early_termination_reward)
    for rule, reward in path_reward:
        print(f"Rule: {' , '.join(rule)},-> ,{target},Reward: {reward}")
        rule_i = {}
        train_precision, train_recall,train_f1,true_positive = ruleEvaluator.calculate_precision_recall_f1(rule, target,ruleEvaluator.train_dataset)
        test_precision, test_recall, test_f1,true_positive = ruleEvaluator.calculate_precision_recall_f1(rule, target,ruleEvaluator.test_dataset)

        if (train_f1 > 0.5) or (train_precision > 0.8):
            train_eval ={"precision": train_precision,
                        "recall": train_recall,
                        "f1_score": train_f1,
                         'true_positive':true_positive}
            test_eval ={"precision": test_precision,
                        "recall": test_recall,
                        "f1_score": test_f1,
                        'true_positive':true_positive}
            rule_i['train_eval'] = train_eval
            rule_i['test_eval'] = test_eval
            rule_i['target'] = target
            rule_i['conditions'] = rule
            # rule_i['predicate']
        rules_bases.append(rule_i)
        # with open(RULE_BASE_PATH,'w') as file:
        #     json.dump(rules_bases,file,indent=4)
    with open(RULE_BASE_PATH,'w') as file:
        json.dump(rules_bases,file,indent=4)

if __name__ == "__main__":

    path = '../config/HDFS/LogRule_HDFS.yaml'
    with open(path, "r", encoding='utf-8-sig') as f:
        config = yaml.safe_load(f)
    print(json.dumps(config, indent=4))

    reward_mode = config["MCTS"]["reward_mode"]
    # Early_termination_reward = config["MCTS"]["Early_termination_reward"]
    Early_termination_reward = 1
    # EPOCH = config["MCTS"]["EPOCH"]
    EPOCH = 500
    # Early_termination_reward = 2
    construct_rules_base(config)

