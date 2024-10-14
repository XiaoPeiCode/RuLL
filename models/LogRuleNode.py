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
    def __init__(self, rule, target, ruleEvaluator, actions_candidates,max_rule_len):
        self.rule = rule  # 当前规则，例如 ['r1', 'r2']
        self.target = target  # 目标关系
        self.ruleEvaluator = ruleEvaluator

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
                                actions_candidates=_actions_candidates,  max_rule_len=self.max_rule_len)
            children.add(child)

        return children

    def find_random_child(self):
        if self.terminal:
            return None
        feature = random.choice(list(set(self.actions_candidates)))
        new_rule = self.rule + [feature]
        _actions_candidates = self.actions_candidates - set(feature)
        return LogRuleNode(rule=new_rule, target=self.target, ruleEvaluator=self.ruleEvaluator,
                           actions_candidates=_actions_candidates,  max_rule_len=self.max_rule_len)

    def is_terminal(self):
        # 终止条件：rule长度或其他逻辑
        # todo:is_terminal只需要计算一次
        # 如果候选集无合适关系也终止

        if self.terminal == None:
            if len(self.rule) >= self.max_rule_len:
                self.terminal = True
            # elif self.ruleEvaluator.evaluate_f1(self.rule, self.target,mcts_mode) >= Early_termination_reward:
            elif self.ruleEvaluator.evaluate(self.rule, self.target,reward_mode) >= Early_termination_reward:
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

