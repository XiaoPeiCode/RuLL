from abc import ABC, abstractmethod
from collections import defaultdict
import math


class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, exploration_weight=0.7):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = {}  # children of each node
        self.exploration_weight = exploration_weight

        self.path_rewards = []  # 用于存储路径和其奖励的列表
        self.rules_rewards = defaultdict(list)  # 改为记录规则和奖励
        self.rules_last_reward = {}  # 存储每条规则最后一次的奖励

        """
        初始化每个节点的总奖励(self.Q)和访问次数(self.N)，
        self.children 字典用于存储每个节点的子节点。
        self.exploration_weight 用于调整探索与利用之间的平衡。
        """

    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"

        """
        此方法用于选择最佳后继节点。首先检查节点是否是终端节点，如果是，则抛出异常。
        如果节点没有子节点，它会尝试随机选择一个子节点。对于已探索的节点，它计算每个子节点的得分（平均奖励），并选择得分最高的子节点。
        """
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children :
            return node.find_random_child()

        # ## XP add
        # if not self.children[node]:
        #     return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            print('n_node',n, self.Q[n], self.N[n])
            return self.Q[n] / self.N[n]  # average reward

        # if self.children[node]:  # 检查列表是否为空
        #     return max(self.children[node], key=score)
        # else:
        #     # 序列为空时的处理逻辑
        #     # 例如，返回None或者抛出一个更具体的异常
        #     return None  # 或者其他默认值

        return max(self.children[node], key=score)

    def do_rollout(self, node):
        "Make the tree one layer better. (Train for one iteration.)"

        """
        选择：选择路径直到找到一个未探索的节点或叶节点。
        扩展：如果节点未被扩展，为其添加所有可能的后继状态。
        模拟：从该节点开始随机模拟，直到游戏结束，返回模拟的奖励。
        回传：将模拟的结果回传给路径上的所有节点，更新它们的访问次数和总奖励
        """
        path = self._select(node)
        print('path',path)
        leaf = path[-1]
        print('leaf',leaf)
        self._expand(leaf)
        reward = self._simulate(leaf)
        print('reward',reward)
        print('path_final',path)
        self._backpropagate(path, reward)
        

    def _select(self, node):
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            # print('node',node)
            # print('self.children',self.children)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path
            # print('self.children[node]',set(self.children[node]))
            # print('self.children.keys()',list(self.children.keys()))
            # print('set(list(self.children.keys()))',set(list(self.children.keys())))
            
            unexplored = set(self.children[node]) - set(list(self.children.keys()))
           
            # print('unexplored',unexplored)
            if unexplored:
                n = unexplored.pop()
                # print('nnnnnn',n)
                path.append(n)
                return path
            node = self._uct_select(node)  # descend a layer deeper

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded
        self.children[node] = node.find_children()
        # print('self.children[node]',self.children[node])

    def _simulate(self, node):
        "Returns the reward for a random simulation (to completion) of `node`"
        reward = None
        while True:
            if node.is_terminal():
                # print(node.reward())
                reward = node.reward()
                return reward
            node = node.find_random_child()
            # print('simulate node',node)
            

    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            # if hasattr(node, 'rule'):  # 检查节点是否有规则属性
            #     rule_key = tuple(node.rule)
            #     self.rules_last_reward[rule_key] = reward  # 更新为最后一次的奖励
        leaf_node = path[-1]
        if leaf_node.is_terminal():  # 只有当路径到达叶节点时才记录
            rule_key = tuple(leaf_node.rule)
            self.rules_last_reward[rule_key] = reward
        # self.path_rewards.append((path, reward))
        # # 记录规则和奖励
        # rule = tuple(node.rule for node in path if hasattr(node, 'rule'))
        # rule = path[-1].rule
        # self.rules_rewards[rule].append(reward)

    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct(n): # n是节点
            "Upper confidence bound for trees"
            """
            当前节点
            """
            print('ucb_selection',n, self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]))
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )
        
        return max(self.children[node], key=uct)

    def get_top_k_paths(self, k):
        # 根据奖励对所有路径进行排序，并返回前k个
        sorted_paths = sorted(self.path_rewards, key=lambda x: x[1], reverse=True)
        return sorted_paths[:]

    def get_top_k_rules(self, k):
        # 计算每条规则的平均奖励，并排序
        avg_rewards = {rule: sum(rewards) / len(rewards) for rule, rewards in self.rules_rewards.items()}
        sorted_rules = sorted(avg_rewards.items(), key=lambda x: x[1], reverse=True)
        return sorted_rules[:k]

    def get_all_rules_rewards(self):
        # Calculate the average reward for each rule and return them
        rules_rewards = {}
        for rule, data in self.all_rules.items():
            if data["count"] > 0:  # Avoid division by zero
                avg_reward = data["reward"] / data["count"]
                rules_rewards[rule] = avg_reward
        return rules_rewards


    def get_all_rules_last_reward(self):
        sorted_rules = sorted(self.rules_last_reward.items(), key=lambda x: x[1], reverse=True)

        return sorted_rules


class Node(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    @abstractmethod
    def find_children(self):
        "All possible successors of this board state"
        return set()

    @abstractmethod
    def find_random_child(self):
        "Random successor of this board state (for more efficient simulation)"
        return None

    @abstractmethod
    def is_terminal(self):
        "Returns True if the node has no children"

        return True

    @abstractmethod
    def reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        return 0

    """
    哈希和等价性 (__hash__ 和 __eq__)：确保节点是可哈希的并且可以比较，这对于在 self.children 和其他字典中存储节点至关重要。
    
    """
    @abstractmethod
    def __hash__(self):
        "Nodes must be hashable"
        return 123456789

    @abstractmethod
    def __eq__(node1, node2):
        "Nodes must be comparable"
        return True