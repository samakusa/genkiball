import sys
import numpy as np

# i"m not import logging,
# to prevent mistype that coding "logging.debug("sample")",
# instead of "logger.debug("sample")"
from logging import basicConfig, getLogger
from logging import DEBUG
import math
import copy
import shogi

basicConfig(
    filename="log/log.log",
    level=DEBUG,
    format="%(asctime)s,%(levelname)s,%(process)d,%(thread)d,"
    + "%(module)s,%(lineno)d,%(funcName)s,%(message)s",
)

logger = getLogger(__name__)

np.random.seed(778)


class Node:
    EXPAND_THRESH = 10
    INVALID_ID = -1

    def __init__(self, _id, board, last_move, parent_id):
        logger.debug(
            "start,_id,%d,board.sfen,%s,last_move,%s,parent_id,%d"
            % (_id, board.sfen(), last_move, parent_id)
        )
        self._id = _id
        self._board = board
        self._last_move = last_move
        self._reward = 0.0
        self._visit = 0
        self._win_rate = (
            np.random.rand()
            if not board.is_game_over()
            else 1.0
            if board.turn == shogi.BLACK
            else 0.0
        )
        self._parent_id = parent_id
        self._child_ids = np.array([], np.uint64)
        logger.debug("end,self._win_rate,%f", self._win_rate)

    def backup(self, reward):
        logger.debug("start,reward,%f" % reward)
        self._visit = self._visit + 1
        self._reward = self._reward + reward
        logger.debug(
            "end,self._visit,%d,self._reward,%f" % (self._visit, self._reward)
        )
        return self

    def get_id(self):
        return self._id

    def get_board(self):
        return self._board

    def get_last_move(self):
        return self._last_move

    def get_reward(self):
        return self._reward

    def get_win_rate(self):
        return self._win_rate

    def get_visit(self):
        return self._visit

    def get_parent_id(self):
        return self._parent_id

    def append_child_id(self, _id):
        self._child_ids = np.append(self._child_ids, _id).astype(np.uint64)
        return self

    def get_child_ids(self):
        return self._child_ids

    def get_ucb(self, total_visit):
        return (
            (self._reward / self._visit)
            + math.sqrt(2 * math.log(total_visit) / self._visit)
            if self._visit != 0
            else sys.float_info.max
        )

    @classmethod
    def test_code(self):
        board = shogi.Board()
        node_0 = Node(
            0,
            copy.deepcopy(board),
            MonteCarloTree.NO_LAST_MOVE,
            MonteCarloTree.NO_PARENT_NODE,
        )
        move = "7g7f"
        board.push_usi(move)
        node_1 = Node(1, copy.deepcopy(board), move, 0)
        node_0.append_child_id(1)

        logger.info(
            "id,%d,board,%s,last_move,%s,reward,%f,win_rate,%f,visit,%d,parent_id,%d,child_ids,%s,ucb,%f"
            % (
                node_0.get_id(),
                node_0.get_board().sfen(),
                node_0.get_last_move(),
                node_0.get_reward(),
                node_0.get_win_rate(),
                node_0.get_visit(),
                node_0.get_parent_id(),
                node_0.get_child_ids(),
                node_0.get_ucb(0),
            )
        )
        logger.info(
            "id,%d,board,%s,last_move,%s,reward,%f,win_rate,%f,visit,%d,parent_id,%d,child_ids,%s,ucb,%f"
            % (
                node_1.get_id(),
                node_1.get_board().sfen(),
                node_1.get_last_move(),
                node_1.get_reward(),
                node_1.get_win_rate(),
                node_1.get_visit(),
                node_1.get_parent_id(),
                node_1.get_child_ids(),
                node_1.get_ucb(1),
            )
        )

        node_0.backup(node_1.get_win_rate())
        node_1.backup(node_1.get_win_rate())

        logger.info(
            "id,%d,board,%s,last_move,%s,reward,%f,win_rate,%f,visit,%d,parent_id,%d,child_ids,%s,ucb,%f"
            % (
                node_0.get_id(),
                node_0.get_board().sfen(),
                node_0.get_last_move(),
                node_0.get_reward(),
                node_0.get_win_rate(),
                node_0.get_visit(),
                node_0.get_parent_id(),
                node_0.get_child_ids(),
                node_0.get_ucb(1),
            )
        )
        logger.info(
            "id,%d,board,%s,last_move,%s,reward,%f,win_rate,%f,visit,%d,parent_id,%d,child_ids,%s,ucb,%f"
            % (
                node_1.get_id(),
                node_1.get_board().sfen(),
                node_1.get_last_move(),
                node_1.get_reward(),
                node_1.get_win_rate(),
                node_1.get_visit(),
                node_1.get_parent_id(),
                node_1.get_child_ids(),
                node_1.get_ucb(1),
            )
        )


class Candidate:
    def __init__(self, win_rate, moves):
        self._moves = moves
        self._value = (
            int(-math.log(1.0 / win_rate - 1.0) * 600)
            if win_rate != 1.0
            else 30000
        )

    def get_main_line(self):
        return self._moves

    def get_value(self):
        return self._value


class MonteCarloTree:
    ROOT_NODE_ID = 0
    NO_LAST_MOVE = ""
    NO_PARENT_NODE = -1

    def __init__(self, board):
        logger.debug("start")
        self._board = board
        self._nodes = np.array([])
        logger.debug("end")

    def init(self):
        logger.debug("start")
        self._nodes = np.append(
            self._nodes,
            Node(
                MonteCarloTree.ROOT_NODE_ID,
                self._board,
                MonteCarloTree.NO_LAST_MOVE,
                MonteCarloTree.NO_PARENT_NODE,
            ),
        )  # root node
        self.expand(self._nodes[MonteCarloTree.ROOT_NODE_ID])
        logger.debug("end")
        return self

    def playout(self, total_visit, top=1):
        logger.debug("start,total_visit,%d,top,%d" % (total_visit, top))
        leaf_node = self.visit(
            self._nodes[MonteCarloTree.ROOT_NODE_ID], total_visit
        )
        if leaf_node.get_visit() >= Node.EXPAND_THRESH:
            self.expand(leaf_node)
        self.backup(leaf_node, leaf_node.get_win_rate())

        child_nodes = np.array(
            list(
                map(
                    lambda i: self._nodes[i],
                    self._nodes[MonteCarloTree.ROOT_NODE_ID].get_child_ids(),
                )
            )
        )
        child_nodes = child_nodes[
            np.argsort(
                np.array(list(map(lambda n: n.get_reward(), child_nodes)))
            )[::-1]
        ][0:top]
        candidates = np.array(
            list(
                map(
                    lambda e: self.main_line(e[1], e[1].get_last_move(), e[0]),
                    enumerate(child_nodes),
                )
            )
        )
        _ = list(
            map(
                lambda e: logger.debug(
                    "ranking,%d,main_line,%s,value,%f"
                    % (e[0], e[1].get_main_line(), e[1].get_value())
                ),
                enumerate(candidates),
            )
        )
        logger.debug("end")
        return candidates

    def expand(self, node):
        logger.debug(
            "start,node_id,%d,board,%s,last_move,%s"
            % (node.get_id(), node.get_board().sfen(), node.get_last_move())
        )
        for move in node.get_board().legal_moves:
            logger.debug("next_move,%s" % move.usi())
            board = copy.deepcopy(node.get_board())
            board.push_usi(move.usi())
            node.append_child_id(len(self._nodes))
            self._nodes = np.append(
                self._nodes,
                Node(self._nodes.shape[0], board, move.usi(), node.get_id()),
            )
        logger.debug("end,child_nodes,%s" % node.get_child_ids())
        return self

    def visit(self, node, total_visit):
        logger.debug(
            "start,node_id,%d,board=%s,last_move,%s,reward,%f,visit,%d"
            % (
                node.get_id(),
                node.get_board().sfen(),
                node.get_last_move(),
                node.get_reward(),
                node.get_visit(),
            )
        )
        if node.get_child_ids().shape[0] == 0:
            logger.debug("end,leaf node")
            return node

        child_nodes = list(map(lambda i: self._nodes[i], node.get_child_ids()))
        max_ucb = max(list(map(lambda n: n.get_ucb(total_visit), child_nodes)))
        node = self.visit(
            np.random.choice(
                np.array(
                    list(
                        filter(
                            lambda n: n.get_ucb(total_visit) == max_ucb,
                            child_nodes,
                        )
                    )
                )
            ),
            total_visit,
        )
        logger.debug(
            "end,node_id,%d,reward,%f,visit,%d"
            % (node.get_id(), node.get_reward(), node.get_visit())
        )
        return node

    def backup(self, node, reward):
        logger.debug(
            "start,node_id,%d,board,%s,last_move,%s,reward,%f"
            % (
                node.get_id(),
                node.get_board().sfen(),
                node.get_last_move(),
                reward,
            )
        )
        if node.get_parent_id() == MonteCarloTree.NO_PARENT_NODE:
            logger.debug("end,root node")
            return self
        node.backup(reward)
        self.backup(self._nodes[node.get_parent_id()], reward)
        logger.debug(
            "end,reward,%f,visit,%d" % (node.get_reward(), node.get_visit())
        )
        return self

    def main_line(self, node, moves, ranking):
        logger.debug(
            "start,node_id,%d,board,%s,last_move,%s,reward,%f,visit,%d,moves,%s,ranking,%d"
            % (
                node.get_id(),
                node.get_board().sfen(),
                node.get_last_move(),
                node.get_reward(),
                node.get_visit(),
                moves,
                ranking,
            )
        )

        if node.get_child_ids().shape[0] == 0:
            logger.debug("end,leaf node")
            return Candidate(node.get_win_rate(), moves)
        child_nodes = np.array(
            list(map(lambda i: self._nodes[i], node.get_child_ids()))
        )
        child_node = child_nodes[
            np.argsort(
                np.array(list(map(lambda n: n.get_reward(), child_nodes)))
            )[::-1]
        ][ranking]
        moves = np.append(moves, child_node.get_last_move())
        candidate = self.main_line(child_node, moves, ranking)
        logger.debug(
            "end,moves,%s,value,%f"
            % (candidate.get_main_line(), candidate.get_value())
        )
        return candidate

    @classmethod
    def test_code(self):
        # tree = MonteCarloTree(copy.deepcopy(shogi.Board())).init()
        # leaf_node = tree.visit(tree._nodes[0], 0)
        # tree.backup(leaf_node, leaf_node.get_win_rate())
        # candidate = tree.main_line(tree._nodes[0], "", 0)

        tree = MonteCarloTree(copy.deepcopy(shogi.Board())).init()
        for i in range(1000):
            tree.playout(i, 3)
            # 25msec per playout


if __name__ == "__main__":
    # Node.test_code()
    MonteCarloTree.test_code()
