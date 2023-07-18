import copy

from exceptions import AgentException


class AlphaBetaAgent:
    def __init__(self, my_token):
        self.my_token = my_token

    def decide(self, connect4):
        if connect4.who_moves != self.my_token:
            raise AgentException('not my round')
        return self.alpha_beta(connect4, 5, float('-inf'), float('inf'))[1]

    def alpha_beta(self, connect4, depth, alpha, beta):
        if depth == 0 or connect4.game_over:
            return self.evaluate(connect4), None
        if connect4.who_moves == self.my_token:
            best_value = float('-inf')
            best_move = None
            for column in connect4.possible_drops():
                connect4_copy = copy.deepcopy(connect4)
                connect4_copy.drop_token(column)
                value, _ = self.alpha_beta(connect4_copy, depth - 1, alpha, beta)
                if value > best_value:
                    best_value = value
                    best_move = column
                alpha = max(alpha, best_value)
                if beta <= alpha:
                    break
            return best_value, best_move
        else:
            best_value = float('inf')
            best_move = None
            for column in connect4.possible_drops():
                connect4_copy = copy.deepcopy(connect4)
                connect4_copy.drop_token(column)
                value, _ = self.alpha_beta(connect4_copy, depth - 1, alpha, beta)
                if value < best_value:
                    best_value = value
                    best_move = column
                beta = min(beta, best_value)
                if beta <= alpha:
                    break
            return best_value, best_move

    def evaluate(self, connect4):
        if connect4.wins == self.my_token:
            return 100
        elif connect4.wins is None:
            return self.additionalEvaluation(connect4)
        else:
            return -100

    def additionalEvaluation(self, connect4):
        allFours = connect4.iter_fours()
        if self.my_token == 'x':
            enemyToken = 'o'
        else:
            enemyToken = 'x'
        evaluation_sum = 0
        for four in allFours:
            positive = four.count(self.my_token)
            negative = four.count(enemyToken)
            if negative == 0:
                evaluation_sum += positive
            if positive == 0:
                evaluation_sum -= negative
        center = connect4.center_column()
        positive = center.count(self.my_token)
        negative = center.count(enemyToken)
        evaluation_sum += 2 * positive
        evaluation_sum -= 2 * negative
        return evaluation_sum
