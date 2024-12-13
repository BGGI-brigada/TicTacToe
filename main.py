import numpy as np
import random
import matplotlib.pyplot as plt
import gym

class TicTacToe:
    def __init__(self):
        self.board = [' '] * 9
        self.current_winner = None

    def available_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == ' ']

    def empty_squares(self):
        return ' ' in self.board

    def make_move(self, square, letter):
        if self.board[square] == ' ':
            self.board[square] = letter
            if self.winner(square, letter):
                self.current_winner = letter
            return True
        return False

    def winner(self, square, letter):
        row_ind = square // 3
        row = self.board[row_ind * 3:(row_ind + 1) * 3]
        if all([spot == letter for spot in row]):
            return True

        col_ind = square % 3
        column = [self.board[col_ind + i * 3] for i in range(3)]
        if all([spot == letter for spot in column]):
            return True

        if square % 2 == 0:
            diagonal1 = [self.board[i] for i in [0, 4, 8]]
            if all([spot == letter for spot in diagonal1]):
                return True

            diagonal2 = [self.board[i] for i in [2, 4, 6]]
            if all([spot == letter for spot in diagonal2]):
                return True

        return False

class RandomAgent:
    def __init__(self):
        pass

    def choose_action(self, available_moves):
        return random.choice(available_moves)

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state, available_moves):
        if random.random() < self.epsilon:
            return random.choice(available_moves)
        q_values = [self.get_q_value(state, a) for a in available_moves]
        max_q = max(q_values)
        return available_moves[q_values.index(max_q)]

    def update_q_value(self, state, action, reward, next_state, next_actions):
        future_rewards = [self.get_q_value(next_state, a) for a in next_actions]
        best_future = max(future_rewards) if future_rewards else 0
        current_q = self.get_q_value(state, action)
        self.q_table[(state, action)] = current_q + self.alpha * (reward + self.gamma * best_future - current_q)

    def get_state(self, board):
        return tuple(board)

def train_agent(episodes=1000):
    agent = QLearningAgent()
    opponent = RandomAgent()
    rewards = []
    avg_rewards = []
    for episode in range(episodes):
        game = TicTacToe()
        state = agent.get_state(game.board)
        total_reward = 0
        while game.empty_squares():
            action = agent.choose_action(state, game.available_moves())
            game.make_move(action, 'X')
            next_state = agent.get_state(game.board)

            if game.current_winner == 'X':
                reward = 1
                agent.update_q_value(state, action, reward, next_state, game.available_moves())
                total_reward += reward
                break
            elif not game.empty_squares():
                reward = 0
                agent.update_q_value(state, action, reward, next_state, game.available_moves())
                total_reward += reward
                break

            opponent_action = opponent.choose_action(game.available_moves())
            game.make_move(opponent_action, 'O')
            next_state = agent.get_state(game.board)

            if game.current_winner == 'O':
                reward = -1
                agent.update_q_value(state, action, reward, next_state, game.available_moves())
                total_reward += reward
                break

            state = next_state
        
        rewards.append(total_reward)
        avg_rewards.append(np.mean(rewards[-100:]))

    plt.plot(range(episodes), avg_rewards)
    plt.xlabel('Количество эпизодов')
    plt.ylabel('Средняя награда')
    plt.title('Кривая зависимости средней награды от количества эпизодов')
    plt.grid()
    plt.show()

    return agent

trained_agent = train_agent()
print("Обучение завершено. Q-таблица содержит", len(trained_agent.q_table), "записей.")
# print("Содержимое Q-таблицы:")
# for key, value in trained_agent.q_table.items():
#     print(f"Состояние: {key}, Действие: {key[1]}, Q-значение: {value}")
