import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Game:
    def __init__(self, payoffs):
        self.payoffs = payoffs

class Learner:
    def __init__(self, game, role):
        self.game = game
        self.role = role
        self.num_actions = game.payoffs[0].shape[role]
        self.current_strategy = None

    def observe_utility(self, utility):
        pass

    def next_strategy(self):
        pass

class RMLearner(Learner):
    def __init__(self, game, role, random_steps = 0):
        super().__init__(game, role)
        self.cum_regret = np.zeros(self.num_actions)
        self.steps = 0
        self.random_steps = random_steps

    def observe_utility(self, utility):
        cur_utility = self.current_strategy @ utility
        cur_regret = utility - cur_utility
        self.cum_regret += cur_regret
        self.steps += 1

    def next_strategy(self):
        # # Random Steps
        # if self.steps < self.random_steps:
        #     random_strategy = np.random.random(self.num_actions)
        #     random_strategy /= random_strategy.sum()
        #     self.current_strategy = random_strategy
        #     return random_strategy

        # Initial Steps
        if self.steps == 1:
            self.current_strategy = np.array([1.0,0.0])
            return self.current_strategy

        if self.steps == 2:
            self.current_strategy = np.array([0.0,1.0])
            return self.current_strategy

        # Regret Matching
        cum_reg_p = np.maximum(self.cum_regret, 0)
        total = cum_reg_p.sum()

        if total == 0:
            strategy = np.ones(self.num_actions)/self.num_actions
        else:
            strategy = cum_reg_p/total

        self.current_strategy = strategy
        return strategy

class LearningAlg:
    def __init__(self, game: Game, learner_types: list, random_steps: int = 0):
        self.game = game
        self.num_learners = len(learner_types)
        assert self.num_learners == game.payoffs[0].ndim, "Incorrect number of learners"
        self.history = []
        self.learners = [learner_type(game, role, random_steps) for role, learner_type in enumerate(learner_types)]
        self.rounds = 0

    def train(self, rounds: int):
        for _ in range(rounds):
            # Next Strategies
            strategies = [learner.next_strategy() for learner in self.learners]
            self.history.append(strategies)

            # Calculate Utilities
            utilities = []
            for i in range(self.num_learners):
                utility_i = self.game.payoffs[i]
                for strategy in strategies[:i]:
                    utility_i = np.tensordot(utility_i, strategy, axes=([0], [0]))
                for strategy in strategies[i+1:]:
                    utility_i = np.tensordot(utility_i, strategy, axes=([1], [0]))
                utilities.append(utility_i)
            # Propogate Utilities
            for role, learner in enumerate(self.learners):
                learner.observe_utility(utilities[role])
        self.rounds += rounds

    def average_plays(self):
        avg_plays = np.zeros((self.rounds,) + self.game.payoffs[0].shape)
        total_play = np.zeros(self.game.payoffs[0].shape, dtype=np.float64)

        for round, strategies in enumerate(self.history):
            play = reduce(np.multiply.outer, strategies)
            total_play += play
            avg_plays[round] = total_play/(round + 1)

        return avg_plays

    def average_strategies(self):
        avg_strategies  = [np.zeros((self.rounds, learner.num_actions)) for learner in self.learners]
        ttl_strategy = [np.zeros(learner.num_actions) for learner in self.learners]

        for round, strategies in enumerate(self.history):
            for role, strategy in enumerate(strategies):
                ttl_strategy[role] += strategy
                avg_strategies[role][round] = ttl_strategy[role]/(round + 1)

        return avg_strategies

def find_weird_games(N, T, eps):
    for _ in range(N):
        payoffs = np.random.random((2,2,2))
        learning_alg = regret_learning(payoffs, T)

        avg_plays = learning_alg.average_plays()
        avg_play = avg_plays[-1]

        if abs(np.linalg.det(avg_play)) > eps:
            avg_strategies = learning_alg.average_strategies()
            visualize_strategies_2_2(avg_strategies)
            return payoffs

def regret_learning(payoffs, T):
    game = Game(payoffs)
    learner_types = [RMLearner for _ in range(payoffs.ndim - 1)]
    learning_alg = LearningAlg(game, learner_types)
    learning_alg.train(T)
    return learning_alg

#=================== Average Strategies Plot ===================
def visualize_strategies_2_2(avg_strategies):
    T = len(avg_strategies[0])
    avg_1_A = avg_strategies[0][:, 0]
    avg_2_A = avg_strategies[1][:, 0]

    # Create 2D plot
    plt.figure(figsize=(8, 8))
    # Plot individual dots for each time step
    plt.scatter(avg_1_A, avg_2_A, c=range(T), cmap='viridis',
                s=20, alpha=0.6, label='Trajectory', zorder=3)

    plt.xlabel('(P(A) for Player 1)', fontsize=12)
    plt.ylabel('(P(A) for Player 2)', fontsize=12)
    plt.title('Regret Matching: Strategy Evolution in 2D Plane', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()


# =================== Animation =====================
def visualize_plays_2_2(avg_plays):
    T = len(avg_plays)
    AA, AB, BA, BB = avg_plays[:, 0, 0], avg_plays[:, 0, 1], avg_plays[:, 1, 0], avg_plays[:, 1, 1]
    data_series = [AA, AB, BA, BB]
    labels = ['AA', 'AB', 'BA', 'BB']

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    bars = []
    for ax, label in zip(axes.flat, labels):
        bar = ax.bar([0], [0], color='black', width=0.4)
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([0])
        ax.set_xticklabels([label], fontsize=12, fontweight='bold')
        ax.set_ylabel('Value')
        bars.append(bar[0])

    # --- Update function ---
    def update(frame):
        for bar, series in zip(bars, data_series):
            bar.set_height(series[frame])
        fig.suptitle(f'Time step: {frame + 1}/{T}', fontsize=14)
        return bars

    # --- Animation speed control ---
    interval = 1000 / T   # milliseconds per frame so total ≈ 10 seconds

    ani = animation.FuncAnimation(fig, update, frames=T, interval=interval, blit=False, repeat=True)
    plt.show()

    ani.save('strategies.mp4', writer='ffmpeg', fps=T/1.5, dpi=150)


if __name__ == "__main__":
    # payoffs = np.array([[[0,0,1], [1,0,0], [0,1,0]], [[0,1,0], [0,0,1], [1,0,0]]])
    # payoffs = np.array([[[1,0],[0,1]], [[1,0],[0,1]]])
    # game = Game(payoffs)
    # learning_alg = LearningAlg(game, [RMLearner, RMLearner], random_steps=2)
    # learning_alg.train(10000)
    # avg_plays = learning_alg.average_plays()
    # visualize_strategies_2_2(learning_alg.average_strategies())

    find_weird_games(10000, 1000, 0.05)
