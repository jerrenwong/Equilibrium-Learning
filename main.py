import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
import string
import os
import matplotlib.image as mpimg
import shutil

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

class MWULearner(Learner):
    def __init__(self, game, role, random_steps = 0, eta = 1):
        super().__init__(game, role)
        self.cum_utilities = np.zeros(self.num_actions)
        self.steps = 0
        self.eta = eta
        self.random_steps = random_steps

    def observe_utility(self, utility):
        self.cum_utilities += utility
        self.cum_utilities -= self.cum_utilities.max()
        self.steps += 1

    def next_strategy(self):
        # Random Steps
        if self.steps < self.random_steps:
            random_strategy = np.random.random(self.num_actions)
            random_strategy /= random_strategy.sum()
            self.current_strategy = random_strategy
            return random_strategy

        # MWU
        strategy = np.exp(self.cum_utilities * self.eta)
        strategy /= strategy.sum()
        self.current_strategy = strategy
        return strategy

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
        # Random Steps
        if self.steps < self.random_steps:
            random_strategy = np.random.random(self.num_actions)
            random_strategy /= random_strategy.sum()
            self.current_strategy = random_strategy
            return random_strategy

        # Initial Steps
        # if self.steps == 1:
        #     self.current_strategy = np.array([1.0,0.0])
        #     return self.current_strategy
        #
        # if self.steps == 2:
        #     self.current_strategy = np.array([0.0,1.0])
        #     return self.current_strategy

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
    def __init__(self, game: Game, learner_types: list, random_steps: int):
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

            # Calculate Utilities (multi-player safe)
            utilities = []
            n = self.num_learners
            axes = string.ascii_lowercase  # 'abcdefghijklmnopqrstuvwxyz'

            for i in range(n):
                payoff_i = self.game.payoffs[i]  # shape (A_0, A_1, ..., A_{n-1})

                # assign indices 'a', 'b', 'c', ... to players 0, 1, 2, ...
                idx = axes[:n]  # e.g. 'abc' for 3 players

                # einsum: payoff_i[a,b,c,...] * Π_{j≠i} strategy_j[index_j] -> vector over index_i
                terms = [idx] + [idx[j] for j in range(n) if j != i]
                equation = ','.join(terms) + '->' + idx[i]
                args = [payoff_i] + [strategies[j] for j in range(n) if j != i]

                utility_i = np.einsum(equation, *args)  # shape (A_i,)
                utilities.append(utility_i)

            # Propagate Utilities
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

    def get_strategy_history(self):
        strategy_history = [np.zeros((self.rounds, learner.num_actions)) for learner in self.learners]

        for round, strategies in enumerate(self.history):
            for role, strategy in enumerate(strategies):
                strategy_history[role][round] = strategy

        return strategy_history

    def get_play_history(self):
        play_history = np.zeros((self.rounds,) + self.game.payoffs[0].shape)

        for round, strategies in enumerate(self.history):
            play = reduce(np.multiply.outer, strategies)
            play_history[round] = play

        return play_history


def learning(payoffs, T, learner_types, random_steps = 0):
    if not isinstance(learner_types, list):
        learner_types = [learner_types] * (payoffs.ndim - 1)
    game = Game(payoffs)
    learning_alg = LearningAlg(game, learner_types, random_steps)
    learning_alg.train(T)
    return learning_alg

def run_multiple_learning(N, n, T, learner_types, jinx = False, random_steps = 0, visualize=True):
    """
    Performs learning on N number of n×n games, then (optionally) plots the static visualization of all results.

    Args:
        N: Number of games to run
        n: Size of the game (n×n action space for each player)
        T: Number of training rounds for each game
        learner_types: Type of learner to use (e.g., RMLearner, MWULearner)
        jinx: If True, use get_jinx_game(n); otherwise random payoffs
        random_steps: random_steps parameter for learners
        visualize: If True, call visualize_*; otherwise just return raw data
    """
    all_avg_plays = []
    all_strategy_histories = []
    all_avg_strategies = []
    print(f"Running learning on {N} randomly initialized {n}×{n} games...")

    for i in range(N):
        # Create payoff matrix for 2-player game: shape (2, n, n)
        if jinx:
            payoffs = get_jinx_game(n).payoffs
        else:
            payoffs = np.random.random((2, n, n))

        # Run learning algorithm
        learning_alg = learning(payoffs, T, learner_types, random_steps)

        # Get average plays over time
        avg_plays = learning_alg.average_plays()
        all_avg_plays.append(avg_plays)

        # Get raw strategy history
        strategy_history = learning_alg.get_strategy_history()
        all_strategy_histories.append(strategy_history)

        # Get average strategies
        avg_strategies = learning_alg.average_strategies()
        all_avg_strategies.append(avg_strategies)

        if (i + 1) % max(1, N // 10) == 0:
            print(f"Completed {i + 1}/{N} games")

    print(f"Done running learning.")

    if visualize:
        print(f"Visualizing results...")

        # Determine MWU parameters (if applicable) from the last run
        eta = None
        try:
            base_type = learner_types[0] if isinstance(learner_types, list) else learner_types
            if issubclass(base_type, MWULearner):
                eta = getattr(learning_alg.learners[0], "eta", None)
        except Exception:
            eta = None

        title_suffix = ""
        if eta is not None:
            title_suffix = rf"(MWU, random steps={random_steps})"

        # Visualize average plays
        visualize_plays_static(all_avg_plays, title_prefix="Average play", title_suffix=title_suffix)

        # Visualize raw strategies
        visualize_strategies_static(all_strategy_histories, title_prefix="Strategy", title_suffix=title_suffix)

        # Visualize average strategies
        visualize_strategies_static(all_avg_strategies, title_prefix="Average strategy", title_suffix=title_suffix)

    return all_avg_plays, all_strategy_histories, all_avg_strategies

def find_weird_games(N, T, d, eps, learner_types):
    for _ in range(N):
        payoffs = np.random.random((2,d,d))
        learning_alg = learning(payoffs, T, learner_types)

        avg_plays = learning_alg.average_plays()
        avg_play = avg_plays[-1]

        if abs(np.linalg.det(avg_play)) > eps:
            avg_strategies = learning_alg.average_strategies()
            visualize_strategies_2_2(avg_strategies)

    return None

def get_jinx_game(n):
    payoffs = np.eye(n)
    payoffs = np.stack([payoffs, payoffs], axis=0)
    game = Game(payoffs)
    return game


def generate_side_by_side_from_run_multiple(
    rs=(0, 1, 10, 100, 1000),
    N=5,
    n=2,
    T=10_000,
    learner_types=MWULearner,   # or RMLearner
    jinx=True,
    base_name="mwu",
    show=True,
):
    os.makedirs("results", exist_ok=True)

    base_type = learner_types[0] if isinstance(learner_types, list) else learner_types
    if base_type.__name__.startswith("RM"):
        learner_label = "RM"
    elif base_type.__name__.startswith("MWU"):
        learner_label = "MWU"
    else:
        learner_label = base_type.__name__

    strategies_imgs = []
    plays_imgs = []

    for r in rs:
        run_multiple_learning(
            N=N,
            n=n,
            T=T,
            learner_types=learner_types,
            jinx=jinx,
            random_steps=r,
        )

        strat_path = os.path.join("results", "strategies_static.png")
        plays_path = os.path.join("results", "plays_static.png")

        if not (os.path.exists(strat_path) and os.path.exists(plays_path)):
            raise FileNotFoundError(
                f"Expected {strat_path} and {plays_path} to exist; "
                "did visualize_* filenames change?"
            )

        strategies_imgs.append(mpimg.imread(strat_path))
        plays_imgs.append(mpimg.imread(plays_path))

    num_r = len(rs)

    # ---------- Big horizontal figure for strategies ----------
    fig_s, axes_s = plt.subplots(
        1,
        num_r,
        figsize=(5 * num_r, 7.5),  # larger logical size
    )
    if num_r == 1:
        axes_s = [axes_s]

    for ax, img, r in zip(axes_s, strategies_imgs, rs):
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(f"r = {r}", fontsize=12)

    fig_s.suptitle(
        f"{learner_label}: Average Player Strategies Over Time\n"
        f"#Simulations = {N}, #Rounds = {T}",
        fontsize=14,
        y=0.85,
    )
    fig_s.tight_layout(rect=[0, 0, 1, 0.90])

    strategies_out = os.path.join("results", f"{base_name}_strategies_multi_r.png")
    fig_s.savefig(strategies_out, dpi=300)  # higher DPI
    if show:
        plt.show()
    else:
        plt.close(fig_s)

    # ---------- Big horizontal figure for empirical play ----------
    fig_p, axes_p = plt.subplots(
        1,
        num_r,
        figsize=(5 * num_r, 7.5),
    )
    if num_r == 1:
        axes_p = [axes_p]

    for ax, img, r in zip(axes_p, plays_imgs, rs):
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(f"r = {r}", fontsize=12)

    fig_p.suptitle(
        f"{learner_label}: Average Empirical Play Over Time\n"
        f"#Simulations = {N}, #Rounds = {T}",
        fontsize=14,
        y=0.85,
    )
    fig_p.tight_layout(rect=[0, 0, 1, 0.90])

    plays_out = os.path.join("results", f"{base_name}_plays_multi_r.png")
    fig_p.savefig(plays_out, dpi=600)  # higher DPI
    if show:
        plt.show()
    else:
        plt.close(fig_p)

    print(f"Saved strategies grid to: {strategies_out}")
    print(f"Saved plays grid to     : {plays_out}")

    return strategies_out, plays_out

#=================== Average Strategies Plot ===================
def visualize_strategies_2_2(strategies):
    T = len(strategies[0])
    strategy_1_A = strategies[0][:, 0]
    strategy_2_A = strategies[1][:, 0]

    # Create 2D plot
    plt.figure(figsize=(8, 8))
    # Plot individual dots for each time step
    plt.scatter(strategy_1_A, strategy_2_A, c=range(T), cmap='viridis',
                s=20, alpha=0.6, label='Trajectory', zorder=3)

    plt.xlabel('(P(A) for Player 1)', fontsize=12)
    plt.ylabel('(P(A) for Player 2)', fontsize=12)
    plt.title('Regret Matching: Strategy Evolution in 2D Plane', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    # plt.show()


# =================== Static Visualization =====================
def visualize_plays_static(plays, title_prefix="", title_suffix=""):
    """
    Plots n×n many 2D graphs showing the probability of playing each action combination over time.
    Can handle multiple plays arrays and plot them with different colors.

    Args:
        plays: numpy array of shape (T, n, n) or (num_runs, T, n, n), or a list of arrays
               where T is the number of time steps and n is the number of actions for each player
    """
    # Normalize input to always be a list of arrays
    plays_array = np.array(plays)

    # Check if input is a single run or multiple runs
    if plays_array.ndim == 3:
        # Single run: shape (T, n, n)
        plays_list = [plays_array]
    elif plays_array.ndim == 4:
        # Multiple runs: shape (num_runs, T, n, n)
        plays_list = [plays_array[i] for i in range(plays_array.shape[0])]
    elif isinstance(plays, list):
        # List of arrays
        plays_list = plays
    else:
        plays_list = [plays_array]

    # Get dimensions from first play array
    T = len(plays_list[0])
    n = plays_list[0].shape[1]  # Assuming square matrix (n actions for each player)
    num_runs = len(plays_list)

    # Create n×n grid of subplots
    fig, axes = plt.subplots(n, n, figsize=(4*n, 4*n))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    # Flatten axes for easier iteration
    if n == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()

    # Plot each action combination over time
    time_steps = np.arange(1, T + 1)

    # Use a colormap for different runs
    colors = plt.cm.tab10(np.linspace(0, 1, num_runs)) if num_runs <= 10 else plt.cm.viridis(np.linspace(0, 1, num_runs))

    for i in range(n):
        for j in range(n):
            idx = i * n + j
            ax = axes_flat[idx]

            # Plot each run with a different color
            for run_idx, play in enumerate(plays_list):
                # Extract probability over time for action combination (i, j)
                prob_over_time = play[:, i, j]

                # Plot the line with different color for each run
                # Use higher zorder so curves are visually above grid/axes
                ax.plot(
                    time_steps,
                    prob_over_time,
                    linewidth=2,
                    color=colors[run_idx],
                    alpha=0.7,
                    zorder=3,
                )

            ax.set_xlabel('Time Step', fontsize=10)
            ax.set_ylabel('Probability', fontsize=10)
            # Use LaTeX-style A_i labels (1-based indexing)
            ax.set_title(
                rf'$(A_{{{i + 1}}}, A_{{{j + 1}}})$',
                fontsize=12,
                fontweight='bold',
            )
            ax.set_ylim(0, 1)
            ax.set_axisbelow(True)
            ax.grid(True, alpha=0.3, zorder=0)

    title = f' Average Empirical Play Over Time'
    if num_runs > 1:
        title += f' ({num_runs} runs)'
    if title_suffix:
        title += f' {title_suffix}'
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.92)
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    # Always save to results folder
    os.makedirs("results", exist_ok=True)
    fig.savefig(os.path.join("results", "plays_static.png"), dpi=150)

    # plt.show()

def visualize_strategies_static(strategies_list, title_prefix="", title_suffix=""):
    """
    Plots each player's raw strategy over time.
    Can handle multiple runs and plot them with different colors.

    Args:
        strategies_list: Either a single strategy history (list of arrays) or a list of strategy histories.
                         Each strategy history is a list of arrays [player1_strategies, player2_strategies, ...]
                         where each array has shape (T, n) - T is the number of time steps and n is the number of actions
    """
    # Normalize input: check if it's a single strategy history or a list of strategy histories
    if len(strategies_list) == 0:
        return

    # Check if first element is a numpy array (meaning strategies_list is a single strategy history)
    # A single strategy history is a list of arrays: [array1, array2, ...]
    # A list of strategy histories is: [[array1, array2, ...], [array1, array2, ...], ...]
    first_elem = strategies_list[0]

    # If first element is an array, then strategies_list is a single strategy history (list of arrays)
    # and needs to be wrapped
    if isinstance(first_elem, np.ndarray):
        strategies_list = [strategies_list]

    # Get dimensions from first run
    num_players = len(strategies_list[0])
    if num_players == 0:
        return

    T = len(strategies_list[0][0])
    n = strategies_list[0][0].shape[1]  # Number of actions
    num_runs = len(strategies_list)

    # Create subplots: one row per player, n columns (one per action)
    fig, axes = plt.subplots(num_players, n, figsize=(4*n, 4*num_players))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    # Handle single player case
    if num_players == 1:
        axes = axes.reshape(1, -1)
    if n == 1:
        axes = axes.reshape(-1, 1)

    # Plot each action combination over time
    time_steps = np.arange(1, T + 1)

    # Use a colormap for different runs
    colors = plt.cm.tab10(np.linspace(0, 1, num_runs)) if num_runs <= 10 else plt.cm.viridis(np.linspace(0, 1, num_runs))

    for player_idx in range(num_players):
        for action_idx in range(n):
            ax = axes[player_idx, action_idx]

            # Plot each run with a different color
            for run_idx, strategy_history in enumerate(strategies_list):
                # Extract probability over time for this player's action
                prob_over_time = strategy_history[player_idx][:, action_idx]

                # Plot the line with different color for each run
                # Use higher zorder so curves are visually above grid/axes
                ax.plot(
                    time_steps,
                    prob_over_time,
                    linewidth=2,
                    color=colors[run_idx],
                    alpha=0.7,
                    zorder=3,
                )

            ax.set_xlabel('Time Step', fontsize=10)
            ax.set_ylabel('Probability', fontsize=10)
            # Use LaTeX-style A_i labels (1-based indexing)
            ax.set_title(
                rf'Player {player_idx + 1}, $A_{{{action_idx + 1}}}$',
                fontsize=12,
                fontweight='bold',
            )
            ax.set_ylim(0, 1)
            ax.set_axisbelow(True)
            ax.grid(True, alpha=0.3, zorder=0)

    title = f'Average Player Strategies Over Time'
    if num_runs > 1:
        title += f' ({num_runs} runs)'
    if title_suffix:
        title += f' {title_suffix}'
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.92)
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    # Always save to results folder
    os.makedirs("results", exist_ok=True)
    fig.savefig(os.path.join("results", "strategies_static.png"), dpi=150)

    # plt.show()

# =================== Animation =====================
def visualize_plays_2_2_animate(plays):
    T = len(plays)
    AA, AB, BA, BB = plays[:, 0, 0], plays[:, 0, 1], plays[:, 1, 0], plays[:, 1, 1]
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
    # plt.show()

    ani.save('strategies.mp4', writer='ffmpeg', fps=T/1.5, dpi=150)


if __name__ == "__main__":
    # payoffs = np.array([[[0,0,1], [1,0,0], [0,1,0]], [[0,1,0], [0,0,1], [1,0,0]]])
    # payoffs = np.array([[[1,0],[0,1]], [[1,0],[0,1]]])

    # payoffs = np.random.random((2,2,2))
    # game = Game(payoffs)
    # learning_alg = LearningAlg(game, [RMLearner, RMLearner])
    # learning_alg.train(10000)
    # avg_strategies = learning_alg.average_strategies()
    # visualize_strategies_2_2(avg_strategies)

    # avg_plays = learning_alg.average_plays()
    # visualize_plays_static(avg_plays)
    generate_side_by_side_from_run_multiple(
        rs=(0, 5, 25, 125),
        N=1,          # number of simulations (same as your example)
        n=2,          # 2x2 games
        T=1_000,     # 10,000 steps as requested
        learner_types=RMLearner,
        jinx=True,
        base_name="rm_2_100",
        show = False
    )
