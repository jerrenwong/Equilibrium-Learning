import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

N = 1
for _ in range(N):
    # Game 1
    payoff_1 = np.array([[1, 0], [0, 1]]) # |A_1| * |A_2|
    payoff_2 = np.array([[1, 0], [0, 1]]) # |A_2| * |A_1|

    # Game 2
    payoff_1 = np.array([[0.48699426, 0.27564575], [0.07094273, 0.86102572]])
    payoff_2 = np.array([[0.32304411, 0.64966661], [0.39678333, 0.14804825]])

    T = 10000

    strategies_1 = np.zeros((T, 2))
    strategies_2 = np.zeros((T, 2))
    plays = np.zeros((T,2,2))


    for t in range(T):

        cum_play = plays.sum(axis = 0)

        if t == 0:
            p_1 = np.random.random()
            p_2 = np.random.random()
            strategy_1 = np.array([p_1, 1-p_1])
            strategy_2 = np.array([p_2, 1-p_2])

        else:
            u_1 = (cum_play * payoff_1).sum()
            u_2 = (cum_play * payoff_2).sum()

            regret_1 = payoff_1 @ cum_play.sum(axis = 0) - u_1
            regret_2 = payoff_2 @ cum_play.sum(axis = 1) - u_2

            regret_1 = np.maximum(regret_1, 0)
            regret_2 = np.maximum(regret_2, 0)

            strategy_1 = regret_1 / regret_1.sum() if regret_1.sum() > 0 else np.array([1/2, 1/2])
            strategy_2 = regret_2 / regret_2.sum() if regret_2.sum() > 0 else np.array([1/2, 1/2])

        strategies_1[t] = strategy_1
        strategies_2[t] = strategy_2
        plays[t] = strategy_1[:,None] @ strategy_2[None,:]

    avg_plays = np.zeros((T,2,2))
    avg_strategies_1 = np.zeros((T,2))
    avg_strategies_2 = np.zeros((T,2))
    for t in range(T):
        avg_plays[t] = plays[:t+1].mean(axis = 0)
        avg_strategies_1[t] = strategies_1[:t+1].mean(axis = 0)
        avg_strategies_2[t] = strategies_2[:t+1].mean(axis = 0)

    #

    # #=================== Average Strategies Plot ===================
    avg_1_A = avg_strategies_1[:, 0]
    avg_2_A = avg_strategies_2[:, 0]

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


    # #=================== Average Plays Plot ===================
    # fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=False, sharey=False)

    # # Left plot: AA vs BB
    # axes[0].scatter(avg_plays[:, 0, 0], avg_plays[:, 1, 1],
    #                 c=range(T), cmap='viridis', s=20, alpha=0.7)
    # axes[0].set_title('AA vs BB')
    # axes[0].set_xlabel('AA')
    # axes[0].set_ylabel('BB')
    # axes[0].set_xlim(0, 1)
    # axes[0].set_ylim(0, 1)
    # # Right plot: AB vs BA
    # axes[1].scatter(avg_plays[:, 0, 1], avg_plays[:, 1, 0],
    #                 c=range(T), cmap='viridis', s=20, alpha=0.7)
    # axes[1].set_title('AB vs BA')
    # axes[1].set_xlabel('AB')
    # axes[1].set_ylabel('BA')
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    # plt.tight_layout()
    # plt.show()

    # AA = avg_plays[:, 0, 0]
    # AB = avg_plays[:, 0, 1]
    # BA = avg_plays[:, 1, 0]
    # BB = avg_plays[:, 1, 1]

    # fig, axes = plt.subplots(4, 1, figsize=(8, 8), sharex=True)

    # axes[0].plot(range(T), AA, color='C0')
    # axes[0].set_ylabel('AA')
    # axes[1].plot(range(T), AB, color='C1')
    # axes[1].set_ylabel('AB')
    # axes[2].plot(range(T), BA, color='C2')
    # axes[2].set_ylabel('BA')
    # axes[3].plot(range(T), BB, color='C3')
    # axes[3].set_ylabel('BB')
    # axes[3].set_xlabel('Time step')

    # plt.tight_layout()
    # plt.show()
    #=================== Animation =====================
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
    interval = 1000 / T  # milliseconds per frame so total ≈ 10 seconds

    ani = animation.FuncAnimation(fig, update, frames=T, interval=interval, blit=False, repeat=True)
    plt.show()

    ani.save('strategies.mp4', writer='ffmpeg', fps=T/1.5, dpi=150)

    determinant = avg_plays[-1][1][1] * avg_plays[-1][0][0] - avg_plays[-1][0][1] * avg_plays[-1][1][0]
    if determinant > 0.01:
        print(payoff_1, payoff_2, determinant)
        break
