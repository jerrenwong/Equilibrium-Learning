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