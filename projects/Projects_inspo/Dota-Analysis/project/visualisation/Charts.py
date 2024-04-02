import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from project.Config import IMAGE_DIR


def draw_activity_chart(df_activity, verbose=True):
    """Generates graph displaying Dota 2 in-game player activity, average and peak values, every month, since 2019.
    Graph is saved to IMAGE_DIR under "activity_plot.png" filename.

        Parameters:
        -----------
        df_activity: pd.DataFrame
            pd.DataFrame containing information about players in-game activity.

        Returns:
        -----------
        None
    """
    graph_filepath = os.path.join(IMAGE_DIR, "activity_plot.png")

    plt.figure(figsize=(16, 6))

    x, y = range(0, len(df_activity["average_activity"])), list(reversed(df_activity["average_activity"] / 1000))
    x2, y2 = range(0, len(df_activity["peak_activity"])), list(reversed(df_activity["peak_activity"] / 1000))

    plt.axvline(6, color="#444444", lw=1, linestyle="--", alpha=0.2)
    plt.axvline(18, color="#444444", lw=1, linestyle="--", alpha=0.2)
    plt.axvline(30, color="#444444", lw=1, linestyle="--", alpha=0.2)
    plt.axvline(42, color="#444444", lw=1, linestyle="--", alpha=0.2)
    plt.axvline(54, color="#444444", lw=1, linestyle="--", alpha=0.2)
    plt.axvline(66, color="#444444", lw=1, linestyle="--", alpha=0.2)
    plt.axvline(78, color="#444444", lw=1, linestyle="--", alpha=0.2)

    plt.scatter(x, y, edgecolor="black", linewidth="1", s=40, alpha=0.9, c="#eb1c09")
    plt.scatter(x2, y2, edgecolor="black", linewidth="1", s=40, alpha=0.9, c="#f79605")

    plt.plot(y, alpha=0.9, c="#eb1c09", label="Average")
    plt.plot(y2, alpha=0.9, c="#f79605", label="Peak", )

    plt.ylabel("Players in-game per hour [in thousands]")
    plt.legend(loc="upper right")
    plt.yticks(np.arange(0, 1500, 100))
    plt.xticks([6, 18, 30, 42, 54, 66, 78], ["2013", "2014", "2015", "2016", "2017", "2018", "2019"])
    plt.title("Player In-Game Activity")

    plt.grid(c="#444444", linestyle='--', linewidth=1, alpha=0.2)

    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)

    if verbose:
        print("Graph saved to file: {}".format(graph_filepath))

    plt.savefig(graph_filepath, dpi=200, bbox_inches="tight")


def draw_player_count_graph(df_total_players, df_mmr, verbose=True):
    """Generates graph displaying number of Dota 2 players: all players, players playing ranked matches, pro players.
    Graph is saved to IMAGE_DIR under "player_count.png" filename.

        Parameters:
        -----------
        df_total_players: pd.DataFrame
            pd.DataFrame containing information about professional players.
        df_mmr: pd.DataFrame
            pd.DataFrame containing information about players mmr distribution.

        Returns:
        -----------
        None
    """
    graph_filepath = os.path.join(IMAGE_DIR, "player_count.png")

    plt.figure(figsize=(16, 6))

    labels = ["Professional Players\n(according to OpenDota service)",
              "Players Playing Ranked Match Mode",
              "All Players\n"]

    color_map = np.array([(146, 0, 166), (186, 47, 138), (216, 91, 105)]) / 255.0

    y = np.array([df_total_players.shape[0], df_mmr["count"].sum(), 11422146])
    y_scaled = y / 1000 ** 2
    x = range(0, len(y))

    for i, color in zip(x, color_map):
        plt.bar([x[i]], [y_scaled[i]], edgecolor="black", linewidth="1", width=0.8, color=color)

    plt.grid(c="#444444", linestyle='--', linewidth=1, alpha=0.2)
    plt.xticks(x, labels, fontsize=12)
    plt.ylabel("Amount of players [in millions]")
    plt.yticks(np.arange(0, 16, 1))
    plt.title("Professional Players vs Non-Professional Players")

    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)

    for val, color, anchor in zip(y, color_map, [(0.1, 0.98), (0.4, 0.98), (0.72, 0.98)]):
        patch = Patch(facecolor=color, edgecolor="black", label="{val:,}".format(val=val))
        legend = plt.legend(handles=[patch], loc="upper left", bbox_to_anchor=anchor, fontsize=18)
        plt.gca().add_artist(legend)

    if verbose:
        print("Graph saved to file: {}".format(graph_filepath))

    plt.savefig(graph_filepath, dpi=200, bbox_inches="tight")


def draw_mmr_distribution_graph(df_mmr, verbose=True):
    """Generates graph displaying Dota 2 players MMR distribution (amount of players per 100 MMR bin).
    Graph is saved to IMAGE_DIR under "mmr.png" filename.

        Parameters:
        -----------
        df_mmr: pd.DataFrame
            pd.DataFrame containing information about players mmr distribution.

        Returns:
        -----------
        None
    """
    graph_filepath = os.path.join(IMAGE_DIR, "mmr.png")

    plt.figure(figsize=(16, 6))

    labels = df_mmr["bin_name"].values
    x, y = range(0, len(df_mmr["count"])), df_mmr["count"] / 1000

    plt.bar(x, y, edgecolor="black", linewidth="1", width=1.0, color="#eb1c09")

    plt.ylabel("Amount of players [in thousands]")
    plt.xlabel("MMR bin")
    plt.xticks(x, labels, rotation="vertical")
    plt.title("Player Distribution by MMR")

    plt.grid(c="#444444", linestyle='--', linewidth=1, alpha=0.2)

    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)

    if verbose:
        print("Graph saved to file: {}".format(graph_filepath))

    plt.savefig(graph_filepath, dpi=200, bbox_inches="tight")


def draw_ranks_distribution_graph(df_ranks, verbose=True):
    """Generates graph displaying Dota 2 players rank distribution (amount of players per in-game rank).
    Graph is saved to IMAGE_DIR under "ranks.png" filename.

        Parameters:
        -----------
        df_ranks: pd.DataFrame
            pd.DataFrame containing information about players rank distribution.

        Returns:
        -----------
        None
    """
    graph_filepath = os.path.join(IMAGE_DIR, "ranks.png")

    plt.figure(figsize=(16, 6))
    color_map = np.array([(47, 0, 135), (98, 0, 164), (146, 0, 166), (186, 47, 138),
                          (216, 91, 105), (238, 137, 73), (246, 189, 39), (228, 250, 21)]) / 255.0

    x, y = range(0, len(df_ranks["count"])), df_ranks["count"] / 1000

    for i, color in zip(range(0, 7 * len(df_ranks["bin_name"]), 7), color_map):
        plt.bar(x[i:i + 7], y[i:i + 7], edgecolor="black", linewidth="1", width=1.0, color=color)

    plt.ylabel("Amount of players [in thousands]")
    plt.xticks(x, df_ranks["bin_name"], rotation="vertical")
    plt.title("Player Distribution by Division")

    plt.grid(c="#444444", linestyle='--', linewidth=1, alpha=0.2)

    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)

    if verbose:
        print("Graph saved to file: {}".format(graph_filepath))

    plt.savefig(graph_filepath, dpi=200, bbox_inches="tight")


def draw_professional_player_mmr(df_total_players, verbose=True):
    """Generates graph displaying Dota 2 professional players mmr distribution (amount of players per 100  MMR bin),
    divided by pro players with teams and without teams. Graph is saved to IMAGE_DIR under "pro_player_mmr.png"
    filename.

        Parameters:
        -----------
        df_total_players: pd.DataFrame
            pd.DataFrame containing information about professional players mmr distribution.

        Returns:
        -----------
        None
    """
    graph_filepath = os.path.join(IMAGE_DIR, "pro_player_mmr.png")

    df_players_with_mmr = df_total_players.dropna(subset=["mmr_estimate"]).copy()
    df_players_with_mmr["mmr_estimate_bin"] = (df_players_with_mmr["mmr_estimate"] / 100).astype(int) * 100

    criteria_in_team = df_players_with_mmr["is_current_team_member"] == True
    df_players_in_team = df_players_with_mmr.loc[criteria_in_team].copy()
    df_players_in_team["mmr_estimate_bin"] = (df_players_in_team["mmr_estimate"] / 100).astype(int) * 100

    players_with_mmr = df_players_with_mmr["mmr_estimate_bin"].value_counts().to_dict()
    players_in_team = df_players_in_team["mmr_estimate_bin"].value_counts().to_dict()

    labels = np.arange(0, df_players_with_mmr["mmr_estimate_bin"].max() + 100, 100)
    x = np.arange(0, len(labels), 1)
    y1 = [(players_with_mmr[label] if label in players_with_mmr else 0) for label in labels]
    y2 = [(players_in_team[label] if label in players_in_team else 0) for label in labels]

    plt.figure(figsize=(16, 6))
    plt.bar(x, y1, width=0.8, edgecolor="black", linewidth="1",
            color=(0.38, 0.0, 0.64), label="All pro players")

    plt.bar(x, y2, width=0.8, edgecolor="black", linewidth="1",
            color=(0.85, 0.36, 0.41), label="Pro players in 5-man team")

    plt.grid(c="#444444", linestyle='--', linewidth=1, alpha=0.2)
    plt.ylabel("Amount of players")
    plt.xlabel("MMR bin")
    plt.xticks(x, labels, rotation="vertical")
    plt.yticks(range(0, 100, 10))
    plt.xlim(20, 84)
    plt.title("Professional Players MMR")
    plt.legend()

    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)

    if verbose:
        print("Graph saved to file: {}".format(graph_filepath))

    plt.savefig(graph_filepath, dpi=200, bbox_inches="tight")


def draw_prizepool_graph(df_ti_prizepool, verbose=True):
    """Generates graph displaying Dota 2 ti tournament yearly prize pool amounts. Graph is saved to IMAGE_DIR
    under "ti_prizepool.png" filename.

        Parameters:
        -----------
        df_ti_prizepool: pd.DataFrame
            pd.DataFrame containing information about ti tournament yearly prize pool amounts.

        Returns:
        -----------
        None
    """
    graph_filepath = os.path.join(IMAGE_DIR, "ti_prizepool.png")

    plt.figure(figsize=(16, 6))

    labels = df_ti_prizepool["year"].values
    x, y = range(0, len(df_ti_prizepool["prize_dolar"])), df_ti_prizepool["prize_dolar"] / 1000 ** 2

    plt.bar(x, y, edgecolor="black", linewidth="1", width=0.8, color="#f79605")

    plt.ylabel("Prize [$ in millions]")
    plt.yticks(np.arange(0, 31, 2))
    plt.xticks(x, labels)
    plt.title("Dota 2 World Championships - The International - Prize Pool")

    plt.grid(c="#444444", linestyle='--', linewidth=1, alpha=0.2)

    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)

    if verbose:
        print("Graph saved to file: {}".format(graph_filepath))

    plt.savefig(graph_filepath, dpi=200, bbox_inches="tight")
