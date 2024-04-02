import os

from project.Config import DATA_DIR

from project.DataProvider import (
    fetch_pro_player_list,
    fetch_pro_team_list,
    fetch_team_players,
    fetch_pro_player_details,
    fetch_mmr_distribution,
    get_prizepool_data,
    get_steamcharts_data
)

from project.DataTransformer import (
    player_data_to_df,
    mmr_data_to_df,
    ranks_data_to_df,
    prizepool_data_to_df,
    steamcharts_data_to_df
)

from project.DataCleaner import (
    clean_players,
    clean_ranks
)

from project.visualisation.Charts import (
    draw_activity_chart,
    draw_prizepool_graph,
    draw_player_count_graph,
    draw_mmr_distribution_graph,
    draw_professional_player_mmr,
    draw_ranks_distribution_graph
)


def etl_process(verbose):
    """Pipeline that consists of the following operation:
        - fetching data about professional players and teams from OpenDotaAPI
        - transforming fetched JSON files, retrieved as dicts, to pd.DataFrame objects
        - combining downloaded player data together
        - parsing manually scrapped and hardcoded data about tournament prize amounts and player in-game activity
        - cleaning pd.DataFrame objects by removing duplicates, selecting columns that can be used to draw conclusions
        - saving all dataframes as .csv files in DATA_DIR
        - generating graphs for stated business questions and saving them in IMAGE_DIR

    Parameters:
    -----------
    verbose: bool
        Default value is False. When switched to True, function will display logs.

    Returns:
    -----------
    df_total_players: pd.DataFrame
        Object containing data about all professional players and teams.
    df_mmr: pd.DataFrame
        Object containing data about all players mmr distribution.
    df_ranks: pd.DataFrame
        Object containing data about all players rank distribution.
    df_prizepool: pd.DataFrame
        Object containing data about Dota 2 TI tournament prize pool yearly amounts.
    df_activity: pd.DataFrame
        Object containing data about Dota 2 player in-game activity since years 2012.
    """
    print("Fetching raw data...")
    pro_player_list = fetch_pro_player_list(verbose=verbose)
    pro_team_list = fetch_pro_team_list(verbose=verbose)
    team_ids, team_players = fetch_team_players(pro_team_list, verbose=verbose)
    account_ids, pro_player_details = fetch_pro_player_details(pro_player_list, verbose=verbose)
    mmr_distribution = fetch_mmr_distribution(verbose=verbose)
    prizepool_dict = get_prizepool_data()
    steamcharts_string = get_steamcharts_data()

    print("Transforming data into pd.DataFrame format...")
    df_total_players = player_data_to_df(pro_player_list, pro_team_list, team_players, pro_player_details, account_ids)
    df_mmr = mmr_data_to_df(mmr_distribution)
    df_ranks = ranks_data_to_df(mmr_distribution)
    df_prizepool = prizepool_data_to_df(prizepool_dict)
    df_activity = steamcharts_data_to_df(steamcharts_string)

    print("Cleaning data...")
    df_total_players = clean_players(df_total_players)
    df_ranks = clean_ranks(df_ranks)

    print("Saving data to files...")
    dataframes = [df_total_players, df_mmr, df_ranks, df_prizepool, df_activity]
    filenames = ["players.csv", "mmr.csv", "ranks.csv", "prizepool.csv", "activity.csv"]
    for df, filename in zip(dataframes, filenames):
        filepath = os.path.join(DATA_DIR, filename)
        df.to_csv(filepath, index=False)
        print("\t- created file: {}".format(filepath))

    return df_total_players, df_mmr, df_ranks, df_prizepool, df_activity


if __name__ == "__main__":
    df_total_players, df_mmr, df_ranks, df_prizepool, df_activity = etl_process(verbose=False)

    draw_prizepool_graph(df_prizepool)
    draw_professional_player_mmr(df_total_players)
    draw_ranks_distribution_graph(df_ranks)
    draw_mmr_distribution_graph(df_mmr)
    draw_player_count_graph(df_total_players, df_mmr)
    draw_activity_chart(df_activity)
