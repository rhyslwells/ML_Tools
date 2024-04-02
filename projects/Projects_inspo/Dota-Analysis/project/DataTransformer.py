from project.data.Parsing import (
    dict_to_dataframe,
    dict_list_to_dataframe,
    steamcharts_to_dataframe,
    account_ids_player_details_to_mmr_dict
)


def player_data_to_df(pro_player_list, pro_team_list, team_players, player_details, account_ids):
    """Combines pd.DataFrames objects containing various data about Dota 2 professional players and professional teams.
    It combines all pd.DataFrames into one on "account_id" key and "team_id" as key. Returns whole data as a single
    pd.DataFrame.

        Parameters:
        -----------
        steam_charts_string: str
           String with steam charts data.

        Returns:
        -----------
        df_total_players: pd.DataFrame
           pd.DataFrame object containing all professional player and team data combined together.
    """
    df_players = dict_to_dataframe(pro_player_list)
    df_teams = dict_to_dataframe(pro_team_list)
    df_team_players = dict_list_to_dataframe(team_players)

    account_id_mmr_dict = account_ids_player_details_to_mmr_dict(account_ids, player_details)
    df_team_player_mmr = dict_to_dataframe(account_id_mmr_dict)

    df_teams.rename(columns={"last_match_time": "team_last_match_time"}, inplace=True)
    df_teams.drop(columns=["name"], inplace=True)
    df_teams = df_teams.set_index("team_id")
    df_players = df_players.set_index("team_id")

    df_total_players = df_players.join(df_teams, on="team_id")

    df_total_players.reset_index(inplace=True)
    df_total_players.set_index("account_id")

    df_team_players.rename(columns={
        "wins": "team_wins",
        "games_played": "team_games_played"
    }, inplace=True)

    df_team_players = df_team_players[["account_id", "team_games_played", "team_wins", "is_current_team_member"]]
    df_team_players = df_team_players.set_index("account_id")

    df_total_players = df_total_players.join(df_team_players, on="account_id")
    df_team_player_mmr = df_team_player_mmr.set_index("account_id")
    df_total_players = df_total_players.join(df_team_player_mmr, on="account_id")

    return df_total_players


def mmr_data_to_df(mmr_distribution):
    """Parses mmr JSON in form of python dict to pd.DataFrame format from extracted mmr part.

        Parameters:
        -----------
        mmr_distribution: dict
            Python dictionary containing players divided by mmr bins.

        Returns:
        -----------
        df_mmr: pd.DataFrame
            Dictionary in form of pd.DataFrame.
    """
    df_mmr = dict_to_dataframe(mmr_distribution["mmr"]["rows"])
    return df_mmr


def ranks_data_to_df(mmr_distribution):
    """Parses mmr JSON in form of python dict to pd.DataFrame format from extracted ranks part.

        Parameters:
        -----------
        mmr_distribution: dict
            Python dictionary containing players divided by mmr bins.

        Returns:
        -----------
        df_mmr: pd.DataFrame
            Dictionary in form of pd.DataFrame.
    """
    df_ranks = dict_to_dataframe(mmr_distribution["ranks"]["rows"])
    return df_ranks


def prizepool_data_to_df(prize_pool_dict):
    """Parses python dict with ti yearly prize pool amounts to pd.DataFrame.

       Parameters:
       -----------
       prize_pool_dict: dict
           Python dictionary containing ti yearly prize pools.

       Returns:
       -----------
       df_prizepool: pd.DataFrame
           Dictionary in form of pd.DataFrame.
    """
    df_prizepool = dict_to_dataframe(prize_pool_dict)
    return df_prizepool


def steamcharts_data_to_df(steam_charts_string):
    """Parses string containing steam charts data about Dota 2 game activity to pd.DataFrame format.

       Parameters:
       -----------
       steam_charts_string: str
           String with steam charts data.

       Returns:
       -----------
       df_activity: pd.DataFrame
           Dictionary in form of pd.DataFrame.
    """
    df_activity = steamcharts_to_dataframe(steam_charts_string)
    return df_activity
