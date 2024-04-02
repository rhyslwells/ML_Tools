from project.data.Generation import rank_name_generator


def clean_players(df_total_players):
    """Solves problem of duplicates within df_total_players pd.DataFrame by:
        - setting "is_current_team_player" col value for player to True only when it has appeared at least once as True
        - dropping duplicated player rows on "account_id"
        - leaving only "account_id", "team_id", "is_current_team_member", "mmr_estimate" columns as only those are
        important from business question perspective

         Parameters:
         -----------
         df_total_players: pd.DataFrame
             pd.DataFrame containing data about Dota 2 professional players.

         Returns:
         -----------
         df_total_players: pd.DataFrame
             pd.DataFrame cleaned.
    """
    team_player_map = df_total_players.groupby("account_id")["is_current_team_member"].any().to_dict()
    df_total_players["is_current_team_member"] = df_total_players["account_id"].map(team_player_map)
    df_total_players.drop_duplicates(subset=["account_id"], inplace=True)
    return df_total_players[["account_id", "team_id", "is_current_team_member", "mmr_estimate"]]


def clean_ranks(df_ranks):
    """Modified df_ranks pd.DataFrame object by:
        - replacing "bin" column with proper indices 0-49
        - replacing "bin_name" with Dota 2 rank names to which bin refers

        Parameters:
        -----------
        df_ranks: pd.DataFrame
            pd.DataFrame containing data about number of players in each Dota 2 rank.

        Returns:
        -----------
        df_ranks: pd.DataFrame
            pd.DataFrame with modified "bin" and "bin_name" column rows.
    """
    rank_names = []
    for rank in ["Herald", "Guardian", "Crusader", "Archon", "Legend", "Ancient", "Divine"]:
        rank_names.extend(rank_name_generator(rank))
    rank_names.append("Immortal")

    df_ranks["bin"] = df_ranks.index.values
    df_ranks["bin_name"] = rank_names

    return df_ranks
