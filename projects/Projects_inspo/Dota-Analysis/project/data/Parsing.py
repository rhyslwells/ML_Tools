import pandas as pd
import numpy as np


def dict_to_dataframe(data_dict):
    """Parses python dictionary to pd.DataFrame format.

       Parameters:
       -----------
       data_dict: dict
            Python dictionary with data.

       Returns:
       -----------
       response: pd.DataFrame
            Data in pd.DataFrame format.
    """
    result = pd.DataFrame(data_dict)
    return result


def dict_list_to_dataframe(data_dict_list):
    """Parses list of python dictionaries to single pd.DataFrame by using iterative concatenation.

       Parameters:
       -----------
       data_dict_list: list
            List containing python dictionaries with data.

       Returns:
       -----------
       df_data: pd.DataFrame
            Data concatenated and parsed to pd.DataFrame format.
    """
    df_data = None
    for data_dict in data_dict_list:
        if df_data is None:
            df_data = pd.DataFrame(data_dict)
        else:
            df_data = pd.concat([df_data, pd.DataFrame(data_dict)], axis=0, sort=False)
    df_data.reset_index(drop=True, inplace=True)
    return df_data


def account_ids_player_details_to_mmr_dict(id_list, detail_list):
    """For each player account_id and player_details_dict it extracts mmr_estimate value and saves in simplified
    dict where account_id is a key and mmr_estimate is a value.

        Parameters:
        -----------
        id_list: list
           List containing player account ids.
        detail_list: list
           List containing player details dictionary.

        Returns:
        -----------
        mmr_dict: dict
           Dictionary with account_id as key and mmr_estimate as value.
    """

    def _get_mmr_estimate(player_detail):
        return player_detail["mmr_estimate"]["estimate"] if "mmr_estimate" in player_detail else np.nan

    mmr_dict = {"account_id": [], "mmr_estimate": []}
    for account_id, player_details in zip(id_list, detail_list):
        mmr_dict["account_id"].append(account_id)
        mmr_dict["mmr_estimate"].append(_get_mmr_estimate(player_details))

    return mmr_dict


def steamcharts_to_dataframe(text):
    """Parses raw text copied from steam charts website into pd.DataFrame object.

        Parameters:
        -----------
        text: str
           Steam charts in form of raw text.

        Returns:
        -----------
        df_data: pd.DataFrame
           DataFrame containing data about game activity.
    """
    months = ["January", "February", "March", "April", "May", "June", "July", "August",
              "September", "October", "November", "December"]
    activity_dict = {"month": [], "year": [], "average_activity": [], "peak_activity": []}

    for i, row in enumerate(text.split("\n")):
        row_parts = row.split()

        if any((m in row_parts) for m in months):
            activity_dict["month"].append(row_parts[0])
            activity_dict["year"].append(int(row_parts[1]))
            activity_dict["average_activity"].append(float(row_parts[2].replace(",", "")))
            activity_dict["peak_activity"].append(float(row_parts[5].replace(",", "")))

    df_data = pd.DataFrame(activity_dict)
    return df_data
