from project.api.Endpoint import (
    ENDPOINT_GET_PRO_PLAYER_LIST,
    ENDPOINT_GET_PRO_TEAM_LIST,
    ENDPOINT_GET_PLAYER_INFO,
    ENDPOINT_GET_PRO_TEAM_PLAYER_LIST,
    ENDPOINT_GET_PLAYER_MMR_DISTRIBUTION
)

from project.data.Scrapping import fetch_data_from_endpoint


def fetch_pro_player_list(verbose=True):
    """Downloads list of pro players.

        Parameters:
        -----------
        verbose: bool
            Switch for displaying network communication and statuses.

        Returns:
        -----------
        response: dict
            Fetched response JSON parsed to python dictionary.
    """
    response = fetch_data_from_endpoint(ENDPOINT_GET_PRO_PLAYER_LIST, verbose=verbose)
    return response


def fetch_pro_team_list(verbose=True):
    """Downloads list of pro teams.

        Parameters:
        -----------
        verbose: bool
            Switch for displaying network communication and statuses.

        Returns:
        -----------
        response: dict
            Fetched response JSON parsed to python dictionary.
    """
    response = fetch_data_from_endpoint(ENDPOINT_GET_PRO_TEAM_LIST, verbose=verbose)
    return response


def fetch_team_players(pro_team_list, verbose=True):
    """For each professional team downloads list of players that were assigned to the team.

        Parameters:
        -----------
        pro_team_list: list
            List of dictionaries containing information about professional teams.
        verbose: bool
            Switch for displaying network communication and statuses.

        Returns:
        -----------
        response: dict
            Fetched response JSON parsed to python dictionary.
    """
    team_ids = [team_json["team_id"] for team_json in pro_team_list]

    team_players = []
    for i, team_id in enumerate(team_ids):

        if verbose:
            print("Downloading data of team {}/{}. ".format(i + 1, len(team_ids)), end="")

        endpoint = ENDPOINT_GET_PRO_TEAM_PLAYER_LIST.format(team_id)
        pro_players_json = fetch_data_from_endpoint(endpoint, verbose=verbose)
        team_players.append(pro_players_json)

    return team_ids, team_players


def fetch_pro_player_details(pro_player_list, verbose=True):
    """Downloads list of pro player details..

        Parameters:
        -----------
        pro_player_list: list
            List of dictionaries containing information about professional players.
        verbose: bool
            Switch for displaying network communication and statuses.

        Returns:
        -----------
        response: dict
            Fetched response JSON parsed to python dictionary.
    """
    account_ids = [player_json["account_id"] for player_json in pro_player_list]

    pro_player_details = []
    for i, account_id in enumerate(account_ids):
        if verbose:
            print("Downloading data of player {}/{}. ".format(i + 1, len(account_ids)), end="")

        endpoint = ENDPOINT_GET_PLAYER_INFO.format(account_id)

        pro_player_details_json = fetch_data_from_endpoint(endpoint, verbose=verbose)
        pro_player_details.append(pro_player_details_json)

    return account_ids, pro_player_details


def fetch_mmr_distribution(verbose=True):
    """Downloads information about players mmr.

        Parameters:
        -----------
        verbose: bool
            Switch for displaying network communication and statuses.

        Returns:
        -----------
        response: dict
            Fetched response JSON parsed to python dictionary.
    """
    response = fetch_data_from_endpoint(ENDPOINT_GET_PLAYER_MMR_DISTRIBUTION, verbose=verbose)
    return response


def get_prizepool_data():
    """Returns hardcoded dict of data which was manually scrapped from:
        - http://dota2.prizetrac.kr/,
        - https://en.wikipedia.org/wiki/The_International_(Dota_2)

        Parameters:
        -----------
        None

        Returns:
        -----------
        prize_dict: dict
            Dict with scrapped data.
    """
    prize_dict = {
        "year": [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019],
        "prize_dolar": [1600000, 1600000, 2874380, 10931105, 18429613, 20770460, 24787916, 25532177, 28200021]
    }
    return prize_dict


def get_steamcharts_data():
    """Returns hardcoded string which was manually scrapped from:
        - https://steamcharts.com/app/570#All

        Parameters:
        -----------
        None

        Returns:
        -----------
        data_string: str
           String with scrapped data.
    """
    data_string = """
    Month	Avg. Players	Gain	% Gain	Peak Players
    June 2019	507,528.4	-40,994.7	-7.47%	865,374
    May 2019	548,523.2	+28,304.2	+5.44%	997,341
    April 2019	520,219.0	-66,286.9	-11.30%	971,545
    March 2019	586,505.8	+21,596.2	+3.82%	1,033,925
    February 2019	564,909.7	+89,162.7	+18.74%	964,921
    January 2019	475,747.0	+36,379.2	+8.28%	874,888
    December 2018	439,367.8	-21,705.7	-4.71%	765,422
    November 2018	461,073.5	+29,899.6	+6.93%	826,053
    October 2018	431,173.9	-35,296.8	-7.57%	739,643
    September 2018	466,470.7	-9,630.3	-2.02%	826,166
    August 2018	476,101.1	+34,386.7	+7.78%	829,281
    July 2018	441,714.3	-32,185.7	-6.79%	701,582
    June 2018	473,900.0	-425.9	-0.09%	796,886
    May 2018	474,325.9	+43,984.9	+10.22%	844,713
    April 2018	430,340.9	-6,921.4	-1.58%	733,214
    March 2018	437,262.3	-1,585.4	-0.36%	773,897
    February 2018	438,847.7	-48,014.2	-9.86%	779,299
    January 2018	486,861.9	-26,212.4	-5.11%	778,627
    December 2017	513,074.3	+25,693.1	+5.27%	864,939
    November 2017	487,381.2	+21,254.5	+4.56%	861,173
    October 2017	466,126.8	-25,323.4	-5.15%	832,550
    September 2017	491,450.2	-65,046.1	-11.69%	829,555
    August 2017	556,496.3	+58,051.9	+11.65%	876,395
    July 2017	498,444.4	-56,844.7	-10.24%	824,297
    June 2017	555,289.1	-12,237.4	-2.16%	923,122
    May 2017	567,526.4	+27,248.4	+5.04%	972,876
    April 2017	540,278.0	-8,157.4	-1.49%	921,318
    March 2017	548,435.4	-43,131.8	-7.29%	956,232
    February 2017	591,567.3	+11,285.8	+1.94%	1,040,877
    January 2017	580,281.5	-13,639.1	-2.30%	1,007,451
    December 2016	593,920.6	+9,669.3	+1.65%	1,014,671
    November 2016	584,251.3	-55,103.8	-8.62%	1,007,270
    October 2016	639,355.1	+16,771.2	+2.69%	1,141,191
    September 2016	622,583.9	-43,429.2	-6.52%	1,064,377
    August 2016	666,013.1	+27,800.4	+4.36%	1,117,519
    July 2016	638,212.7	-2,014.3	-0.31%	1,084,198
    June 2016	640,227.0	+16,428.3	+2.63%	1,095,994
    May 2016	623,798.7	-33,145.7	-5.05%	1,075,307
    April 2016	656,944.4	-15,610.5	-2.32%	1,164,041
    March 2016	672,554.9	-36,623.4	-5.16%	1,291,328
    February 2016	709,178.3	+97,003.5	+15.85%	1,248,394
    January 2016	612,174.8	+38,830.5	+6.77%	1,067,949
    December 2015	573,344.3	+33,807.9	+6.27%	999,452
    November 2015	539,536.3	+17,594.6	+3.37%	943,635
    October 2015	521,941.7	+13,784.9	+2.71%	917,306
    September 2015	508,156.8	-98,787.1	-16.28%	888,728
    August 2015	606,944.0	+51,953.0	+9.36%	933,942
    July 2015	554,991.0	-13,457.3	-2.37%	877,264
    June 2015	568,448.3	-11,900.1	-2.05%	913,997
    May 2015	580,348.4	+54,286.7	+10.32%	967,674
    April 2015	526,061.7	-45,651.4	-7.99%	929,677
    March 2015	571,713.2	-57,257.3	-9.10%	1,213,940
    February 2015	628,970.4	+70,466.1	+12.62%	1,262,612
    January 2015	558,504.3	+34,564.0	+6.60%	961,737
    December 2014	523,940.3	-4,849.5	-0.92%	936,583
    November 2014	528,789.8	+33,096.8	+6.68%	963,810
    October 2014	495,693.0	+17,694.6	+3.70%	880,655
    September 2014	477,998.5	-12,885.4	-2.62%	864,261
    August 2014	490,883.9	-46,134.8	-8.59%	774,319
    July 2014	537,018.7	+23,235.6	+4.52%	874,975
    June 2014	513,783.1	+31,395.8	+6.51%	833,145
    May 2014	482,387.2	+60,677.0	+14.39%	843,024
    April 2014	421,710.2	+11,954.7	+2.92%	734,998
    March 2014	409,755.6	-11,358.6	-2.70%	698,197
    February 2014	421,114.2	+27,253.9	+6.92%	738,682
    January 2014	393,860.3	+27,253.8	+7.43%	673,496
    December 2013	366,606.5	+18,360.1	+5.27%	685,503
    November 2013	348,246.4	+18,568.7	+5.63%	702,792
    October 2013	329,677.6	+17,252.9	+5.52%	581,615
    September 2013	312,424.8	-18,295.3	-5.53%	566,715
    August 2013	330,720.1	+92,920.0	+39.07%	520,532
    July 2013	237,800.1	+27,575.3	+13.12%	422,617
    June 2013	210,224.8	+15,861.0	+8.16%	326,160
    May 2013	194,363.8	+19,528.1	+11.17%	325,815
    April 2013	174,835.7	-6,043.2	-3.34%	299,667
    March 2013	180,878.9	+13,905.9	+8.33%	325,598
    February 2013	166,973.0	+19,224.8	+13.01%	283,870
    January 2013	147,748.1	+25,823.7	+21.18%	260,989
    December 2012	121,924.4	+20,847.0	+20.62%	213,521
    November 2012	101,077.4	+25,112.0	+33.06%	169,631
    October 2012	75,965.4	+14,097.8	+22.79%	171,860
    September 2012	61,867.7	+6,099.1	+10.94%	118,724
    August 2012	55,768.6	+3,047.6	+5.78%	108,689
    July 2012	52,721.1	-	-	75,041
    """
    return data_string
