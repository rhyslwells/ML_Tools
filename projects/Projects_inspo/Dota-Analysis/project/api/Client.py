import requests


def get(url, verbose=False):
    """Access data at specified url, fetches and returns as a Python dictionary. Allows verbose mode to display log
    messages. Raises exception on failure.

       Parameters:
       -----------
       url: string
            String containing endpoint to server.
       verbose: bool
            Default value is False. When switched to True, function will display logs.

       Returns:
       -----------
       response: dict
            Python dictionary with parsed JSON message.
    """
    try:
        if verbose:
            print("Fetching data from: {}".format(url))
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if verbose:
            print(e)
        raise e