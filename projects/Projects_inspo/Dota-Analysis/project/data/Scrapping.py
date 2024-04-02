import time
import requests

from project.api.Client import get

RETRY_WAIT = 60


def fetch_data_from_endpoint(url, retry_limit=3, verbose=True):
    """Downloads data from specified url. If there will be connection issue with server, function will retry connection
     every 60 seconds for retry_limit times.

        Parameters:
        -----------
        url: string
            String containing endpoint to server.
        retry_limit: int
            Number of retries in case server response suggests error.
        verbose: bool
            Allows log messages to be displayed if set to True. Default - True.

        Returns:
        -----------
        response: dict
            Python dictionary with parsed JSON message.
    """
    retry_count = 0
    while True:
        try:
            response_json = get(url, verbose=verbose)
            if verbose:
                print("Success!")

            return response_json
        except requests.exceptions.HTTPError as e:
            retry_count = retry_count + 1

            if retry_count >= retry_limit:
                raise e

            if verbose:
                print("Waiting {} seconds before retry.".format(RETRY_WAIT))

            time.sleep(60)
