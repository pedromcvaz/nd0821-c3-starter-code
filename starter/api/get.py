import requests
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)-10s %(message)s")
logger = logging.getLogger()

URL = 'http://myapi.herokuapp.com/'

headers = {'content-type': 'application/json'}

if __name__ == '__main__':
    response = requests.get(URL)

    if response.status_code == 200:
        result = response.json()
        logging.info(f"Predicted value: {result}")
    else:
        logging.error(
            f'Error! Response code is: {response.status_code}')