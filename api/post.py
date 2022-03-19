import json
import requests
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)-10s %(message)s")
logger = logging.getLogger()

url = 'https://udacityml.herokuapp.com/predict'
payload = {
    "age": 26,
    "workclass": "Private",
    "fnlgt": 172987,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Married-civ-spouse",
    "occupation": "Tech-support",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 50,
    "native-country": "United-States",
    "salary": "<=50K"
}
headers = {'result-type': 'application/json'}

if __name__ == '__main__':
    response = requests.post(url, data=json.dumps(payload), headers=headers)

    if response.status_code == 200:
        result = response.json()['result']
        logging.info(f"Predicted value for the sample: {result}")
    else:
        logging.error(
            f'Error! Response code is {response.status_code}')