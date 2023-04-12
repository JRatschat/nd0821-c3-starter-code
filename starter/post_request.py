import requests

input_data = {
        "age": 32,
        "workclass": "Private",
        "fnlgt": 280464,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "Black",
        "sex": "Male",
        "capital-gain": 14084,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }

response = requests.post(
    "https://nanodegree-ml-deploy.onrender.com/api",
    json=input_data
)

print(f'Response status code: {response.status_code}')
print(f'Response body: {response.json()}')
