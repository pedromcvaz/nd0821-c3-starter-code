
def test_get_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome! This api provides an interface for scoring the census data from Udacity's ML DevOps program"}


def test_response_neg(client, payload):
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()['result'] == 0


def test_response_pos(client, payload_2):
    response = client.post("/predict", json=payload_2)
    assert response.status_code == 200
    assert response.json()['result'] == 1


def test_response_error(client, payload_3):
    response = client.post("/predict", json=payload_3)
    assert response.status_code == 422
