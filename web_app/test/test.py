import pytest
from your_application import create_app


@pytest.fixture(scope="session")
def app(app):
    _app = app
    _app.config['TESTING'] = True
    return _app


@pytest.fixture(scope="session")
def client(app):
    with app.test_client() as client:
        yield client


@pytest.fixture(autouse=True)
def clear_chat_history_before_and_after_tests(client):
    client.delete("/clear")
    yield
    client.delete("/clear")


@pytest.mark.parametrize("endpoint,expected_length", [
    ("/top_cos_sim_bi_cr", 6),
    ("/top_l2_bi_cr", 6),
    ("/top_l2_psa_bi_cr", 6),
    ("/top_cr", 6),
])
def test_repeated_calls_return_expected_length(client, endpoint, expected_length):
    for _ in range(3):
        response = client.post(endpoint, json={"query": "test", "user": "test_user"})
    assert len(response.json['response']) == expected_length


@pytest.mark.parametrize("top_n,endpoint", [
    (1, "/top_cos_sim_bi_cr"), (5, "/top_cos_sim_bi_cr"), (10, "/top_cos_sim_bi_cr"),
    (1, "/top_l2_bi_cr"), (5, "/top_l2_bi_cr"), (10, "/top_l2_bi_cr"),
    (1, "/top_l2_psa_bi_cr"), (5, "/top_l2_psa_bi_cr"), (10, "/top_l2_psa_bi_cr"),
    (1, "/top_cr"), (5, "/top_cr"), (10, "/top_cr"),
])
def test_top_n_responses(client, top_n, endpoint):
    response = client.post(endpoint, json={"query": "test", "user": "test_user", "top_n": top_n})
    assert response.status_code == 200
