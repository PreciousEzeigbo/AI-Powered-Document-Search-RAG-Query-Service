def test_health_check_returns_service_status(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "service": "RAG Document Search",
    }
