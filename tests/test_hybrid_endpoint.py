from fastapi.testclient import TestClient

from app.main import app


def test_hybrid_classification_endpoint_uses_hybrid_model(monkeypatch):
    def fake_hybrid_classify(image_file, model_type="hybrid", backend="grounded_sam"):
        assert model_type == "hybrid"
        return {
            "found": True,
            "prediction": "hueso largo",
            "probabilities": {"hueso largo": 0.91, "cráneo": 0.09},
            "cv_features": {"found": True},
        }

    monkeypatch.setattr("app.main.hybrid_classify_image", fake_hybrid_classify)

    client = TestClient(app)
    response = client.post(
        "/hybrid/classify",
        files={"file": ("bone.jpg", b"fake-image-bytes", "image/jpeg")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"] == "hueso largo"
    assert payload["probabilities"]["hueso largo"] == 0.91
