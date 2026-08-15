from fastapi.testclient import TestClient

from serving.backend.main import app


def test_hybrid_classification_endpoint_uses_hybrid_model(monkeypatch):
    def fake_classify_two_stage(image_file, backend="grounded_sam", bg_gray=128):
        return {
            "found": True,
            "prediction": "hueso largo",
            "probabilities": {"hueso largo": 0.91, "cráneo": 0.09},
            "cv_features": {"found": True},
            "fine_bone": None,
            "taxon": None,
        }

    monkeypatch.setattr("serving.backend.main.classify_image_two_stage", fake_classify_two_stage)

    client = TestClient(app)
    response = client.post(
        "/hybrid/classify",
        files={"file": ("bone.jpg", b"fake-image-bytes", "image/jpeg")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"] == "hueso largo"
    assert payload["probabilities"]["hueso largo"] == 0.91
