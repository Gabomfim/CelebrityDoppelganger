from fastapi.testclient import TestClient

from api import LINKEDIN_URL, app


class StubMatcher:
    def match(self, image_bytes: bytes):
        assert image_bytes == b"in-memory-image"
        return [{"name": "Example Person", "similarity": 92.0}]


def test_match_accepts_raw_image_and_declares_no_storage():
    with TestClient(app) as client:
        app.state.matcher = StubMatcher()
        response = client.post(
            "/api/match", content=b"in-memory-image", headers={"content-type": "image/jpeg"}
        )
    assert response.status_code == 200
    assert response.json()["image_stored"] is False
    assert response.json()["neighbor_count"] == 3


def test_match_rejects_non_image_content():
    with TestClient(app) as client:
        app.state.matcher = StubMatcher()
        response = client.post(
            "/api/match", content=b"not-an-image", headers={"content-type": "text/plain"}
        )
    assert response.status_code == 415


def test_social_links_are_configured(monkeypatch):
    monkeypatch.delenv("LINKEDIN_URL", raising=False)
    with TestClient(app) as client:
        response = client.get("/api/config")
    assert response.json()["linkedin_url"] == LINKEDIN_URL
    assert "github.com/Gabomfim/CelebrityDoppelganger" in response.json()["github_url"]


def test_camera_controls_have_safe_initial_state():
    with TestClient(app) as client:
        html = client.get("/").text
        css = client.get("/styles.css").text
    assert 'id="camera-button"' in html
    assert 'id="snap-button"' in html and "hidden disabled" in html
    assert 'id="match-button"' in html and "hidden" in html
    assert "[hidden] { display:none !important; }" in css
