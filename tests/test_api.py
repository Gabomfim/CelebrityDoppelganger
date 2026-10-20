import base64
from pathlib import Path

from fastapi.testclient import TestClient

from api import LINKEDIN_URL, app

PROJECT_ROOT = Path(__file__).parents[1]


class StubMatcher:
    def match(self, image_bytes: bytes):
        assert image_bytes == b"in-memory-image"
        return [{"name": "Example Person", "similarity": 92.0}]

    def match_prototype(self, images: list[bytes]):
        assert images == [b"selfie-one", b"selfie-two", b"selfie-three"]
        return [{"name": "Prototype Person", "similarity": 94.0}]


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


def test_match_prototype_averages_three_in_memory_selfies():
    images = [
        base64.b64encode(value).decode()
        for value in (b"selfie-one", b"selfie-two", b"selfie-three")
    ]
    with TestClient(app) as client:
        app.state.matcher = StubMatcher()
        response = client.post("/api/match-prototype", json={"images": images})
    assert response.status_code == 200
    assert response.json()["prototype_image_count"] == 3
    assert response.json()["neighbor_count"] == 3
    assert response.json()["image_stored"] is False


def test_match_prototype_requires_exactly_three_selfies():
    with TestClient(app) as client:
        app.state.matcher = StubMatcher()
        response = client.post("/api/match-prototype", json={"images": ["b25l"]})
    assert response.status_code == 400


def test_social_links_are_configured(monkeypatch):
    monkeypatch.delenv("LINKEDIN_URL", raising=False)
    with TestClient(app) as client:
        response = client.get("/api/config")
    assert response.json()["linkedin_url"] == LINKEDIN_URL
    assert "github.com/Gabomfim/CelebrityDoppelganger" in response.json()["github_url"]


def test_cache_headers_keep_private_api_dynamic_and_cache_public_assets():
    with TestClient(app) as client:
        config = client.get("/api/config")
        stylesheet = client.get("/styles.css")
    assert config.headers["cache-control"] == "no-store"
    assert stylesheet.headers["cache-control"] == "public, max-age=300"


def test_portuguese_locale_is_suggested_without_exposing_location():
    with TestClient(app) as client:
        response = client.get(
            "/api/config",
            headers={"cloudfront-viewer-country": "BR", "accept-language": "en-US"},
        )
    assert response.json()["suggested_locale"] == "pt-BR"
    assert "country" not in response.json()


def test_browser_language_is_a_local_development_fallback():
    with TestClient(app) as client:
        portuguese = client.get("/api/config", headers={"accept-language": "pt-PT"})
        english = client.get("/api/config", headers={"accept-language": "en-US"})
    assert portuguese.json()["suggested_locale"] == "pt-BR"
    assert english.json()["suggested_locale"] == "en"


def test_analytics_accepts_only_anonymous_allowlisted_data(capsys):
    with TestClient(app) as client:
        response = client.post(
            "/api/analytics",
            json={
                "event": "camera_failed",
                "session_id": "12345678-1234-1234-1234-123456789abc",
                "browser": "LinkedIn",
                "device": "mobile",
                "os": "iOS",
                "error_code": "permission_denied",
                "selfie": "must-not-be-logged",
                "user_agent": "must-not-be-logged",
            },
        )
    assert response.status_code == 204
    output = capsys.readouterr().out
    assert '"browser":"LinkedIn"' in output
    assert "permission_denied" in output
    assert "must-not-be-logged" not in output


def test_analytics_rejects_unknown_events():
    with TestClient(app) as client:
        response = client.post(
            "/api/analytics",
            json={"event": "store_selfie", "session_id": "12345678-1234-1234-1234-123456789abc"},
        )
    assert response.status_code == 400


def test_production_server_does_not_log_visitor_ip_addresses():
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text()
    assert '"--no-access-log"' in dockerfile


def test_camera_controls_have_safe_initial_state():
    with TestClient(app) as client:
        html = client.get("/").text
        css = client.get("/styles.css").text
    assert 'id="camera-button"' in html
    assert '<video id="camera" playsinline autoplay muted>' in html
    assert 'id="snap-button"' in html and "hidden disabled" in html
    assert 'id="match-button"' in html and "hidden" in html
    assert 'id="countdown"' in html and 'id="shot-strip"' in html
    assert "Take 3 selfies" in html and "Upload 3 photos" in html
    assert "Remove sunglasses" in html and "exactly one face" in html
    assert "Made by Gabriel Silveira with contrastive supervised prototype learning." in html
    assert '<span class="step" aria-label="Step 1">01</span>' in html
    assert "Public figure" not in client.get("/app.js").text
    assert "A celebrated face with an uncanny resemblance" not in client.get("/app.js").text
    assert "Settings → Websites → Camera" in client.get("/app.js").text
    assert "waitForDecodedVideoFrame" in client.get("/app.js").text
    assert '"black_preview"' in client.get("/app.js").text
    assert 'analyticsPreference === "off"' in client.get("/app.js").text
    assert "if (!analyticsEnabled) return" in client.get("/app.js").text
    assert "[hidden] { display:none !important; }" in css
    assert 'data-locale="en"' in html and 'data-locale="pt-BR"' in html
    assert "/i18n.js" in html


def test_portuguese_translation_covers_the_experience():
    with TestClient(app) as client:
        javascript = client.get("/i18n.js").text
        methodology = client.get("/methodology.html").text
    assert "Seus três sósias famosos mais convincentes" in javascript
    assert "Privacidade faz parte da arquitetura" in javascript
    assert "A REVELAÇÃO DO TAPETE VERMELHO" in javascript
    assert 'data-locale="pt-BR"' in methodology


def test_methodology_page_explains_validated_training_pipeline():
    with TestClient(app) as client:
        response = client.get("/methodology.html")
    assert response.status_code == 200
    html = response.text
    assert "Supervised contrastive objective" in html
    assert "VGGFace2" in html
    assert "5.34 → 9.55" in html
    assert "Selfies and the temporary prototype" in html


def test_results_include_private_social_share_experience():
    with TestClient(app) as client:
        html = client.get("/").text
        javascript = client.get("/app.js").text
    assert 'id="share-reveal-button"' in html
    assert 'id="share-dialog"' in html
    for platform in ("instagram", "facebook", "whatsapp", "x", "threads", "native"):
        assert f'data-share-platform="{platform}"' in html
    assert "createShareCard" in javascript
    assert "navigator.canShare" in javascript
    assert "Your selfies are never included or stored" in html
