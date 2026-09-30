"""Deployed, the backend also serves the built web app (web/dist) on the same address."""

from __future__ import annotations

from pathlib import Path

import pytest
from app import main
from fastapi.testclient import TestClient


@pytest.fixture
def dist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "dist"
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text("<!doctype html><div id=root></div>", encoding="utf-8")
    (root / "assets" / "index-abc123.js").write_text("console.log(1)", encoding="utf-8")
    (root / "favicon.svg").write_text("<svg/>", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("not for the web", encoding="utf-8")
    monkeypatch.setenv("WEB_DIST", str(root))
    return root


def test_pages_and_assets_are_served(dist) -> None:
    with TestClient(main.app) as client:
        page = client.get("/")
        assert page.status_code == 200 and "id=root" in page.text
        assert page.headers["cache-control"] == "no-cache"
        js = client.get("/assets/index-abc123.js")
        assert js.status_code == 200 and js.text == "console.log(1)"
        assert "immutable" in js.headers["cache-control"]
        assert client.get("/favicon.svg").text == "<svg/>"
        assert "id=root" in client.get("/some/app/route").text  # the app routes by URL hash
        assert client.head("/").status_code == 200


def test_the_api_and_sockets_are_never_shadowed(dist) -> None:
    with TestClient(main.app) as client:
        assert client.get("/api/health").json()["status"] == "ok"
        missing = client.get("/api/does-not-exist")
        assert missing.status_code == 404 and missing.json() == {"detail": "Not Found"}
        assert client.get("/ws").status_code == 404
        assert client.get("/api").status_code == 404


def test_files_outside_the_build_are_unreachable(dist) -> None:
    with TestClient(main.app) as client:
        for path in ("/../secret.txt", "/%2e%2e/secret.txt", "/assets/../../secret.txt"):
            r = client.get(path)
            assert "not for the web" not in r.text


def test_without_a_build_nothing_is_served(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WEB_DIST", str(tmp_path / "missing"))
    with TestClient(main.app) as client:
        assert client.get("/").status_code == 404
        assert client.get("/api/health").status_code == 200
