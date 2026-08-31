from __future__ import annotations

import http.client
import threading

import chimera_agent.server as submission_server


def _serve_once() -> tuple[submission_server.ThreadingHTTPServer, threading.Thread]:
    httpd = submission_server.ThreadingHTTPServer(("127.0.0.1", 0), submission_server.SubmissionHandler)
    thread = threading.Thread(target=httpd.handle_request, daemon=True)
    thread.start()
    return httpd, thread


def test_health_endpoint_returns_ready(monkeypatch):
    monkeypatch.setattr(submission_server, "is_ready", lambda: True)
    httpd, thread = _serve_once()
    host, port = httpd.server_address
    conn = http.client.HTTPConnection(host, port)
    conn.request("GET", "/health")
    response = conn.getresponse()
    assert response.status == 200
    assert response.read() == b'{"status": "ok"}'
    thread.join(timeout=2)
    httpd.server_close()


def test_health_endpoint_returns_loading_until_model_is_ready(monkeypatch):
    monkeypatch.setattr(submission_server, "is_ready", lambda: False)
    httpd, thread = _serve_once()
    host, port = httpd.server_address
    conn = http.client.HTTPConnection(host, port)
    conn.request("GET", "/health")
    response = conn.getresponse()
    assert response.status == 503
    assert response.read() == b'{"status": "loading"}'
    thread.join(timeout=2)
    httpd.server_close()


def test_invoke_runs_pipeline_and_returns_created(monkeypatch):
    calls = []
    monkeypatch.setattr(submission_server, "run_inference", lambda config: calls.append(config))
    monkeypatch.setattr(submission_server, "is_ready", lambda: True)
    httpd, thread = _serve_once()
    host, port = httpd.server_address
    conn = http.client.HTTPConnection(host, port)
    conn.request("POST", "/invoke", body=b"{}", headers={"Content-Type": "application/json"})
    response = conn.getresponse()
    assert response.status == 201
    assert response.read() == b'{"status": "success"}'
    assert calls == ["/opt/app/configs/config.yaml"]
    thread.join(timeout=2)
    httpd.server_close()


def test_invoke_returns_service_unavailable_until_model_is_ready(monkeypatch):
    monkeypatch.setattr(submission_server, "is_ready", lambda: False)
    httpd, thread = _serve_once()
    host, port = httpd.server_address
    conn = http.client.HTTPConnection(host, port)
    conn.request("POST", "/invoke", body=b"{}", headers={"Content-Type": "application/json"})
    response = conn.getresponse()
    assert response.status == 503
    assert response.read() == b'{"error": "model is still loading"}'
    thread.join(timeout=2)
    httpd.server_close()
