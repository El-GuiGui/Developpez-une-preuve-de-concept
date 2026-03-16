import os
import time
import subprocess
import signal

import httpx

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))


def test_api_starts_and_health():
    env = dict(os.environ)
    env["SKIP_MODEL_LOAD"] = "1"

    p = subprocess.Popen(
        [
            "python",
            "-m",
            "uvicorn",
            "api.main:app",
            "--host",
            API_HOST,
            "--port",
            str(API_PORT),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        # wait for startup
        ok = False
        for _ in range(30):
            try:
                r = httpx.get(f"http://{API_HOST}:{API_PORT}/health", timeout=1.0)
                if r.status_code == 200:
                    ok = True
                    break
            except Exception:
                pass
            time.sleep(0.5)

        assert ok, "API did not start or /health not reachable"

    finally:
        p.send_signal(signal.SIGINT)
        try:
            p.wait(timeout=5)
        except Exception:
            p.kill()
