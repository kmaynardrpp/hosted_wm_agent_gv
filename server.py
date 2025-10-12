# server.py — FastAPI bridge for InfoZone (auto-detect SPA dist)
import os
import re
import sys
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles

# ---------- Project root ----------
PROJECT_ROOT = Path(os.environ.get("INFOZONE_ROOT") or Path(__file__).resolve().parent).resolve()
os.environ["INFOZONE_ROOT"] = str(PROJECT_ROOT)
MAIN_PY = PROJECT_ROOT / "main.py"
UPLOAD_DIR = PROJECT_ROOT / "uploads"
RUNS_DIR = PROJECT_ROOT / ".runs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="InfoZone Web Bridge", version="0.4.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve files under project root for /files/...
app.mount("/files", StaticFiles(directory=str(PROJECT_ROOT)), name="files")

# ---------- Auto-detect SPA dist ----------
def _find_frontend_dir() -> Optional[Path]:
    # 1) explicit env override
    env_dir = os.environ.get("FRONTEND_DIR", "").strip()
    if env_dir:
        p = Path(env_dir).resolve()
        if (p / "index.html").exists():
            return p

    # 2) common locations
    candidates = [
        PROJECT_ROOT / "web" / "infozone-web" / "dist",
        PROJECT_ROOT / "web" / "infozone-web-gv" / "dist",
    ]
    # 3) glob any web/*/dist with index.html
    web_root = PROJECT_ROOT / "web"
    if web_root.exists():
        for sub in web_root.iterdir():
            p = sub / "dist"
            if (p / "index.html").exists():
                candidates.append(p)

    for c in candidates:
        if c and (c / "index.html").exists():
            return c
    return None

FRONTEND_DIR = _find_frontend_dir()
if FRONTEND_DIR:
    # mount static assets (vite puts hashed files under /assets)
    assets = FRONTEND_DIR / "assets"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

    @app.get("/", response_model=None)
    def _root() -> Response:
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/{full_path:path}", response_model=None)
    def _spa(full_path: str) -> Response:
        # Always serve index.html for SPA routes
        index = FRONTEND_DIR / "index.html"
        return FileResponse(index if index.exists() else FRONTEND_DIR / "404.html")

# ---------- Link parsing ----------
FILELINK_RE = re.compile(r"\[[^\]]+\]\(file:///([^)\r\n]+)\)")

def _artifact_type(path: str) -> str:
    ext = (path.rsplit(".", 1)[-1] if "." in path else "").lower()
    if ext == "pdf": return "pdf"
    if ext in {"png","jpg","jpeg","gif","webp"}: return "image"
    return "file"

def _to_http_url(local_path: str) -> Optional[str]:
    try:
        p = Path(local_path.replace("\\", "/")).resolve()
        root = PROJECT_ROOT.resolve()
        if str(p).lower().startswith(str(root).lower()):
            rel = p.relative_to(root).as_posix()
            return f"/files/{rel}"
        return None
    except Exception:
        return None

# ---------- API ----------
@app.get("/api/ping", response_model=None)
def ping() -> Response:
    return JSONResponse({"ok": True, "root": str(PROJECT_ROOT), "has_main": MAIN_PY.exists(), "spa": str(FRONTEND_DIR) if FRONTEND_DIR else None})

@app.post("/api/run", response_model=None)
def run(prompt: str = Form(...), files: List[UploadFile] = File(default=[])) -> Response:
    if not MAIN_PY.exists():
        return PlainTextResponse("main.py not found under INFOZONE_ROOT", status_code=500)

    job_dir = UPLOAD_DIR / f"job_{os.getpid()}_{os.urandom(3).hex()}"
    job_dir.mkdir(parents=True, exist_ok=True)

    csv_paths: List[Path] = []
    for f in files or []:
        dest = job_dir / f.filename
        with dest.open("wb") as w:
            shutil.copyfileobj(f.file, w)
        csv_paths.append(dest.resolve())

    cmd = [sys.executable, str(MAIN_PY), prompt] + [str(p) for p in csv_paths]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    job_out = RUNS_DIR / job_dir.name
    job_out.mkdir(parents=True, exist_ok=True)
    env["INFOZONE_OUT_DIR"] = str(job_out)

    proc = subprocess.run(
        cmd, cwd=str(PROJECT_ROOT), env=env,
        capture_output=True, text=True, encoding="utf-8", errors="ignore"
    )

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    links = FILELINK_RE.findall(stdout)
    for line in stdout.splitlines():
        if "file:///" in line and "](" not in line:
            try:
                part = line.split("file:///", 1)[1]
                part = part.split(")")[0].strip()
                links.append(part)
            except Exception:
                pass

    artifacts = []
    seen = set()
    for link in links:
        if link in seen: continue
        seen.add(link)
        http = _to_http_url(link)
        if http:
            artifacts.append({"url": http, "type": _artifact_type(link), "filename": Path(link).name})

    ok = (proc.returncode == 0)
    summary = "Report created." if ok and artifacts else "Runner finished (check logs)."
    status = 200 if ok else 500

    return JSONResponse({"ok": ok, "summary": summary, "artifacts": artifacts, "logs": (stdout + ("\n\n[stderr]\n" + stderr if stderr else "")).strip()}, status_code=status)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
