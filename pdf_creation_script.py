# ------------------------------------------------------------------------------
# pdf_creation_script.py — Robust PDF builder for InfoZone
#
# Public API (unchanged):
#   - safe_build_pdf(report: dict, output_path: str, logo_path: str|None = None, dpi: int = 120) -> None
#   - build_pdf(report: dict, output_path: str, logo_path: str|None = None, dpi: int = 120) -> None
#
# Notes:
#   * Root resolution prefers BATCH_WALMART_ROOT > INFOZONE_ROOT > file parent > CWD.
#   * If logo_path is None or missing, tries ROOT/redpoint_logo.png automatically.
#   * Expects 'report' dict like:
#       {
#         "title": "Walmart Renovation RTLS Summary",
#         "meta": "string for header/footer",
#         "sections": [
#           {"type":"summary", "title":"Summary", "bullets":[...]},
#           {"type":"table", "title":"Evidence", "data":[{...}], "headers":[...], "rows_per_page": 24},
#           {"type":"charts","title":"Figures","figures":[<matplotlib.figure.Figure>, ...]}
#         ]
#       }
#   * Figures must be live Matplotlib Figure objects (not file paths/Axes).
#   * Atomic write: writes to a temporary file and then replaces output_path.
# ------------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import io
import os
import shutil
import tempfile

import numpy as np
from PIL import Image

# Root resolution (batch/CLI friendly)
def _resolve_root() -> Path:
    for env_name in ("BATCH_WALMART_ROOT", "INFOZONE_ROOT"):
        v = os.environ.get(env_name, "").strip()
        if v:
            p = Path(v).resolve()
            if p.exists():
                return p
    p = Path(__file__).resolve().parent
    if p.exists():
        return p
    return Path.cwd().resolve()

ROOT = _resolve_root()

# ReportLab imports
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.lib import colors

# Matplotlib (Agg, no GUI)
import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402


# ----------------------------- helpers: layout --------------------------------
PT = 72.0  # points per inch
PAGE_W, PAGE_H = letter  # (612 x 792 pt) = (8.5 x 11 in)

MARGIN_L = 36  # 0.5 in
MARGIN_R = 36
MARGIN_T = 48  # header
MARGIN_B = 50  # footer

CONTENT_X = MARGIN_L
CONTENT_Y = MARGIN_B
CONTENT_W = PAGE_W - MARGIN_L - MARGIN_R
CONTENT_H = PAGE_H - MARGIN_T - MARGIN_B

HEADER_LOGO_H = 28  # pt
HEADER_GAP = 6


def _fit_into(w: float, h: float, box_w: float, box_h: float) -> Tuple[float, float]:
    """Scale (w,h) to fit inside (box_w,box_h) keeping aspect, return new (w,h)."""
    if w <= 0 or h <= 0:
        return 0, 0
    s = min(box_w / w, box_h / h)
    return w * s, h * s


def _draw_header_footer(c: canvas.Canvas, title: str, meta: str, logo: Optional[Image.Image], page_no: int, total_estimate: Optional[int]) -> None:
    # Header title
    c.setFont("Helvetica-Bold", 12)
    c.drawString(MARGIN_L, PAGE_H - MARGIN_T + HEADER_GAP, title[:140])

    # Logo on the right (if any)
    if logo is not None:
        iw, ih = logo.size
        tw, th = _fit_into(iw, ih, 140, HEADER_LOGO_H)
        if tw > 0 and th > 0:
            bio = io.BytesIO()
            logo.save(bio, format="PNG")
            bio.seek(0)
            c.drawImage(ImageReader(bio), PAGE_W - MARGIN_R - tw, PAGE_H - MARGIN_T + HEADER_GAP - (th - HEADER_LOGO_H) / 2, tw, th, mask="auto")

    # Thin line
    c.setStrokeColor(colors.lightgrey)
    c.setLineWidth(0.5)
    c.line(MARGIN_L, PAGE_H - MARGIN_T - 4, PAGE_W - MARGIN_R, PAGE_H - MARGIN_T - 4)

    # Footer meta + page number
    c.setFont("Helvetica", 8)
    foot = meta or ""
    c.setFillColor(colors.grey)
    c.drawString(MARGIN_L, MARGIN_B - 20, foot[:180])
    c.setFillColor(colors.black)
    if total_estimate:
        c.drawRightString(PAGE_W - MARGIN_R, MARGIN_B - 20, f"Page {page_no}/{total_estimate}")
    else:
        c.drawRightString(PAGE_W - MARGIN_R, MARGIN_B - 20, f"Page {page_no}")


# ----------------------------- helpers: figures -------------------------------
def _figure_to_rgb_image(fig, dpi: int = 120) -> Image.Image:
    """
    Rasterize a Matplotlib Figure to a PIL RGB image using Agg backend.
    Works on MPL >= 3.9 (uses buffer_rgba()).
    """
    canvas_agg = FigureCanvasAgg(fig)
    fig.set_dpi(dpi)
    canvas_agg.draw()
    w, h = [int(v) for v in fig.get_size_inches() * dpi]
    buf = np.asarray(canvas_agg.buffer_rgba()).reshape((h, w, 4))
    # Convert RGBA -> RGB (white background)
    rgb = np.empty((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = (buf[:, :, 0] * (buf[:, :, 3] / 255.0) + 255 * (1 - buf[:, :, 3] / 255.0)).astype(np.uint8)
    rgb[:, :, 1] = (buf[:, :, 1] * (buf[:, :, 3] / 255.0) + 255 * (1 - buf[:, :, 3] / 255.0)).astype(np.uint8)
    rgb[:, :, 2] = (buf[:, :, 2] * (buf[:, :, 3] / 255.0) + 255 * (1 - buf[:, :, 3] / 255.0)).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


# ------------------------------- draw sections --------------------------------
def _draw_summary(c: canvas.Canvas, section: Dict[str, Any], y: float) -> float:
    title = str(section.get("title", "Summary") or "Summary")
    items: Sequence[str] = section.get("bullets", []) or []
    c.setFont("Helvetica-Bold", 11)
    c.drawString(CONTENT_X, y, title[:120])
    y -= 14
    c.setFont("Helvetica", 10)
    for s in items:
        text = str(s or "")
        if not text:
            continue
        lines = text.splitlines() or [text]
        for ln in lines:
            if y < CONTENT_Y + 40:
                c.showPage()
                y = PAGE_H - MARGIN_T - 10
            c.drawString(CONTENT_X + 12, y, f"• {ln[:160]}")
            y -= 13
    return y


def _draw_table(c: canvas.Canvas, section: Dict[str, Any], y: float) -> float:
    title = str(section.get("title", "Table") or "Table")
    headers: Sequence[str] = [str(h) for h in (section.get("headers") or [])]
    rows: Sequence[Dict[str, Any]] = section.get("data") or []
    rpp = int(section.get("rows_per_page", 24) or 24)

    if not headers or not rows:
        return y

    c.setFont("Helvetica-Bold", 11)
    c.drawString(CONTENT_X, y, title[:120])
    y -= 14

    # Calculate column widths
    col_w = CONTENT_W / max(1, len(headers))
    row_h = 12

    def draw_header(_y: float) -> float:
        c.setFont("Helvetica-Bold", 9)
        c.setFillColor(colors.white)
        c.setStrokeColor(colors.lightgrey)
        c.rect(CONTENT_X, _y - row_h, CONTENT_W, row_h, stroke=1, fill=1)
        c.setFillColor(colors.black)
        for i, h in enumerate(headers):
            c.drawString(CONTENT_X + i * col_w + 2, _y - row_h + 3, str(h)[:32])
        return _y - row_h - 2

    def draw_row(data_row: Dict[str, Any], _y: float) -> float:
        c.setFont("Helvetica", 8)
        c.setStrokeColor(colors.lightgrey)
        c.rect(CONTENT_X, _y - row_h, CONTENT_W, row_h, stroke=1, fill=0)
        for i, h in enumerate(headers):
            val = str(data_row.get(h, ""))
            c.drawString(CONTENT_X + i * col_w + 2, _y - row_h + 3, val[:40])
        return _y - row_h - 2

    count = 0
    header_y = y
    y = draw_header(y)
    for r in rows:
        if count and (count % rpp == 0 or y < CONTENT_Y + 40):
            c.showPage()
            _draw_header_footer(c, "", "", None, 0, None)  # just re-draw line boundaries
            y = PAGE_H - MARGIN_T - 10
            y = draw_header(y)
        y = draw_row(r, y)
        count += 1
    return y


def _draw_charts(c: canvas.Canvas, section: Dict[str, Any], y: float, dpi: int) -> float:
    title = str(section.get("title", "Figures") or "Figures")
    figs: List[Any] = [f for f in (section.get("figures") or []) if hasattr(f, "savefig")]
    if not figs:
        return y

    c.setFont("Helvetica-Bold", 11)
    c.drawString(CONTENT_X, y, title[:120])
    y -= 12

    box_w, box_h = CONTENT_W, (CONTENT_H - 24)  # generous space per page

    for i, fig in enumerate(figs, 1):
        img = _figure_to_rgb_image(fig, dpi=dpi)
        iw, ih = img.size
        tw, th = _fit_into(iw, ih, box_w, box_h)

        if y - th < CONTENT_Y + 24:
            c.showPage()
            y = PAGE_H - MARGIN_T - 10

        bio = io.BytesIO()
        img.save(bio, format="PNG")
        bio.seek(0)
        c.drawImage(ImageReader(bio), CONTENT_X + (box_w - tw) / 2, y - th, tw, th, mask="auto")
        y -= (th + 12)

        # After drawing, keep the figure alive (caller may close later)
    return y


# ------------------------------- build / safe ---------------------------------
def build_pdf(report: Dict[str, Any], output_path: str, logo_path: Optional[str] = None, dpi: int = 120) -> None:
    """
    Render the given report dict as a PDF to output_path. Atomic write.
    """
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    # Resolve logo
    logo_img: Optional[Image.Image] = None
    logo_candidates: List[Path] = []
    if logo_path:
        logo_candidates.append(Path(logo_path))
    logo_candidates += [ROOT / "redpoint_logo.png", Path.cwd() / "redpoint_logo.png"]
    for cand in logo_candidates:
        try:
            if cand.exists():
                logo_img = Image.open(cand).convert("RGBA")
                break
        except Exception:
            continue

    title = str(report.get("title", "InfoZone Report") or "InfoZone Report")
    meta  = str(report.get("meta", "") or "")
    sections: List[Dict[str, Any]] = report.get("sections") or []

    # Atomic write: temp then replace
    tmp_path = out.with_suffix(out.suffix + ".tmp")
    c = canvas.Canvas(str(tmp_path), pagesize=letter)

    page_no = 1
    total_estimate = None  # could be len(sections)+N; keep None to avoid lying

    # First page header
    _draw_header_footer(c, title, meta, logo_img, page_no, total_estimate)
    y = PAGE_H - MARGIN_T - 10

    for sec in sections:
        st = (sec.get("type") or "").lower()
        if st == "summary":
            y = _draw_summary(c, sec, y)
        elif st == "table":
            y = _draw_table(c, sec, y)
        elif st == "charts":
            y = _draw_charts(c, sec, y, dpi=dpi)
        else:
            # generic text section
            t = str(sec.get("title", "Notes") or "Notes")
            body = str(sec.get("text", "") or "")
            c.setFont("Helvetica-Bold", 11); c.drawString(CONTENT_X, y, t[:120]); y -= 14
            c.setFont("Helvetica", 10)
            for ln in (body.splitlines() or []):
                if y < CONTENT_Y + 40:
                    c.showPage(); page_no += 1
                    _draw_header_footer(c, title, meta, logo_img, page_no, total_estimate)
                    y = PAGE_H - MARGIN_T - 10
                c.drawString(CONTENT_X, y, ln[:140]); y -= 12

        # New page if space is tight between sections
        if y < CONTENT_Y + 80:
            c.showPage(); page_no += 1
            _draw_header_footer(c, title, meta, logo_img, page_no, total_estimate)
            y = PAGE_H - MARGIN_T - 10

    # Ensure at least one page is saved
    c.showPage()
    c.save()

    # Atomic replace
    try:
        shutil.move(str(tmp_path), str(out))
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def safe_build_pdf(report: Dict[str, Any], output_path: str, logo_path: Optional[str] = None, dpi: int = 120) -> None:
    """
    Thin wrapper kept for API compatibility. The generator wraps this in its own
    try/except and will attempt a lite report if this raises.
    """
    build_pdf(report, output_path=output_path, logo_path=logo_path, dpi=dpi)
