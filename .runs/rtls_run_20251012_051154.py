#!/usr/bin/env python3
import sys, os
from pathlib import Path

# Resolve project root and enable local imports
ROOT = Path(os.environ.get("INFOZONE_ROOT", ""))
if not ROOT or not (ROOT / "guidelines.txt").exists():
    script_dir = Path(__file__).resolve().parent
    ROOT = script_dir if (script_dir / "guidelines.txt").exists() else script_dir.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Common paths
GUIDELINES = ROOT / "guidelines.txt"
CONTEXT    = ROOT / "context.txt"
FLOORJSON  = ROOT / "floorplans.json"
LOGO       = ROOT / "redpoint_logo.png"
CONFIG     = ROOT / "report_config.json"
LIMITS_PY  = ROOT / "report_limits.py"
ZONES_JSON = ROOT / "zones.json"

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""

# Matplotlib ≥3.9 backend shim
import matplotlib; matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg as _FCA; import numpy as _np
_FCA.tostring_rgb = getattr(_FCA,"tostring_rgb", lambda self: _np.asarray(self.buffer_rgba())[..., :3].tobytes())

import matplotlib as _mpl
_get_cmap = getattr(getattr(_mpl, "colormaps", _mpl), "get_cmap", None)

import traceback
import json
import math
import pandas as pd
import matplotlib.pyplot as plt

# Helper imports (local)
from extractor import extract_tracks
from pdf_creation_script import safe_build_pdf
from report_limits import make_lite, apply_budgets

# ------------- CSV path gathering -------------
def _gather_csv_paths() -> list[Path]:
    paths: list[Path] = []
    # from argv
    for a in sys.argv[1:]:
        try:
            p = Path(a).resolve()
            if p.suffix.lower() == ".csv":
                paths.append(p)
        except Exception:
            continue
    # env fallback
    if not paths:
        env = os.environ.get("INFOZONE_CSVS", "")
        if env:
            for part in env.split(os.pathsep):
                try:
                    p = Path(part).resolve()
                    if p.suffix.lower() == ".csv":
                        paths.append(p)
                except Exception:
                    continue
    return paths

csv_paths = _gather_csv_paths()
if not csv_paths:
    # As a last resort, try the user's provided path in prompt-like environments (no-op if not present)
    default_hint = r"C:\Users\KevinMaynard\Desktop\AI\WM_GPT_AGENT\positions_2025-09-29.csv"
    try:
        p = Path(default_hint)
        if p.exists():
            csv_paths = [p.resolve()]
    except Exception:
        pass

if not csv_paths:
    print("Error Report:")
    print("No CSV paths provided.")
    raise SystemExit(1)

# OUT_DIR policy
OUT_ENV = os.environ.get("INFOZONE_OUT_DIR", "").strip()
out_dir = Path(OUT_ENV).resolve() if OUT_ENV else Path(csv_paths[0]).resolve().parent
out_dir.mkdir(parents=True, exist_ok=True)

# Load config if present
def _load_config() -> dict:
    try:
        if CONFIG.exists():
            return json.loads(CONFIG.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        pass
    # Sensible defaults if missing
    return {
        "prefer_floorplan": True,
        "floorplan_margin": 0.10,
        "overlay_point_size": 8,
        "overlay_alpha": 0.85,
        "overlay_color_by": "trade",
        "overlay_subsample": 20000,
        "draw_trails": False,
        "trail_seconds": 900,
        "draw_zones": True,
        "zone_face_alpha": 0.20,
        "zone_edge_alpha": 0.65,
        "top_n": 10,
        "pie_max_trades": 8,
        "pie_max_single_share": 0.90,
        "line_min_points": 2,
        "small_multiples_cols": 2,
        "max_figures": 6,
        "figsize_overlay": [9, 7],
        "figsize_bar": [7, 5],
        "figsize_line": [7, 5],
        "figsize_pie": [5, 5],
        "figsize_box": [7, 5]
    }

cfg = _load_config()
overlay_max = int(cfg.get("overlay_subsample", 20000))
line_min_points = int(cfg.get("line_min_points", 2))
pie_max_trades = int(cfg.get("pie_max_trades", 8))
pie_max_single_share = float(cfg.get("pie_max_single_share", 0.90))

# ------------- Aggregation structures -------------
total_samples = 0
first_ts = None
last_ts = None
unique_uids: set[str] = set()
unique_trades: set[str] = set()
trade_counts: dict[str, int] = {}
hour_counts: dict[pd.Timestamp, int] = {}
overlay_parts: list[pd.DataFrame] = []

# ------------- Helper: add to hourly counts -------------
def _add_hour_counts(ts_series: pd.Series):
    # Ensure tz-aware UTC (or coerce)
    s = pd.to_datetime(ts_series, utc=True, errors="coerce")
    s = s.dropna()
    if s.empty:
        return
    hrs = s.dt.floor("h")
    vc = hrs.value_counts()
    for ts, cnt in vc.items():
        hour_counts[ts] = hour_counts.get(ts, 0) + int(cnt)

# ------------- Ingest loop (per file) -------------
schema_checked = False
for csv_path in csv_paths:
    try:
        raw = extract_tracks(str(csv_path), mac_map_path=str(ROOT / "trackable_objects.json"))
        audit = raw.get("audit", {}) if isinstance(raw, dict) else {}
        if not audit or not audit.get("mac_map_loaded", False):
            print("Error Report:")
            print("MAC/UID map failed to load or is empty; cannot proceed without mapped trackables.")
            raise SystemExit(1)

        rows = raw.get("rows", [])
        df = pd.DataFrame(rows)
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]

        # Emergency floor crop (global)
        xn = pd.to_numeric(df.get("x", ""), errors="coerce")
        yn = pd.to_numeric(df.get("y", ""), errors="coerce")
        df = df.loc[(xn >= 12000) & (yn >= 15000)].copy()

        # Timestamp canon
        src = df["ts_iso"] if "ts_iso" in df.columns else (df["ts"] if "ts" in df.columns else "")
        df["ts_utc"] = pd.to_datetime(src, utc=True, errors="coerce")

        # Required columns check (after first file)
        if not schema_checked:
            cols = set(df.columns.astype(str))
            if not ((("trackable" in cols) or ("trackable_uid" in cols)) and ("trade" in cols) and ("x" in cols) and ("y" in cols)):
                print("Error Report:")
                print("Missing required columns for analysis.")
                print(f"Columns detected: {','.join(df.columns.astype(str))}")
                raise SystemExit(1)
            schema_checked = True

        if df.empty:
            # Nothing from this file after crop; continue to next file
            continue

        # Narrow to needed columns
        need_cols = [c for c in ["trackable_uid", "trackable", "trade", "x", "y", "ts_utc"] if c in df.columns]
        use = df[need_cols].copy()

        # Clean types
        use["x"] = pd.to_numeric(use["x"], errors="coerce")
        use["y"] = pd.to_numeric(use["y"], errors="coerce")
        use = use.dropna(subset=["x", "y", "ts_utc"])

        if use.empty:
            continue

        # Aggregations
        total_samples += len(use)
        # Time range
        smin = use["ts_utc"].min()
        smax = use["ts_utc"].max()
        if pd.notna(smin):
            first_ts = smin if first_ts is None else min(first_ts, smin)
        if pd.notna(smax):
            last_ts = smax if last_ts is None else max(last_ts, smax)
        # Unique IDs/trades
        if "trackable_uid" in use.columns:
            unique_uids.update([str(v) for v in use["trackable_uid"].dropna().astype(str).unique().tolist()])
        if "trade" in use.columns:
            ts = use["trade"].fillna("").astype(str)
            unique_trades.update(ts.unique().tolist())
            vc = ts.value_counts()
            for k, v in vc.items():
                trade_counts[k] = trade_counts.get(k, 0) + int(v)

        # Hourly counts
        _add_hour_counts(use["ts_utc"])

        # Overlay sampling (per-file capped sample)
        take = min(len(use), overlay_max)
        if take > 0:
            if len(use) > take:
                # uniform stride sample (deterministic)
                idx = _np.linspace(0, len(use) - 1, take).astype(int)
                part = use.iloc[idx][["x", "y"] + (["trade"] if "trade" in use.columns else [])].copy()
            else:
                part = use[["x", "y"] + (["trade"] if "trade" in use.columns else [])].copy()
            overlay_parts.append(part)

        # Clean up per-file
        del df, use
    except SystemExit:
        raise
    except Exception as e:
        print("Error Report:")
        print(f"Ingestion failed: {e.__class__.__name__}: {e}")
        traceback.print_exc(limit=2)
        raise SystemExit(1)

# Combined overlay DF
overlay_df = pd.concat(overlay_parts, ignore_index=True) if overlay_parts else pd.DataFrame(columns=["x", "y", "trade"])

# ------------- Floorplan loading -------------
def _find_floorplan_image() -> Path | None:
    for nm in ("floorplan.png", "floorplan.jpg", "floorplan.jpeg"):
        p = ROOT / nm
        if p.exists():
            return p
    # Also check common uppercase/lowercase variants
    for nm in ("Floorplan.png", "Floorplan.jpg", "Floorplan.jpeg"):
        p = ROOT / nm
        if p.exists():
            return p
    return None

def _load_floorplan_extent():
    """
    Returns (img, (x_min, x_max, y_min, y_max)) or (None, None) on failure.
    """
    if not FLOORJSON.exists():
        return None, None
    try:
        data = json.loads(FLOORJSON.read_text(encoding="utf-8", errors="ignore"))
        plans = data.get("floorplans") or data.get("plans") or []
        fp0 = plans[0] if isinstance(plans, list) and plans else (plans if isinstance(plans, dict) else None)
        if not fp0:
            return None, None
        width = float(fp0.get("width", 0))
        height = float(fp0.get("height", 0))
        x_c = float(fp0.get("image_offset_x", 0))
        y_c = float(fp0.get("image_offset_y", 0))
        image_scale = float(fp0.get("image_scale", 0))
        scale = image_scale * 100.0  # mm/pixel
        x_min = (x_c - width/2.0) * scale
        x_max = (x_c + width/2.0) * scale
        y_min = (y_c - height/2.0) * scale
        y_max = (y_c + height/2.0) * scale

        img_path = _find_floorplan_image()
        if not img_path or not img_path.exists():
            return None, None
        try:
            img = plt.imread(str(img_path))
        except Exception:
            img = None
        return img, (x_min, x_max, y_min, y_max)
    except Exception:
        return None, None

fp_img, fp_extent = _load_floorplan_extent()

# ------------- Chart builders -------------
def _colors_for_categories(categories: list[str]):
    cmap = _get_cmap("tab10") if callable(_get_cmap) else None
    out = {}
    uniq = list(dict.fromkeys(categories))
    for i, c in enumerate(uniq):
        if cmap is not None:
            out[c] = cmap(i % 10)
        else:
            out[c] = (0.2 + 0.6 * ((i*37) % 10)/10.0, 0.2, 0.8, 1.0)
    return out

def make_floorplan_overlay(df: pd.DataFrame, img, extent, cfg: dict) -> plt.Figure | None:
    if df is None or df.empty or img is None or extent is None:
        return None
    # ensure numeric
    df = df.copy()
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["x", "y"])
    if df.empty:
        return None

    # Limit to overlay_subsample
    max_pts = int(cfg.get("overlay_subsample", 20000))
    if len(df) > max_pts:
        idx = _np.linspace(0, len(df) - 1, max_pts).astype(int)
        df = df.iloc[idx]

    color_by = str(cfg.get("overlay_color_by", "trade"))
    if color_by not in df.columns:
        color_by = "none"

    fig = plt.figure(figsize=tuple(cfg.get("figsize_overlay", (9, 7))))
    ax = fig.add_subplot(111)
    x_min, x_max, y_min, y_max = extent
    ax.imshow(img, extent=[x_min, x_max, y_min, y_max], origin="upper")

    if color_by == "none":
        ax.scatter(df["x"], df["y"], s=float(cfg.get("overlay_point_size", 8)),
                   alpha=float(cfg.get("overlay_alpha", 0.85)))
    else:
        cats = df[color_by].astype(str).tolist()
        palette = _colors_for_categories(cats)
        labels_present = []
        for cat, g in df.groupby(color_by):
            ax.scatter(g["x"], g["y"], s=float(cfg.get("overlay_point_size", 8)),
                       alpha=float(cfg.get("overlay_alpha", 0.85)),
                       color=palette.get(cat), label=str(cat))
            labels_present.append(str(cat))
        if len(labels_present) <= 12:
            ax.legend(loc="upper left", frameon=True, fontsize=8)

    mx = float(cfg.get("floorplan_margin", 0.10))
    xr = (x_max - x_min); yr = (y_max - y_min)
    ax.set_xlim(x_min - mx * xr, x_max + mx * xr)
    ax.set_ylim(y_min - mx * yr, y_max + mx * yr)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
    ax.set_title("Floorplan Overlay")
    fig.tight_layout()
    return fig

def make_hourly_line(hour_counts: dict[pd.Timestamp, int], cfg: dict) -> plt.Figure | None:
    if not hour_counts:
        return None
    # Prepare series
    idx = sorted(hour_counts.keys())
    if len(idx) < line_min_points:
        return None
    vals = [hour_counts[k] for k in idx]
    s = pd.Series(vals, index=pd.to_datetime(idx, utc=True))
    # Convert to UTC naive for plotting safety
    x = s.index.tz_convert("UTC").tz_localize(None)
    y = s.values

    fig = plt.figure(figsize=tuple(cfg.get("figsize_line", (7, 5))))
    ax = fig.add_subplot(111)
    ax.plot(x, y, color="#1f77b4", linewidth=1.8)
    ax.set_title("Hourly Sample Counts (UTC)")
    ax.set_xlabel("Hour (UTC)")
    ax.set_ylabel("Samples")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig

def make_trade_share(trade_counts: dict[str, int], cfg: dict) -> plt.Figure | None:
    if not trade_counts:
        return None
    items = sorted(trade_counts.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k if k else "unknown" for k, _ in items]
    sizes = [v for _, v in items]
    total = sum(sizes) if sizes else 0
    if total == 0:
        return None
    shares = [v / total for v in sizes]

    # Decide pie vs bar
    if len(labels) <= pie_max_trades and max(shares) <= pie_max_single_share:
        fig = plt.figure(figsize=tuple(cfg.get("figsize_pie", (5, 5))))
        ax = fig.add_subplot(111)
        # Small explode for emphasis
        explode = [0.02 if i == 0 else 0 for i in range(len(labels))]
        ax.pie(sizes, labels=labels, autopct=lambda p: f"{p:.0f}%", startangle=90, explode=explode, textprops={"fontsize": 8})
        ax.set_title("Trade Share (by samples)")
        ax.axis("equal")
        fig.tight_layout()
        return fig
    else:
        top_n = int(cfg.get("top_n", 10))
        labels_top = labels[:top_n]
        sizes_top = sizes[:top_n]
        fig = plt.figure(figsize=tuple(cfg.get("figsize_bar", (7, 5))))
        ax = fig.add_subplot(111)
        ax.barh(labels_top[::-1], sizes_top[::-1], color="#2ca02c", alpha=0.85)
        ax.set_title("Top Trades (by samples)")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Trade")
        fig.tight_layout()
        return fig

# ------------- Build figures -------------
figs: list = []
if total_samples > 0:
    # 1) Floorplan overlay if assets present
    if bool(cfg.get("prefer_floorplan", True)) and fp_img is not None and fp_extent is not None and not overlay_df.empty:
        f1 = make_floorplan_overlay(overlay_df, fp_img, fp_extent, cfg)
        if f1 is not None and getattr(f1, "savefig", None):
            figs.append(f1)
    # 2) Hourly line
    f2 = make_hourly_line(hour_counts, cfg)
    if f2 is not None and getattr(f2, "savefig", None):
        figs.append(f2)
    # 3) Trade share
    f3 = make_trade_share(trade_counts, cfg)
    if f3 is not None and getattr(f3, "savefig", None):
        figs.append(f3)

# ------------- Summary bullets -------------
def _fmt_ts(ts) -> str:
    try:
        if pd.isna(ts) or ts is None:
            return ""
        # Keep UTC, but show ISO short
        return pd.to_datetime(ts, utc=True).strftime("%Y-%m-%d %H:%MZ")
    except Exception:
        return str(ts)

bullets: list[str] = []
if total_samples <= 0:
    bullets.append("No position samples available after filtering. A minimal summary has been generated.")
else:
    bullets.append(f"Files processed: {len(csv_paths)}")
    bullets.append(f"Samples analyzed: {total_samples:,}")
    bullets.append(f"Time span (UTC): {_fmt_ts(first_ts)} to {_fmt_ts(last_ts)}")
    bullets.append(f"Unique trackables: {len(unique_uids)}")
    bullets.append(f"Trades observed: {len([t for t in unique_trades if t])}")
    if trade_counts:
        items = sorted(trade_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
        top_str = "; ".join([f"{k or 'unknown'}: {v:,}" for k, v in items])
        bullets.append(f"Top trades by samples: {top_str}")
    if hour_counts:
        bullets.append(f"Hours with data: {len(hour_counts)}")

# ------------- Build report dict -------------
report_date = (pd.to_datetime(last_ts, utc=True).strftime("%Y-%m-%d") if last_ts is not None else pd.Timestamp.utcnow().strftime("%Y-%m-%d"))
title_str = "Walmart Renovation RTLS — Position Summary"
meta_lines = []
try:
    meta_lines.append(f"ROOT: {str(ROOT.resolve())}")
except Exception:
    pass
meta_lines.append(f"CSV count: {len(csv_paths)}")
meta_lines.append(f"Generated: {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%MZ')}")
meta_text = " | ".join(meta_lines)

sections = []
sections.append({"type": "summary", "title": "Key Findings", "bullets": bullets})
# Add charts only if any figures exist
figs_live = [f for f in figs if getattr(f, "savefig", None)]
if figs_live:
    sections.append({"type": "charts", "title": "Visuals", "figures": figs_live})

report = {"title": title_str, "meta": meta_text, "sections": sections}
# Apply budgets (cap figures/text if necessary)
report = apply_budgets(report)

# ------------- Save PNGs first -------------
png_paths: list[Path] = []
if figs_live:
    for i, fig in enumerate(figs_live, 1):
        pth = out_dir / f"info_zone_report_{report_date}_plot{i:02d}.png"
        try:
            fig.savefig(str(pth), dpi=120)
            png_paths.append(pth)
        except Exception:
            # Continue; PDF still can embed figures
            pass

# ------------- Build PDF -------------
pdf_path = out_dir / f"info_zone_report_{report_date}.pdf"
try:
    safe_build_pdf(report, str(pdf_path), logo_path=str(LOGO))
except Exception as e:
    print("Error Report:")
    print(f"PDF build failed: {e.__class__.__name__}: {e}")
    traceback.print_exc(limit=2)
    try:
        report = make_lite(report)
        safe_build_pdf(report, str(pdf_path), logo_path=str(LOGO))
    except Exception as e2:
        print("Error Report:")
        print(f"Lite PDF failed: {e2.__class__.__name__}: {e2}")
        traceback.print_exc(limit=2)
        raise SystemExit(1)

# ------------- Success links (Windows/Linux safe) -------------
def file_uri(p: Path) -> str:
    return "file:///" + str(p.resolve()).replace("\\", "/")

print(f"[Download the PDF]({file_uri(pdf_path)})")
for i, pth in enumerate(png_paths, 1):
    print(f"[Download Plot {i}]({file_uri(pth)})")