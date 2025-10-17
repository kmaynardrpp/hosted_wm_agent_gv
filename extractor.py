# ------------------------------------------------------------------------------
# extractor.py
#
# Local, MAC/UID-aware extractor for Walmart RTLS positions (x, y, z).
# Populates:
#   - trackable / trackable_uid via local trackable_objects.json (MAC and UID)
#   - trade from FINAL trackable (regex/prefix canonicalization)
#   - ts_utc (ISO 'Z'), ts_iso (alias), ts_short ("MM-DD HH:MM")
#   - zone_name normalization (STRICT mapping to canonical names like "Sales Floor",
#     "Vestibule", "Receiving", "FET", ...), and filters out Trailer rows.
#
# Returns: {"rows": List[dict], "audit": dict}
# ------------------------------------------------------------------------------

from __future__ import annotations

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# =========================== ROOT resolution (LOCAL) ===========================
def _resolve_root() -> Path:
    v = os.environ.get("INFOZONE_ROOT", "").strip()
    if v:
        p = Path(v).resolve()
        if p.exists():
            return p
    p = Path(__file__).resolve().parent
    return p if p.exists() else Path.cwd().resolve()

ROOT = _resolve_root()
EXTRACTOR_SIGNATURE = "extractor/v7-mac-guess+uid-safe-name+zone-map-strict"

# ============================== MAC/UID map (LOCAL) ============================
def _norm_mac(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r"[^0-9a-f]", "", str(s).lower())

def _load_maps(mac_map_path: Optional[str | os.PathLike] = None) -> tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    candidates: List[Path] = []
    if mac_map_path:
        candidates.append(Path(mac_map_path))
    candidates.extend([ROOT / "trackable_objects.json", Path.cwd() / "trackable_objects.json"])

    mac_map: Dict[str, Dict[str, str]] = {}
    uid_map: Dict[str, Dict[str, str]] = {}

    for p in candidates:
        try:
            if not p.exists():
                continue
            data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
            items = data.get("trackable_objects") or data.get("objects") or data
            if isinstance(items, list):
                for it in items:
                    mac = _norm_mac(it.get("mac") or it.get("mac_address"))
                    uid = str(it.get("uid", "") or it.get("id", "") or "")
                    name = str(it.get("name", "") or it.get("display_name", "") or "")
                    if mac:
                        mac_map[mac] = {"name": name, "uid": uid}
                    if uid:
                        uid_map[uid] = {"name": name, "uid": uid}
            elif isinstance(items, dict):
                for k, v in items.items():
                    if not isinstance(v, dict):
                        continue
                    mac = _norm_mac(k)
                    uid = str(v.get("uid", "") or v.get("id", "") or "")
                    name = str(v.get("name", "") or v.get("display_name", "") or "")
                    if mac:
                        mac_map[mac] = {"name": name, "uid": uid}
                    if uid:
                        uid_map[uid] = {"name": name, "uid": uid}
            if mac_map or uid_map:
                break
        except Exception:
            continue
    return mac_map, uid_map

# ============================== Trade inference ===============================
_TRADE_REGEX = [
    (re.compile(r"\bcarp(enter|entry|entry_\d+)?\b|carpentr(y)?", re.I), "carpentry"),
    (re.compile(r"\belectric(ians?|ian)?\b|\belec\b|\belectric\b", re.I), "electrician"),
    (re.compile(r"\bplumb(ing|er)?\b", re.I), "plumbing"),
    (re.compile(r"\bpainter\b|\bpainting\b|\bpaint\b", re.I), "painter"),
    (re.compile(r"\bmechanical\b|\bhvac\b|\bmech\b", re.I), "mechanical"),
    (re.compile(r"\bclean(ing|er)?\b|\bclean\b", re.I), "cleaning"),
    (re.compile(r"\brefrig(eration)?\b|\brefrig\b", re.I), "refrigeration"),
    (re.compile(r"\bstrip(ing|e)?\b|\bstripe\b", re.I), "striping"),
    (re.compile(r"\broof(ing)?\b|\broof\b", re.I), "roofing"),
    (re.compile(r"\bfloor(ing)?\b|\bfloor\b", re.I), "flooring"),
    (re.compile(r"\bconcrete\b", re.I), "concrete"),
    (re.compile(r"\b(building\s*signage|blgd\s*signage|signage)\b|\bbuilding\b", re.I), "building_signage"),
    (re.compile(r"\bfire\s*sprink(ler)?\b|\bfire\s*spk\b|\bfire\b|\bsprinkler\b", re.I), "fire_sprinkler"),
    (re.compile(r"\bems\b", re.I), "ems"),
    (re.compile(r"\bsteel\b", re.I), "steel"),
]
_TRADE_PREFIX_MAP = {
    "carpentry":"carpentry","carpenter":"carpentry","carp":"carpentry",
    "electrician":"electrician","electric":"electrician","elec":"electrician",
    "plumbing":"plumbing","plumber":"plumbing","plumb":"plumbing",
    "painter":"painter","painting":"painter","paint":"painter",
    "mechanical":"mechanical","mech":"mechanical","hvac":"mechanical",
    "cleaning":"cleaning","cleaner":"cleaning","clean":"cleaning",
    "refrigeration":"refrigeration","refrig":"refrigeration",
    "striping":"striping","stripe":"striping",
    "roofing":"roofing","roof":"roofing",
    "flooring":"flooring","floor":"flooring",
    "concrete":"concrete",
    "signage":"building_signage","building":"building_signage","building_signage":"building_signage","blgd":"building_signage",
    "fire":"fire_sprinkler","sprinkler":"fire_sprinkler","fire_sprinkler":"fire_sprinkler","fire_spk":"fire_sprinkler",
    "ems":"ems",
    "steel":"steel",
}
def _infer_trade_from_trackable(label: str) -> str:
    if not label:
        return ""
    s = str(label).replace("_", " ").replace("-", " ").strip()
    prefix = re.split(r"\s+", s, maxsplit=1)[0].lower()
    if prefix in _TRADE_PREFIX_MAP:
        return _TRADE_PREFIX_MAP[prefix]
    for pat, canon in _TRADE_REGEX:
        if pat.search(s):
            return canon
    return ""

# ============================ Zone normalization (STRICT) =====================
_ZONE_MAP: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(100\s+gr\s+)?sales\s+floor\b", re.I), "Sales Floor"),
    (re.compile(r"\bvestibule\b", re.I), "Vestibule"),
    (re.compile(r"\bfet\b", re.I), "FET"),
    (re.compile(r"\bpersonnel\b", re.I), "Personnel"),
    (re.compile(r"\bpharmacy\b", re.I), "Pharmacy"),
    (re.compile(r"\b(restroom|customer\s+restroom)s?\b", re.I), "Restroom"),
    (re.compile(r"\bbreak\s*room\b|\bbreakroom\b", re.I), "Breakroom"),
    (re.compile(r"\bpickup\b", re.I), "Pickup"),
    (re.compile(r"\bdeli\b", re.I), "Deli"),
    (re.compile(r"\breceiving\b", re.I), "Receiving"),
    (re.compile(r"\btrailer\b", re.I), "Trailer"),
]
_ZONE_FALLBACK_STOP = {
    "zone","area","aisle","hall","hallway","bay","dock","office",
    "department","dept","section","room","floor","gm","gr","store",
    "backroom","front","front-end","pickup","storage","information","active","inactive"
}

def _desired_zone_display(raw: str) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    # direct canonical match
    for pat, display in _ZONE_MAP:
        if pat.search(s):
            return display
    # strip "Zone <num> - " and numeric noise
    tail = re.sub(r"^\s*zone\s+\d+(\.\d+)?\s*-\s*", "", s, flags=re.I)
    tail = re.sub(r"^\s*(\d{1,4}(\.\d+)?(\s+[A-Z]{2,})?\s*)+", "", tail).strip()
    if not tail:
        return ""
    joined = " ".join(re.findall(r"[A-Za-z]+", tail)).title()
    if "Sales" in joined and "Floor" in joined:
        return "Sales Floor"
    for target in ["Vestibule","Personnel","Pharmacy","Breakroom","Receiving","Deli","Restroom","Pickup","FET"]:
        if re.search(fr"\b{target}\b", joined, flags=re.I):
            return target
    words = [w for w in re.findall(r"[A-Za-z]+", tail) if w.lower() not in _ZONE_FALLBACK_STOP]
    return (words[-1].title() if words else tail.title())

# --- Load zone UID->Name lookup so we can map UIDs to names before normalization
def _load_zone_name_lookup(zones_path: Optional[str | os.PathLike] = None) -> Dict[str, str]:
    candidates: List[Path] = []
    if zones_path:
        candidates.append(Path(zones_path))
    candidates.extend([ROOT / "zones.json", Path.cwd() / "zones.json"])
    for p in candidates:
        try:
            if not p.exists():
                continue
            data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
            zones = data.get("zones") or []
            lookup: Dict[str, str] = {}
            for z in zones:
                uid = str(z.get("uid", "") or "")
                name = str(z.get("name", "") or "")
                if uid and name:
                    # store canonical display for that UID
                    lookup[uid] = _desired_zone_display(name)
            if lookup:
                return lookup
        except Exception:
            continue
    return {}

# ============================ CSV reading helpers =============================
def _read_csv_comma(csv_path: str, row_limit: Optional[int]) -> pd.DataFrame:
    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(
                csv_path, sep=",", header=0, dtype=str, engine="python",
                on_bad_lines="skip", comment="#", encoding=enc, na_filter=False,
            ); break
        except Exception:
            df = None
    if df is None:
        df = pd.read_csv(csv_path, sep=None, header=0, dtype=str, engine="python",
                         on_bad_lines="skip", comment="#", na_filter=False)
    if row_limit and len(df) > row_limit:
        df = df.iloc[:row_limit, :].copy()
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    return df

def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        c = lower.get(name.lower())
        if c:
            return c
    for c in df.columns:
        lc = c.lower()
        for name in candidates:
            if name.lower() in lc:
                return c
    return None

_MAC_SHAPED = re.compile(r"^(?:[0-9A-Fa-f]{2}([:-]?)){5}[0-9A-Fa-f]{2}$|^[0-9A-Fa-f]{12}$")

def _guess_mac_col_by_pattern(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    for c in cols:
        s = df[c].astype(str)
        s = s[s.str.len() > 0].head(4000)
        if s.empty:
            continue
        hit = (s.str.match(_MAC_SHAPED)).mean()
        if hit >= 0.05:
            return c
    return None

# ================================ Public API ==================================
def extract_tracks(csv_path: str,
                   mac_map_path: Optional[str | os.PathLike] = None,
                   row_limit: Optional[int] = None,
                   **kwargs) -> dict:
    df = _read_csv_comma(csv_path, row_limit=row_limit)

    # Ensure string dtype
    for c in df.columns:
        if not pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].astype(str)

    # Identify columns
    mac_aliases = ["mac","mac_address","macaddr","bt_mac","ble_mac","bluetooth_mac","bdaddr","btaddr","addr","address","bleaddress","ble_address"]
    col_mac  = _find_col(df, mac_aliases) or _guess_mac_col_by_pattern(df)

    uid_aliases = ["trackable_uid","uid","user_id","device_id","tag_id","entity_id","badge_id","tag_uid"]
    col_uidc = _find_col(df, uid_aliases)
    ts_aliases  = ["ts_utc","ts","timestamp","time","datetime","date_time"]
    col_ts   = _find_col(df, ts_aliases)
    col_zone_src = _find_col(df, ["zone_name","zone","area","area_name","region","location"])

    # Canonical outputs
    if col_uidc is None:
        df["trackable_uid"] = ""
        col_uidc = "trackable_uid"

    col_trade = _find_col(df, ["trade","role","craft","discipline"]) or "trade"
    if col_trade not in df.columns:
        df[col_trade] = ""

    # Timestamps
    df["ts_utc"] = ""
    df["ts_iso"] = ""
    df["ts_short"] = ""
    if col_ts:
        ts_parsed = pd.to_datetime(df[col_ts], utc=True, errors="coerce")
        mask = ts_parsed.notna()
        if mask.any():
            iso = ts_parsed[mask].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            df.loc[mask, "ts_utc"] = iso
            df.loc[mask, "ts_iso"] = iso
            df.loc[mask, "ts_short"] = ts_parsed[mask].dt.strftime("%m-%d %H:%M")

    # Load maps (LOCAL)
    mac_map, uid_map = _load_maps(mac_map_path)

    # Normalized keys
    mac_norm = df[col_mac].map(_norm_mac) if col_mac else pd.Series([""] * len(df), index=df.index)
    uid_vals = df[col_uidc].astype(str) if col_uidc else pd.Series([""] * len(df), index=df.index)

    # Choose a safe name source (never zone/area/region/id/uid)
    def _choose_trackable_source(cols: List[str], zone_present: bool) -> Optional[str]:
        lower = {c.lower(): c for c in cols}
        exact_pref = ["trackable","trackable_name","device_name","tag_name","asset_name","display_name"]
        for nm in exact_pref:
            c = lower.get(nm)
            if c:
                return c
        if not zone_present:
            c = lower.get("name")
            if c:
                return c
        for c in cols:
            lc = c.lower()
            if lc.endswith("_name") and not any(b in lc for b in ("zone","area","region","id","uid")):
                return c
        return None

    col_trackable_src = _choose_trackable_source(list(df.columns), zone_present=(col_zone_src is not None))

    # Build mapped name/uid from MAC and UID
    def _map_from_mac(uid_series=False) -> pd.Series:
        vals = []
        for mac in mac_norm.tolist():
            m = mac_map.get(mac) if mac else None
            vals.append((m.get("uid") if uid_series else m.get("name")) if m else "")
        return pd.Series(vals, index=df.index, dtype="object")

    def _map_from_uid(uid_series=False) -> pd.Series:
        vals = []
        for u in uid_vals.tolist():
            m = uid_map.get(u) if u else None
            vals.append((m.get("uid") if uid_series else m.get("name")) if m else "")
        return pd.Series(vals, index=df.index, dtype="object")

    mapped_name_mac = _map_from_mac(uid_series=False)
    mapped_uid_mac  = _map_from_mac(uid_series=True)
    mapped_name_uid = _map_from_uid(uid_series=False)
    mapped_uid_uid  = _map_from_uid(uid_series=True)

    # Final canonical 'trackable'
    if col_trackable_src and not any(k in col_trackable_src.lower() for k in ("zone","area","region","id","uid")):
        cur = df[col_trackable_src].astype(str).str.strip()
    else:
        cur = pd.Series([""] * len(df), index=df.index, dtype="object")
    candidate = cur.where(cur.ne(""), mapped_name_mac)
    candidate = candidate.where(candidate.ne(""), mapped_name_uid)
    df["trackable"] = candidate.fillna("")

    # Ensure UID column
    curu = df[col_uidc].astype(str).str.strip()
    fillu = curu.where(curu.ne(""), mapped_uid_mac)
    fillu = fillu.where(fillu.ne(""), mapped_uid_uid)
    df[col_uidc] = fillu.fillna("")

    # Trade inference for blank/unknown
    tcur = df[col_trade].astype(str).str.strip()
    need = tcur.eq("") | tcur.str.lower().eq("unknown")
    if need.any():
        df.loc[need, col_trade] = df.loc[need, "trackable"].map(_infer_trade_from_trackable).fillna("")

    # ---- Zone normalization & Trailer filtering (STRICT + UID guard) ----
    removed_trailer = 0
    if col_zone_src:
        uid2name = _load_zone_name_lookup()  # map zone UID -> canonical name if available
        raw_vals = df[col_zone_src].astype(str)

        UIDISH = re.compile(r"^[A-Za-z0-9_\-]{16,}$")  # looks like a UID token (no spaces, long)
        def _normalize_zone_value(v: str) -> str:
            s = (v or "").strip()
            if not s:
                return ""
            # If it's an exact UID we know, map to its human name first
            if s in uid2name:
                return _desired_zone_display(uid2name[s])
            # If it looks like a UID but we don't know it, DO NOT try to "normalize" (avoid 'Q')
            if UIDISH.match(s) and (" " not in s):
                return s  # leave as-is
            # Otherwise treat as a human label and normalize
            return _desired_zone_display(s)

        znorm = raw_vals.map(_normalize_zone_value)
        if col_zone_src != "zone_name":
            df["zone_name"] = znorm
        else:
            df[col_zone_src] = znorm

        # Filter Trailer rows (after normalization or UID mapping)
        mask_trailer = df["zone_name"].str.contains(r"\bTrailer\b", case=False, na=False)
        removed_trailer = int(mask_trailer.sum())
        if removed_trailer:
            df = df.loc[~mask_trailer].copy()

    # Rows → list-of-dicts
    out_rows = df.to_dict(orient="records")

    # Audit
    macs_seen = int((mac_norm != "").sum())
    mac_hits = int(sum(1 for m in mac_norm if m and m in mac_map))
    uids_seen = int((uid_vals.astype(bool)).sum())
    uid_hits = int(sum(1 for u in uid_vals if u and u in uid_map))
    trade_nonempty_rate = float((df[col_trade].astype(str).str.strip() != "").mean()) if len(df) else 0.0

    audit = {
        "source_file": str(csv_path),
        "root": str(ROOT),
        "extractor_signature": EXTRACTOR_SIGNATURE,
        "mac_map_loaded": bool(mac_map),
        "uid_map_loaded": bool(uid_map),
        "macs_seen": macs_seen,
        "mac_hits": mac_hits,
        "uids_seen": uids_seen,
        "uid_hits": uid_hits,
        "rows_total": int(len(df)),
        "rows_removed_trailer": removed_trailer,
        "columns_detected": list(map(str, df.columns.tolist())),
        "mac_col_selected": col_mac or "",
        "trade_nonempty_rate": round(trade_nonempty_rate, 4),
        "notes": "zone_name normalized (UID-aware); Trailer rows removed; trackable from safe/MAC/UID; ts_utc canonical.",
    }

    return {"rows": out_rows, "audit": audit}
