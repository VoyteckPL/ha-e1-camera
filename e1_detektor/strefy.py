"""Strefy obrazu kamery i ustawienia powiadomien per strefa x kategoria.

Wspolrzedne punktow: 0..1000 (x, y) wzgledem calego kadru.
Konfiguracja w /data/strefy.json (przezywa aktualizacje add-onu).
"""
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None

log = logging.getLogger("e1.strefy")

DATA = Path(os.environ.get("E1_DATA", "/data"))
PLIK = DATA / "strefy.json"
HA_CFG = Path(os.environ.get("E1_HA", "/ha_config"))

KATEGORIE = ("osoba", "pojazd", "zwierze")
POZA = "poza"
POZA_NAZWA = "Poza strefami"
POZA_KOLOR = "#6b7280"
DOM = {"n": 1, "a": "00:00", "b": "23:59", "d": "1111111", "m": 0, "c": 60, "s": 1, "t": 0, "h": 1}
KOLORY = ["#2563eb", "#16a34a", "#f59e0b", "#dc2626", "#8b5cf6", "#0891b2", "#db2777", "#65a30d"]
MAX_STREF = 8
MAX_PKT = 40

_cfg = None
_ostatnie = {}  # (strefa, kat) -> czas ostatniego powiadomienia
_poprz = {}  # kat -> (strefa, czas) do histerezy


def _tz():
    try:
        z = json.load(open(HA_CFG / ".storage" / "core.config"))["data"]["time_zone"]
        return ZoneInfo(z)
    except Exception:
        return ZoneInfo("Europe/Warsaw") if ZoneInfo else None


def parse_granica(s):
    pts = []
    for p in (s or "").split(";"):
        if "," in p:
            try:
                a, b = p.split(",")[:2]
                pts.append([float(a), float(b)])
            except ValueError:
                pass
    return pts if len(pts) >= 3 else []


def domyslna(granica=""):
    pts = parse_granica(granica)
    ust = {}
    strefy = []
    if pts:
        strefy.append({"id": "z1", "nazwa": "Podjazd", "kolor": KOLORY[0], "prio": 1, "pts": pts})
        ust["z1"] = {
            "osoba": dict(DOM, c=120),
            "pojazd": dict(DOM, c=45),
            "zwierze": dict(DOM, c=120),
        }
        ust[POZA] = {
            "osoba": dict(DOM, c=120),
            "pojazd": dict(DOM, n=0, h=0, c=45),
            "zwierze": dict(DOM, n=0, h=0, c=120),
        }
    else:
        ust[POZA] = {"osoba": dict(DOM, c=120), "pojazd": dict(DOM, c=45), "zwierze": dict(DOM, c=120)}
    return {"strefy": strefy, "ust": ust}


def _ust_ok(u):
    out = dict(DOM)
    if not isinstance(u, dict):
        return out
    try:
        out["n"] = 1 if int(u.get("n", 1)) else 0
        out["t"] = 1 if int(u.get("t", 0)) else 0
        out["h"] = 1 if int(u.get("h", 1)) else 0
        out["s"] = 1 if int(u.get("s", 1)) else 0
        out["m"] = max(0, min(2, int(u.get("m", 0))))
        out["c"] = max(0, min(86400, int(u.get("c", 60))))
    except (TypeError, ValueError):
        pass
    for k in ("a", "b"):
        v = str(u.get(k, DOM[k]))
        out[k] = v if re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", v) else DOM[k]
    d = str(u.get("d", DOM["d"]))
    out["d"] = d if re.fullmatch(r"[01]{7}", d) else DOM["d"]
    return out


def czysc(c):
    """Waliduje i normalizuje konfiguracje z panelu."""
    strefy, ids = [], set()
    for i, z in enumerate((c or {}).get("strefy", [])[:MAX_STREF]):
        pts = []
        for p in (z.get("pts") or [])[:MAX_PKT]:
            try:
                pts.append([max(0.0, min(1000.0, float(p[0]))), max(0.0, min(1000.0, float(p[1])))])
            except (TypeError, ValueError, IndexError):
                pass
        if len(pts) < 3:
            continue
        zid = re.sub(r"[^a-z0-9_]", "", str(z.get("id", "")).lower())[:12] or f"z{i + 1}"
        while zid in ids or zid == POZA:
            zid += "x"
        ids.add(zid)
        kol = str(z.get("kolor", ""))
        strefy.append(
            {
                "id": zid,
                "nazwa": (str(z.get("nazwa", "")).strip()[:30] or f"Strefa {i + 1}"),
                "kolor": kol if re.fullmatch(r"#[0-9a-fA-F]{6}", kol) else KOLORY[i % len(KOLORY)],
                "prio": max(0, min(99, int(z.get("prio", 1) or 0))),
                "pts": pts,
            }
        )
    ust = {}
    src = (c or {}).get("ust", {})
    for zid in [s["id"] for s in strefy] + [POZA]:
        ust[zid] = {k: _ust_ok((src.get(zid) or {}).get(k)) for k in KATEGORIE}
    return {"strefy": strefy, "ust": ust}


def wczytaj(granica=""):
    global _cfg
    if _cfg is None:
        try:
            _cfg = czysc(json.load(open(PLIK)))
        except Exception:
            _cfg = None
    if _cfg is None:
        return czysc(domyslna(granica))
    return _cfg


def zapisz(c):
    global _cfg
    c = czysc(c)
    DATA.mkdir(parents=True, exist_ok=True)
    tmp = PLIK.with_suffix(".tmp")
    json.dump(c, open(tmp, "w"), ensure_ascii=False)
    os.replace(tmp, PLIK)
    _cfg = c
    return c


def w_wielokacie(px, py, pts):
    ins = False
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[i - 1]
        if (a[1] > py) != (b[1] > py) and px < (b[0] - a[0]) * (py - a[1]) / (b[1] - a[1]) + a[0]:
            ins = not ins
    return ins


def przypisz(box, iw, ih, cfg, kat, teraz=None):
    """Zwraca id strefy dla ramki: wiekszosc probek na 'stopach' obiektu, z histereza."""
    strefy = sorted(cfg["strefy"], key=lambda z: -z["prio"])
    if not strefy:
        return POZA
    x1, y1, x2, y2 = box[:4]
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    pk = []
    for fx in (0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9):
        for fy in (0.88, 0.94, 1.0):
            pk.append(((x1 + fx * w) / iw * 1000, (y1 + fy * h) / ih * 1000))
    licz = {}
    for px, py in pk:
        zid = POZA
        for z in strefy:
            if w_wielokacie(px, py, z["pts"]):
                zid = z["id"]
                break
        licz[zid] = licz.get(zid, 0) + 1
    prio = {z["id"]: z["prio"] for z in strefy}
    prio[POZA] = -1
    rank = sorted(licz.items(), key=lambda kv: (-kv[1], -prio.get(kv[0], 0)))
    best = rank[0][0]
    t = teraz if teraz is not None else time.time()
    if len(rank) > 1 and rank[0][1] - rank[1][1] <= 3:
        pv = _poprz.get(kat)
        if pv and t - pv[1] < 90 and pv[0] in (rank[0][0], rank[1][0]):
            best = pv[0]
    _poprz[kat] = (best, t)
    return best


def nazwa(cfg, zid):
    if zid == POZA:
        return POZA_NAZWA
    for z in cfg["strefy"]:
        if z["id"] == zid:
            return z["nazwa"]
    return POZA_NAZWA


def _min(s):
    return int(s[:2]) * 60 + int(s[3:5])


def decyzja(cfg, zid, kat, znaleziono, ctx=None, teraz=None):
    """Czy powiadomic i czy zapisac do historii. ctx: {'ciemno': bool}."""
    ctx = ctx or {}
    u = (cfg["ust"].get(zid) or {}).get(kat) or dict(DOM)
    tz = _tz()
    now = teraz or (datetime.now(tz) if tz else datetime.now())
    od, do, tm = _min(u["a"]), _min(u["b"]), now.hour * 60 + now.minute
    okno = (od <= tm <= do) if od <= do else (tm >= od or tm <= do)
    dzien = u["d"][now.weekday()] == "1"
    ciemno = bool(ctx.get("ciemno"))
    pora = u["m"] == 0 or (u["m"] == 1 and ciemno) or (u["m"] == 2 and not ciemno)
    potw = znaleziono or not u["t"]
    historia = bool(u["h"]) and potw
    t = time.time()
    odstep = t - _ostatnie.get((zid, kat), 0) >= u["c"]
    pow_ = bool(u["n"]) and okno and dzien and pora and odstep and potw
    if pow_:
        _ostatnie[(zid, kat)] = t
    return {"powiadom": pow_, "historia": historia, "dzwiek": int(u["s"])}


def api(dane):
    cmd = str(dane.get("cmd", "get"))
    if cmd == "set":
        c = zapisz(dane.get("dane") or {})
    else:
        c = wczytaj(str(dane.get("granica", "")))
    return {"ok": True, "strefy": c["strefy"], "ust": c["ust"], "poza": {"nazwa": POZA_NAZWA, "kolor": POZA_KOLOR}}
