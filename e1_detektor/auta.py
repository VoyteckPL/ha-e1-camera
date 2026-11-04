"""Rozpoznawanie aut domowników po kolorze (histogram) + uczenie z etykiet nadanych w panelu."""
import json
import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger("e1.auta")

PLIK = Path(os.environ.get("E1_DATA", "/data")) / "auta.json"
DOMYSLNE = [
    {"id": "kia", "nazwa": "Kia Sportage 5", "opis": "zielona", "prior": [2, 3, 4]},
    {"id": "seat", "nazwa": "Seat Altea", "opis": "niebieski", "prior": [6, 7, 8]},
    {"id": "mazda", "nazwa": "Mazda 5", "opis": "srebrna", "prior": [14]},
]
PROG = 0.5
MAX_PROBEK = 30
OBCE = "obce"


def _dane():
    try:
        d = json.load(open(PLIK))
    except Exception:
        d = {}
    d.setdefault("auta", DOMYSLNE)
    d.setdefault("probki", {})
    return d


def _zapisz(d):
    tmp = PLIK.with_suffix(".tmp")
    json.dump(d, open(tmp, "w"))
    os.replace(tmp, PLIK)


def cecha(path):
    """Wektor 15 liczb: 12 przedzialow barwy (kolorowe piksele) + ciemny/bialy/szary."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    img = img.crop((int(w * 0.12), int(h * 0.15), int(w * 0.88), int(h * 0.88))).resize((64, 40))
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    v = np.zeros(15, dtype=np.float32)
    kol = (S > 0.28) & (V > 0.22)
    hb = np.minimum((H * 12).astype(int), 11)
    for i in range(12):
        v[i] = float(np.sum(kol & (hb == i)))
    ciemny = (V <= 0.22) | ((S <= 0.28) & (V < 0.35))
    bialy = (~kol) & (~ciemny) & (V > 0.7)
    v[12] = float(np.sum(ciemny))
    v[13] = float(np.sum(bialy))
    v[14] = float(np.sum((~kol) & (~ciemny) & (~bialy)))
    s = v.sum()
    return (v / s).tolist() if s > 0 else v.tolist()


def _inter(a, b):
    return float(np.minimum(np.array(a), np.array(b)).sum())


def rozpoznaj(path):
    """Zwraca dict auta (id, nazwa, opis) albo None."""
    try:
        v = cecha(path)
    except Exception:
        log.exception("cecha")
        return None
    d = _dane()
    best, bs = None, 0.0
    obce = max([_inter(v, p) for p in d["probki"].get(OBCE, [])] or [0.0])
    for a in d["auta"]:
        pr = d["probki"].get(a["id"], [])
        if pr:
            sc = max(_inter(v, p) for p in pr)
        elif a.get("prior"):
            sc = sum(v[i] for i in a["prior"])
        else:
            continue
        if sc > bs:
            best, bs = a, sc
    if best is None or bs < PROG or obce > bs + 0.03:
        return None
    return {"id": best["id"], "nazwa": best["nazwa"], "opis": best.get("opis", ""), "score": round(bs, 2)}


def etykieta(path, auto_id):
    d = _dane()
    if auto_id != OBCE and not any(a["id"] == auto_id for a in d["auta"]):
        return None
    pr = d["probki"].setdefault(auto_id, [])
    pr.append(cecha(path))
    del pr[:-MAX_PROBEK]
    _zapisz(d)
    return next((a for a in d["auta"] if a["id"] == auto_id), {"id": OBCE, "nazwa": "Obce auto", "opis": ""})


def lista():
    d = _dane()
    return [dict(a, probek=len(d["probki"].get(a["id"], []))) for a in d["auta"]]


def etykieta_pelna(a):
    if not a:
        return ""
    return a["nazwa"] + (" (" + a["opis"] + ")" if a.get("opis") else "")
