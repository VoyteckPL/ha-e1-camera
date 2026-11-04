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
    {"id": "mazda", "nazwa": "Mazda 5", "opis": "srebrna", "prior": []},
]
PROG = 0.62
MARGINES = 0.1
MAX_PROBEK = 30
OBCE = "obce"


def _dane():
    try:
        d = json.load(open(PLIK))
    except Exception:
        d = {}
    d.setdefault("auta", DOMYSLNE)
    d.setdefault("probki", {})
    d["probki"] = {k: [x for x in v if len(x) == NB] for k, v in d["probki"].items()}  # stary format cech odpada
    return d


def _zapisz(d):
    tmp = PLIK.with_suffix(".tmp")
    json.dump(d, open(tmp, "w"))
    os.replace(tmp, PLIK)


NB = 17  # 12 przedzialow barwy + 5 poziomow jasnosci dla szarosci/czerni/bieli


def _hist(hsv, m):
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    v = np.zeros(NB, dtype=np.float32)
    kol = (S > 0.17) & (V > 0.18) & m
    hb = np.minimum((H * 12).astype(int), 11)
    for i in range(12):
        v[i] = float(np.sum(kol & (hb == i)))
    vb = np.minimum((V * 5).astype(int), 4)
    for i in range(5):
        v[12 + i] = float(np.sum((~kol) & m & (vb == i)))
    s = v.sum()
    return (v / s).tolist() if s > 0 else v.tolist()


def cecha(path):
    """Cechy z pliku wycinka (srodek kadru) - dla starszych zdarzen bez wielokata."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    img = img.crop((int(w * 0.12), int(h * 0.15), int(w * 0.88), int(h * 0.88))).resize((64, 40))
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
    return _hist(hsv, np.ones(hsv.shape[:2], bool))


def cecha_maska(img, poly, n):
    """Cechy tylko z pikseli wewnatrz obrysu auta (bez tla): img - pelny kadr, poly/n - wspolrzedne 0-1."""
    from PIL import ImageDraw
    img = img.convert("RGB")
    iw, ih = img.size
    if poly and len(poly) >= 3:
        pts = [(p[0] * iw, p[1] * ih) for p in poly]
        x1, y1 = min(p[0] for p in pts), min(p[1] for p in pts)
        x2, y2 = max(p[0] for p in pts), max(p[1] for p in pts)
    else:
        x1, y1, x2, y2 = n[0] * iw, n[1] * ih, n[2] * iw, n[3] * ih
        w, h = x2 - x1, y2 - y1
        x1, y1, x2, y2 = x1 + w * 0.12, y1 + h * 0.15, x2 - w * 0.12, y2 - h * 0.12
        pts = None
    crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
    sc = min(1.0, 96 / max(crop.size))
    crop = crop.resize((max(1, int(crop.size[0] * sc)), max(1, int(crop.size[1] * sc))))
    mask = Image.new("L", crop.size, 255 if pts is None else 0)
    if pts:
        ImageDraw.Draw(mask).polygon([((px - int(x1)) * sc, (py - int(y1)) * sc) for px, py in pts], fill=255)
    hsv = np.asarray(crop.convert("HSV"), dtype=np.float32) / 255.0
    return _hist(hsv, np.asarray(mask) > 0)


def _inter(a, b):
    return float(np.minimum(np.array(a), np.array(b)).sum())


def rozpoznaj(path):
    try:
        return rozpoznaj_wektor(cecha(path))
    except Exception:
        log.exception("cecha")
        return None


def rozpoznaj_wektor(v):
    """Zwraca dict auta (id, nazwa, opis) albo None. Wymaga wyraznej przewagi nad pozostalymi autami."""
    d = _dane()
    scores = []  # (wynik, auto lub None dla "obce")
    ob = [_inter(v, p) for p in d["probki"].get(OBCE, [])]
    if ob:
        scores.append((max(ob), None))
    for a in d["auta"]:
        pr = d["probki"].get(a["id"], [])
        if pr:
            sc = max(_inter(v, p) for p in pr)
        elif a.get("prior"):
            sc = sum(v[i] for i in a["prior"])
        else:
            continue
        scores.append((sc, a))
    if not scores:
        return None
    scores.sort(key=lambda x: -x[0])
    bs, best = scores[0]
    second = scores[1][0] if len(scores) > 1 else 0.0
    if best is None or bs < PROG or (bs - second) < (MARGINES if bs < 0.85 else 0.03):
        log.info("nierozpoznane: %s", [(a['id'] if a else 'obce', round(x, 2)) for x, a in scores])
        return None
    return {"id": best["id"], "nazwa": best["nazwa"], "opis": best.get("opis", ""), "score": round(bs, 2)}


def etykieta(path, auto_id):
    return etykieta_wektor(cecha(path), auto_id)


def etykieta_wektor(vec, auto_id):
    d = _dane()
    if auto_id != OBCE and not any(a["id"] == auto_id for a in d["auta"]):
        return None
    pr = d["probki"].setdefault(auto_id, [])
    pr.append(vec)
    del pr[:-MAX_PROBEK]
    _zapisz(d)
    return next((a for a in d["auta"] if a["id"] == auto_id), {"id": OBCE, "nazwa": "Obce auto", "opis": ""})


def reset(auto_id, opis=None):
    """Kasuje probki auta (wraca do priorytetu barwy); opcjonalnie ustawia opis."""
    d = _dane()
    d["probki"].pop(auto_id, None)
    for a in d["auta"]:
        if a["id"] == auto_id and opis is not None:
            a["opis"] = opis
    _zapisz(d)


def lista():
    d = _dane()
    return [dict(a, probek=len(d["probki"].get(a["id"], []))) for a in d["auta"]]


def etykieta_pelna(a):
    if not a:
        return ""
    return a["nazwa"] + (" (" + a["opis"] + ")" if a.get("opis") else "")
