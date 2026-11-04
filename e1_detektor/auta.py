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
    d.setdefault("auto_probki", {})
    for k in ("probki", "auto_probki"):
        d[k] = {a: [x for x in v if len(x) in (NB, NB + NE)] for a, v in d[k].items()}  # stary format cech odpada
    return d


def _zapisz(d):
    tmp = PLIK.with_suffix(".tmp")
    json.dump(d, open(tmp, "w"))
    os.replace(tmp, PLIK)


NB = 17  # 12 przedzialow barwy + 5 poziomow jasnosci dla szarosci/czerni/bieli
NE = 1280  # dlugosc odcisku z sieci (MobileNetV2, uśredniona mapa cech)
EMB_PLIK = Path(os.environ.get("E1_EMB", "/app/emb.onnx"))
EMB_BAZA = 0.5  # cosinus zwykle miedzy roznymi autami ~0.5 -> przeskalowanie do 0..1
AUTO_MAX = 15  # ile probek moze dodac sam system (reczne nigdy nie sa wypierane)
AUTO_PROG = 0.9
PRIOR_PROG = 0.55  # priorytet barwy liczy sie tylko gdy zadne auto z probkami nie pasuje
_emb = {"s": None, "nazwa": None, "probowano": False}


def _sesja():
    if not _emb["probowano"]:
        _emb["probowano"] = True
        try:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = 2
            sess = ort.InferenceSession(str(EMB_PLIK), sess_options=so, providers=["CPUExecutionProvider"])
            _emb["s"], _emb["nazwa"] = sess, sess.get_outputs()[-1].name
            log.info("model odciskow aut zaladowany")
        except Exception as e:
            log.warning("brak modelu odciskow (%s) - tylko kolor", e)
    return _emb["s"]


def odcisk(img, mask=None):
    """Wektor cech z sieci dla obrazu auta (tlo poza maska wyszarzone), L2-znormalizowany; None gdy brak modelu."""
    sess = _sesja()
    if sess is None:
        return None
    try:
        im = img.convert("RGB")
        if mask is not None:
            bg = Image.new("RGB", im.size, (124, 116, 104))
            im = Image.composite(im, bg, mask)
        w, h = im.size
        sc = 224 / max(w, h)
        im = im.resize((max(1, int(w * sc)), max(1, int(h * sc))))
        can = Image.new("RGB", (224, 224), (124, 116, 104))
        can.paste(im, ((224 - im.size[0]) // 2, (224 - im.size[1]) // 2))
        a = (np.asarray(can, dtype=np.float32) / 255.0 - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        out = sess.run([_emb["nazwa"]], {sess.get_inputs()[0].name: a.transpose(2, 0, 1)[None].astype(np.float32)})[0]
        v = np.asarray(out, dtype=np.float32).reshape(-1)
        if v.size != NE:
            log.warning("odcisk ma %d wymiarow, oczekiwano %d", v.size, NE)
            return None
        return (v / (np.linalg.norm(v) + 1e-9)).tolist()
    except Exception:
        log.exception("odcisk")
        return None


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
    img = img.crop((int(w * 0.12), int(h * 0.15), int(w * 0.88), int(h * 0.88)))
    e = odcisk(img)
    hsv = np.asarray(img.resize((64, 40)).convert("HSV"), dtype=np.float32) / 255.0
    return _hist(hsv, np.ones(hsv.shape[:2], bool)) + (e or [])


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
    e = None
    if _sesja() is not None:  # do sieci wiekszy wycinek (min. 160 px dluzszego boku)
        big = img.crop((int(x1), int(y1), int(x2), int(y2)))
        sc2 = min(1.0, 320 / max(big.size))
        big = big.resize((max(1, int(big.size[0] * sc2)), max(1, int(big.size[1] * sc2))))
        m2 = Image.new("L", big.size, 255 if pts is None else 0)
        if pts:
            ImageDraw.Draw(m2).polygon([((px - int(x1)) * sc2, (py - int(y1)) * sc2) for px, py in pts], fill=255)
        e = odcisk(big, m2)
    return _hist(hsv, np.asarray(mask) > 0) + (e or [])


def _inter(a, b):
    return float(np.minimum(np.array(a), np.array(b)).sum())


def _podobienstwo(v, p):
    """Podobienstwo 0..1 miedzy wektorem v a probka p (kolor, a gdy oba maja odcisk - tez ksztalt)."""
    kol = _inter(v[:NB], p[:NB])
    if len(v) == NB + NE and len(p) == NB + NE:
        cs = float(np.dot(v[NB:], p[NB:]))
        ksz = max(0.0, (cs - EMB_BAZA) / (1.0 - EMB_BAZA))
        return 0.75 * ksz + 0.25 * kol
    return kol


def rozpoznaj(path):
    try:
        return rozpoznaj_wektor(cecha(path))
    except Exception:
        log.exception("cecha")
        return None


def _probki(d, k, v):
    """Wszystkie probki auta k (reczne + automatyczne); gdy wektor ma odcisk, preferuje probki z odciskiem."""
    pr = d["probki"].get(k, []) + d["auto_probki"].get(k, [])
    if len(v) == NB + NE:
        z = [p for p in pr if len(p) == NB + NE]
        if z:
            return z
    return pr


def rozpoznaj_wektor(v, uczen=True):
    """Zwraca dict auta (id, nazwa, opis) albo None. Wymaga wyraznej przewagi nad pozostalymi autami.
    Przy bardzo pewnym rozpoznaniu system sam dodaje probke (osobna, ograniczona pula)."""
    d = _dane()
    scores = []  # (wynik, auto lub None dla "obce")
    ob = [_podobienstwo(v, p) for p in _probki(d, OBCE, v)]
    if ob:
        scores.append((max(ob), None))
    bez = []  # auta bez probek - oceniane po priorytecie barwy, ale tylko gdy zadne auto z probkami nie pasuje
    for a in d["auta"]:
        pr = _probki(d, a["id"], v)
        if pr:
            scores.append((max(_podobienstwo(v, p) for p in pr), a))
        elif a.get("prior"):
            bez.append((sum(v[i] for i in a["prior"]), a))
    if not scores or max(x for x, _ in scores) < PRIOR_PROG:
        scores += bez
    if not scores:
        return None
    scores.sort(key=lambda x: -x[0])
    bs, best = scores[0]
    second = scores[1][0] if len(scores) > 1 else 0.0
    log.info("wyniki: %s emb=%s", [(a['id'] if a else 'obce', round(x, 2)) for x, a in scores], len(v) == NB + NE)
    if best is None or bs < PROG or (bs - second) < (MARGINES if bs < 0.85 else 0.03):
        log.info("nierozpoznane: %s", [(a['id'] if a else 'obce', round(x, 2)) for x, a in scores])
        return None
    if uczen and len(v) == NB + NE and bs >= AUTO_PROG and (bs - second) >= MARGINES and bs < 0.985:
        ap = d["auto_probki"].setdefault(best["id"], [])
        ap.append(v)
        del ap[:-AUTO_MAX]
        _zapisz(d)
        log.info("auto-probka dla %s (wynik %.2f)", best["id"], bs)
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
    d["auto_probki"].pop(auto_id, None)
    for a in d["auta"]:
        if a["id"] == auto_id and opis is not None:
            a["opis"] = opis
    _zapisz(d)


def lista():
    d = _dane()
    return [dict(a, probek=len(d["probki"].get(a["id"], [])), auto_probek=len(d["auto_probki"].get(a["id"], []))) for a in d["auta"]]


def etykieta_pelna(a):
    if not a:
        return ""
    return a["nazwa"] + (" (" + a["opis"] + ")" if a.get("opis") else "")
