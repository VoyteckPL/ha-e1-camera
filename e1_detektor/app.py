"""E1 Kamera - lokalny detektor.

Przyjmuje od Home Assistanta nazwę zdarzenia, znajduje na zdjęciu (a gdy go tam nie
ma, na klatkach z nagrania) osobę, samochód albo zwierzę, wycina kadr z marginesem
i przygotowuje wideo do odtwarzania w przeglądarce. Wszystko lokalnie (ONNX, YOLOX).
"""
import asyncio
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import onnxruntime as ort
from aiohttp import web
from PIL import Image

import auta
import bufor
import reolink
import strefy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("e1")

OPTS = {}
try:
    OPTS = json.load(open("/data/options.json"))
except Exception:
    pass

MODEL_NAME = OPTS.get("model", "yolox_s")
THREADS = int(OPTS.get("threads", 4))
MARGIN = float(OPTS.get("margin", 0.2))
MIN_CONF = float(OPTS.get("min_conf", 0.35))
KEEP_DAYS = int(OPTS.get("keep_days", 7))

HA = Path(os.environ.get("E1_HA", "/ha_config"))
MEDIA = Path(os.environ.get("E1_MEDIA", "/media"))
SNAP = HA / "www" / "snapshots"
MODEL_DIR = Path(os.environ.get("E1_MODELS", "/app"))

# klasy COCO
KLASY = {
    "osoba": [0],
    "pojazd": [2, 3, 5, 7],
    "zwierze": [14, 15, 16, 17, 18, 19],
}
RODZAJ = {"osoba": "osoba", "pojazd": "pojazd", "zwierzę": "zwierze", "zwierze": "zwierze"}

so = ort.SessionOptions()
so.intra_op_num_threads = THREADS
so.inter_op_num_threads = 1
SESS = ort.InferenceSession(str(MODEL_DIR / f"{MODEL_NAME}.onnx"), so, providers=["CPUExecutionProvider"])
INP = SESS.get_inputs()[0]
IN_H, IN_W = int(INP.shape[2]), int(INP.shape[3])
log.info("model %s wejscie %sx%s, watki %s", MODEL_NAME, IN_W, IN_H, THREADS)
SESS_T = None  # lekki model do przegladania klatek z bufora
try:
    if MODEL_NAME != "yolox_tiny" and (MODEL_DIR / "yolox_tiny.onnx").exists():
        SESS_T = ort.InferenceSession(str(MODEL_DIR / "yolox_tiny.onnx"), so, providers=["CPUExecutionProvider"])
        log.info("model skanu klatek: yolox_tiny")
except Exception:
    SESS_T = None

LOCK = asyncio.Lock()
GLOCK = asyncio.Lock()  # poprawa kadru z klipu: po jednym zdarzeniu naraz
LAST_B = {}  # rodzaj -> ostatnie zdarzenie (nie nadpisuj nowszego powiadomienia starszym)


def _grids(IN_H=None, IN_W=None):
    IN_H = IN_H or globals()["IN_H"]
    IN_W = IN_W or globals()["IN_W"]
    gs, st = [], []
    for s in (8, 16, 32):
        h, w = IN_H // s, IN_W // s
        xv, yv = np.meshgrid(np.arange(w), np.arange(h))
        g = np.stack((xv, yv), 2).reshape(-1, 2)
        gs.append(g)
        st.append(np.full((g.shape[0], 1), s))
    return np.concatenate(gs, 0).astype(np.float32), np.concatenate(st, 0).astype(np.float32)


GRID, STRIDE = _grids()
_GR = {}


def _siatka(h, w):
    if (h, w) == (IN_H, IN_W):
        return GRID, STRIDE
    if (h, w) not in _GR:
        _GR[(h, w)] = _grids(h, w)
    return _GR[(h, w)]


def nms(boxes, scores, thr=0.45):
    idx = scores.argsort()[::-1]
    keep = []
    while idx.size:
        i = idx[0]
        keep.append(i)
        if idx.size == 1:
            break
        rest = idx[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        a1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        a2 = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (a1 + a2 - inter + 1e-9)
        idx = rest[iou <= thr]
    return keep


def detect(img: Image.Image, klasy, conf_thr, sess=None):
    """Zwraca listę (x1, y1, x2, y2, conf) w pikselach obrazu."""
    ses = sess or SESS
    inp = ses.get_inputs()[0]
    th, tw = int(inp.shape[2]), int(inp.shape[3])
    grid, stride = _siatka(th, tw)
    iw, ih = img.size
    r = min(tw / iw, th / ih)
    nw, nh = max(1, int(iw * r)), max(1, int(ih * r))
    small = img.convert("RGB").resize((nw, nh), Image.BILINEAR)
    canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
    canvas[:nh, :nw] = np.asarray(small)
    x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32)  # BGR, CHW
    out = ses.run(None, {inp.name: x})[0][0]
    out[:, :2] = (out[:, :2] + grid) * stride
    out[:, 2:4] = np.exp(out[:, 2:4]) * stride
    obj = out[:, 4]
    cls = out[:, 5:]
    sc = cls[:, klasy] * obj[:, None]
    best = sc.max(1)
    m = best >= conf_thr
    if not m.any():
        return []
    o, b = out[m], best[m]
    xyxy = np.stack((o[:, 0] - o[:, 2] / 2, o[:, 1] - o[:, 3] / 2, o[:, 0] + o[:, 2] / 2, o[:, 1] + o[:, 3] / 2), 1) / r
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, iw)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, ih)
    keep = nms(xyxy, b)
    return [(*map(float, xyxy[i]), float(b[i])) for i in keep]


def box_score(box, iw, ih):
    x1, y1, x2, y2, c = box
    frac = max(0.0, (x2 - x1) * (y2 - y1)) / (iw * ih)
    size = 0.5 + 0.5 * min(1.0, math.sqrt(frac) * 3)
    edge = 0.7 if (x1 < 3 or y1 < 3 or x2 > iw - 3 or y2 > ih - 3) else 1.0
    return c * size * edge


def wybierz_kadr(boxes, iw, ih):
    """Z wykrytych ramek (jedna klatka) wybiera kadr: najlepsza albo suma bliskich."""
    best = max(boxes, key=lambda b: box_score(b, iw, ih))
    sel = [b for b in boxes if b[4] >= max(MIN_CONF, best[4] * 0.6)]
    if len(sel) >= 2:
        u = (min(b[0] for b in sel), min(b[1] for b in sel), max(b[2] for b in sel), max(b[3] for b in sel))
        if (u[2] - u[0]) * (u[3] - u[1]) <= 0.4 * iw * ih:
            return best, u, len(sel)
    return best, best[:4], 1


STAT_PLIK = Path(os.environ.get("E1_DATA", "/data")) / "statyczne.json"
STAT_MIN = 3  # tyle wczesniejszych zdarzen z ramka w tym samym miejscu = obiekt stoi (np. zaparkowane auto)
_stat = None


def _stat_lista():
    global _stat
    if _stat is None:
        try:
            _stat = json.load(open(STAT_PLIK))
        except Exception:
            _stat = []
    return _stat


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def _norm(bx, iw, ih):
    return [bx[0] / iw, bx[1] / ih, bx[2] / iw, bx[3] / ih]


def ruchome(boxes, iw, ih, b):
    """Ramki pojazdow bez tych, ktore w tym samym miejscu byly w >= STAT_MIN wczesniejszych zdarzeniach (12 h)."""
    now = time.time()
    hist = [h for h in _stat_lista() if now - h["t"] < 12 * 3600 and h["b"] != b]
    out = []
    for bx in boxes:
        n = _norm(bx, iw, ih)
        if len({h["b"] for h in hist if _iou(n, h["box"]) > 0.8}) < STAT_MIN:
            out.append(bx)
    return out


def zapamietaj(boxes, iw, ih, b):
    global _stat
    now = time.time()
    s = [h for h in _stat_lista() if now - h["t"] < 24 * 3600 and h["b"] != b]
    s += [{"t": now, "b": b, "box": _norm(bx, iw, ih)} for bx in boxes]
    _stat = s[-400:]
    try:
        tmp = STAT_PLIK.with_suffix(".tmp")
        json.dump(_stat, open(tmp, "w"))
        os.replace(tmp, STAT_PLIK)
    except Exception:
        log.exception("zapis statycznych")


def zapisz_kadr(img, rect, out_path):
    iw, ih = img.size
    x1, y1, x2, y2 = rect
    w, h = x2 - x1, y2 - y1
    mx, my = w * MARGIN, h * MARGIN
    box = (
        int(max(0, x1 - mx)),
        int(max(0, y1 - my)),
        int(min(iw, x2 + mx)),
        int(min(ih, y2 + my)),
    )
    # maly obiekt (daleko): wycinek co najmniej ~20% szerokosci kadru, zeby nie byl skrajnym, rozmytym powiekszeniem
    mn = int(iw * 0.2)
    bx0, by0, bx1, by1 = box
    if bx1 - bx0 < mn:
        c = (bx0 + bx1) / 2
        bx0, bx1 = int(max(0, min(c - mn / 2, iw - mn))), int(min(iw, max(c + mn / 2, mn)))
    if by1 - by0 < mn:
        c = (by0 + by1) / 2
        by0, by1 = int(max(0, min(c - mn / 2, ih - mn))), int(min(ih, max(c + mn / 2, mn)))
    box = (bx0, by0, bx1, by1)
    crop = img.convert("RGB").crop(box)
    crop.save(out_path, "JPEG", quality=90)
    return box


def ffprobe_ok(path: Path):
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,codec_type", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode != 0:
            return None
        return json.loads(r.stdout).get("streams", [])
    except Exception:
        return None


async def czekaj_na_klip(b, limit):
    clip = MEDIA / f"e1_rec_{b}.mp4"
    end = time.time() + limit
    last = -1
    while time.time() < end:
        if clip.exists():
            sz = clip.stat().st_size
            if sz > 0 and sz == last and ffprobe_ok(clip):
                return clip
            last = sz
        await asyncio.sleep(1.0)
    return clip if clip.exists() and ffprobe_ok(clip) else None


def rowne_klatki_bsf(clip: Path):
    """Kamera daje 25 kl./s, ale w pliku znaczniki czasu leza na siatce 30 kl./s (odstepy 33/67 ms) - widac to jako mikro-przyciecia.
    Zwraca filtr przepisujacy czas na rowne 40 ms (bez przekodowania); tylko dla wideo bez klatek B i ~25 kl./s."""
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=has_b_frames,nb_read_packets:format=duration",
                            "-count_packets", "-of", "json", str(clip)], capture_output=True, text=True, timeout=60)
        j = json.loads(r.stdout)
        st = j["streams"][0]
        n, dur = int(st["nb_read_packets"]), float(j["format"]["duration"])
        if int(st.get("has_b_frames", 0)) == 0 and dur > 1 and 24.0 <= n / dur <= 26.0:
            return "setts=ts=N/(25*TB)"
    except Exception:
        pass
    return None


def przygotuj_wideo_sync(clip: Path, out: Path):
    streams = ffprobe_ok(clip) or []
    vid = next((s for s in streams if s.get("codec_type") == "video"), {})
    tmp = out.with_suffix(".tmp.mp4")
    if vid.get("codec_name") == "h264":
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(clip), "-map", "0:v:0", "-map", "0:a:0?",
               "-c:v", "copy", "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart", str(tmp)]
        _bsf = rowne_klatki_bsf(clip)
        if _bsf:
            cmd[cmd.index("-c:v") + 2:cmd.index("-c:v") + 2] = ["-bsf:v", _bsf]
    else:
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(clip), "-map", "0:v:0", "-map", "0:a:0?",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "27", "-vf", "scale=-2:720,format=yuv420p",
               "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart", str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        log.error("ffmpeg wideo: %s", r.stderr[-300:])
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(out)
    return True


async def sasiedni_klip(t0, limit):
    """Gdy kamera nie nagrala wlasnego klipu (rownolegle zdarzenia), uzyj klipu z tego samego momentu."""
    end = time.time() + limit
    while time.time() < end:
        best = None
        for f in MEDIA.glob("e1_rec_*.mp4"):
            try:
                if t0 - 20 <= f.stat().st_mtime <= t0 + 30 and ffprobe_ok(f):
                    if best is None or f.stat().st_mtime > best.stat().st_mtime:
                        best = f
            except OSError:
                pass
        if best:
            return best
        await asyncio.sleep(2.0)
    return None


async def przygotuj_wideo(b):
    try:
        t0 = time.time()
        clip = await czekaj_na_klip(b, 45)
        if not clip:
            clip = await sasiedni_klip(t0, 20)
            if clip:
                log.info("klip sasiedni dla %s: %s", b, clip.name)
        if not clip:
            log.warning("brak klipu dla %s", b)
            return
        out = SNAP / f"e1_v_{b}.mp4"
        if b in reolink.GOTOWE:
            return
        ok = await asyncio.get_running_loop().run_in_executor(None, przygotuj_wideo_sync, clip, out)
        if ok:
            try:
                dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)], capture_output=True, text=True, timeout=20).stdout.strip()
            except Exception:
                dur = "?"
            log.info("wideo gotowe %s (%.1f MB, %s s)", out.name, out.stat().st_size / 1e6, dur)
    except Exception:
        log.exception("wideo %s", b)


def analiza_sync(b, plik, rodzaj, granica, klasy):
    t0 = time.time()
    SNAP.mkdir(parents=True, exist_ok=True)
    snap = SNAP / plik
    wynik = None  # (img, boxes, nazwa_pliku_pelnego)
    zapas = None  # gdy na zdjeciu sa tylko stojace pojazdy: uzywane, jesli klip nic ruchomego nie da
    if snap.exists():
        img = Image.open(snap)
        img.load()
        boxes = detect(img, klasy, MIN_CONF)
        if boxes:
            if klasy == KLASY["pojazd"] and not ruchome(boxes, *img.size, b):
                zapas = (img, boxes, plik)
            else:
                wynik = (img, boxes, plik)
    return wynik, zapas


def analiza_klip_sync(clip: Path, b, klasy):
    tmpd = tempfile.mkdtemp(prefix="e1_")
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(clip), "-vf", "fps=0.8", "-q:v", "3", f"{tmpd}/f_%02d.jpg"],
            capture_output=True, text=True, timeout=120,
        )
        files = sorted(Path(tmpd).glob("f_*.jpg"))
        best = None
        zapas = None  # najlepsza klatka tylko ze stojacymi pojazdami
        for f in files:
            img = Image.open(f)
            img.load()
            boxes = detect(img, klasy, MIN_CONF)
            surowe = boxes
            if boxes and klasy == KLASY["pojazd"]:
                boxes = ruchome(boxes, *img.size, b)
                if not boxes:
                    sc = max(box_score(x, *img.size) for x in surowe)
                    if zapas is None or sc > zapas[0]:
                        zapas = (sc, img, surowe, surowe)
                    continue
            if not boxes:
                continue
            sc = max(box_score(x, *img.size) for x in boxes)
            if best is None or sc > best[0]:
                best = (sc, img, boxes, surowe)
        if best is None:
            best = zapas
        if best:
            if klasy == KLASY["pojazd"]:
                zapamietaj(best[3], *best[1].size, b)
            nazwa = f"e1_best_{b}.jpg"
            best[1].convert("RGB").save(SNAP / nazwa, "JPEG", quality=90)
            return best[1], best[2], nazwa
        return None
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)


_TEST = {}


def _cel_drogi(cfg):
    """Srodek strefy 'Droga' (0..1); bez takiej strefy - srodek kadru."""
    for z in cfg.get("strefy", []):
        if "droga" in z.get("nazwa", "").lower() and z.get("pts"):
            xs = [p[0] for p in z["pts"]]
            ys = [p[1] for p in z["pts"]]
            return ((min(xs) + max(xs)) / 2000.0, (min(ys) + max(ys)) / 2000.0)
    return (0.5, 0.5)


PRZED_S = 8.0
KIER = {}  # b -> 'prawo' / 'lewo' / '?' (kierunek jazdy w kadrze)
PEND = {}  # b -> (czas klatki, mapa ruchu, ramka) do podmiany na pelna klatke z nagrania kamery


EXTRA = {}  # b -> lista dodatkowych aut jadacych droga w tym samym zdarzeniu [{key, ts, kier}]


def _dodatkowe_auta(b, files, male, bg, od, FPS, kref, n_main, granica, scan, cel):
    """Sledzi plamy ruchu w strefie Droga na klatkach lekkiego strumienia; kazdy tor poza torem glownego auta to osobny przejazd
    (dwa auta w kadrze naraz albo jedno po drugim w jednym zdarzeniu). Zwraca [{key, ts, kier}], zapisuje zdjecie i wycinek kazdego."""
    import cv2
    cfg = strefy.wczytaj(granica)
    droga = next((z["pts"] for z in cfg.get("strefy", []) if "droga" in z.get("nazwa", "").lower() and z.get("pts")), None)
    if not droga:
        return []
    kern = np.ones((3, 3), np.uint8)
    tory = []
    for k, a in enumerate(male):
        m = (np.abs(a - bg) > 28).astype(np.uint8)
        if not (0.004 <= float(m.mean()) <= 0.6):
            continue
        nl, _lab, st, cen = cv2.connectedComponentsWithStats(cv2.dilate(m, kern), connectivity=8)
        for i in range(1, nl):
            x, y, w, h, ar = [int(v) for v in st[i]]
            if ar < 25 or not strefy.w_wielokacie((x + w / 2) / 80 * 1000, (y + h) / 60 * 1000, droga):
                continue
            bl = (k, float(cen[i][0]) / 80, float(cen[i][1]) / 60, ar, (x / 80, y / 60, (x + w) / 80, (y + h) / 60))
            best, bd = None, 0.3
            for t in tory:
                if t[-1][0] < k and k - t[-1][0] <= 3:
                    d = math.hypot(bl[1] - t[-1][1], (bl[2] - t[-1][2]) * 0.7)
                    if d < bd:
                        best, bd = t, d
            if best is not None:
                best.append(bl)
            else:
                tory.append([bl])
    log.info("bufor: %s surowe tory w drodze: %d", b, len(tory))
    tory = [t for t in tory if len(t) >= 3 or (len(t) >= 2 and abs(t[-1][1] - t[0][1]) >= 0.05)]
    if not tory:
        return []
    mx, my = (n_main[0] + n_main[2]) / 2, (n_main[1] + n_main[3]) / 2
    glowny, gd = None, 0.2
    for ti, t in enumerate(tory):
        for bl in t:
            if abs(bl[0] - kref) <= 1:
                d = math.hypot(bl[1] - mx, (bl[2] - my) * 0.7)
                if d < gd:
                    glowny, gd = ti, d
    log.info("bufor: %s tory ruchu w drodze: %s, glowny=%s", b, [(len(t), round(t[0][0] / FPS, 1), round(t[-1][0] / FPS, 1), round(t[0][1], 2), round(t[-1][1], 2)) for t in tory], glowny)
    wynik = []
    st160 = None
    for ti, t in enumerate(tory):
        if ti == glowny or len(wynik) >= 3:
            continue
        bl = max(t, key=lambda q: q[3] * max(0.15, 1.0 - math.hypot(q[1] - cel[0], (q[2] - cel[1]) * 0.7) * 1.4))
        k = bl[0]
        img = Image.open(files[k])
        img.load()
        img = img.convert("RGB")
        iw, ih = img.size
        bb = bl[4]
        ex, ey = (bb[2] - bb[0]) * 0.15, (bb[3] - bb[1]) * 0.15
        bbx = (bb[0] - ex, bb[1] - ey, bb[2] + ex, bb[3] + ey)
        wyb = None
        for bx in detect(img, KLASY["pojazd"], MIN_CONF * 0.9, sess=scan):
            nb = _norm(bx, iw, ih)
            cxb, cyb = (nb[0] + nb[2]) / 2, (nb[1] + nb[3]) / 2
            if _iou(nb, bbx) > 0.15 or (bbx[0] <= cxb <= bbx[2] and bbx[1] <= cyb <= bbx[3]):
                if wyb is None or _iou(nb, bbx) > _iou(_norm(wyb, iw, ih), bbx):
                    wyb = bx
        if wyb is None:
            continue
        nb = _norm(wyb, iw, ih)
        if abs(k - kref) <= 4 and _iou(nb, n_main) > 0.3:
            continue  # to samo auto co glowne
        if not strefy.w_wielokacie((nb[0] + nb[2]) / 2 * 1000, nb[3] * 1000, droga):
            continue
        if st160 is None:
            st160 = np.stack([np.asarray(Image.open(f).convert("L").resize((160, 120)), dtype=np.float32) for f in files])
            med160 = np.median(st160, 0)
        key = f"{b}_x{len(wynik) + 1}"
        fg_ref = np.clip(np.abs(st160[k] - med160) / 64.0, 0, 1)
        img.save(SNAP / f"e1_best_{key}.jpg", quality=92)
        _best, rect, _n = wybierz_kadr([wyb], iw, ih)
        zapisz_kadr(img, rect, SNAP / f"e1_crop_{key}.jpg")
        ts_k = od + k / FPS
        KIER[key] = kierunek_z_ruchu([(q[0], q[1], q[2]) for q in t])
        PEND[key] = (ts_k, fg_ref, list(nb))
        wynik.append({"key": key, "ts": ts_k, "kier": KIER[key]})
        log.info("bufor: %s dodatkowe auto %s @%.1f s, kierunek %s, tor %d klatek", b, key, ts_k - (od + PRZED_S), KIER[key], len(t))
    return wynik


def najlepsza_klatka_sync(b, klasy, granica, t0=None, extra=False):
    """Szuka na lekkim strumieniu (bufor 's') chwili, gdy pojazd jest najblizej srodka drogi, a pelna klatke bierze z glownego."""
    if t0 is None:
        t = reolink.czas_zdarzenia(b)
        if t is None:
            return None
        t0 = time.mktime(t.timetuple())
    tmpd = tempfile.mkdtemp(prefix="e1b_")
    ts = time.time()
    FPS = 3
    try:
        pojazd = klasy == KLASY["pojazd"]
        # zdarzenie z kamery przychodzi z opoznieniem - auto bywa juz w kadrze (lub za nim) przed t0; osoby i zwierzeta ida wolniej, okno krotsze
        od = t0 - (PRZED_S if pojazd else 3.0)
        do = min(t0 + (5.5 if pojazd else 3.5), time.time() - 3.0)
        files = bufor.klatki(od, do, FPS, tmpd, pre="s")
        if not files:
            return None
        cx, cy = _cel_drogi(strefy.wczytaj(granica)) if pojazd else (0.5, 0.55)
        wag = 1.4 if pojazd else 0.7  # jak bardzo liczy sie bycie na srodku (auto: srodek drogi, osoba/zwierze: glownie cala w kadrze)
        scan = SESS_T or SESS
        # 1) tani wstepny wybor: ruch wzgledem tla (mediana klatek) - ktore klatki maja ruchomy obiekt najblizej srodka drogi
        male = []
        for f in files:
            im = Image.open(f)
            im.load()
            male.append(np.asarray(im.convert("L").resize((80, 60)), dtype=np.float32))
        bg = np.median(np.stack(male), 0)
        ocena_ruchu = []
        traj = []
        for k, a in enumerate(male):
            m = np.abs(a - bg) > 28
            ar = float(m.mean())
            if ar < 0.004 or ar > 0.6:
                continue
            ys, xs = np.nonzero(m)
            mx, my = float(xs.mean()) / 80, float(ys.mean()) / 60
            traj.append((k, mx, my))
            ocena_ruchu.append((min(ar * 20, 1.0) * max(0.15, 1.0 - math.hypot(mx - cx, (my - cy) * 0.7) * wag), k))
        ocena_ruchu.sort(reverse=True)
        KIER[b] = kierunek_z_ruchu(traj)
        wybor = []
        for _, k in ocena_ruchu:
            if all(abs(k - j) >= 2 for j in wybor):
                wybor.append(k)
            if len(wybor) >= 5:
                break
        if not wybor:  # brak wyraznego ruchu: kilka rownomiernie rozlozonych klatek
            wybor = sorted({int(i * (len(files) - 1) / 4) for i in range(5)})
        kand = []
        for k in wybor:
            img = Image.open(files[k])
            img.load()
            iw, ih = img.size
            boxes = detect(img, klasy, MIN_CONF * 0.9, sess=scan)
            if klasy == KLASY["pojazd"]:
                boxes = ruchome(boxes, iw, ih, b)
            for bx in boxes:
                n = _norm(bx, iw, ih)
                mx, my = (n[0] + n[2]) / 2, (n[1] + n[3]) / 2
                dist = math.hypot(mx - cx, (my - cy) * 0.7)
                kand.append((box_score(bx, iw, ih) * max(0.15, 1.0 - dist * wag), od + k / FPS, n, bx[4], k))
        log.info("bufor: %s skan %d klatek lekkiego strumienia, detekcja %d (%.1f s), kandydatow %d", b, len(files), len(wybor), time.time() - ts, len(kand))
        if not kand:
            return None
        kand.sort(key=lambda x: -x[0])
        _, tc, n, conf, kref = kand[0]
        pf = SNAP / f"e1_best_{b}.jpg"
        st160 = np.stack([np.asarray(Image.open(f).convert("L").resize((160, 120)), dtype=np.float32) for f in files])
        fg_ref = np.clip(np.abs(st160[kref] - np.median(st160, 0)) / 64.0, 0, 1)
        # glowny strumien z bufora bywa rozjechany czasowo (i nie zawsze ma auto) - obraz bierzemy z lekkiego, a po nagraniu z kamery podmieniamy na pelna jakosc
        img = Image.open(files[kref])
        img.load()
        img = img.convert("RGB")
        img.save(pf, quality=92)
        PEND[b] = (tc, fg_ref, list(n))
        iw, ih = img.size
        wybrany = None
        # pelny model na pelnej klatce: ramka najblizsza tej z lekkiego strumienia
        boxes = detect(img, klasy, MIN_CONF)
        surowe = boxes
        if klasy == KLASY["pojazd"]:
            boxes = ruchome(boxes, iw, ih, b)
        sx = [n[0] * iw, n[1] * ih, n[2] * iw, n[3] * ih]
        if boxes:
            kb = max(boxes, key=lambda x: _iou(x[:4], sx))
            if _iou(kb[:4], sx) > 0.25:
                wybrany = kb
        if wybrany is None:
            wybrany = (*sx, conf)  # zapas: ramka z lekkiego strumienia przeskalowana na pelna klatke
        if klasy == KLASY["pojazd"]:
            zapamietaj(surowe or [wybrany], iw, ih, b)
        log.info("bufor: %s najlepsza klatka @%.1f s po zdarzeniu, razem %.1f s, conf %.2f", b, tc - t0, time.time() - ts, wybrany[4])
        if extra and pojazd:
            try:
                EXTRA[b] = _dodatkowe_auta(b, files, male, bg, od, FPS, kref, list(n), granica, scan, (cx, cy))
            except Exception:
                log.exception("dodatkowe auta %s", b)
        return img, [wybrany], pf.name
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)


def kierunek_z_ruchu(traj):
    """traj: (nr klatki, x, y) srodka ruchu; zwraca 'prawo' / 'lewo' / '?' (w kadrze, w prawo = z lewej do prawej)."""
    if len(traj) < 3:
        return "?"
    traj = sorted(traj)
    n = max(1, len(traj) // 3)
    d = sum(t[1] for t in traj[-n:]) / n - sum(t[1] for t in traj[:n]) / n
    return "prawo" if d > 0.06 else "lewo" if d < -0.06 else "?"


# ---------------- zakladka Przejazdy: lista aut jadacych droga + rozpoznanie przez Gemini ----------------
PZ_PLIK = Path("/data/przejazdy.json")
PZ_LOCK = asyncio.Lock()
PZ_DNI = 30
PZ_STRUKTURA = {
    "marka": {"description": "Marka samochodu (np. Skoda, Volkswagen). Gdy nie da sie ustalic: nieznana.", "required": True, "selector": {"text": {}}},
    "model": {"description": "Najbardziej prawdopodobny model i generacja (np. Octavia III). Gdy nie da sie ustalic: nieznany.", "required": True, "selector": {"text": {}}},
    "kolor": {"description": "Kolor nadwozia po polsku (np. srebrny, granatowy).", "required": True, "selector": {"text": {}}},
    "typ": {"description": "Typ pojazdu.", "required": True, "selector": {"select": {"options": ["osobowy", "kombi", "SUV", "dostawczy", "ciezarowy", "autobus", "motocykl", "inny"]}}},
    "pewnosc": {"description": "Pewnosc rozpoznania marki i modelu, 0-100.", "required": True, "selector": {"number": {"min": 0, "max": 100}}},
    "tablica": {"description": "Numer rejestracyjny TYLKO jesli jest w pelni czytelny na zdjeciu. W przeciwnym razie pusty tekst. Nie zgaduj.", "required": False, "selector": {"text": {}}},
}


def pz_wczytaj():
    try:
        return json.loads(PZ_PLIK.read_text())
    except Exception:
        return []


def pz_zapisz(lista):
    PZ_PLIK.parent.mkdir(parents=True, exist_ok=True)
    gr = time.time() - PZ_DNI * 86400
    lista = [e for e in lista if e.get("ts", 0) >= gr][-1000:]
    tmp = PZ_PLIK.with_suffix(".tmp")
    tmp.write_text(json.dumps(lista, ensure_ascii=False))
    tmp.replace(PZ_PLIK)


async def pz_ustaw(b, **pola):
    async with PZ_LOCK:
        lista = pz_wczytaj()
        for e in lista:
            if e.get("b") == b:
                e.update(pola)
                break
        else:
            lista.append(dict(b=b, **pola))
        pz_zapisz(lista)


def pz_miniatura(b):
    """Mala miniatura trzymana dluzej niz zdjecia zdarzen (do tabeli)."""
    src = SNAP / f"e1_crop_{b}.jpg"
    if not src.exists():
        return False
    im = Image.open(src)
    im.load()
    im = im.convert("RGB")
    im.thumbnail((360, 360))
    im.save(SNAP / f"e1_pz_{b}.jpg", "JPEG", quality=82)
    return True


def pz_dla_ai(b):
    """Kopia wycinka w /media (tam widzi go HA) w rozsadnym rozmiarze; zwraca media_content_id."""
    src = SNAP / f"e1_crop_{b}.jpg"
    im = Image.open(src)
    im.load()
    im = im.convert("RGB")
    im.thumbnail((1024, 1024))
    d = MEDIA / "e1_ai"
    d.mkdir(parents=True, exist_ok=True)
    im.save(d / f"{b}.jpg", "JPEG", quality=88)
    return f"media-source://media_source/local/e1_ai/{b}.jpg"


async def przejazd_ai(b):
    mid = await asyncio.get_running_loop().run_in_executor(None, pz_dla_ai, b)
    zad = (
        "Na zdjeciu z kamery monitoringu jest samochod przejezdzajacy ulica. Rozpoznaj go. "
        "Jesli widac kilka aut, opisz to na srodku/najwieksze. Zgaduj najbardziej prawdopodobny model, a pewnosc oceniaj uczciwie."
    )
    async with reolink.HA() as ha:
        r = await ha.cmd({
            "type": "call_service", "domain": "ai_task", "service": "generate_data", "return_response": True,
            "service_data": {
                "entity_id": "ai_task.google_ai_task", "task_name": "przejazd", "instructions": zad,
                "structure": PZ_STRUKTURA,
                "attachments": [{"media_content_id": mid, "media_content_type": "image/jpeg"}],
            },
        })
    return (r.get("response") or {}).get("data") or {}


async def przejazd_wpis(b, czekaj=True, ts=None):
    """Dodaje przejazd do tabeli od razu (godzina, kierunek), potem dopisuje rozpoznanie z Gemini na najlepszym dostepnym wycinku."""
    if ts is None:
        t = reolink.czas_zdarzenia(b)
        ts = time.mktime(t.timetuple()) if t else time.time()
    await pz_ustaw(b, ts=ts, czas=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)), kier=KIER.get(b, "?"), crop=f"e1_crop_{b}.jpg", status="analiza")
    if czekaj:
        for _ in range(45):  # do ~90 s: czekamy na wycinek w pelnej jakosci z nagrania kamery
            if b not in PEND:
                break
            await asyncio.sleep(2)
    try:
        await asyncio.get_running_loop().run_in_executor(None, pz_miniatura, b)
        d = await przejazd_ai(b)
        await pz_ustaw(b, marka=str(d.get("marka", "")), model=str(d.get("model", "")), kolor=str(d.get("kolor", "")), typ=str(d.get("typ", "")),
                       tablica=re.sub(r"[^A-Z0-9]", "", str(d.get("tablica", "") or "").upper())[:9],
                       pewnosc=int(float(d.get("pewnosc", 0) or 0)), status="ok", mini=f"e1_pz_{b}.jpg")
        log.info("przejazd %s: %s", b, d)
    except Exception as e:
        log.exception("przejazd_ai %s", b)
        await pz_ustaw(b, status="blad", blad=str(e)[:120], mini=f"e1_pz_{b}.jpg" if (SNAP / f"e1_pz_{b}.jpg").exists() else "")


def dopieszcz_sync(b, klasy, k=None):
    """Po nagraniu z kamery: znajduje w nim klatke z autem tak jak na wybranej klatce lekkiego strumienia i podmienia snap/wycinek na pelna jakosc."""
    k = k or b
    tc, fg_ref, n = PEND.pop(k)
    fs = reolink.START.get(b)
    clip = SNAP / f"e1_v_{b}.mp4"
    if fs is None or not clip.exists():
        return False
    off = tc - time.mktime(fs.timetuple())
    pol, fps = 3.0, 6
    od = max(0.0, off - pol)
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{od:.2f}", "-t", f"{2 * pol:.2f}", "-i", str(clip),
                        "-vf", f"fps={fps},scale=160:120,format=gray", "-f", "rawvideo", "pipe:1"], capture_output=True, timeout=120)
    m = len(r.stdout) // 19200
    if m < 3:
        return False
    arr = np.frombuffer(r.stdout[:m * 19200], np.uint8).reshape(m, 120, 160).astype(np.float32)
    bg = np.median(arr, 0)
    x0, y0, x1, y1 = n
    ex, ey = max(0.08, (x1 - x0) * 0.8), max(0.08, (y1 - y0) * 0.8)
    ax0, ax1 = int(max(0.0, x0 - ex) * 160), int(min(1.0, x1 + ex) * 160) + 1
    ay0, ay1 = int(max(0.0, y0 - ey) * 120), int(min(1.0, y1 + ey) * 120) + 1
    ref = fg_ref[ay0:ay1, ax0:ax1]
    nr = float(np.sqrt((ref * ref).sum())) + 1e-6
    pod = []
    for fr in arr:
        fg = np.clip(np.abs(fr - bg) / 64.0, 0, 1)[ay0:ay1, ax0:ax1]
        pod.append(float((fg * ref).sum() / (np.sqrt((fg * fg).sum()) * nr + 1e-6)))
    i = int(np.argmax(pod))
    log.info("dopieszczanie %s: klatek %d, najlepsza %d (%+.1f s od oczekiwanej), podobienstwo %.2f", k, m, i, od + i / fps - off, pod[i])
    if pod[i] < 0.4:
        return False
    pf = SNAP / f"e1_best_{k}.jpg"
    tmp = SNAP / f"e1_best_{k}.tmp.jpg"
    r2 = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{od + i / fps:.2f}", "-i", str(clip), "-frames:v", "1", "-q:v", "2", str(tmp)], capture_output=True, timeout=120)
    if r2.returncode != 0 or not tmp.exists() or tmp.stat().st_size < 1000:
        return False
    img = Image.open(tmp)
    img.load()
    img = img.convert("RGB")
    iw, ih = img.size
    sx = [x0 * iw, y0 * ih, x1 * iw, y1 * ih]
    boxes = detect(img, klasy, MIN_CONF)
    boxes = ruchome(boxes, iw, ih, b) or boxes
    wyb = None
    if boxes:
        kb = max(boxes, key=lambda x: _iou(x[:4], sx))
        if _iou(kb[:4], sx) > 0.25:
            wyb = kb
    if wyb is None:
        wyb = (*sx, 0.5)
    best, rect, _n = wybierz_kadr([wyb], iw, ih)
    zapisz_kadr(img, rect, SNAP / f"e1_crop_{k}.jpg")
    tmp.replace(pf)
    log.info("dopieszczanie %s: podmieniono snap i wycinek na klatke w pelnej jakosci (%dx%d)", k, iw, ih)
    return True


async def rl_i_dopieszcz(b, rodzaj, klasy):
    ok = await reolink.reolink_wideo(b, rodzaj, SNAP / f"e1_v_{b}.mp4", MEDIA, przygotuj_wideo_sync, force=True)  # nagranie z kamery zawsze, bo z niego bierzemy pelna jakosc
    if ok and b in PEND:
        try:
            await asyncio.get_running_loop().run_in_executor(None, dopieszcz_sync, b, klasy)
        except Exception:
            log.exception("dopieszczanie %s", b)
    for kk in [x for x in list(PEND) if x.startswith(b + "_x")]:
        if ok:
            try:
                await asyncio.get_running_loop().run_in_executor(None, dopieszcz_sync, b, klasy, kk)
            except Exception:
                log.exception("dopieszczanie %s", kk)
        PEND.pop(kk, None)
    PEND.pop(b, None)


def klip_analiza_sync(b, fps=2):
    """Skanuje caly klip zdarzenia (osoby i pojazdy), sklada tory i zwraca ich opis (do raportu / osobnych zdarzen)."""
    clip = SNAP / f"e1_v_{b}.mp4"
    if not clip.exists():
        return {"blad": "brak klipu"}
    tmpd = tempfile.mkdtemp(prefix="e1k_")
    try:
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(clip), "-vf", f"fps={fps},scale=1280:-2", "-q:v", "3", f"{tmpd}/f_%03d.jpg"], capture_output=True, timeout=240)
        files = sorted(Path(tmpd).glob("f_*.jpg"))
        if not files:
            return {"blad": "brak klatek"}
        log.info("klip_analiza %s: %d klatek, skanuje", b, len(files))
        tory = []
        for kl, nazwa in (("osoba", "osoba"), ("pojazd", "pojazd")):
            tt = []
            for i, f in enumerate(files):
                if i % 20 == 0:
                    log.info("klip_analiza %s: %s klatka %d/%d", b, nazwa, i, len(files))
                im = Image.open(f)
                im.load()
                iw, ih = im.size
                for bx in detect(im, KLASY[kl], 0.3):
                    n = _norm(bx, iw, ih)
                    best, bd = None, 0.25
                    for t in tt:
                        if i - t["f"][-1][0] <= 3 and t["f"][-1][0] < i:
                            lb = t["f"][-1][1]
                            d = math.hypot((n[0] + n[2]) / 2 - (lb[0] + lb[2]) / 2, ((n[1] + n[3]) / 2 - (lb[1] + lb[3]) / 2) * 0.75)
                            if _iou(n, lb) > 0.1 or d < 0.1:
                                if d < bd:
                                    best, bd = t, d
                    if best is not None:
                        best["f"].append((i, n, bx[4]))
                    else:
                        tt.append({"k": nazwa, "f": [(i, n, bx[4])]})
            tory += tt
        wyn = []
        dl = len(files) / fps
        for t in tory:
            f = t["f"]
            if len(f) < 2:
                continue
            c0 = ((f[0][1][0] + f[0][1][2]) / 2, (f[0][1][1] + f[0][1][3]) / 2)
            c1 = ((f[-1][1][0] + f[-1][1][2]) / 2, (f[-1][1][1] + f[-1][1][3]) / 2)
            dxy = math.hypot(c1[0] - c0[0], (c1[1] - c0[1]) * 0.75)
            mx = max(abs((a[1][0] + a[1][2]) / 2 - c0[0]) for a in f)
            t0, t1 = f[0][0] / fps, f[-1][0] / fps
            statyczny = max(math.hypot((a[1][0] + a[1][2]) / 2 - c0[0], (a[1][1] + a[1][3]) / 2 - c0[1]) for a in f) < 0.03
            wyn.append({"k": t["k"], "od": round(t0, 1), "do": round(t1, 1), "n": len(f), "x0": round(c0[0], 2), "y0": round(c0[1], 2), "x1": round(c1[0], 2), "y1": round(c1[1], 2),
                        "przes": round(dxy, 2), "stoi": statyczny, "kier": "prawo" if c1[0] - c0[0] > 0.06 else "lewo" if c1[0] - c0[0] < -0.06 else "?",
                        "conf": round(max(a[2] for a in f), 2), "box": [round(v, 2) for v in max(f, key=lambda a: (a[1][2] - a[1][0]) * (a[1][3] - a[1][1]) * a[2])[1]],
                        "t_best": round(max(f, key=lambda a: (a[1][2] - a[1][0]) * (a[1][3] - a[1][1]) * a[2])[0] / fps, 1)})
        wyn.sort(key=lambda x: x["od"])
        return {"dl": round(dl, 1), "klatek": len(files), "tory": wyn}
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)


def diag_sync(b, od_off=-9.0, do_off=7.0, krok=1):
    """Diagnostyka: dla kazdej klatki lekkiego strumienia w oknie - ruch (powierzchnia, srodek) i detekcje pojazdow."""
    t = reolink.czas_zdarzenia(b)
    t0 = time.mktime(t.timetuple())
    tmpd = tempfile.mkdtemp(prefix="e1d_")
    try:
        od, do = t0 + od_off, t0 + do_off
        files = bufor.klatki(od, do, 3, tmpd, pre="s")
        out = []
        male = []
        for f in files:
            im = Image.open(f)
            im.load()
            male.append(np.asarray(im.convert("L").resize((80, 60)), dtype=np.float32))
        bg = np.median(np.stack(male), 0) if male else None
        scan = SESS_T or SESS
        for k, f in enumerate(files):
            m = np.abs(male[k] - bg) > 28
            ar = float(m.mean())
            ys, xs = np.nonzero(m)
            r = {"t": round(od + k / 3 - t0, 1), "ar": round(ar, 3)}
            if ar > 0.004:
                r["mx"] = round(float(xs.mean()) / 80, 2)
                r["my"] = round(float(ys.mean()) / 60, 2)
            if k % krok == 0:
                im = Image.open(f)
                im.load()
                iw, ih = im.size
                bx = detect(im, KLASY["pojazd"], MIN_CONF * 0.9, sess=scan)
                r["det"] = [[round(v, 2) for v in _norm(x, iw, ih)] + [round(x[4], 2)] for x in bx]
            out.append(r)
        return {"t0": t0, "kl": len(files), "ramki": out}
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)


def diag2_sync(b, od_off=-10.0, do_off=8.0, fps=8):
    """Diagnostyka: energia ruchu (roznica od tla) klatek GLOWNEGO strumienia w oknie, do porownania z lekkim."""
    t = reolink.czas_zdarzenia(b)
    t0 = time.mktime(t.timetuple())
    od, do = t0 + od_off, t0 + do_off
    k = bufor.okno(od, do, "m")
    if not k:
        return {"blad": "brak kawalkow"}
    ss = max(0.0, od - k[0][0])
    ts = time.time()
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "pipe:0", "-ss", f"{ss:.2f}", "-t", f"{do - od:.2f}",
                        "-vf", f"fps={fps},scale=160:120,format=gray", "-f", "rawvideo", "pipe:1"], input=bufor._scal(k), capture_output=True, timeout=170)
    n = len(r.stdout) // 19200
    arr = np.frombuffer(r.stdout[:n * 19200], np.uint8).reshape(n, 120, 160).astype(np.float32)
    bg = np.median(arr, 0)
    out = []
    for i, a in enumerate(arr):
        m = np.abs(a - bg) > 14
        ar = float(m.mean())
        o = {"t": round(od + i / fps - t0, 1), "ar": round(ar, 3), "e": round(float(np.abs(a - bg).mean()), 2), "m": int(np.abs(a - bg).max()), "br": round(float(a.mean()), 1)}
        if ar > 0.0005:
            ys, xs = np.nonzero(m)
            o["mx"] = round(float(xs.mean()) / 160, 2)
            o["my"] = round(float(ys.mean()) / 120, 2)
        out.append(o)
    return {"t0": t0, "k0": k[0][0] - t0, "sek": round(time.time() - ts, 1), "n": n, "ramki": [[o["t"], o["ar"], o["e"], o["m"], o.get("mx"), o.get("my")] for o in out if o["m"] >= 30][:80]}


def popraw_kadr_sync(clip, b, n0):
    """Z klipu wybiera klatke z najwieksza liczba osob (razem w jednym kadrze).

    Zwraca dict: crop/plik (grupowe, tylko gdy osob jest wiecej niz n0), osoby (osobne kadry, gdy >= 2) i n.
    """
    tmpd = tempfile.mkdtemp(prefix="e1g_")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(clip), "-t", "18", "-vf", "fps=1", "-q:v", "3", f"{tmpd}/g_%02d.jpg"],
            capture_output=True, text=True, timeout=120,
        )
        best = None
        for f in sorted(Path(tmpd).glob("g_*.jpg")):
            img = Image.open(f)
            img.load()
            boxes = detect(img, KLASY["osoba"], MIN_CONF)
            if not boxes:
                continue
            bx, rect, n = wybierz_kadr(boxes, *img.size)
            key = (n, box_score(bx, *img.size))
            if best is None or key > best[0]:
                sel = sorted((x for x in boxes if x[4] >= max(MIN_CONF, bx[4] * 0.6)), key=lambda x: x[0])
                best = (key, img, rect, n, sel)
        if not best:
            return None
        _, img, rect, n, sel = best
        grupa = n > n0
        osoby = []
        if n >= 2:
            for i, x in enumerate(sel[:6]):
                nm = f"e1_os_{b}_{i}.jpg"
                zapisz_kadr(img, x[:4], SNAP / nm)
                osoby.append(nm)
        if not grupa and not osoby:
            return None
        out = {"n": n, "osoby": osoby, "grupa": grupa, "crop": "", "plik": ""}
        if grupa:
            out["crop"], out["plik"] = f"e1_crop_{b}_g.jpg", f"e1_best_{b}_g.jpg"
            zapisz_kadr(img, rect, SNAP / out["crop"])
            img.convert("RGB").save(SNAP / out["plik"], "JPEG", quality=90)
        return out
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)


async def klip_do_analizy(b, limit=45):
    """Zywy klip e1_rec; gdy go nie ma - czeka (do ~150 s) na wideo z Reolinka e1_v_{b}.mp4."""
    clip = await czekaj_na_klip(b, limit)
    if clip:
        return clip
    p = SNAP / f"e1_v_{b}.mp4"
    t0 = time.time()
    while time.time() - t0 < 150:
        if b in reolink.GOTOWE and p.exists():
            return p
        await asyncio.sleep(5)
    return None


async def popraw_kadr(b, n0, rodzaj, lok, hm, powiadom, glosno):
    """W tle: gdy w klipie jest klatka z wiekszą liczba osob niz na zdjeciu, podmienia kadr w historii i powiadomieniu."""
    try:
        clip = await klip_do_analizy(b)
        if not clip:
            return
        loop = asyncio.get_running_loop()
        async with GLOCK:
            w = await loop.run_in_executor(None, popraw_kadr_sync, clip, b, n0)
        if not w:
            log.info("kadr %s: bez zmian (na zdjeciu %s os.)", b, n0)
            return
        n = w["n"]
        log.info("kadr %s: z klipu osob %s -> %s, grupowe=%s, osobne=%s", b, n0, n, w["grupa"], len(w["osoby"]))
        async with reolink.HA() as ha:
            await ha.cmd({
                "type": "fire_event",
                "event_type": "kamera_zdarzenie_popr",
                "event_data": {
                    "b": b, "crop": w["crop"], "plik": w["plik"], "osoby": w["osoby"], "n": n, "rodzaj": rodzaj, "lok": lok, "hm": hm,
                    "pow": 1 if (powiadom and w["grupa"] and LAST_B.get(rodzaj) == b) else 0, "glosno": 1 if glosno else 0,
                },
            })
    except Exception:
        log.exception("popraw_kadr")


async def grupa_test(lista):
    """Test: uruchamia analize grupy na istniejacych klipach (po kolei, w tle); wynik w logu."""
    loop = asyncio.get_running_loop()
    for bb in lista:
        clip = SNAP / f"e1_v_{bb}.mp4"
        if not clip.exists():
            log.info("grupa-test %s: brak klipu", bb)
            continue
        try:
            t0 = time.time()
            async with GLOCK:
                w = await loop.run_in_executor(None, popraw_kadr_sync, clip, bb, 1)
            log.info("grupa-test %s -> %s (%.0f s)", bb, w, time.time() - t0)
        except Exception:
            log.exception("grupa-test %s", bb)


PARK_PLIK = Path(os.environ.get("E1_DATA", "/data")) / "zaparkowane.json"
RUCH_ETYKIETA = {"wjazd": "Wjazd", "wyjazd": "Wyjazd", "przejazd": "Przejazd"}


def _park_lista():
    try:
        return json.load(open(PARK_PLIK))
    except Exception:
        return []


def _park_zapisz(lista):
    try:
        tmp = PARK_PLIK.with_suffix(".tmp")
        json.dump(lista, open(tmp, "w"))
        os.replace(tmp, PARK_PLIK)
    except Exception:
        log.exception("zapis zaparkowanych")
    _park_sensor(lista)


def _park_sensor(lista):
    """Wystawia licznik jako sensor.kamera_zaparkowane (REST przez proxy Supervisora)."""
    import urllib.request
    try:
        body = json.dumps({
            "state": len(lista),
            "attributes": {
                "friendly_name": "Kamera zaparkowane auta",
                "unit_of_measurement": "aut",
                "icon": "mdi:car-multiple",
                "od": [datetime.fromtimestamp(p["t"]).strftime("%Y-%m-%d %H:%M") for p in lista],
                "auta": [
                    {"nazwa": p.get("nazwa") or "Nierozpoznane auto", "od": datetime.fromtimestamp(p["t"]).strftime("%H:%M"),
                     "dzien": datetime.fromtimestamp(p["t"]).strftime("%Y-%m-%d"), "crop": p.get("crop", "")}
                    for p in lista
                ],
            },
        }).encode()
        rq = urllib.request.Request(
            reolink.URL + "/api/states/sensor.kamera_zaparkowane", data=body, method="POST",
            headers={"Authorization": "Bearer " + reolink.TOKEN, "Content-Type": "application/json"},
        )
        urllib.request.urlopen(rq, timeout=10).read()
    except Exception as e:
        log.warning("sensor zaparkowane: %s", e)


def _podjazd_id(cfg):
    for z in cfg["strefy"]:
        if "podjazd" in z["nazwa"].lower():
            return z["id"]
    return cfg["strefy"][0]["id"] if cfg["strefy"] else None


def _park_dodaj(park, ra, b, box, crop):
    if ra:
        park = [p for p in park if p.get("auto") != ra["id"]]
    park.append({"t": time.time(), "b": b, "box": box, "auto": ra["id"] if ra else None,
                 "nazwa": auta.etykieta_pelna(ra) if ra else "", "crop": crop})
    return park


def nieruchome_sync(b):
    """Auto stoi: rozpoznaje je ze zdjecia zdarzenia; znane auto dopisuje do rejestru zaparkowanych."""
    crop = SNAP / f"e1_crop_{b}.jpg"
    if not crop.exists():
        return None
    ra = auta.rozpoznaj(crop)
    if ra:
        park = [p for p in _park_lista() if time.time() - p["t"] < 7 * 86400]
        if not any(p.get("auto") == ra["id"] for p in park):
            park = _park_dodaj(park, ra, b, [0, 0, 1, 1], f"e1_crop_{b}.jpg")
            _park_zapisz(park)
    return ra


def sledz_sync(clip, b, granica=""):
    """Sledzi pojazdy w klipie (fps=2, 20 s). Zwraca dict z kierunkiem (wjazd/wyjazd/przejazd) albo None, gdy nic sie nie rusza."""
    tmpd = tempfile.mkdtemp(prefix="e1t_")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(clip), "-t", "20", "-vf", "fps=2", "-q:v", "3", f"{tmpd}/t_%02d.jpg"],
            capture_output=True, text=True, timeout=120,
        )
        cfg = strefy.wczytaj(granica)
        pid = _podjazd_id(cfg)
        tracks = []  # kazdy: {"det": [(idx, box)], "img": {idx: img}}
        imgs = {}
        for i, f in enumerate(sorted(Path(tmpd).glob("t_*.jpg"))):
            img = Image.open(f)
            img.load()
            imgs[i] = img
            iw, ih = img.size
            used = set()
            for bx in sorted(detect(img, KLASY["pojazd"], MIN_CONF), key=lambda x: -x[4]):
                cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
                bestt, bd = None, 0.12 * iw
                for ti, t in enumerate(tracks):
                    if ti in used or i - t["det"][-1][0] > 3:
                        continue
                    lb = t["det"][-1][1]
                    d = math.hypot(cx - (lb[0] + lb[2]) / 2, cy - (lb[1] + lb[3]) / 2)
                    if d < bd:
                        bestt, bd = ti, d
                if bestt is None:
                    tracks.append({"det": [(i, bx)]})
                    used.add(len(tracks) - 1)
                else:
                    tracks[bestt]["det"].append((i, bx))
                    used.add(bestt)
        if not imgs:
            return None
        iw, ih = imgs[0].size
        wyniki = []
        for t in tracks:
            if len(t["det"]) < 3:
                continue
            a, z = t["det"][0][1], t["det"][-1][1]
            disp = math.hypot((z[0] + z[2]) / 2 - (a[0] + a[2]) / 2, (z[1] + z[3]) / 2 - (a[1] + a[3]) / 2) / iw
            if disp < 0.06:
                continue  # stoi
            za = strefy.przypisz(a, iw, ih, cfg, "trk")
            zz = strefy.przypisz(z, iw, ih, cfg, "trk2")
            if pid and zz == pid and za != pid:
                kier = "wjazd"
            elif pid and za == pid and zz != pid:
                kier = "wyjazd"
            else:
                kier = "przejazd"
            idx, bb = max(t["det"], key=lambda d: box_score(d[1], iw, ih))
            wyniki.append({"kier": kier, "disp": disp, "start": a, "end": z, "idx": idx, "box": bb})
        if not wyniki:
            return None
        pri = {"wjazd": 2, "wyjazd": 2, "przejazd": 1}
        w = max(wyniki, key=lambda r: (pri[r["kier"]], r["disp"]))
        img = imgs[w["idx"]]
        _, rect, _ = wybierz_kadr([w["box"]], iw, ih)
        crop, plik = f"e1_crop_{b}_r.jpg", f"e1_best_{b}_r.jpg"
        zapisz_kadr(img, rect, SNAP / crop)
        img.convert("RGB").save(SNAP / plik, "JPEG", quality=90)
        # rozpoznanie auta + rejestr zaparkowanych
        ra = auta.rozpoznaj(SNAP / crop)
        park = [p for p in _park_lista() if time.time() - p["t"] < 7 * 86400]
        if w["kier"] == "wjazd":
            park = _park_dodaj(park, ra, b, _norm(w["end"], iw, ih), crop)
        elif w["kier"] == "wyjazd":
            sn = _norm(w["start"], iw, ih)
            if ra and any(p.get("auto") == ra["id"] for p in park):
                park = [p for p in park if p.get("auto") != ra["id"]]
            else:
                ok = [p for p in park if _iou(sn, p["box"]) > 0.3]
                if ok:
                    park.remove(max(ok, key=lambda p: _iou(sn, p["box"])))
                elif park:
                    park.remove(min(park, key=lambda p: p["t"]))
        _park_zapisz(park)
        return {"kier": w["kier"], "crop": crop, "plik": plik, "zaparkowane": len(park), "auto": ra}
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)


POJ_PLIK = Path(os.environ.get("E1_DATA", "/data")) / "pojazdy.json"
KOL = {"kia": "#22c55e", "seat": "#3b82f6", "mazda": "#9ca3af", "obce": "#f59e0b", None: "#ef4444"}


def _poj_load():
    try:
        return json.load(open(POJ_PLIK))
    except Exception:
        return {}


def _poj_save(m):
    try:
        keys = sorted(m, key=lambda k: m[k].get("t", 0))[-150:]
        m = {k: m[k] for k in keys}
        tmp = POJ_PLIK.with_suffix(".tmp")
        json.dump(m, open(tmp, "w"))
        os.replace(tmp, POJ_PLIK)
    except Exception:
        log.exception("zapis pojazdy")


def segmentuj(img, n):
    """Przyblizony (kanciasty) obrys karoserii: GrabCut w ramce auta -> kontur -> uproszczony wielokat (wspolrzedne 0-1)."""
    try:
        import cv2
        iw, ih = img.size
        x1, y1, x2, y2 = n[0] * iw, n[1] * ih, n[2] * iw, n[3] * ih
        w, h = x2 - x1, y2 - y1
        cx1, cy1, cx2, cy2 = int(max(0, x1 - w * 0.08)), int(max(0, y1 - h * 0.08)), int(min(iw, x2 + w * 0.08)), int(min(ih, y2 + h * 0.08))
        c = np.asarray(img.convert("RGB").crop((cx1, cy1, cx2, cy2)))
        sc = min(1.0, 480 / max(c.shape[:2]))
        sm = cv2.resize(c, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) if sc < 1 else c
        bgr = cv2.cvtColor(sm, cv2.COLOR_RGB2BGR)
        H, W = bgr.shape[:2]
        rect = (int((x1 - cx1) * sc + W * 0.02), int((y1 - cy1) * sc + H * 0.02), int(w * sc - W * 0.04), int(h * sc - H * 0.04))
        mask = np.zeros((H, W), np.uint8)
        cv2.grabCut(bgr, mask, rect, np.zeros((1, 65)), np.zeros((1, 65)), 5, cv2.GC_INIT_WITH_RECT)
        mk = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
        mk = cv2.morphologyEx(mk, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        cs, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            return None
        cn = max(cs, key=cv2.contourArea)
        if cv2.contourArea(cn) < 0.3 * rect[2] * rect[3]:
            return None
        per = cv2.arcLength(cn, True)
        eps = 0.006
        ap = cv2.approxPolyDP(cn, eps * per, True)
        while len(ap) > 26 and eps < 0.05:
            eps += 0.002
            ap = cv2.approxPolyDP(cn, eps * per, True)
        ap = ap[:, 0, :]
        return [[round((cx1 + p[0] / sc) / iw, 4), round((cy1 + p[1] / sc) / ih, 4)] for p in ap]
    except Exception:
        log.exception("segmentuj")
        return None


def rysuj(img, items):
    """Sama nazwa auta w srodku jego sylwetki (bez obrysu)."""
    from PIL import ImageDraw, ImageFont
    base = img.convert("RGBA")
    iw, ih = base.size
    ov = Image.new("RGBA", base.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    try:
        font = ImageFont.load_default(size=max(12, iw // 90))
    except Exception:
        font = ImageFont.load_default()
    lw = max(2, iw // 650)
    zajete = []
    for it in items:
        x1, y1, x2, y2 = it["n"][0] * iw, it["n"][1] * ih, it["n"][2] * iw, it["n"][3] * ih
        hx = KOL.get(it.get("auto_id"), KOL[None]).lstrip("#")
        rgb = tuple(int(hx[k:k + 2], 16) for k in (0, 2, 4))
        poly = it.get("poly")
        if poly and len(poly) >= 3:
            pts = [(p[0] * iw, p[1] * ih) for p in poly]
            x1, y1 = min(p[0] for p in pts), min(p[1] for p in pts)  # obrysu nie rysujemy, poligon sluzy do wyliczenia srodka
        txt = {"kia": "KIA", "seat": "SEAT", "mazda": "MAZDA", "obce": "OBCE"}.get(it.get("auto_id"), "?")
        tb = dr.textbbox((0, 0), txt, font=font)
        tw, th = tb[2] - tb[0] + 8, tb[3] - tb[1] + 6
        if poly and len(poly) >= 3:
            ar = cxs = cys = 0.0
            for k in range(len(pts)):  # srodek ciezkosci pola wielokata (wzor Gaussa), nie srednia wierzcholkow
                xa, ya = pts[k]
                xb, yb = pts[(k + 1) % len(pts)]
                cr = xa * yb - xb * ya
                ar += cr
                cxs += (xa + xb) * cr
                cys += (ya + yb) * cr
            if abs(ar) > 1e-6:
                cx, cy = cxs / (3 * ar), cys / (3 * ar)
            else:
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        else:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        x1, ty = cx - tw / 2, cy - th / 2  # opis w srodku auta
        for (lx, ly, lw2, lh) in zajete:
            if x1 < lx + lw2 and x1 + tw > lx and ty < ly + lh and ty + th > ly:
                ty = ly + lh  # nie nakladaj etykiet
        zajete.append((x1, ty, tw, th))
        dr.rectangle([x1, ty, x1 + tw, ty + th], fill=rgb + (200,))
        dr.text((x1 + 4, ty + 2), txt, fill=(0, 0, 0, 255), font=font)
    return Image.alpha_composite(base, ov).convert("RGB")


def _poj_publiczne(it):
    return {"i": it["i"], "crop": it["crop"], "nazwa": it.get("nazwa", ""), "auto_id": it.get("auto_id")}


def _bez_duplikatow(boxes):
    """Odrzuca ramki w >60% zawarte w ramce o wyzszej pewnosci (to samo auto wykryte dwa razy)."""
    kept = []
    for bx in sorted(boxes, key=lambda x: -x[4]):
        area = max(1.0, (bx[2] - bx[0]) * (bx[3] - bx[1]))
        dup = False
        for k in kept:
            ix = max(0.0, min(bx[2], k[2]) - max(bx[0], k[0]))
            iy = max(0.0, min(bx[3], k[3]) - max(bx[1], k[1]))
            ak = max(1.0, (k[2] - k[0]) * (k[3] - k[1]))
            if ix * iy / area > 0.6 or ix * iy / (area + ak - ix * iy) > 0.35:
                dup = True
                break
        if not dup:
            kept.append(bx)
    return sorted(kept, key=lambda x: x[0])


def adnotuj_sync(b, src: Path):
    """Wykrywa wszystkie auta na pelnym kadrze, rozpoznaje kazde, zapisuje wycinki i kadr z obrysami."""
    if not src.exists():
        return None
    img = Image.open(src)
    img.load()
    iw, ih = img.size
    items = []
    for i, bx in enumerate(_bez_duplikatow(detect(img, KLASY["pojazd"], MIN_CONF))):
        _, rect, _ = wybierz_kadr([bx], iw, ih)
        crop = f"e1_crop_{b}_c{i}.jpg"
        zapisz_kadr(img, rect, SNAP / crop)
        nn = _norm(bx, iw, ih)
        poly = segmentuj(img, nn)
        f = auta.cecha_maska(img, poly, nn)
        ra = auta.rozpoznaj_wektor(f)
        log.info("auto %s box=%s conf=%.2f rozpoznano=%s", b, [round(x, 2) for x in nn], bx[4], ra)
        items.append({"i": i, "crop": crop, "n": nn, "poly": poly, "f": f, "auto_id": ra["id"] if ra else None,
                      "nazwa": auta.etykieta_pelna(ra) if ra else ""})
    if not items:
        return None
    m = _poj_load()
    m[b] = {"src": src.name, "v": 0, "t": time.time(), "items": items}
    _poj_save(m)
    out = f"e1_best_{b}_a0.jpg"
    rysuj(img, items).save(SNAP / out, "JPEG", quality=90)
    return {"plik": out, "pojazdy": [_poj_publiczne(x) for x in items], "auto": ", ".join(x["nazwa"] for x in items if x["nazwa"]),
            "items": items}


def przemaluj_sync(b, i, auto_id, nazwa):
    """Po recznej etykiecie: aktualizuje nazwe auta nr i i rysuje kadr od nowa (nowa nazwa pliku - omija cache)."""
    m = _poj_load()
    e = m.get(b)
    if not e:
        return None
    for it in e["items"]:
        if it["i"] == i:
            it["auto_id"], it["nazwa"] = auto_id, nazwa
    e["v"] += 1
    _poj_save(m)
    img = Image.open(SNAP / e["src"])
    img.load()
    out = f"e1_best_{b}_a{e['v']}.jpg"
    rysuj(img, e["items"]).save(SNAP / out, "JPEG", quality=90)
    return {"plik": out, "pojazdy": [_poj_publiczne(x) for x in e["items"]], "auto": ", ".join(x["nazwa"] for x in e["items"] if x["nazwa"])}


def _poj_event(b, w, lok, hm, ruch=None):
    d = {"b": b, "crop": "", "plik": w["plik"], "osoby": [], "n": 1, "rodzaj": "Pojazd", "lok": lok, "hm": hm,
         "auto": w["auto"], "pojazdy": w["pojazdy"], "pow": 0, "glosno": 0}
    if ruch:
        d["ruch"] = ruch
    return d


async def sledz_pojazd(b, lok, hm, granica="", pelny=""):
    """W tle: ruch pojazdu z klipu (wjazd/wyjazd/przejazd), rozpoznanie aut i kadr z obrysami; aktualizuje wpis w historii."""
    try:
        loop = asyncio.get_running_loop()
        if pelny:  # od razu: obrysy i nazwy aut ze zdjecia zdarzenia (bez czekania na klip)
            async with GLOCK:
                ad0 = await loop.run_in_executor(None, adnotuj_sync, b, SNAP / pelny)
            if ad0:
                async with reolink.HA() as ha:
                    await ha.cmd({"type": "fire_event", "event_type": "kamera_zdarzenie_popr", "event_data": _poj_event(b, ad0, lok, hm)})
        clip = await klip_do_analizy(b)
        w = None
        if clip:
            async with GLOCK:
                w = await loop.run_in_executor(None, sledz_sync, clip, b, granica)
        if w:
            src = SNAP / w["plik"]
            ruch = RUCH_ETYKIETA[w["kier"]]
        else:
            await loop.run_in_executor(None, nieruchome_sync, b)
            src = SNAP / pelny if pelny else SNAP / f"e1_crop_{b}.jpg"
            ruch = None
        async with GLOCK:
            ad = await loop.run_in_executor(None, adnotuj_sync, b, src)
        log.info("sledzenie %s: ruch=%s, aut na kadrze=%s", b, ruch, len(ad["pojazdy"]) if ad else 0)
        if not ad and not w:
            return
        ev = _poj_event(b, ad, lok, hm, ruch) if ad else {
            "b": b, "crop": "", "plik": w["plik"], "osoby": [], "n": 1, "rodzaj": "Pojazd", "lok": lok, "hm": hm,
            "ruch": ruch, "auto": auta.etykieta_pelna(w.get("auto")), "pow": 0, "glosno": 0}
        if w:
            ev["crop"] = w["crop"]
        async with reolink.HA() as ha:
            await ha.cmd({"type": "fire_event", "event_type": "kamera_zdarzenie_popr", "event_data": ev})
    except Exception:
        log.exception("sledz_pojazd")


async def analyze(request):
    d = await request.json()
    if str(d.get("b", "")) == "zaparkowane":
        if str(d.get("plik", "")) == "reset":
            _park_zapisz([])
        elif str(d.get("plik", "")) == "skan":
            async with GLOCK:
                await asyncio.get_running_loop().run_in_executor(None, skan_sync)
        elif str(d.get("plik", "")) == "sensor":
            _park_sensor(_park_lista())
        return web.json_response({"ok": True, "zaparkowane": _park_lista()})
    if str(d.get("b", "")) == "obrys":
        lista = [x.strip() for x in str(d.get("plik", "")).split(",") if x.strip() and "/" not in x]

        async def _o():
            loop = asyncio.get_running_loop()
            for bb in lista:
                try:
                    async with GLOCK:
                        ad = await loop.run_in_executor(None, adnotuj_sync, bb, SNAP / f"{bb}.jpg")
                    log.info("obrys-test %s -> %s", bb, ad and ad["plik"])
                    if ad:
                        async with reolink.HA() as ha:
                            await ha.cmd({"type": "fire_event", "event_type": "kamera_zdarzenie_popr", "event_data": _poj_event(bb, ad, "", "")})
                except Exception:
                    log.exception("obrys %s", bb)

        asyncio.create_task(_o())
        return web.json_response({"ok": True, "start": lista})
    if str(d.get("b", "")) == "sledz":
        lista = [x.strip() for x in str(d.get("plik", "")).split(",") if x.strip() and "/" not in x]

        async def _t():
            loop = asyncio.get_running_loop()
            for bb in lista:
                clip = SNAP / f"e1_v_{bb}.mp4"
                if clip.exists():
                    async with GLOCK:
                        w = await loop.run_in_executor(None, sledz_sync, clip, bb, "")
                    log.info("sledz-test %s -> %s", bb, w)

        asyncio.create_task(_t())
        return web.json_response({"ok": True, "start": lista})
    if str(d.get("b", "")) == "api":
        try:
            dane = json.loads(str(d.get("plik", "{}")) or "{}")
            if dane.get("cmd") == "auta":
                return web.json_response({"ok": True, "auta": auta.lista()})
            if dane.get("cmd") == "bufor":
                return web.json_response({"ok": True, "bufor": bufor.status()})
            if dane.get("cmd") == "bufor_test":
                # test na zywo w tle: klip i najlepsza klatka dla zdarzenia sprzed `sek` sekund; wynik: cmd bufor_wynik
                _t0 = time.time() - float(dane.get("sek", 20))
                _b = "e1_" + time.strftime("%Y%m%d_%H%M%S", time.localtime(_t0)) + "_test"
                _TEST.clear()

                async def _bt():
                    _ts = time.time()
                    try:
                        _wy = await asyncio.get_running_loop().run_in_executor(None, najlepsza_klatka_sync, _b, KLASY["pojazd"], "", _t0)
                        _TEST["klatka"] = bool(_wy)
                        _TEST["t_klatka"] = round(time.time() - _ts, 1)
                        _TEST["klip"] = await bufor.klip(_b, SNAP / f"e1_v_{_b}.mp4", _t0)
                        _TEST["t_calosc"] = round(time.time() - _ts, 1)
                    except Exception as _e:
                        log.exception("bufor_test")
                        _TEST["blad"] = str(_e)
                    _TEST["b"] = _b
                asyncio.create_task(_bt())
                return web.json_response({"ok": True, "b": _b})
            if dane.get("cmd") == "zloz_bench":
                def _bench():
                    _k = bufor.okno(time.time() - 60, time.time() - 20)
                    _r = {"kawalki": len(_k), "mb": round(sum(p.stat().st_size for _, _, p in _k) / 1e6, 1)}
                    _d = Path(tempfile.mkdtemp(prefix="e1z_"))
                    try:
                        _l = _d / "l.lst"
                        bufor._lista(_k, _l)
                        for _nm, _cmd in (
                            ("concat_faststart", ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(_l), "-c", "copy", "-movflags", "+faststart", str(_d / "a.mp4")]),
                            ("concat_bez_faststart", ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(_l), "-c", "copy", str(_d / "b.mp4")]),
                            ("concat_maly_probe", ["ffmpeg", "-y", "-loglevel", "error", "-probesize", "32", "-analyzeduration", "0", "-f", "concat", "-safe", "0", "-i", str(_l), "-c", "copy", "-movflags", "+faststart", str(_d / "c.mp4")]),
                        ):
                            _t = time.time()
                            _p = subprocess.run(_cmd, capture_output=True, text=True, timeout=120)
                            _r[_nm] = [round(time.time() - _t, 2), _p.returncode, round(reolink.dlugosc(_d / _cmd[-1].split("/")[-1]), 1)]
                        _t = time.time()
                        with open(_d / "all.ts", "wb") as _o:
                            for _, _, _p2 in _k:
                                _o.write(_p2.read_bytes())
                        _t2 = time.time()
                        _p = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(_d / "all.ts"), "-c", "copy", "-movflags", "+faststart", str(_d / "d.mp4")], capture_output=True, text=True, timeout=120)
                        _r["cat_potem_remux"] = [round(_t2 - _t, 2), round(time.time() - _t2, 2), _p.returncode, round(reolink.dlugosc(_d / "d.mp4"), 1)]
                    finally:
                        shutil.rmtree(_d, ignore_errors=True)
                    return _r
                return web.json_response({"ok": True, "bench": await asyncio.get_running_loop().run_in_executor(None, _bench)})
            if dane.get("cmd") == "diag":
                _b = str(dane.get("b", ""))

                async def _dg():
                    try:
                        _TEST["diag"] = await asyncio.get_running_loop().run_in_executor(None, diag_sync, _b, float(dane.get("od", -9)), float(dane.get("do", 7)), int(dane.get("krok", 1)))
                    except Exception as _e:
                        log.exception("diag")
                        _TEST["diag"] = {"blad": str(_e)}
                _TEST.pop("diag", None)
                asyncio.create_task(_dg())
                return web.json_response({"ok": True})
            if dane.get("cmd") == "diag2":
                _b = str(dane.get("b", ""))

                async def _dg2():
                    try:
                        _TEST["diag2"] = await asyncio.get_running_loop().run_in_executor(None, diag2_sync, _b, float(dane.get("od", -10)), float(dane.get("do", 8)), int(dane.get("fps", 8)))
                    except Exception as _e:
                        log.exception("diag2")
                        _TEST["diag2"] = {"blad": str(_e)}
                _TEST.pop("diag2", None)
                asyncio.create_task(_dg2())
                return web.json_response({"ok": True})
            if dane.get("cmd") == "dopiesc":
                _b = str(dane.get("b", ""))

                async def _dp():
                    try:
                        _w = await asyncio.get_running_loop().run_in_executor(None, najlepsza_klatka_sync, _b, KLASY["pojazd"], "")
                        if _w:
                            await rl_i_dopieszcz(_b, "pojazd", KLASY["pojazd"])
                    except Exception:
                        log.exception("dopiesc")
                asyncio.create_task(_dp())
                return web.json_response({"ok": True})
            if dane.get("cmd") == "przelicz":
                _b = str(dane.get("b", ""))
                if not _b or "/" in _b:
                    return web.json_response({"ok": False, "blad": "b"})
                _ld = asyncio.get_running_loop()
                _w = await _ld.run_in_executor(None, najlepsza_klatka_sync, _b, KLASY["pojazd"], "")
                if not _w:
                    return web.json_response({"ok": False, "blad": "brak klatki (bufor?)"})
                _img, _bx, _pf = _w
                _dop = None
                if dane.get("fs") and _b in PEND:
                    _tb = reolink.czas_zdarzenia(_b)
                    reolink.START[_b] = datetime.combine(_tb.date(), datetime.strptime(str(dane["fs"]), "%H:%M:%S").time())
                    _dop = await _ld.run_in_executor(None, dopieszcz_sync, _b, KLASY["pojazd"])
                    if _dop:
                        return web.json_response({"ok": True, "dopieszczone": True, "crop": f"e1_crop_{_b}.jpg", "pelny": _pf})
                _iw, _ih = _img.size
                _best, _rect, _n = wybierz_kadr(_bx, _iw, _ih)
                _cb = await _ld.run_in_executor(None, zapisz_kadr, _img, _rect, SNAP / f"e1_crop_{_b}.jpg")
                return web.json_response({"ok": True, "crop": f"e1_crop_{_b}.jpg", "pelny": _pf, "box": [round(v) for v in _best[:4]], "crop_box": list(_cb), "conf": round(_best[4], 3), "dopieszczone": _dop})
            if dane.get("cmd") == "wideo_diag":
                _b = str(dane.get("b", ""))
                def _wd():
                    out = {}
                    for nm in (f"e1_v_{_b}.mp4", f"e1_rec_{_b}.mp4"):
                        f = SNAP / nm
                        if not f.exists():
                            continue
                        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,profile:format=duration,bit_rate", "-of", "json", str(f)], capture_output=True, text=True, timeout=60)
                        info = json.loads(r.stdout or "{}")
                        r2 = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time,dts_time,flags", "-of", "csv=p=0", str(f)], capture_output=True, text=True, timeout=60)
                        pts, kf = [], []
                        for ln in r2.stdout.splitlines():
                            q = ln.split(",")
                            try:
                                pts.append(float(q[0]))
                                if "K" in q[-1]:
                                    kf.append(float(q[0]))
                            except Exception:
                                pass
                        pts.sort()
                        d = [round(b2 - a, 3) for a, b2 in zip(pts, pts[1:])]
                        dd = sorted(d)
                        out[nm] = {"info": info, "klatek": len(pts), "dt_med": dd[len(dd) // 2] if dd else None, "dt_max": dd[-5:] if dd else None, "hist": {str(k): d.count(k) for k in sorted(set(d))},
                                   "dt_duze": [(round(pts[i], 2), x) for i, x in enumerate(d) if x > 0.2][:25], "kf": [round(x, 2) for x in kf[:40]]}
                    return out
                _TEST["wd"] = await asyncio.get_running_loop().run_in_executor(None, _wd)
                return web.json_response({"ok": True})
            if dane.get("cmd") in ("kam_enc", "kam_enc_set"):
                import aiohttp as _ah
                async with _ah.ClientSession() as _s:
                    _tok = await reolink._kam_token(_s)
                    _bs = reolink._kam["base"]
                    async with _s.post(f"{_bs}/api.cgi?cmd=GetEnc&token={_tok}", json=[{"cmd": "GetEnc", "action": 1, "param": {"channel": 0}}], ssl=False, timeout=_ah.ClientTimeout(total=15)) as _r:
                        _j = await _r.json(content_type=None)
                    _enc = _j[0]["value"]["Enc"]
                    _TEST["enc_przed"] = _enc
                    if dane.get("cmd") == "kam_enc_set":
                        _enc["mainStream"]["frameRate"] = int(dane.get("fps", 25))
                        async with _s.post(f"{_bs}/api.cgi?cmd=SetEnc&token={_tok}", json=[{"cmd": "SetEnc", "param": {"Enc": _enc}}], ssl=False, timeout=_ah.ClientTimeout(total=15)) as _r:
                            _TEST["enc_set"] = await _r.json(content_type=None)
                return web.json_response({"ok": True, "enc": _TEST.get("enc_przed"), "set": _TEST.get("enc_set")})
            if dane.get("cmd") == "napraw_wideo":
                _b = str(dane.get("b", ""))
                def _nw():
                    f = SNAP / f"e1_v_{_b}.mp4"
                    bsf = rowne_klatki_bsf(f)
                    if not bsf:
                        return "pomijam (nie 25 kl./s albo klatki B)"
                    tmp = f.with_suffix(".fix.mp4")
                    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(f), "-map", "0", "-c", "copy", "-bsf:v", bsf, "-movflags", "+faststart", str(tmp)], capture_output=True, text=True, timeout=120)
                    if r.returncode != 0:
                        tmp.unlink(missing_ok=True)
                        return "blad: " + r.stderr[-200:]
                    tmp.replace(f)
                    return "ok"
                _TEST["nw"] = await asyncio.get_running_loop().run_in_executor(None, _nw)
                return web.json_response({"ok": True, "wynik": _TEST["nw"]})
            if dane.get("cmd") == "przejazd_extra":
                _b = str(dane.get("b", ""))
                async def _pe():
                    async with LOCK:
                        await asyncio.get_running_loop().run_in_executor(None, najlepsza_klatka_sync, _b, KLASY["pojazd"], "", None, True)
                    ex = EXTRA.pop(_b, [])
                    _TEST["extra"] = ex
                    for e in ex:
                        asyncio.create_task(przejazd_wpis(e["key"], ts=e["ts"]))
                    if ex:
                        asyncio.create_task(rl_i_dopieszcz(_b, "pojazd", KLASY["pojazd"]))
                asyncio.create_task(_pe())
                return web.json_response({"ok": True})
            if dane.get("cmd") == "obraz_info":
                _b = str(dane.get("b", ""))
                _o = {}
                for _n in (f"e1_crop_{_b}.jpg", f"e1_best_{_b}.jpg", f"e1_pz_{_b}.jpg"):
                    _f = SNAP / _n
                    if _f.exists():
                        _im = Image.open(_f)
                        _o[_n] = {"px": list(_im.size), "bajty": _f.stat().st_size, "mtime": time.strftime("%H:%M:%S", time.localtime(_f.stat().st_mtime))}
                return web.json_response({"ok": True, "info": _o})
            if dane.get("cmd") == "klip_analiza":
                _b = str(dane.get("b", ""))
                async def _ka():
                    try:
                        _TEST["ka"] = await asyncio.get_running_loop().run_in_executor(None, klip_analiza_sync, _b)
                        log.info("klip_analiza %s: gotowe, tory %d", _b, len((_TEST["ka"] or {}).get("tory", [])))
                    except Exception:
                        log.exception("klip_analiza %s", _b)
                _TEST["ka"] = None
                asyncio.create_task(_ka())
                return web.json_response({"ok": True})
            if dane.get("cmd") == "bufor_wynik":
                return web.json_response({"ok": True, "wynik": _TEST})
            if dane.get("cmd") == "sd_test":
                import aiohttp as _ah
                async with _ah.ClientSession() as _s:
                    _fl = await reolink.sd_szukaj(_s, datetime.now() - timedelta(minutes=int(dane.get("min", 60))), datetime.now())
                    if not _fl:
                        return web.json_response({"ok": True, "pliki": 0})
                    _f = _fl[-1]
                    _p = MEDIA / "e1_sdtest.mp4"
                    _t0 = time.time()
                    _n = await reolink.sd_pobierz(_s, _f["name"], _p)
                    _d = reolink.dlugosc(_p)
                    _p.unlink(missing_ok=True)
                return web.json_response({"ok": True, "pliki": len(_fl), "plik": _f["name"], "bajty": _n, "sek": round(_d, 1), "czas_pobrania": round(time.time() - _t0, 1)})
            if dane.get("cmd") == "parking_skan":
                async with GLOCK:
                    await asyncio.get_running_loop().run_in_executor(None, skan_sync)
                return web.json_response({"ok": True})
            if dane.get("cmd") == "przejazdy":
                lst = pz_wczytaj()
                lst.sort(key=lambda e: -e.get("ts", 0))
                return web.json_response({"ok": True, "teraz": time.time(), "lista": lst[: int(dane.get("n", 200))]})
            if dane.get("cmd") == "przejazd_test":
                _b = str(dane.get("b", ""))
                asyncio.create_task(przejazd_wpis(_b, czekaj=False))
                return web.json_response({"ok": True})
            if dane.get("cmd") == "parking":
                hh = _hist_load()
                return web.json_response({"ok": True, "teraz": time.time(), "sesje": hh["sesje"], "auta": auta.lista()})
            if dane.get("cmd") == "reset":
                auta.reset(str(dane.get("auto", "")), dane.get("opis"))
                return web.json_response({"ok": True, "auta": auta.lista()})
            if dane.get("cmd") == "etykieta":
                bb, cr, av = str(dane.get("b", "")), str(dane.get("crop", "")), str(dane.get("auto", ""))
                if "/" in cr or not cr:
                    return web.json_response({"ok": False, "blad": "zly plik"})
                vec = None
                if dane.get("i") is not None:
                    ent = _poj_load().get(bb)
                    if ent:
                        vec = next((x.get("f") for x in ent["items"] if x["i"] == int(dane["i"])), None)
                a = auta.etykieta_wektor(vec, av) if vec else auta.etykieta(SNAP / cr, av)
                if a is None:
                    return web.json_response({"ok": False, "blad": "nieznane auto"})
                nm = auta.etykieta_pelna(a) if av != auta.OBCE else "Obce auto"
                idx = dane.get("i")
                ev = {"b": bb, "crop": "", "plik": "", "osoby": [], "n": 1, "rodzaj": "Pojazd", "lok": "", "hm": "",
                      "auto": nm, "pow": 0, "glosno": 0}
                if idx is not None:
                    w2 = await asyncio.get_running_loop().run_in_executor(
                        None, przemaluj_sync, bb, int(idx), av, nm)
                    if w2:
                        ev.update(plik=w2["plik"], pojazdy=w2["pojazdy"], auto=w2["auto"])
                async with reolink.HA() as ha:
                    await ha.cmd({"type": "fire_event", "event_type": "kamera_zdarzenie_popr", "event_data": ev})
                return web.json_response({"ok": True, "auto": nm, "auta": auta.lista()})
            return web.json_response(strefy.api(json.loads(str(d.get("plik", "{}")) or "{}")))
        except Exception as e:
            log.exception("api")
            return web.json_response({"ok": False, "blad": str(e)})
    if str(d.get("b", "")) == "lista":
        pre = str(d.get("plik", ""))
        pliki = sorted(f.name for f in SNAP.glob("e1_*.jpg") if f.name.startswith("e1_" + pre) and "_crop_" not in f.name and "_best_" not in f.name)
        return web.json_response({"ok": True, "pliki": pliki})
    if str(d.get("b", "")) == "grupa":
        lista = [x.strip() for x in str(d.get("plik", "")).split(",") if x.strip() and "/" not in x]
        asyncio.create_task(grupa_test(lista))
        return web.json_response({"ok": True, "start": lista})
    b = str(d.get("b", "")).strip()
    plik = str(d.get("plik", "")).strip()
    rodzaj = RODZAJ.get(str(d.get("rodzaj", "osoba")).lower(), "osoba")
    granica = str(d.get("granica", ""))
    ctx = {}
    if granica.startswith("{"):
        try:
            ctx = json.loads(granica)
            granica = str(ctx.get("g", ""))
        except ValueError:
            granica = ""
    wait = float(d.get("wait", 25))
    if not b or not plik or "/" in b or "/" in plik:
        return web.json_response({"ok": False, "blad": "zle parametry"}, status=400)
    klasy = KLASY[rodzaj]
    loop = asyncio.get_running_loop()
    t0 = time.time()
    zrodlo = "zdjecie"
    asyncio.create_task(reolink.sd_wideo(b, SNAP / f"e1_v_{b}.mp4", MEDIA, przygotuj_wideo_sync))  # zapas: gdy bufor zawiedzie
    asyncio.create_task(bufor.klip(b, SNAP / f"e1_v_{b}.mp4"))  # klip z bufora: od razu, dokleja sie w trakcie
    async with LOCK:
        wynik = zapas = None
        if bufor.gotowy("m") and bufor.gotowy("s"):
            tz0 = reolink.czas_zdarzenia(b)
            if tz0:
                await bufor.czekaj(min(time.mktime(tz0.timetuple()) + (4.0 if rodzaj == "pojazd" else 3.5), time.time() + 6), 7.0)
            try:
                wynik = await loop.run_in_executor(None, najlepsza_klatka_sync, b, klasy, granica, None, True)
            except Exception:
                log.exception("najlepsza klatka %s", b)
            if wynik:
                zrodlo = "bufor"
        if wynik is None:
            wynik, zapas = await loop.run_in_executor(None, analiza_sync, b, plik, rodzaj, granica, klasy)
        if wynik is None and wait > 0:
            clip = await czekaj_na_klip(b, wait)
            if clip:
                zrodlo = "klip"
                wynik = await loop.run_in_executor(None, analiza_klip_sync, clip, b, klasy)
        if wynik is None and zapas is not None:
            wynik = zapas
            zrodlo = "zdjecie"
    res = {"ok": True, "found": False, "rodzaj": rodzaj, "video": f"e1_v_{b}.mp4", "zrodlo": zrodlo}
    if wynik:
        img, boxes, pelny = wynik
        iw, ih = img.size
        if rodzaj == "pojazd" and zrodlo not in ("klip", "bufor"):
            zapamietaj(boxes, iw, ih, b)
            boxes = ruchome(boxes, iw, ih, b) or boxes
        best, rect, n = wybierz_kadr(boxes, iw, ih)
        crop = f"e1_crop_{b}.jpg"
        cb = await loop.run_in_executor(None, zapisz_kadr, img, rect, SNAP / crop)
        res.update(
            found=True,
            conf=round(best[4], 3),
            n=n,
            strefa_id=strefy.przypisz(best, iw, ih, strefy.wczytaj(granica), rodzaj),
            crop=crop,
            pelny=pelny,
            box=[round(v) for v in best[:4]],
            crop_box=list(cb),
            size=[iw, ih],
        )
    cfg = strefy.wczytaj(granica)
    if not res.get("strefa_id"):
        # brak potwierdzenia detektora: pierwsza strefa z listy (glowna) albo "poza"
        res["strefa_id"] = cfg["strefy"][0]["id"] if cfg["strefy"] else strefy.POZA
    res["strefa"] = strefy.nazwa(cfg, res["strefa_id"])
    res["lok"] = res["strefa"]
    res.update(strefy.decyzja(cfg, res["strefa_id"], rodzaj, res["found"], ctx))
    res["czas"] = round(time.time() - t0, 2)
    LAST_B[rodzaj] = b
    if rodzaj == "osoba" and res["found"]:
        tz = strefy._tz()
        hm = (datetime.now(tz) if tz else datetime.now()).strftime("%H:%M")
        asyncio.create_task(popraw_kadr(b, res.get("n", 1), rodzaj, res["strefa"], hm, res.get("powiadom"), res.get("dzwiek", 1) == 1))
    if rodzaj == "pojazd" and res["found"]:
        tz = strefy._tz()
        hm2 = (datetime.now(tz) if tz else datetime.now()).strftime("%H:%M")
        asyncio.create_task(sledz_pojazd(b, res["strefa"], hm2, granica, res.get("pelny", "")))
    asyncio.create_task(przygotuj_wideo(b))
    asyncio.create_task(rl_i_dopieszcz(b, rodzaj, klasy))
    if rodzaj == "pojazd" and res["found"] and "droga" in str(res.get("strefa", "")).lower():
        asyncio.create_task(przejazd_wpis(b))
    if rodzaj == "pojazd" and res["found"]:
        for ex in EXTRA.pop(b, []):
            asyncio.create_task(przejazd_wpis(ex["key"], ts=ex["ts"]))
    log.info("analiza %s %s -> %s", b, rodzaj, {k: res.get(k) for k in ("found", "conf", "strefa", "powiadom", "historia", "zrodlo", "czas")})
    return web.json_response(res)


async def health(request):
    return web.json_response({"ok": True, "model": MODEL_NAME, "wejscie": [IN_W, IN_H]})


async def sprzatanie():
    while True:
        try:
            gr = time.time() - KEEP_DAYS * 86400
            for pat in ("e1_crop_*.jpg", "e1_best_*.jpg", "e1_v_*.mp4"):
                for f in SNAP.glob(pat):
                    if f.stat().st_mtime < gr:
                        f.unlink(missing_ok=True)
            for f in SNAP.glob("e1_pz_*.jpg"):
                if f.stat().st_mtime < time.time() - PZ_DNI * 86400:
                    f.unlink(missing_ok=True)
            for f in (MEDIA / "e1_ai").glob("*.jpg"):
                if f.stat().st_mtime < time.time() - 3600:
                    f.unlink(missing_ok=True)
            for f in MEDIA.glob("e1_rec_*.mp4"):
                if f.stat().st_mtime < time.time() - 3 * 3600:
                    f.unlink(missing_ok=True)
        except Exception:
            log.exception("sprzatanie")
        await asyncio.sleep(3600)


def instaluj_panel():
    try:
        src = Path("/app/kamera_panel.html")
        dst = HA / "www" / "kamera_panel.html"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        log.info("panel zainstalowany: %s", dst)
    except Exception:
        log.exception("instaluj_panel")


def usun_stare():
    """Jednorazowe sprzatanie po starej wersji z AI (wzorce, kadry e1_ai_f_*, stare crop)."""
    import re

    n = 0
    try:
        for pat in ("e1_ai_f_*", "e1_crop_bf_*", "e1_klatki_*", "e1_test_*"):
            for f in SNAP.glob(pat):
                f.unlink(missing_ok=True)
                n += 1
        for f in SNAP.glob("e1_crop_*.jpg"):
            if re.search(r"_(osoba|pojazd|zwierze)_\d+\.jpg$", f.name):
                f.unlink(missing_ok=True)
                n += 1
        wz = HA / "www" / "wzorce"
        if wz.exists():
            shutil.rmtree(wz, ignore_errors=True)
            n += 1
        for f in MEDIA.glob("wz__*"):
            f.unlink(missing_ok=True)
            n += 1
    except Exception:
        log.exception("usun_stare")
    log.info("sprzatanie starych plikow: %d", n)


CAM = os.environ.get("E1_CAM", "camera.e1_outdoor_wysoka_rozdzielczosc")


HIST_PLIK = PARK_PLIK.with_name("park_hist.json")
SKAN_MIN = 10          # co ile minut skan podjazdu
ELIM_MIN = 0.45        # minimalny wynik, zeby przypisac auto "przez eliminacje"
ELIM_OBCE = 0.05       # i nie gorszy niz 'obce' - tyle
STAB_D = 0.05          # ten sam obiekt = srodek obrysu w tym promieniu (ulamek kadru)
SES_MISS = 2           # tyle skanow z rzedu bez auta = odjechalo


def _hist_load():
    try:
        h = json.load(open(HIST_PLIK))
        h.setdefault("skany", [])
        h.setdefault("sesje", [])
        return h
    except Exception:
        return {"skany": [], "sesje": []}


def _hist_save(h):
    try:
        tmp = HIST_PLIK.with_suffix(".tmp")
        json.dump(h, open(tmp, "w"))
        os.replace(tmp, HIST_PLIK)
    except Exception:
        log.exception("zapis historii parkingu")


def _srodek(n):
    return [(n[0] + n[2]) / 2, (n[1] + n[3]) / 2]


def _blisko(c, lista):
    return any(abs(c[0] - x[0]) < STAB_D and abs(c[1] - x[1]) < STAB_D for x in lista)


def _eliminacja(items, poprzednie):
    """Auta bez pewnego rozpoznania, ktore stoja w miejscu (byly w poprzednim skanie), dostaja wolne znane auto
    o najlepszym wyniku (kazde auto najwyzej raz na skan). Rozwiazuje remisy typu Mazda 0.55 vs obce 0.52."""
    znane = {x["id"]: x for x in auta.lista()}
    wolne = set(znane) - {it["auto_id"] for it in items if it["auto_id"]}
    kand = []
    for it in items:
        if it["auto_id"] or not it.get("f") or not _blisko(_srodek(it["n"]), poprzednie):
            continue
        w = auta.wyniki(it["f"])
        ob = w.get(auta.OBCE, 0.0)
        for k in wolne:
            sc = w.get(k, 0.0)
            if sc >= ELIM_MIN and sc >= ob - ELIM_OBCE:
                kand.append((sc, it["i"], k))
    uzyte_i, uzyte_k = set(), set()
    for sc, i, k in sorted(kand, reverse=True):
        if i in uzyte_i or k in uzyte_k:
            continue
        uzyte_i.add(i)
        uzyte_k.add(k)
        for it in items:
            if it["i"] == i:
                it["auto_id"] = k
                it["nazwa"] = auta.etykieta_pelna(znane[k])
                log.info("eliminacja: auto %s -> %s (wynik %.2f)", i, k, sc)


def _sesje_aktualizuj(h, park, now):
    ses = h["sesje"]
    otw = {s["k"]: s for s in ses if s.get("do") is None}
    widziane = set()
    for p in park:
        k = p.get("auto") or "?"
        widziane.add(k)
        s = otw.get(k)
        if s is None:
            s = {"k": k, "nazwa": p.get("nazwa") or "Nierozpoznane auto", "od": p["t"], "do": None, "ost": now, "miss": 0}
            ses.append(s)
        s["ost"], s["miss"] = now, 0
        if p.get("nazwa"):
            s["nazwa"] = p["nazwa"]
    for k, s in otw.items():
        if k in widziane:
            continue
        s["miss"] = s.get("miss", 0) + 1
        if s["miss"] >= SES_MISS:
            s["do"] = s["ost"] + 300
    h["sesje"] = [s for s in ses if (s.get("do") or now) > now - 30 * 86400]


def skan_sync():
    """Zdjecie z kamery teraz -> auta (rozpoznane po obrysie) -> rejestr zaparkowanych + historia postojow."""
    import urllib.request
    rq = urllib.request.Request(reolink.URL + "/api/camera_proxy/" + CAM, headers={"Authorization": "Bearer " + reolink.TOKEN})
    data = urllib.request.urlopen(rq, timeout=20).read()
    tmp = SNAP / "e1_skan.jpg"
    tmp.write_bytes(data)
    iw, ih = Image.open(tmp).size
    cfg = strefy.wczytaj("")
    pid = _podjazd_id(cfg)
    stare = _park_lista()
    h = _hist_load()
    now = time.time()
    ad = adnotuj_sync("skan", tmp)
    items = ad["items"] if ad else []
    poprzednie = [c for sk in h["skany"][-3:] for c in sk["c"]] + [_srodek(p["box"]) for p in stare if p.get("box")]
    _eliminacja(items, poprzednie)
    park = []
    for it in items:
        n = it["n"]
        px = [n[0] * iw, n[1] * ih, n[2] * iw, n[3] * ih]
        stoi = _blisko(_srodek(n), poprzednie)
        if not it["auto_id"] and pid and strefy.przypisz(px, iw, ih, cfg, "skan") != pid:
            continue  # nieznane auto poza podjazdem (np. na ulicy) nie liczy sie jako zaparkowane
        t = now
        if it["auto_id"]:
            t = next((p["t"] for p in stare if p.get("auto") == it["auto_id"]), t)
        else:
            t = next((p["t"] for p in stare if not p.get("auto") and p.get("box") and
                      abs(_srodek(p["box"])[0] - _srodek(n)[0]) < STAB_D and abs(_srodek(p["box"])[1] - _srodek(n)[1]) < STAB_D), t)
        park.append({"t": t, "b": "skan", "box": n, "auto": it["auto_id"], "nazwa": it["nazwa"], "crop": it["crop"]})
    h["skany"] = (h["skany"] + [{"t": now, "c": [_srodek(it["n"]) for it in items]}])[-8:]
    _sesje_aktualizuj(h, park, now)
    _hist_save(h)
    _park_zapisz(park)
    return park


async def skan_petla():
    await asyncio.sleep(30)
    while True:
        try:
            async with GLOCK:
                await asyncio.get_running_loop().run_in_executor(None, skan_sync)
        except Exception as e:
            log.warning("skan: %s", e)
        await asyncio.sleep(SKAN_MIN * 60)


async def on_start(app):
    instaluj_panel()
    usun_stare()
    app["sprz"] = asyncio.create_task(sprzatanie())
    app["skan"] = asyncio.create_task(skan_petla())
    app["bufor"] = asyncio.create_task(bufor.petla("m"))
    app["bufor_s"] = asyncio.create_task(bufor.petla("s"))


def main():
    app = web.Application(client_max_size=1024 * 1024)
    app.add_routes([web.post("/analyze", analyze), web.get("/health", health)])
    app.on_startup.append(on_start)
    web.run_app(app, host="0.0.0.0", port=8099)


if __name__ == "__main__":
    main()
