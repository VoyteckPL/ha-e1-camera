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
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
from aiohttp import web
from PIL import Image

import auta
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

LOCK = asyncio.Lock()
GLOCK = asyncio.Lock()  # poprawa kadru z klipu: po jednym zdarzeniu naraz
LAST_B = {}  # rodzaj -> ostatnie zdarzenie (nie nadpisuj nowszego powiadomienia starszym)


def _grids():
    gs, st = [], []
    for s in (8, 16, 32):
        h, w = IN_H // s, IN_W // s
        xv, yv = np.meshgrid(np.arange(w), np.arange(h))
        g = np.stack((xv, yv), 2).reshape(-1, 2)
        gs.append(g)
        st.append(np.full((g.shape[0], 1), s))
    return np.concatenate(gs, 0).astype(np.float32), np.concatenate(st, 0).astype(np.float32)


GRID, STRIDE = _grids()


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


def detect(img: Image.Image, klasy, conf_thr):
    """Zwraca listę (x1, y1, x2, y2, conf) w pikselach obrazu."""
    iw, ih = img.size
    r = min(IN_W / iw, IN_H / ih)
    nw, nh = max(1, int(iw * r)), max(1, int(ih * r))
    small = img.convert("RGB").resize((nw, nh), Image.BILINEAR)
    canvas = np.full((IN_H, IN_W, 3), 114, dtype=np.uint8)
    canvas[:nh, :nw] = np.asarray(small)
    x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32)  # BGR, CHW
    out = SESS.run(None, {INP.name: x})[0][0]
    out[:, :2] = (out[:, :2] + GRID) * STRIDE
    out[:, 2:4] = np.exp(out[:, 2:4]) * STRIDE
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


def przygotuj_wideo_sync(clip: Path, out: Path):
    streams = ffprobe_ok(clip) or []
    vid = next((s for s in streams if s.get("codec_type") == "video"), {})
    tmp = out.with_suffix(".tmp.mp4")
    if vid.get("codec_name") == "h264":
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(clip), "-map", "0:v:0", "-map", "0:a:0?",
               "-c:v", "copy", "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart", str(tmp)]
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
    """Cienki obrys kazdego auta + mala, polprzezroczysta nazwa (zeby nie zaslaniala obrazu)."""
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
            dr.line(pts + [pts[0]], fill=rgb + (235,), width=lw, joint="curve")
            x1, y1 = min(p[0] for p in pts), min(p[1] for p in pts)
        else:
            dr.rectangle([x1, y1, x2, y2], outline=rgb + (235,), width=lw)
        txt = {"kia": "KIA", "seat": "SEAT", "mazda": "MAZDA", "obce": "OBCE"}.get(it.get("auto_id"), "?")
        tb = dr.textbbox((0, 0), txt, font=font)
        tw, th = tb[2] - tb[0] + 8, tb[3] - tb[1] + 6
        if poly and len(poly) >= 3:
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
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
    async with LOCK:
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
        if rodzaj == "pojazd" and zrodlo != "klip":
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
    asyncio.create_task(reolink.reolink_wideo(b, rodzaj, SNAP / f"e1_v_{b}.mp4", MEDIA, przygotuj_wideo_sync))
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


def skan_sync():
    """Zdjecie z kamery teraz -> auta (rozpoznane po obrysie) -> rejestr zaparkowanych."""
    import urllib.request
    rq = urllib.request.Request(reolink.URL + "/api/camera_proxy/" + CAM, headers={"Authorization": "Bearer " + reolink.TOKEN})
    data = urllib.request.urlopen(rq, timeout=20).read()
    tmp = SNAP / "e1_skan.jpg"
    tmp.write_bytes(data)
    iw, ih = Image.open(tmp).size
    cfg = strefy.wczytaj("")
    pid = _podjazd_id(cfg)
    stare = _park_lista()
    ad = adnotuj_sync("skan", tmp)
    park = []
    for it in (ad["items"] if ad else []):
        n = it["n"]
        px = [n[0] * iw, n[1] * ih, n[2] * iw, n[3] * ih]
        if not it["auto_id"] and pid and strefy.przypisz(px, iw, ih, cfg, "skan") != pid:
            continue  # nieznane auto poza podjazdem (np. na ulicy) nie liczy sie jako zaparkowane
        t = time.time()
        if it["auto_id"]:
            t = next((p["t"] for p in stare if p.get("auto") == it["auto_id"]), t)
        park.append({"t": t, "b": "skan", "box": n, "auto": it["auto_id"], "nazwa": it["nazwa"], "crop": it["crop"]})
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
        await asyncio.sleep(20 * 60)


async def on_start(app):
    instaluj_panel()
    usun_stare()
    app["sprz"] = asyncio.create_task(sprzatanie())
    app["skan"] = asyncio.create_task(skan_petla())


def main():
    app = web.Application(client_max_size=1024 * 1024)
    app.add_routes([web.post("/analyze", analyze), web.get("/health", health)])
    app.on_startup.append(on_start)
    web.run_app(app, host="0.0.0.0", port=8099)


if __name__ == "__main__":
    main()
