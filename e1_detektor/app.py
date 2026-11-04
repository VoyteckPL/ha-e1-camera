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
from pathlib import Path

import numpy as np
import onnxruntime as ort
from aiohttp import web
from PIL import Image

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
    if snap.exists():
        img = Image.open(snap)
        img.load()
        boxes = detect(img, klasy, MIN_CONF)
        if boxes:
            wynik = (img, boxes, plik)
    return wynik, t0


def analiza_klip_sync(clip: Path, b, klasy):
    tmpd = tempfile.mkdtemp(prefix="e1_")
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(clip), "-vf", "fps=0.8", "-q:v", "3", f"{tmpd}/f_%02d.jpg"],
            capture_output=True, text=True, timeout=120,
        )
        files = sorted(Path(tmpd).glob("f_*.jpg"))
        best = None
        for f in files:
            img = Image.open(f)
            img.load()
            boxes = detect(img, klasy, MIN_CONF)
            if not boxes:
                continue
            sc = max(box_score(x, *img.size) for x in boxes)
            if best is None or sc > best[0]:
                best = (sc, img, boxes)
        if best:
            nazwa = f"e1_best_{b}.jpg"
            best[1].convert("RGB").save(SNAP / nazwa, "JPEG", quality=90)
            return best[1], best[2], nazwa
        return None
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)


async def analyze(request):
    d = await request.json()
    if str(d.get("b", "")) == "api":
        try:
            return web.json_response(strefy.api(json.loads(str(d.get("plik", "{}")) or "{}")))
        except Exception as e:
            log.exception("api")
            return web.json_response({"ok": False, "blad": str(e)})
    if str(d.get("b", "")) == "lista":
        pre = str(d.get("plik", ""))
        pliki = sorted(f.name for f in SNAP.glob("e1_*.jpg") if f.name.startswith("e1_" + pre) and "_crop_" not in f.name and "_best_" not in f.name)
        return web.json_response({"ok": True, "pliki": pliki})
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
        wynik, _ = await loop.run_in_executor(None, analiza_sync, b, plik, rodzaj, granica, klasy)
        if wynik is None and wait > 0:
            clip = await czekaj_na_klip(b, wait)
            if clip:
                zrodlo = "klip"
                wynik = await loop.run_in_executor(None, analiza_klip_sync, clip, b, klasy)
    res = {"ok": True, "found": False, "rodzaj": rodzaj, "video": f"e1_v_{b}.mp4", "zrodlo": zrodlo}
    if wynik:
        img, boxes, pelny = wynik
        iw, ih = img.size
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


async def on_start(app):
    instaluj_panel()
    usun_stare()
    app["sprz"] = asyncio.create_task(sprzatanie())


def main():
    app = web.Application(client_max_size=1024 * 1024)
    app.add_routes([web.post("/analyze", analyze), web.get("/health", health)])
    app.on_startup.append(on_start)
    web.run_app(app, host="0.0.0.0", port=8099)


if __name__ == "__main__":
    main()
