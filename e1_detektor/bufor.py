"""Bufor kolowy: ciagle nagrywanie glownego strumienia RTSP kamery (kopiowanie bez przekodowania, kawalki ~2 s).

Starsze kawalki sa kasowane (KEEP_S). Z bufora skladamy klip zdarzenia (od razu, w trakcie nagrywania) i wyciagamy
klatki do wyboru najlepszego kadru. Nazwy plikow: m_<epoch poczatku>.ts (strftime %s, czas zegara Pi).
"""
import asyncio
import logging
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import quote, urlparse

import reolink

log = logging.getLogger("e1.bufor")

DIR = Path(os.environ.get("E1_BUF", "/media/e1_buf"))
SEG = 2          # s: dlugosc kawalka (faktycznie do najblizszej klatki kluczowej)
KEEP_S = 45 * 60  # s: tyle trzymamy
PRE = 6.0        # s przed zdarzeniem w klipie
POST = 28.0      # s po poczatku zdarzenia w klipie
RTSP_PORT = 554
RE = re.compile(r"[ms]_(\d+)\.ts$")
_stan = {"start": 0.0, "restarty": 0, "blad": ""}


def _url(pre="m"):
    k = reolink._kam
    if not k["base"] and not reolink._kam_dane():
        return None
    host = urlparse(k["base"]).hostname
    st = "main" if pre == "m" else "sub"
    return f"rtsp://{quote(k['user'], safe='')}:{quote(k['pw'], safe='')}@{host}:{RTSP_PORT}/Preview_01_{st}"


def segmenty(pre="m"):
    out = []
    try:
        for p in DIR.glob(pre + "_*.ts"):
            m = RE.search(p.name)
            if m:
                out.append((int(m.group(1)), p))
    except OSError:
        pass
    out.sort()
    return out


def okno(od, do, pre="m"):
    """Zamkniete kawalki nachodzace na [od, do]: lista (poczatek, koniec, sciezka). Ostatni (otwarty) pomijamy."""
    s = segmenty(pre)
    out = []
    for i in range(len(s) - 1):
        t, p = s[i]
        e = min(s[i + 1][0], t + 8)
        if e > od and t < do:
            out.append((t, e, p))
    return out


def gotowy(pre="m"):
    s = segmenty(pre)
    return len(s) >= 3 and time.time() - s[-1][0] < 15


def sprzataj():
    gr = time.time() - KEEP_S
    for pre in ("m", "s"):
        for t, p in segmenty(pre):
            if t < gr:
                p.unlink(missing_ok=True)


def status():
    out = {"restarty": _stan["restarty"], "blad": _stan["blad"], "teraz": round(time.time()), "tz": time.tzname[0]}
    for pre in ("m", "s"):
        s = segmenty(pre)
        sz = 0
        for _, p in s:
            try:
                sz += p.stat().st_size
            except OSError:
                pass
        out[pre] = {"kawalki": len(s), "od": s[0][0] if s else 0, "do": s[-1][0] if s else 0, "mb": round(sz / 1e6, 1), "gotowy": gotowy(pre)}
    return out


async def petla(pre="m"):
    DIR.mkdir(parents=True, exist_ok=True)
    while True:
        url = _url(pre)
        if not url:
            await asyncio.sleep(30)
            continue
        if pre == "m":
            kodek = ["-map", "0:v:0", "-map", "0:a:0?", "-c:v", "copy", "-c:a", "aac", "-b:a", "64k"]
        else:
            kodek = ["-map", "0:v:0", "-c:v", "copy", "-an"]
        cmd = ["ffmpeg", "-nostdin", "-loglevel", "warning", "-rtsp_transport", "tcp", "-i", url, *kodek,
               "-f", "segment", "-segment_time", str(SEG), "-segment_format", "mpegts",
               "-strftime", "1", str(DIR / (pre + "_%s.ts"))]
        logf = Path(tempfile.gettempdir()) / f"bufor_ffmpeg_{pre}.log"
        p = None
        start = time.time()
        try:
            with open(logf, "wb") as lf:
                p = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL, stderr=lf)
                log.info("bufor[%s]: ffmpeg wystartowal", pre)
                tsp = 0.0
                while True:
                    try:
                        await asyncio.wait_for(p.wait(), 5)
                        break
                    except asyncio.TimeoutError:
                        pass
                    if pre == "m" and time.time() - tsp > 60:
                        tsp = time.time()
                        sprzataj()
                    s = segmenty(pre)
                    ost = max(s[-1][0], start) if s else start
                    try:
                        if s:
                            ost = max(ost, s[-1][1].stat().st_mtime)
                    except OSError:
                        pass
                    if time.time() - ost > 25:
                        log.warning("bufor[%s]: brak nowych kawalkow od 25 s - restart ffmpeg", pre)
                        p.kill()
                        await p.wait()
                        break
        except Exception as e:
            log.exception("bufor")
            _stan["blad"] = str(e)[:100]
        finally:
            try:
                if p and p.returncode is None:
                    p.kill()
            except Exception:
                pass
        try:
            tail = logf.read_text(errors="ignore")[-300:].replace(url, "***")
        except Exception:
            tail = ""
        _stan["restarty"] += 1
        _stan["blad"] = tail.strip()[-100:]
        log.warning("bufor[%s]: ffmpeg zakonczony (%s), ponawiam za 5 s: %s", pre, p.returncode if p else "?", tail.strip()[-200:])
        await asyncio.sleep(5)


def _lista(kawalki, plik: Path):
    plik.write_text("".join(f"file '{p}'\n" for _, _, p in kawalki))


def _scal(kawalki):
    """Kawalki mpegts mozna po prostu skleic bajtami (znaczniki czasu ciagle). Duzo szybciej niz demuxer concat (probe kazdego pliku)."""
    buf = bytearray()
    for _, _, p in kawalki:
        try:
            buf += p.read_bytes()
        except OSError:
            pass
    return bytes(buf)


def zloz(kawalki, out: Path):
    tmp = out.with_suffix(".tmp.mp4")
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "pipe:0", "-c", "copy", "-movflags", "+faststart", str(tmp)],
                       input=_scal(kawalki), capture_output=True, timeout=120)
    if r.returncode != 0:
        log.error("bufor zloz: %s", r.stderr[-300:].decode(errors="ignore"))
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(out)
    return True


async def klip(b, out: Path, t0=None):
    """Sklada klip zdarzenia z bufora i dokleja go w miare nagrywania. Zwraca True, gdy klip jest pelny."""
    if t0 is None:
        t = reolink.czas_zdarzenia(b)
        if t is None:
            return False
        t0 = time.mktime(t.timetuple())
    od, do = t0 - PRE, t0 + POST
    start = time.time()
    pub = 0.0
    loop = asyncio.get_running_loop()
    while time.time() - start < POST + 45:
        if b in reolink.GOTOWE:
            return True
        k = okno(od, do)
        if k:
            cov = k[-1][1] - k[0][0]
            pelny = k[-1][1] >= do
            if cov >= pub + 3 or (pelny and cov > pub):
                tz_ = time.time()
                if await loop.run_in_executor(None, zloz, k, out):
                    pub = cov
                    log.info("bufor: %s klip %.0f s (%d kawalkow), skladanie %.1f s, %.1f s od przyjscia zdarzenia", b, cov, len(k), time.time() - tz_, time.time() - start)
                    if pelny:
                        reolink.GOTOWE.add(b)
                        return True
        elif time.time() - start > 25 and t0 < time.time() - 60 and not segmenty():
            return False
        await asyncio.sleep(1.0)
    return False


def klatki(od, do, fps, tmpd, pre="m"):
    """Wyciaga klatki z okna [od, do] (jpg). Zwraca liste sciezek; klatka k ~ czas od + k/fps."""
    k = okno(od, do, pre)
    if not k:
        return []
    ss = max(0.0, od - k[0][0])
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "pipe:0",
                        "-ss", f"{ss:.2f}", "-t", f"{do - od:.2f}", "-vf", f"fps={fps}", "-q:v", "4",
                        str(Path(tmpd) / (pre + "_%03d.jpg"))], input=_scal(k), capture_output=True, timeout=120)
    if r.returncode != 0:
        log.error("bufor klatki: %s", r.stderr[-300:].decode(errors="ignore"))
    return sorted(Path(tmpd).glob(pre + "_*.jpg"))


def klatka_pelna(t, out: Path):
    """Jedna klatka w pelnej rozdzielczosci z glownego strumienia w chwili t (epoch). True, gdy sie udalo."""
    s = segmenty("m")
    for i in range(len(s) - 1, -1, -1):
        if s[i][0] <= t:
            off = max(0.0, t - s[i][0])
            for p in (s[i][1], ) if off < 7 else ():
                r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(p), "-ss", f"{off:.2f}", "-frames:v", "1",
                                    "-q:v", "3", str(out)], capture_output=True, text=True, timeout=60)
                if r.returncode == 0 and out.exists() and out.stat().st_size > 1000:
                    return True
            return False
    return False


async def czekaj(do, limit=10.0):
    """Czeka, az bufor obejmie czas `do` (lub limit)."""
    end = time.time() + limit
    while time.time() < end:
        k = okno(do - 1, do)
        if k and k[-1][1] >= do:
            return True
        await asyncio.sleep(0.5)
    return False
