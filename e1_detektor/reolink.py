"""Pobieranie nagran zdarzen z karty SD kamery Reolink przez media_source Home Assistanta."""
import asyncio
import logging
import os
import re
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

import aiohttp

import strefy

log = logging.getLogger("e1.reolink")

TOKEN = os.environ.get("SUPERVISOR_TOKEN", "")
WS = os.environ.get("E1_HA_WS", "ws://supervisor/core/websocket")
URL = os.environ.get("E1_HA_URL", "http://supervisor/core")
TYTUL = {"osoba": "Person", "pojazd": "Vehicle", "zwierze": "Animal"}


class HA:
    def __init__(self):
        self.i = 0

    async def __aenter__(self):
        self.s = aiohttp.ClientSession()
        self.ws = await self.s.ws_connect(WS, heartbeat=30, max_msg_size=0)
        await self.ws.receive_json()  # auth_required
        await self.ws.send_json({"type": "auth", "access_token": TOKEN})
        r = await self.ws.receive_json()
        if r.get("type") != "auth_ok":
            raise RuntimeError("auth HA: %s" % r.get("type"))
        return self

    async def __aexit__(self, *a):
        await self.ws.close()
        await self.s.close()

    async def cmd(self, payload):
        self.i += 1
        await self.ws.send_json(dict(payload, id=self.i))
        while True:
            m = await asyncio.wait_for(self.ws.receive_json(), 40)
            if m.get("id") == self.i and m.get("type") == "result":
                if not m.get("success"):
                    raise RuntimeError(str(m.get("error")))
                return m.get("result")


def czas_zdarzenia(b):
    m = re.match(r"e1_(\d{8})_(\d{6})_", b)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def dlugosc(p):
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                           capture_output=True, text=True, timeout=20)
        return float(r.stdout.strip())
    except Exception:
        return 0.0


async def wpis_kamery(ha):
    r = await ha.cmd({"type": "media_source/browse_media", "media_content_id": "media-source://reolink"})
    cid = r["children"][0]["media_content_id"]  # media-source://reolink/CAM|<entry>|0
    return cid.split("/", 3)[-1].split("|")[1]


async def pliki_dnia(ha, entry, d):
    mid = f"media-source://reolink/DAY|{entry}|0|main|{d.year}|{d.month}|{d.day}"
    r = await ha.cmd({"type": "media_source/browse_media", "media_content_id": mid})
    out = []
    for c in r.get("children") or []:
        p = c["media_content_id"].split("|")
        try:
            fs = datetime.strptime(p[-2], "%Y%m%d%H%M%S")
            fe = datetime.strptime(p[-1], "%Y%m%d%H%M%S")
        except (ValueError, IndexError):
            continue
        out.append({"fs": fs, "fe": fe, "id": c["media_content_id"], "tytul": c.get("title", "")})
    return out


def wybierz(pliki, t, kat):
    cand = [f for f in pliki if f["fs"] - timedelta(seconds=3) <= t <= f["fe"] + timedelta(seconds=3)]
    if not cand:
        return None
    slowo = TYTUL.get(kat, "")
    cand.sort(key=lambda f: (slowo not in f["tytul"], abs((f["fs"] - (t - timedelta(seconds=6))).total_seconds())))
    return cand[0]


async def pobierz(ha, mid, cel: Path):
    r = await ha.cmd({"type": "media_source/resolve_media", "media_content_id": mid, "expires": 120})
    url = r["url"]
    if url.startswith("/"):
        url = URL + url
    tmp = cel.with_suffix(".part")
    async with aiohttp.ClientSession() as s:
        # adres z authSig jest podpisany: bez naglowka Authorization (z nim HA odpowiada 400)
        hdr = {} if "authSig=" in url else {"Authorization": "Bearer " + TOKEN}
        async with s.get(url, headers=hdr, timeout=aiohttp.ClientTimeout(total=300)) as resp:
            if resp.status >= 400:
                raise RuntimeError("pobieranie %s: %s" % (resp.status, (await resp.text())[:200]))
            resp.raise_for_status()
            with open(tmp, "wb") as fh:
                async for chunk in resp.content.iter_chunked(1 << 20):
                    fh.write(chunk)
    tmp.replace(cel)
    return cel


CONV = asyncio.Lock()
GOTOWE = set()  # zdarzenia, dla ktorych jest juz wideo z Reolinka


async def reolink_wideo(b, kat, out: Path, media: Path, konwertuj, limit=300):
    """Czeka na nagranie Reolinka dla zdarzenia b, pobiera je i podmienia wideo out."""
    t = czas_zdarzenia(b)
    if not TOKEN or t is None:
        return False
    tz = strefy._tz()
    start = time.time()
    await asyncio.sleep(10)
    while time.time() - start < limit:
        try:
            async with HA() as ha:
                entry = await wpis_kamery(ha)
                teraz = datetime.now(tz).replace(tzinfo=None)
                f = wybierz(await pliki_dnia(ha, entry, t.date()), t, kat)
                if f and f["fe"] <= teraz - timedelta(seconds=4):
                    raw = media / f"e1_rl_{b}.mp4"
                    await pobierz(ha, f["id"], raw)
                    d = dlugosc(raw)
                    if d >= (f["fe"] - f["fs"]).total_seconds() - 4:
                        async with CONV:
                            ok = await asyncio.get_running_loop().run_in_executor(None, konwertuj, raw, out)
                        raw.unlink(missing_ok=True)
                        if ok:
                            GOTOWE.add(b)
                            log.info("wideo reolink gotowe %s (%s, %.0f s, po %.0f s)", out.name, f["tytul"], d, time.time() - start)
                            return True
                    else:
                        log.info("nagranie reolink jeszcze niepelne %s (%.0f s)", b, d)
                    raw.unlink(missing_ok=True)
        except Exception as e:
            log.warning("reolink %s: %s", b, e)
        await asyncio.sleep(10)
    log.warning("brak nagrania reolink dla %s", b)
    return False
