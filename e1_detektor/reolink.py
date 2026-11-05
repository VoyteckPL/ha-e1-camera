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
    sciezka = url if url.startswith("/") else "/" + url.split("/", 3)[-1]
    # 1) bezposrednio do HA z podpisanym adresem, 2) przez proxy Supervisora z tokenem
    proby = [("http://homeassistant:8123" + sciezka, {}), (URL + sciezka, {"Authorization": "Bearer " + TOKEN})]
    tmp = cel.with_suffix(".part")
    blad = ""
    async with aiohttp.ClientSession() as s:
        for u, hdr in proby:
            try:
                async with s.get(u, headers=hdr, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                    if resp.status >= 400:
                        blad += " [%s %s: %s]" % (u.split("/")[2], resp.status, (await resp.text())[:120])
                        continue
                    with open(tmp, "wb") as fh:
                        async for chunk in resp.content.iter_chunked(1 << 20):
                            fh.write(chunk)
                tmp.replace(cel)
                return cel
            except aiohttp.ClientError as e:
                blad += " [%s: %s]" % (u.split("/")[2], e)
    raise RuntimeError("pobieranie nieudane:" + blad)


CONV = asyncio.Lock()
GOTOWE = set()  # zdarzenia, dla ktorych jest juz wideo z Reolinka


async def reolink_wideo(b, kat, out: Path, media: Path, konwertuj, limit=300):
    """Czeka na nagranie Reolinka dla zdarzenia b, pobiera je i podmienia wideo out."""
    t = czas_zdarzenia(b)
    if not TOKEN or t is None:
        return False
    tz = strefy._tz()
    start = time.time()
    await asyncio.sleep(5)
    while time.time() - start < limit and b not in GOTOWE:
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
        await asyncio.sleep(3)
    log.warning("brak nagrania reolink dla %s", b)
    return False


# ---------------- bezposrednio z kamery po IP (karta SD) ----------------
import json

_kam = {"base": "", "user": "", "pw": "", "token": "", "exp": 0.0}


def _kam_dane():
    try:
        d = json.load(open("/ha_config/.storage/core.config_entries"))
        for e in d["data"]["entries"]:
            if e.get("domain") == "reolink":
                x = e["data"]
                sch = "https" if x.get("use_https") else "http"
                _kam.update(base=f"{sch}://{x['host']}:{x.get('port') or (443 if sch == 'https' else 80)}", user=x["username"], pw=x["password"])
                return True
    except Exception as ex:
        log.warning("sd: brak danych kamery: %s", ex)
    return False


async def _kam_token(s):
    if not _kam["base"] and not _kam_dane():
        raise RuntimeError("brak danych kamery")
    if time.time() > _kam["exp"] - 30:
        body = [{"cmd": "Login", "param": {"User": {"Version": "0", "userName": _kam["user"], "password": _kam["pw"]}}}]
        async with s.post(_kam["base"] + "/api.cgi?cmd=Login", json=body, ssl=False, timeout=aiohttp.ClientTimeout(total=15)) as r:
            j = await r.json(content_type=None)
        t = j[0]["value"]["Token"]
        _kam.update(token=t["name"], exp=time.time() + int(t.get("leaseTime", 3600)))
    return _kam["token"]


def _tm(t):
    return {"year": t.year, "mon": t.month, "day": t.day, "hour": t.hour, "min": t.minute, "sec": t.second}


def _dt(x):
    return datetime(x["year"], x["mon"], x["day"], x["hour"], x["min"], x["sec"])


async def sd_szukaj(s, od, do):
    tok = await _kam_token(s)
    prm = {"Search": {"channel": 0, "onlyStatus": 0, "streamType": "main", "StartTime": _tm(od), "EndTime": _tm(do)}}
    body = [{"cmd": "Search", "action": 1, "param": prm}]
    async with s.post(f"{_kam['base']}/api.cgi?cmd=Search&token={tok}", json=body, ssl=False, timeout=aiohttp.ClientTimeout(total=15)) as r:
        j = await r.json(content_type=None)
    fl = (j[0].get("value") or {}).get("SearchResult", {}).get("File", []) or []
    return [{"name": f["name"], "fs": _dt(f["StartTime"]), "fe": _dt(f["EndTime"]), "size": int(f.get("size") or 0)} for f in fl]


async def sd_pobierz(s, name, cel: Path):
    tok = await _kam_token(s)
    tmp = cel.with_suffix(".part")
    url = f"{_kam['base']}/api.cgi?cmd=Download&source={name}&output={name.split('/')[-1]}&token={tok}"
    async with s.get(url, ssl=False, timeout=aiohttp.ClientTimeout(total=120)) as r:
        if r.status >= 400:
            raise RuntimeError("download %s" % r.status)
        n = 0
        with open(tmp, "wb") as fh:
            async for ch in r.content.iter_chunked(1 << 20):
                fh.write(ch)
                n += len(ch)
    tmp.replace(cel)
    return n


async def sd_wideo(b, out: Path, media: Path, konwertuj, limit=240):
    """Odpytuje kamere po IP co ~1.5 s. Gdy plik zdarzenia jest na SD - pobiera go (takze w trakcie nagrywania, jesli kamera
    na to pozwala), publikuje coraz dluzsze wideo, a po zakonczeniu nagrania - pelne. Wszystko jest logowane ('sd:')."""
    t = czas_zdarzenia(b)
    if t is None:
        return False
    start = time.time()
    raw = media / f"e1_sd_{b}.mp4"
    pub = 0.0          # dlugosc ostatnio opublikowanego wideo
    sig_prev, stale = None, 0
    pierwszy = None
    try:
        async with aiohttp.ClientSession() as s:
            while time.time() - start < limit and b not in GOTOWE:
                try:
                    teraz = datetime.now()
                    fl = await sd_szukaj(s, t - timedelta(seconds=40), min(teraz, t + timedelta(minutes=10)) + timedelta(seconds=5))
                    cand = [f for f in fl if f["fs"] <= t + timedelta(seconds=8) and f["fe"] >= t - timedelta(seconds=4)]
                    if cand:
                        f = max(cand, key=lambda x: x["fs"])
                        if pierwszy is None:
                            pierwszy = time.time()
                            log.info("sd: %s plik widoczny po %.1f s: %s start=%s end=%s size=%d teraz=%s", b, pierwszy - start,
                                     f["name"].split("/")[-1], f["fs"].time(), f["fe"].time(), f["size"], teraz.time().replace(microsecond=0))
                        sig = (f["fe"], f["size"])
                        stale = stale + 1 if sig == sig_prev else 0
                        if sig != sig_prev:
                            sig_prev = sig
                            n = await sd_pobierz(s, f["name"], raw)
                            d = dlugosc(raw)
                            oczek = (f["fe"] - f["fs"]).total_seconds()
                            log.info("sd: %s pobrano %.1f MB, ffprobe=%.1f s (oczekiwane %.0f s, %.1f s od startu)", b, n / 1e6, d, oczek, time.time() - start)
                            if d >= 2 and d >= pub + 3 and b not in GOTOWE:
                                async with CONV:
                                    ok = await asyncio.get_running_loop().run_in_executor(None, konwertuj, raw, out)
                                if ok:
                                    pub = d
                                    log.info("sd: %s opublikowano %.1f s", b, d)
                        # koniec: nic sie nie zmienia, plik skonczony
                        if stale >= 2 and f["fe"] <= datetime.now() - timedelta(seconds=4) and pub >= (f["fe"] - f["fs"]).total_seconds() - 4:
                            GOTOWE.add(b)
                            log.info("sd: %s KONIEC, pelne wideo %.1f s po %.1f s", b, pub, time.time() - start)
                            return True
                except Exception as e:
                    log.warning("sd %s: %s", b, e)
                await asyncio.sleep(1.5)
    finally:
        raw.unlink(missing_ok=True)
    return False
