"""E1 strojenie 0.1 - tylko zbiera dane (kontekst + metryki obrazu + zdarzenia + etykiety). Nie zmienia ustawien kamery."""
import asyncio
import json
import logging
import os
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

import aiohttp
import numpy as np
from aiohttp import web
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("e1.tuner")

DATA = Path(os.environ.get("E1_DATA", "/data"))
TOKEN = os.environ.get("SUPERVISOR_TOKEN", "")
API = "http://supervisor/core/api"
WS = "ws://supervisor/core/websocket"
try:
    OPTS = json.load(open("/data/options.json"))
except Exception:
    OPTS = {}
INTERWAL = int(OPTS.get("probkowanie_min", 10)) * 60

CAM = "camera.e1_outdoor_wysoka_rozdzielczosc"
HIST = "sensor.kamera_historia_zdarzen"
LUX = "sensor.jasnosc_zewnetrzna"
DESZCZ = "binary_sensor.my_rainsensor_multisensor_rain_0"
POGODA = "weather.openweathermap"
SWIATLA = {"podjazd": "light.podjazd", "reflektor": "light.e1_outdoor_reflektor"}
LICZBY = {
    "czul_osoby": "number.e1_outdoor_czulosc_wykrywania_osob_przez_ai",
    "czul_pojazdy": "number.e1_outdoor_czulosc_wykrywania_pojazdow_przez_ai",
    "czul_zwierzeta": "number.e1_outdoor_czulosc_wykrywania_zwierzat_przez_ai",
    "czul_ruch": "number.e1_outdoor_czulosc_na_ruch",
}
F_PROBKI, F_ZDARZ, F_ETYK = DATA / "probki.jsonl", DATA / "zdarzenia.jsonl", DATA / "etykiety.jsonl"
MAX_LINII = 60000

_sess = None
_last_ctx = {}
_stat = {"probki": 0, "zdarzenia": 0, "etykiety": 0, "start": time.time(), "blad": ""}


def H():
    return {"Authorization": "Bearer " + TOKEN}


async def stan(eid):
    try:
        async with _sess.get(f"{API}/states/{eid}", headers=H(), timeout=aiohttp.ClientTimeout(total=15)) as r:
            return await r.json() if r.status == 200 else None
    except Exception:
        return None


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _wiek_s(iso):
    try:
        return int(time.time() - datetime.fromisoformat(iso).timestamp())
    except Exception:
        return None


async def kontekst():
    """Biezacy kontekst: pora, jasnosc zewnetrzna, deszcz (i od kiedy), pogoda, swiatla, czulosci kamery."""
    now = datetime.now().astimezone()
    c = {"t": int(time.time()), "czas": now.strftime("%Y-%m-%d %H:%M:%S"), "godz": round(now.hour + now.minute / 60, 2),
         "dzien_roku": now.timetuple().tm_yday, "dow": now.weekday()}
    names = {"lux": LUX, "deszcz": DESZCZ, "pogoda": POGODA, "slonce": "sun.sun"}
    names.update({"sw_" + k: v for k, v in SWIATLA.items()})
    names.update(LICZBY)
    res = dict(zip(names, await asyncio.gather(*[stan(v) for v in names.values()])))
    s = res["lux"]
    c["lux"] = _num(s["state"]) if s else None
    s = res["deszcz"]
    if s:
        c["deszcz"] = 1 if s["state"] == "on" else 0
        c["deszcz_zmiana_s"] = _wiek_s(s.get("last_changed", ""))
    s = res["pogoda"]
    if s:
        a = s.get("attributes", {})
        c.update(pogoda=s["state"], temp=a.get("temperature"), wiatr=a.get("wind_speed"), porywy=a.get("wind_gust_speed"),
                 wilg=a.get("humidity"), chmury=a.get("cloud_coverage"), widocz=a.get("visibility"))
    s = res["slonce"]
    if s:
        c["slonce_elew"] = s.get("attributes", {}).get("elevation")
    for k in SWIATLA:
        s = res["sw_" + k]
        if s:
            c["sw_" + k] = 1 if s["state"] == "on" else 0
            c["sw_" + k + "_zmiana_s"] = _wiek_s(s.get("last_changed", ""))
    for k in LICZBY:
        s = res[k]
        if s:
            c[k] = _num(s["state"])
    return c


def metryki(data):
    """Metryki obrazu z jednego kadru: jasnosc, przepalenia, zaciemnienia, ostrosc, szum."""
    im = Image.open(BytesIO(data)).convert("L")
    w, h = im.size
    sc = 640 / max(w, h)
    if sc < 1:
        im = im.resize((int(w * sc), int(h * sc)))
    a = np.asarray(im, dtype=np.float32)
    lap = a[1:-1, 1:-1] * 4 - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:]
    ciemne = a[1:-1, 1:-1] < 60  # szum mierzymy w ciemnych obszarach (tam jest najlepiej widoczny)
    szum = float(np.median(np.abs(lap[ciemne])) if ciemne.sum() > 500 else np.median(np.abs(lap)))
    return {"jasn": round(float(a.mean()), 1), "p5": round(float(np.percentile(a, 5)), 1), "p95": round(float(np.percentile(a, 95)), 1),
            "przepal": round(float((a > 250).mean()), 4), "czern": round(float((a < 5).mean()), 4),
            "ostrosc": round(float(lap.var()), 1), "szum": round(szum, 2), "kontrast": round(float(a.std()), 1)}


async def obraz():
    try:
        async with _sess.get(f"{API}/camera_proxy/{CAM}", headers=H(), timeout=aiohttp.ClientTimeout(total=30)) as r:
            if r.status != 200:
                return None
            return metryki(await r.read())
    except Exception as e:
        log.warning("obraz: %s", e)
        return None


def zapisz(plik, obj):
    try:
        with open(plik, "a") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        if plik.stat().st_size > 20_000_000:  # prosta rotacja: zostaw ostatnie MAX_LINII
            L = plik.read_text().splitlines()[-MAX_LINII:]
            plik.write_text("\n".join(L) + "\n")
    except Exception:
        log.exception("zapis %s", plik)


async def petla_probek():
    global _last_ctx
    await asyncio.sleep(15)
    while True:
        try:
            c = await kontekst()
            m = await obraz()
            if m:
                c["obraz"] = m
            _last_ctx = c
            zapisz(F_PROBKI, c)
            _stat["probki"] += 1
            await pchnij()
        except Exception as e:
            _stat["blad"] = str(e)
            log.exception("probka")
        await asyncio.sleep(INTERWAL)


def _znane():
    s = set()
    try:
        for l in open(F_ZDARZ):
            s.add(json.loads(l)["b"])
    except Exception:
        pass
    return s


async def petla_zdarzen():
    """Co 30 s sprawdza historie zdarzen kamery; nowe zapisuje razem z biezacym kontekstem."""
    znane = _znane()
    pierwszy = not znane  # przy pierwszym starcie nie zalewaj plikami historii - bierzemy tylko nowe
    await asyncio.sleep(20)
    while True:
        try:
            s = await stan(HIST)
            lista = (s or {}).get("attributes", {}).get("zdarzenia", []) or []
            if pierwszy:
                znane |= {e["b"] for e in lista}
                pierwszy = False
            else:
                nowe = [e for e in lista if e["b"] not in znane]
                if nowe:
                    c = await kontekst()
                    for e in reversed(nowe):
                        znane.add(e["b"])
                        zapisz(F_ZDARZ, {"b": e["b"], "czas": e.get("czas"), "rodzaj": e.get("rodzaj"), "lok": e.get("lok"), "conf": e.get("conf"),
                                         "pow": e.get("pow"), "auto": e.get("auto"), "ctx": c})
                        _stat["zdarzenia"] += 1
                    await pchnij()
        except Exception as e:
            _stat["blad"] = str(e)
            log.exception("zdarzenia")
        await asyncio.sleep(30)


async def petla_etykiet():
    """Websocket: etykiety 'falszywy alarm' z panelu (event kamera_etykieta)."""
    while True:
        try:
            async with _sess.ws_connect(WS, heartbeat=30) as ws:
                await ws.receive_json()
                await ws.send_json({"type": "auth", "access_token": TOKEN})
                if (await ws.receive_json()).get("type") != "auth_ok":
                    raise RuntimeError("auth")
                await ws.send_json({"id": 1, "type": "subscribe_events", "event_type": "kamera_etykieta"})
                async for msg in ws:
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue
                    d = json.loads(msg.data)
                    if d.get("type") != "event":
                        continue
                    e = d["event"]["data"]
                    zapisz(F_ETYK, {"t": int(time.time()), "b": e.get("b"), "falszywy": int(e.get("falszywy", 0)), "rodzaj": e.get("rodzaj"),
                                    "lok": e.get("lok"), "conf": e.get("conf"), "czas": e.get("czas")})
                    _stat["etykiety"] += 1
                    log.info("etykieta: %s falszywy=%s", e.get("b"), e.get("falszywy"))
                    await pchnij()
        except Exception as e:
            log.warning("ws: %s", e)
        await asyncio.sleep(10)


SENSOR = "sensor.kamera_strojenie"


def podsumowanie():
    """Dane dla panelu: ostatnie 7 dni zdarzen (z liczba oznaczonych jako falszywe) + seria metryk obrazu."""
    od = time.time() - 7 * 86400
    etyk = {}
    for e in _wczytaj(F_ETYK, 5000):
        etyk[e["b"]] = e.get("falszywy", 0)
    zd = {}
    for e in _wczytaj(F_ZDARZ, 5000):
        try:
            if datetime.strptime(e["czas"], "%Y-%m-%d %H:%M:%S").timestamp() < od:
                continue
        except Exception:
            continue
        r = zd.setdefault(e.get("rodzaj") or "?", {"n": 0, "fal": 0})
        r["n"] += 1
        r["fal"] += 1 if etyk.get(e["b"]) else 0
    seria = []
    for p in _wczytaj(F_PROBKI, 72):
        o = p.get("obraz") or {}
        seria.append({"t": p["t"], "jasn": o.get("jasn"), "lux": p.get("lux"), "szum": o.get("szum"), "ostr": o.get("ostrosc"),
                      "prz": o.get("przepal"), "czern": o.get("czern"), "dz": p.get("deszcz")})
    return {"tryb": OPTS.get("tryb", "zbieranie"), "ctx": _last_ctx, "stat": _stat, "zd7": zd, "etykiet": len(etyk), "seria": seria}


async def pchnij():
    try:
        p = podsumowanie()
        async with _sess.post(f"{API}/states/{SENSOR}", headers=H(), timeout=aiohttp.ClientTimeout(total=15),
                              json={"state": _stat["probki"], "attributes": dict(p, friendly_name="Kamera strojenie", icon="mdi:tune-variant")}) as r:
            if r.status >= 300:
                log.warning("sensor %s", r.status)
    except Exception as e:
        log.warning("pchnij: %s", e)


def _wczytaj(plik, n):
    try:
        return [json.loads(l) for l in open(plik).read().splitlines()[-n:]]
    except Exception:
        return []


async def status(request):
    return web.json_response({"ok": True, "tryb": OPTS.get("tryb", "zbieranie"), "stat": _stat, "ctx": _last_ctx,
                              "ostatnie_zdarzenia": _wczytaj(F_ZDARZ, 5), "ostatnie_etykiety": _wczytaj(F_ETYK, 5)})


async def eksport(request):
    n = min(5000, int(request.query.get("n", 500)))
    plik = {"probki": F_PROBKI, "zdarzenia": F_ZDARZ, "etykiety": F_ETYK}.get(request.query.get("co", "probki"), F_PROBKI)
    return web.json_response(_wczytaj(plik, n))


async def on_start(app):
    global _sess
    DATA.mkdir(parents=True, exist_ok=True)
    _sess = aiohttp.ClientSession()
    app["t"] = [asyncio.create_task(f()) for f in (petla_probek, petla_zdarzen, petla_etykiet)]


def main():
    app = web.Application()
    app.add_routes([web.get("/status", status), web.get("/dane", eksport)])
    app.on_startup.append(on_start)
    web.run_app(app, host="0.0.0.0", port=8100)


if __name__ == "__main__":
    main()
