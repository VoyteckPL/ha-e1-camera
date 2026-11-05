"""E1 strojenie 0.1 - tylko zbiera dane (kontekst + metryki obrazu + zdarzenia + etykiety). Nie zmienia ustawien kamery."""
import asyncio
import json
import math
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


# ---- bezposredni dostep do kamery (Isp: ekspozycja, wzmocnienie, migawka...) - dane logowania z integracji Reolink w HA ----
_cam = {"base": None, "user": None, "pw": None, "token": None, "exp": 0}


def _wczytaj_kamere():
    try:
        d = json.load(open("/ha_config/.storage/core.config_entries"))
        for e in d["data"]["entries"]:
            if e.get("domain") == "reolink":
                x = e["data"]
                sch = "https" if x.get("use_https") else "http"
                _cam.update(base=f"{sch}://{x['host']}:{x.get('port') or (443 if sch == 'https' else 80)}", user=x["username"], pw=x["password"])
                return True
    except Exception as ex:
        log.warning("brak danych kamery: %s", ex)
    return False


async def cam_cmd(cmd, param=None, action=0, ch=0):
    if not _cam["base"] and not _wczytaj_kamere():
        raise RuntimeError("brak danych kamery")
    if time.time() > _cam["exp"] - 30:
        body = [{"cmd": "Login", "param": {"User": {"Version": "0", "userName": _cam["user"], "password": _cam["pw"]}}}]
        async with _sess.post(_cam["base"] + "/api.cgi?cmd=Login", json=body, ssl=False, timeout=aiohttp.ClientTimeout(total=15)) as r:
            j = await r.json(content_type=None)
        log.info("login kamera: %s", json.dumps(j)[:300])
        t = j[0]["value"]["Token"]
        _cam.update(token=t["name"], exp=time.time() + int(t.get("leaseTime", 3600)))
    body = [{"cmd": cmd, "action": action, "param": param or {"channel": ch}}]
    async with _sess.post(f"{_cam['base']}/api.cgi?cmd={cmd}&token={_cam['token']}", json=body, ssl=False, timeout=aiohttp.ClientTimeout(total=15)) as r:
        j = await r.json(content_type=None)
    return j[0]


async def isp_get():
    j = await cam_cmd("GetIsp", action=1)
    return j["value"]["Isp"] if "value" in j else j


async def isp_set(zmiany):
    """Pobiera pelny Isp, nakłada zmiany (slownik, zagnieżdżone np. {"gain": {"max": 80}}), wysyla z powrotem."""
    isp = await isp_get()
    for k, v in zmiany.items():
        if isinstance(v, dict) and isinstance(isp.get(k), dict):
            isp[k].update(v)
        else:
            isp[k] = v
    return await cam_cmd("SetIsp", param={"Isp": isp})


SENSOR = "sensor.kamera_strojenie"
AUTO = "input_boolean.kamera_auto_presety"
N = "number.e1_outdoor_"
E_OS, E_PO, E_ZW, E_RU = (N + "czulosc_wykrywania_osob_przez_ai", N + "czulosc_wykrywania_pojazdow_przez_ai",
                          N + "czulosc_wykrywania_zwierzat_przez_ai", N + "czulosc_na_ruch")
E_JA, E_KO, E_NA, E_OT = N + "jasnosc_obrazu", N + "kontrast_obrazu", N + "nasycenie_obrazu", N + "ostrosc_obrazu"
# --- plynne auto-strojenie: wartosci docelowe wyliczane z ciaglej jasnosci zewnetrznej, deszczu/sniegu i nauczonych poprawek ---
DAY = {E_OS: 50, E_PO: 50, E_ZW: 50, E_RU: 35, E_JA: 128, E_KO: 128, E_NA: 128, E_OT: 128}
NIGHT = {E_OS: 70, E_PO: 75, E_ZW: 65, E_RU: 40, E_JA: 165, E_KO: 125, E_NA: 120, E_OT: 120}
DELTA_WET = {E_OS: -5, E_PO: -10, E_ZW: -10, E_RU: -10}  # przy deszczu/sniegu (skalowane 0..1)
SENS = (E_OS, E_PO, E_ZW, E_RU)
DRC_DAY, DRC_NOC = 128, 170
LUX_JASNO, LUX_CIEMNO = 60.0, 8.0  # powyzej: pelny dzien, ponizej: pelna noc, pomiedzy plynnie (skala log)
DESZCZ_PO_S = 20 * 60
HOLD_S = 3 * 3600  # reczna zmiana suwaka wstrzymuje auto dla tego suwaka na 3 h
RODZAJ_ENT = {"Osoba": E_OS, "Pojazd": E_PO, "Zwierzę": E_ZW}
F_OFF = DATA / "offsety.json"
_auto = {"f": None, "wet": 0.0, "od": None, "opis": "", "wlaczone": False, "applied": {}, "hold": {}, "drc": None, "ucz_dzien": None}


def f_noc(lux):
    l = max(float(lux), 1.0)
    f = (math.log10(LUX_JASNO) - math.log10(l)) / (math.log10(LUX_JASNO) - math.log10(LUX_CIEMNO))
    return min(1.0, max(0.0, f))


def mokro(c):
    r = 0.0
    if c.get("deszcz"):
        r = 1.0
    elif c.get("deszcz") == 0 and c.get("deszcz_zmiana_s") is not None and c["deszcz_zmiana_s"] < DESZCZ_PO_S:
        r = 1.0 - c["deszcz_zmiana_s"] / DESZCZ_PO_S
    p = str(c.get("pogoda") or "")
    if p in ("snowy", "snowy-rainy", "hail"):
        r = max(r, 1.0)
    elif p in ("rainy", "pouring", "lightning-rainy"):
        r = max(r, 0.6)
    return r


def _offsety():
    try:
        return json.load(open(F_OFF))
    except Exception:
        return {}


def ucz_offsety():
    """Raz na dobe: z ostatnich 7 dni (zdarzenia + Twoje 'falszywe') koryguje czulosc osobno dla dnia i nocy. Maly krok, twarde granice."""
    od = time.time() - 7 * 86400
    etyk = {}
    for e in _wczytaj(F_ETYK, 5000):
        etyk[e["b"]] = e.get("falszywy", 0)
    kub = {}
    for e in _wczytaj(F_ZDARZ, 5000):
        ent = RODZAJ_ENT.get(e.get("rodzaj"))
        try:
            if not ent or datetime.strptime(e["czas"], "%Y-%m-%d %H:%M:%S").timestamp() < od:
                continue
        except Exception:
            continue
        pora = "noc" if ((e.get("ctx") or {}).get("lux") or 0) < LUX_CIEMNO + 7 else "dzien"
        k = kub.setdefault((ent, pora), [0, 0])
        k[0] += 1
        k[1] += 1 if etyk.get(e["b"]) else 0
    off = _offsety()
    for (ent, pora), (n, fal) in kub.items():
        key = f"{ent}|{pora}"
        v = off.get(key, 0)
        if n >= 8 and fal / n > 0.35:
            v = max(-20, v - 5)
        elif n >= 12 and fal / n < 0.08:
            v = min(10, v + 3)
        off[key] = v
        log.info("uczenie %s: n=%d falszywych=%d -> poprawka %+d", key, n, fal, v)
    json.dump(off, open(F_OFF, "w"))


async def ustaw(eid, val):
    """Ustaw encje number (REST service call). True gdy sie udalo."""
    try:
        async with _sess.post(f"{API}/services/{eid.split('.')[0]}/set_value", headers=H(),
                              json={"entity_id": eid, "value": val}, timeout=aiohttp.ClientTimeout(total=20)) as r:
            return r.status < 300
    except Exception as ex:
        log.warning("ustaw %s: %s", eid, ex)
        return False


def cele(f, w):
    off = _offsety()
    t = {}
    for e in DAY:
        v = DAY[e] + f * (NIGHT[e] - DAY[e])
        if e in SENS:
            v += (1 - f) * off.get(f"{e}|dzien", 0) + f * off.get(f"{e}|noc", 0)
            v += DELTA_WET.get(e, 0) * w
            v = min(95, max(20, v))
        t[e] = int(round(v))
    return t


async def petla_auto():
    """Co 2 min: plynnie dopasowuje czulosci i obraz do jasnosci/pogody. Reczne zmiany suwakow wstrzymuja auto dla danego suwaka na 3 h."""
    await asyncio.sleep(45)
    while True:
        try:
            on = (await stan(AUTO) or {}).get("state") == "on"
            _auto["wlaczone"] = on
            if on:
                c = await kontekst()
                lux = c.get("lux")
                if lux is not None:
                    f0 = f_noc(lux)
                    f = f0 if _auto["f"] is None else _auto["f"] + 0.5 * (f0 - _auto["f"])  # wygladzanie
                    w = mokro(c)
                    _auto.update(f=f, wet=w)
                    now = time.time()
                    if time.localtime().tm_hour == 4 and _auto["ucz_dzien"] != time.strftime("%Y-%m-%d"):
                        _auto["ucz_dzien"] = time.strftime("%Y-%m-%d")
                        ucz_offsety()
                    zm = {}
                    for e, tv in cele(f, w).items():
                        cur = _num((await stan(e) or {}).get("state"))
                        ap = _auto["applied"].get(e)
                        if cur is not None and ap is not None and abs(cur - ap) >= 2:  # ktos zmienil recznie
                            _auto["hold"][e] = now + HOLD_S
                            _auto["applied"].pop(e, None)
                        if now < _auto["hold"].get(e, 0) or cur is None:
                            continue
                        if abs(tv - cur) >= 3 or (e == E_RU and tv != cur and abs(tv - cur) >= 2):
                            if await ustaw(e, tv):
                                _auto["applied"][e] = tv
                                zm[e.split(".")[1][-18:]] = tv
                    # DRC kamery: wlacza sie przy zmroku (histereza 0.2/0.35), sila rosnie z ciemnoscia
                    if _auto["drc"] is None:  # pierwszy obieg: odczytaj aktualny stan DRC z kamery
                        try:
                            ii = await isp_get()
                            _auto["drc"] = int(ii.get("drc", 0)) if ii.get("backLight") == "DynamicRangeControl" else 0
                        except Exception as ex:
                            log.warning("isp: %s", ex)
                            _auto["drc"] = 0
                    drc_on = _auto["drc"] > 0
                    want = f >= 0.35 or (drc_on and f >= 0.2)
                    drc_v = int(round(DRC_DAY + f * (DRC_NOC - DRC_DAY))) if want else 0
                    if (drc_v > 0) != drc_on or (drc_v > 0 and abs(drc_v - (_auto["drc"] or 0)) >= 8):
                        try:
                            await isp_set({"backLight": "DynamicRangeControl" if drc_v else "Off", "drc": drc_v or DRC_DAY})
                            _auto["drc"] = drc_v
                            zm["drc"] = drc_v
                        except Exception as ex:
                            log.warning("drc: %s", ex)
                    _auto["opis"] = f"{'Noc' if f > 0.5 else 'Dzień'} {round(f * 100)}%" + (f" · deszcz/śnieg {round(w * 100)}%" if w > 0.05 else "")
                    if zm:
                        _auto["od"] = int(now)
                        zapisz(DATA / "zmiany.jsonl", {"t": int(now), "f": round(f, 2), "wet": round(w, 2), "lux": lux, "zm": zm})
                        log.info("auto: %s -> %s", _auto["opis"], zm)
                    await pchnij()
            else:
                _auto.update(f=None, applied={}, hold={}, drc=None)
        except Exception:
            log.exception("auto")
        await asyncio.sleep(120)


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
    return {"tryb": OPTS.get("tryb", "zbieranie"), "ctx": _last_ctx, "stat": _stat, "zd7": zd, "etykiet": len(etyk), "seria": seria,
            "auto": {"wlaczone": _auto.get("wlaczone", False), "opis": _auto["opis"], "od": _auto["od"], "f": _auto["f"], "wet": _auto["wet"]}}


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


async def isp_api(request):
    """GET /isp - odczyt; POST /isp {"zmiany": {...}} - zapis (z zapamietaniem poprzednich wartosci w /data/isp_prev.json)."""
    try:
        if request.method == "POST":
            d = await request.json()
            zm = d.get("zmiany") or {}
            prev = await isp_get()
            json.dump(prev, open(DATA / "isp_prev.json", "w"))
            res = await isp_set(zm)
            return web.json_response({"ok": True, "wynik": res, "isp": await isp_get()})
        j = await cam_cmd("GetIsp", action=1)
        return web.json_response({"ok": True, "isp": j.get("value", {}).get("Isp"), "zakres": j.get("range", {}).get("Isp")})
    except Exception as e:
        log.exception("isp")
        return web.json_response({"ok": False, "blad": str(e)})


async def cam_odczyt(request):
    """GET /odczyt?c=GetAiAlarm&ai=people - tylko komendy Get* (diagnostyka/odkrywanie ustawien kamery)."""
    c = request.query.get("c", "")
    if not c.startswith("Get") or not c.isalnum():
        return web.json_response({"ok": False, "blad": "tylko Get*"})
    try:
        param = {"channel": 0}
        if request.query.get("ai"):
            param["ai_type"] = request.query["ai"]
        j = await cam_cmd(c, param=param, action=1)
        return web.json_response({"ok": True, "odp": j})
    except Exception as e:
        return web.json_response({"ok": False, "blad": str(e)})


async def cam_szukaj(request):
    """GET /szukaj?min=15 - lista nagran na karcie SD z ostatnich N minut (tylko odczyt)."""
    try:
        n = min(240, max(1, int(request.query.get("min", 15))))
        k = time.time()
        a, b = time.localtime(k - n * 60), time.localtime(k)
        tm = lambda t: {"year": t.tm_year, "mon": t.tm_mon, "day": t.tm_mday, "hour": t.tm_hour, "min": t.tm_min, "sec": t.tm_sec}
        prm = {"Search": {"channel": 0, "onlyStatus": 0, "streamType": request.query.get("st", "main"), "StartTime": tm(a), "EndTime": tm(b)}}
        j = await cam_cmd("Search", param=prm, action=1)
        return web.json_response({"ok": True, "teraz": time.strftime("%H:%M:%S"), "odp": j})
    except Exception as e:
        return web.json_response({"ok": False, "blad": str(e)})


async def eksport(request):
    n = min(5000, int(request.query.get("n", 500)))
    plik = {"probki": F_PROBKI, "zdarzenia": F_ZDARZ, "etykiety": F_ETYK}.get(request.query.get("co", "probki"), F_PROBKI)
    return web.json_response(_wczytaj(plik, n))


async def on_start(app):
    global _sess
    DATA.mkdir(parents=True, exist_ok=True)
    _sess = aiohttp.ClientSession()
    app["t"] = [asyncio.create_task(f()) for f in (petla_probek, petla_zdarzen, petla_etykiet, petla_auto)]


def main():
    app = web.Application()
    app.add_routes([web.get("/status", status), web.get("/dane", eksport), web.get("/isp", isp_api), web.get("/odczyt", cam_odczyt), web.get("/szukaj", cam_szukaj), web.post("/isp", isp_api)])
    app.on_startup.append(on_start)
    web.run_app(app, host="0.0.0.0", port=8100)


if __name__ == "__main__":
    main()
