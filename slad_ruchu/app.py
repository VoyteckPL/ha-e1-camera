"""Ślad ruchu - po zakończeniu zdarzenia Frigate składa klatki z nagrania w jeden obraz
(obiekt w kilku miejscach z godzinami) i podmienia nim załącznik powiadomienia.

Wejście: MQTT frigate/reviews (type=end). Frigate: /api/events/<id> (trasa obiektu)
i /api/<kamera>/recordings/<czas>/snapshot.jpg (klatka z nagrania).
Wyjście: /config/www/slad_ruchu/<review>.jpg (w HA jako /local/slad_ruchu/...) oraz
wywołanie script.kamera_powiad_wyslij z tym samym tagiem, więc podmienia powiadomienie.
"""
import io
import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import paho.mqtt.client as mqtt
import requests
from PIL import Image

import kompozycja

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("slad")

OPTS = {}
try:
    OPTS = json.load(open("/data/options.json"))
except Exception:
    pass

FRIGATE = OPTS.get("frigate_url", "http://192.168.1.31:5050").rstrip("/")
TOKEN = os.environ.get("SUPERVISOR_TOKEN", "")
HA_API = os.environ.get("E1_HA_API", "http://supervisor/core/api")
HAH = {"Authorization": f"Bearer {TOKEN}"}
OUT = Path(os.environ.get("E1_OUT", "/config/www/slad_ruchu"))
OUT.mkdir(parents=True, exist_ok=True)
TZ = ZoneInfo(os.environ.get("TZ", "Europe/Warsaw"))
GOTOWE = {}  # review id -> czas, żeby nie robić dwa razy
LOCK = threading.Lock()


def ha_stan(entity):
    try:
        r = requests.get(f"{HA_API}/states/{entity}", headers=HAH, timeout=10)
        return r.json().get("state") if r.ok else None
    except Exception:
        return None


def mqtt_dane():
    if os.environ.get("E1_MQTT_HOST"):
        return os.environ["E1_MQTT_HOST"], int(os.environ.get("E1_MQTT_PORT", 1883)), None, None
    r = requests.get("http://supervisor/services/mqtt", headers=HAH, timeout=10).json()["data"]
    return r["host"], int(r["port"]), r.get("username"), r.get("password")


def pobierz_klatke(kamera, ts):
    try:
        r = requests.get(f"{FRIGATE}/api/{kamera}/recordings/{ts:.3f}/snapshot.jpg", timeout=20)
        if r.ok and r.headers.get("content-type", "").startswith("image"):
            return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception as e:
        log.warning("klatka %s: %s", ts, e)
    return None


def trasa(event_id):
    """[(czas, (x, y, w, h))] z path_data zdarzenia; punkt trasy to środek dolnej krawędzi ramki."""
    r = requests.get(f"{FRIGATE}/api/events/{event_id}", timeout=20)
    r.raise_for_status()
    d = r.json().get("data") or {}
    box = d.get("box") or [0, 0, 0.1, 0.2]
    w, h = box[2], box[3]
    wynik = []
    for p in d.get("path_data") or []:
        (x, y), ts = p[0], p[1]
        wynik.append((float(ts), (x - w / 2, y - h, w, h)))
    return sorted(wynik)


def zrob(review):
    rid = review["id"]
    kamera = review["camera"]
    detekcje = (review.get("data") or {}).get("detections") or []
    if not detekcje:
        return
    punkty = []
    for ev in detekcje:
        try:
            punkty += trasa(ev)
        except Exception as e:
            log.warning("trasa %s: %s", ev, e)
    punkty.sort()
    if len(punkty) < 2:
        log.info("%s: za krótka trasa (%d pkt), pomijam", rid, len(punkty))
        return
    wybrane = kompozycja.wybierz_probki(punkty, int(OPTS.get("max_klatek", 7)))
    probki = []
    for ts, ramka in wybrane:
        img = pobierz_klatke(kamera, ts)
        if img is not None:
            opis = datetime.fromtimestamp(ts, TZ).strftime("%H:%M:%S")
            probki.append(kompozycja.Probka(ts, img, ramka, opis))
    if len(probki) < 2:
        log.info("%s: za mało klatek (%d), pomijam", rid, len(probki))
        return
    # dodatkowe klatki tylko do tła: początek i koniec zdarzenia, gdy obiekt jest poza kadrem
    tla = []
    for ts in (review.get("start_time"), review.get("end_time")):
        if ts:
            i = pobierz_klatke(kamera, float(ts))
            if i is not None:
                tla.append(i)
    wynik = kompozycja.zloz(probki, tla, int(OPTS.get("szerokosc", 1280)))
    sciezka = OUT / f"{rid}.jpg"
    wynik.save(sciezka, "JPEG", quality=85)
    log.info("zapisano %s (%d klatek)", sciezka, len(probki))
    wyslij(review, rid, probki)


def wyslij(review, rid, probki):
    """Podmień powiadomienie (ten sam tag) obrazem śladu, bez dźwięku."""
    if ha_stan("input_boolean.kamera_powiad_wlaczone") != "on":
        log.info("powiadomienia wyłączone lub HA nie odpowiada, obraz zostaje w /local/slad_ruchu/")
        return
    ev = ((review.get("data") or {}).get("detections") or [""])[0]
    dane = {
        "obiekt": "ślad ruchu", "emoji": "🎞️", "strefa": ", ".join((review.get("data") or {}).get("zones") or []).capitalize(),
        "kamera": review["camera"].upper(), "godzina": probki[0].opis[:5], "pora": "",
        "tryb": "Ciche",
        "obraz": f"/local/slad_ruchu/{rid}.jpg", "tag": f"frigate_{rid}",
        "klik": f"/galeria-frigate/galeria?event={ev}", "alert_once": True, "test": False,
    }
    try:
        r = requests.post(f"{HA_API}/services/script/kamera_powiad_wyslij", headers=HAH, json=dane, timeout=20)
        log.info("powiadomienie: HTTP %s", r.status_code)
    except Exception as e:
        log.warning("powiadomienie: %s", e)


def czy_wybrane():
    if not OPTS.get("tylko_gdy_wybrany", True):
        return True
    return ha_stan(OPTS.get("encja_zalacznika", "input_select.kamera_powiad_zalacznik")) == OPTS.get("opcja_zalacznika", "Ślad ruchu")


def obsluz(review):
    rid = review["id"]
    with LOCK:
        if rid in GOTOWE:
            return
        GOTOWE[rid] = time.time()
    time.sleep(int(OPTS.get("opoznienie_s", 15)))  # Frigate dopisuje nagranie z opóźnieniem
    try:
        zrob(review)
    except Exception:
        log.exception("błąd składania %s", rid)


def sprzatanie():
    limit = time.time() - int(OPTS.get("dni_przechowywania", 7)) * 86400
    for f in OUT.glob("*.jpg"):
        try:
            if f.stat().st_mtime < limit:
                f.unlink()
        except Exception:
            pass
    with LOCK:
        for k in [k for k, v in GOTOWE.items() if v < time.time() - 86400]:
            GOTOWE.pop(k, None)


def on_message(client, userdata, msg):
    try:
        p = json.loads(msg.payload)
    except Exception:
        return
    if p.get("type") != "end":
        return
    review = p.get("after") or {}
    if OPTS.get("tylko_alerty", True) and review.get("severity") != "alert":
        return
    if not czy_wybrane():
        return
    threading.Thread(target=obsluz, args=(review,), daemon=True).start()


def main():
    host, port, user, pwd = mqtt_dane()
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2) if hasattr(mqtt, "CallbackAPIVersion") else mqtt.Client()
    if user:
        c.username_pw_set(user, pwd)
    c.on_message = on_message
    c.on_connect = lambda cl, *a, **k: cl.subscribe("frigate/reviews")
    c.connect(host, port, 60)
    log.info("słucham frigate/reviews na %s:%s, Frigate %s", host, port, FRIGATE)
    t = threading.Thread(target=lambda: [(sprzatanie(), time.sleep(3600)) for _ in iter(int, 1)], daemon=True)
    t.start()
    c.loop_forever()


if __name__ == "__main__":
    main()
