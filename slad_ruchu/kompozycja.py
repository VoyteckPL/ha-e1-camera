"""Składanie klatek zdarzenia w jeden obraz (ślad ruchu).

Tło to mediana wszystkich klatek, więc poruszający się obiekt z niego znika.
Z każdej próbki wklejamy na tło tylko piksele, które różnią się od tła
w okolicy ramki obiektu, i dopisujemy nad nim godzinę.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

PROG_ROZNICY = 32       # 0-255, różnica od tła liczona jako maks. z kanałów
MIN_POKRYCIE = 0.03     # gdy maska jest mniejsza (np. obiekt stoi), wklej miękką elipsę
MARGINES = 0.25         # powiększenie ramki z każdej strony


@dataclass
class Probka:
    czas: float
    obraz: Image.Image
    ramka: Optional[tuple]  # (x, y, w, h) znormalizowane 0-1, albo None
    opis: str               # np. "12:25:31"


def _do_szerokosci(img: Image.Image, szer: int) -> Image.Image:
    img = img.convert("RGB")
    if img.width == szer:
        return img
    h = round(img.height * szer / img.width)
    return img.resize((szer, h), Image.LANCZOS)


def _czcionka(rozmiar: int):
    try:
        return ImageFont.load_default(size=rozmiar)
    except TypeError:  # starszy Pillow
        return ImageFont.load_default()


def _obszar(ramka, W, H):
    x, y, w, h = ramka
    mx, my = w * MARGINES, h * MARGINES
    x0 = max(0, int((x - mx) * W))
    y0 = max(0, int((y - my) * H))
    x1 = min(W, int((x + w + mx) * W))
    y1 = min(H, int((y + h + my) * H))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return x0, y0, x1, y1


def _maska(klatka: np.ndarray, tlo: np.ndarray, obszar) -> np.ndarray:
    """Maska 0-1 (float) w obrębie obszaru."""
    x0, y0, x1, y1 = obszar
    a = klatka[y0:y1, x0:x1].astype(np.int16)
    b = tlo[y0:y1, x0:x1].astype(np.int16)
    roznica = np.abs(a - b).max(axis=2)
    m = Image.fromarray(((roznica > PROG_ROZNICY) * 255).astype(np.uint8), "L")
    # usuń drobny szum, domknij sylwetkę, zmiękcz krawędzie
    m = m.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.MinFilter(3))
    m = m.filter(ImageFilter.GaussianBlur(2))
    arr = np.asarray(m, dtype=np.float32) / 255.0
    if arr.mean() < MIN_POKRYCIE:
        # obiekt prawie nie różni się od tła: wklej miękką elipsę z całej ramki
        e = Image.new("L", (x1 - x0, y1 - y0), 0)
        d = ImageDraw.Draw(e)
        d.ellipse((0, 0, x1 - x0 - 1, y1 - y0 - 1), fill=255)
        e = e.filter(ImageFilter.GaussianBlur(max(2, (x1 - x0) // 10)))
        arr = np.asarray(e, dtype=np.float32) / 255.0
    return arr


def zloz(probki: list, tla: list, szerokosc: int = 1280) -> Image.Image:
    """probki: lista Probka; tla: dodatkowe klatki tylko do wyliczenia tła."""
    if not probki:
        raise ValueError("brak probek")
    probki = sorted(probki, key=lambda p: p.czas)
    obrazy = [_do_szerokosci(p.obraz, szerokosc) for p in probki]
    W, H = obrazy[0].size
    tla_r = [_do_szerokosci(t, szerokosc) for t in tla]
    tla_r = [t for t in tla_r if t.size == (W, H)]
    stos = np.stack([np.asarray(o) for o in obrazy + tla_r])
    tlo = np.median(stos, axis=0).astype(np.uint8)
    wynik = tlo.astype(np.float32)

    etykiety = []
    punkty = []
    for p, o in zip(probki, obrazy):
        kl = np.asarray(o)
        obszar = _obszar(p.ramka, W, H) if p.ramka else (0, 0, W, H)
        if obszar is None:
            continue
        m = _maska(kl, tlo, obszar)[..., None]
        x0, y0, x1, y1 = obszar
        wynik[y0:y1, x0:x1] = kl[y0:y1, x0:x1] * m + wynik[y0:y1, x0:x1] * (1 - m)
        if p.ramka:
            x, y, w, h = p.ramka
            etykiety.append((int((x + w / 2) * W), int(y * H), p.opis))
            punkty.append((int((x + w / 2) * W), int((y + h) * H)))
        else:
            etykiety.append((W // 2, 10, p.opis))

    obraz = Image.fromarray(np.clip(wynik, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
    nakladka = Image.new("RGBA", obraz.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(nakladka)
    if len(punkty) > 1:
        d.line(punkty, fill=(255, 200, 0, 150), width=max(2, W // 400), joint="curve")
        for px, py in punkty:
            r = max(3, W // 300)
            d.ellipse((px - r, py - r, px + r, py + r), fill=(255, 200, 0, 200))
    f = _czcionka(max(14, W // 55))
    for cx, top, tekst in etykiety:
        l, t, r, b = d.textbbox((0, 0), tekst, font=f)
        tw, th = r - l, b - t
        pad = max(3, th // 4)
        x0 = min(max(0, cx - tw // 2 - pad), W - tw - 2 * pad)
        y0 = max(0, top - th - 3 * pad)
        d.rounded_rectangle((x0, y0, x0 + tw + 2 * pad, y0 + th + 2 * pad), radius=pad, fill=(0, 0, 0, 150))
        d.text((x0 + pad - l, y0 + pad - t), tekst, font=f, fill=(255, 255, 255, 255))
    return Image.alpha_composite(obraz, nakladka).convert("RGB")


def wybierz_probki(punkty: list, max_klatek: int) -> list:
    """punkty: [(czas, ramka)] posortowane; zwraca podzbiór rozłożony równo w czasie."""
    if len(punkty) <= max_klatek:
        return punkty
    t0, t1 = punkty[0][0], punkty[-1][0]
    krok = (t1 - t0) / (max_klatek - 1) if t1 > t0 else 0
    wybrane, cel = [], t0
    for p in punkty:
        if p[0] >= cel - 1e-6:
            wybrane.append(p)
            cel = p[0] + krok
    if wybrane[-1] is not punkty[-1]:
        if len(wybrane) >= max_klatek:
            wybrane[-1] = punkty[-1]
        else:
            wybrane.append(punkty[-1])
    return wybrane[:max_klatek]
