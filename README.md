# E1 Kamera

Aplikacja (add-on) Home Assistant: lokalny detektor osób, samochodów i zwierząt (YOLOX, ONNX) dla kamery Reolink E1 Outdoor.

- wycina z zdjęcia zdarzenia kadr z 20% marginesem,
- gdy obiektu nie widać na zdjęciu, szuka go na klatkach z nagrania,
- przygotowuje wideo zdarzenia do odtwarzania w przeglądarce,
- wszystko lokalnie, bez chmury i bez AI.

Instalacja: Ustawienia → Aplikacje → Sklep → ⋮ → Repozytoria → `https://github.com/VoyteckPL/ha-e1-camera`.

## Ślad ruchu (slad_ruchu)

Składa zdarzenie z Frigate w jeden obraz (obiekt w kilku miejscach z godzinami) do powiadomień. Szczegóły: slad_ruchu/DOCS.md.
