# E1 Kamera - detektor

API (port 8099, tylko sieć wewnętrzna HA):

`POST /analyze` `{ "b": "<baza nazwy>", "plik": "<zdjęcie w /config/www/snapshots>", "rodzaj": "Osoba|Pojazd|Zwierzę", "granica": "x,y;x,y;...", "wait": 25 }`

Zwraca ramkę, kadr (`e1_crop_<b>.jpg`), lokalizację (podjazd/droga) i nazwę wideo (`e1_v_<b>.mp4`, gotowe po kilku sekundach).

Opcje: `model` (yolox_s dokładniejszy, yolox_tiny szybszy), `threads`, `margin` (0.2 = 20%), `min_conf`.
