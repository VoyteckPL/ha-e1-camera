# Ślad ruchu

Słucha `frigate/reviews` (MQTT). Gdy alert się kończy, pobiera trasę obiektu z Frigate (`/api/events/<id>`), klatki z nagrania (`/api/<kamera>/recordings/<czas>/snapshot.jpg`), składa je w jeden obraz z godzinami i zapisuje w `/config/www/slad_ruchu/` (w HA: `/local/slad_ruchu/...`). Potem wywołuje `script.kamera_powiad_wyslij` z tym samym tagiem, więc obraz podmienia wcześniejsze powiadomienie, bez dźwięku.

Działa tylko, gdy w panelu „Powiadomienia kamer” załącznik to „Ślad ruchu” (opcja `tylko_gdy_wybrany`).
Obraz powstaje kilka sekund po końcu zdarzenia (`opoznienie_s`, Frigate musi dopisać nagranie).
