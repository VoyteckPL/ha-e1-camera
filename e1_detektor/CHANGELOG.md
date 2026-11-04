## 0.1.0
- Pierwsza wersja: detekcja YOLOX, kadr z marginesem, wideo, lokalizacja podjazd/droga.

## 0.2.0
- Nowy panel: tylko osoby / samochody / zwierzęta, konfigurator powiadomień, arkusz zdarzenia z zoomem zdjęcia i wideo (mini pasek przewijania). Panel jest kopiowany do `www/kamera_panel.html` przy starcie.

## 0.2.1
- Jednorazowe sprzątanie plików po starej wersji z AI.

## 0.2.2
- Pomocnicze: `b=lista` w /analyze zwraca zdjęcia o danym prefiksie.

## 0.2.3
- Wideo: gdy kamera nie nagra własnego klipu (równoległe zdarzenia), używany jest klip z tego samego momentu.

## 0.3.0
- Strefy: dowolna liczba nazwanych stref z kolorami, przyciąganiem punktów i osobnymi ustawieniami powiadomień dla każdej strefy i kategorii (osoba / samochód / zwierzę).
- Przypisanie do strefy wg większości punktów przy stopach obiektu, z histerezą na granicy.
- Decyzja o powiadomieniu i historii zapada w add-onie.
- Wideo: nagranie 15 s po wykryciu.

## 0.3.1
- Edytor stref: mniejsze punkty, zoom (dwa palce, kółko, przyciski) i przesuwanie, brak zaznaczania tekstu/obrazu przy klikaniu.

## 0.3.2
- Edytor: zoom kółkiem obsługiwany na poziomie okna (pewniejszy), znacznik wersji panelu.

## 0.3.3
- Log: czas trwania wideo.

## 0.4.0
- Wideo: główne źródło to nagranie Reolinka z karty SD (z pre-recordem). Klip z HA zostaje jako szybki podgląd i zapas. Wymaga dostępu add-onu do API HA.

## 0.4.1
- Pobieranie nagrań: podpisany adres bez nagłówka Authorization.

## 0.4.2
- Pobieranie nagrań: najpierw bezpośrednio z HA (podpisany adres), potem przez proxy Supervisora; czytelniejsze błędy.

## 0.4.3
- Kadr pojazdu: detektor uczy się stojących pojazdów (ten sam obiekt w tym samym miejscu w co najmniej 3 wcześniejszych zdarzeniach z ostatnich 12 h, np. zaparkowane auto) i wybiera na zdjęcie pojazd w ruchu. Gdy na zdjęciu są tylko stojące pojazdy, szukany jest ruchomy w klipie; jeśli go brak, pokazywany jest stojący jak dotychczas.

## 0.4.4
- Panel (v27): arkusz zdarzenia na pełnym ekranie (zdjęcie i wideo), zoom szczypnięciem, kółkiem i podwójnym dotknięciem.
- Ikona zapisu zdjęcia lub wideo (udostępnianie na telefonie / pobieranie na komputerze).
- Przełącznik „Wycinek” / „Cały kadr”.

## 0.4.5
- Panel (v28): przycisk wyciszenia dźwięku w odtwarzaczu wideo (zapamiętuje wybór).

## 0.5.0
- Zdarzenia z osobą: po chwili detektor przegląda klip (do 20 s) i wybiera klatkę, na której widać najwięcej osób razem. Zdjęcie grupowe podmienia kadr w historii i powiadomieniu (to samo `tag`, bez ponownego dźwięku).
- Gdy w klatce jest co najmniej 2 osoby, powstają też osobne kadry każdej osoby; w arkuszu zdarzenia (panel v29) pojawiają się jako miniatury z zoomem i zapisem.
- Pomocniczo: `b=grupa` w /analyze uruchamia tę analizę na istniejącym klipie (test).

## 0.5.1
- Analiza grupy z klipu: 1 klatka/s przez pierwsze 18 s (szybciej na Pi); `b=grupa` działa w tle i zapisuje wynik w logu.

## 0.6.0
- Sledzenie pojazdow w klipie: wjazd / wyjazd / przejazd wzgledem strefy Podjazd (pole `ruch` w historii), rejestr zaparkowanych aut (/data/zaparkowane.json, `b=zaparkowane`), test `b=sledz`.

## 0.6.1
- Sledzenie i poprawa kadru: gdy brak zywego klipu, czeka na wideo z Reolinka.

## 0.6.2
- Licznik zaparkowanych aut jako sensor.kamera_zaparkowane (b=zaparkowane, plik=reset|sensor).

## 0.7.0
- Rozpoznawanie aut domownikow (Kia Sportage 5, Seat Altea, Mazda 5) po kolorze + uczenie z etykiet w panelu; pasek zaparkowanych aut w panelu; nazwy aut w rejestrze i sensorze.

## 0.7.1
- Skan zaparkowanych aut co 20 min i na zadanie (b=zaparkowane, plik=skan): zdjecie z kamery -> auta w strefie Podjazd -> rejestr. Mazda 5: srebrna.

## 0.7.2
- Skan: znane auta liczone w calym kadrze, nieznane tylko na podjezdzie.

## 0.7.3
- Panel v32: przyciski nazwy auta tez na kartach i osi czasu.

## 0.8.0
- Obrysy wszystkich aut na kadrze (kolor i nazwa), osobne oznaczanie kazdego auta (pojazdy w historii), panel v33.

## 0.8.1
- Obrys karoserii (kanciasty wielokat z GrabCut, opencv) zamiast prostokata.

## 0.8.2
- Skan zapisuje tez kadr z obrysami (e1_best_skan_a0.jpg).

## 0.8.3
- Panel v34: ponawia pobranie listy aut, gdy dodatek chwilowo niedostepny (znikaly przyciski oznaczania).

## 0.8.4
- Test b=obrys: dorysowanie obrysow do istniejacych zdarzen.

## 0.8.5
- Usuwanie podwojnych wykryc tego samego auta (ramka zawarta w innej).

## 0.8.6
- Dedupe aut: tez przy IoU > 0.35.
