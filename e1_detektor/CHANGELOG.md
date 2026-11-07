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

## 0.8.7
- Obrysy i nazwy aut od razu po zdarzeniu (przed klipem).

## 0.8.8
- Rozpoznawanie: Mazda bez priora koloru (ciemne auta nie sa Mazda/Kia), wyzszy prog; etykiety na zdjeciu sie nie nakladaja.

## 0.8.9
- Obrys prostszy: wypukly, max 8 punktow.

## 0.9.0
- Panel v35: przyciski aut w jednym rzedzie pod miniatura (bez lamania tekstu).

## 0.9.1
- Panel v36: oznaczanie aut chowane (przycisk Oznacz auta / Ukryj).

## 0.9.2
- Rozpoznawanie aut po kolorach wewnatrz obrysu (bez tla) z wymaganym marginesem przewagi; skan i etykiety uzywaja wektorow cech; przycisk Obce sie zaznacza (panel v37).

## 0.9.3
- Obrysy dokladniejsze (do ~26 punktow, GrabCut 480 px), cienkie linie, mala polprzezroczysta nazwa.

## 0.9.4
- Opisy na zdjeciu: KIA / MAZDA / SEAT w srodku auta, male, polprzezroczyste.

## 0.9.5
- Opisy aut wyraznie widoczne (mniej przezroczyste, ciemny tekst).

## 0.9.6
- Opisy aut w środku obrysu – środek ciężkości pola (nie średnia wierzchołków).

## 0.9.7
- Log wykrytych aut przy skanie (diagnostyka).

## 0.9.8
- Log wyników rozpoznawania dla nierozpoznanych aut.

## 0.9.9
- Przy bardzo wysokim dopasowaniu (>=0.85) wystarczy mały margines nad drugim autem.

## 0.10.0
- API: reset próbek wybranego auta.

## 0.11.0
- Rozpoznawanie aut po kształcie: odcisk z sieci MobileNetV2 + kolor (gdy modelu brak - sam kolor).
- Bardzo pewne rozpoznania system dodaje sam jako osobną pulę próbek (max 15 na auto), ręczne nigdy nie są wypierane.

## 0.11.1
- Priorytet barwy (auta bez próbek) tylko gdy żadne auto z próbkami nie pasuje (>=0.55) - Kia nie jest już brana za Seata.

## 0.11.2
- Bez obrysów na zdjęciach, same etykiety aut.

## 0.12.0
- Panel: przycisk "Fałszywy" przy zdarzeniu (wysyła event kamera_etykieta do add-onu strojenia).

## 0.12.1
- Panel: widok "Strojenie" (kontekst, czułości, jakość obrazu, zdarzenia 7 dni).

## 0.12.2
- Panel: przycisk "Na żywo" (natywny podgląd HA z kamery; zapasowo odświeżane zdjęcia).

## 0.13.0
- Panel przebudowany pod telefon: dolna nawigacja (Zdarzenia / Na żywo / Ustawienia), domyślnie oś czasu, zwarte przyciski.
- Zakładka Na żywo: podgląd, suwaki czułości AI i obrazu, presety (Dzień, Noc, Deszcz, Czuły, Spokojny, Własny), cofnij.

## 0.13.1
- Panel: przełącznik Auto-presety w zakładce Na żywo.

## 0.13.2
- Preset Noc: jasność 165, kontrast 125, nasycenie 120 (sprawdzone na obrazie z kamery, kolor zachowany).

## 0.14.0
- Panel: opis auto-strojenia (płynne dopasowanie).

## 0.14.1
- Podglad na zywo: zoom szczypaniem, przeciaganie, dwuklik = reset.

## 0.14.2
- Pelny ekran na zywo: wlasny overlay z zoomem, auto-obrot do landscape.

## 0.14.3
- Ping panelu przez callWS (bez czerwonego toastu przy chwilowym braku lacznosci).

## 0.15.0
- Zakladka Parking: os czasu postojow kazdego auta (24 h / 7 dni), status stoi/odjechalo.
- Rozpoznawanie przez eliminacje: auto stojace w miejscu (widziane w poprzednim skanie), ktorego nie rozpoznano pewnie, dostaje wolne znane auto o najlepszym wyniku (kazde najwyzej raz). Naprawia Mazde i nocnego Seata.
- Skan podjazdu co 10 min, historia postojow w /data/park_hist.json.

## 0.15.1
- Wideo z Reolinka: sprawdzanie co 3 s (bylo 10 s), start po 5 s (bylo 10 s).
- Panel: po otwarciu zdarzenia stale sprawdza czy wideo jest gotowe - przycisk Wideo pokazuje stan (⏳/✓/✗), wideo laduje sie samo, licznik oczekiwania, brak zamrozenia przy podmianie pliku.

## 0.16.0
- Wideo bezposrednio z kamery po IP (Search/Download z karty SD), odpytywanie co 1.5 s; jesli kamera daje plik w trakcie nagrywania - publikuje coraz dluzsze wideo i dociaga reszte. Logi "sd:" mierza kiedy plik sie pojawia. Sciezka przez HA zostaje jako zapas.
- Panel: dociaganie - po skonczeniu krotszej wersji sam przeladowuje dluzsza od tego samego miejsca.

## 0.16.1
- api cmd sd_test (diagnostyka pobierania z SD po IP).

## 0.16.2
- Podglad w trakcie nagrywania: pobieranie z kamery startuje od razu po zdarzeniu (bez czekania na analize). Pomiar: czesciowe wideo ~32 s od zdarzenia, pelne ~46 s.
- Po koncu nagrania pelny plik dostarcza zwykle sciezka przez HA (szybsza); sd bierze go sam po 25 s, gdyby HA nie dalo rady.

## 0.16.3
- Pobieranie z kamery po IP (sd) tylko jako zapas po 70 s - rownolegle pobieranie spowalnialo glowna sciezke (kamera oddaje ok. 0.5 MB/s). Jakosc nagran bez zmian (4096 kbps).

## 0.17.0
- Bufor kołowy: dodatek stale nagrywa główny strumień RTSP kamery w pełnej jakości (kopiowanie, kawałki ~2 s, ostatnie 45 min na dysku). Klip zdarzenia składa się z bufora od razu (kilka sekund przed ruchem) i dokleja się w trakcie nagrywania; ścieżki z Reolinka/karty SD zostają jako zapas.
- Przejeżdżające pojazdy: kadr wybierany z klatek bufora wokół zdarzenia (najbliżej środka strefy „Droga", cały w kadrze) zamiast jednego zdjęcia z chwili zgłoszenia.
- /analyze: polecenia `bufor` (stan) i `bufor_test` (próba na żywo).

## 0.17.1
- Test bufora w tle, logi skanu.

## 0.17.2
- Detekcja obsluguje modele o innym rozmiarze wejscia (yolox_tiny 416) - do szybkiego przegladania klatek z bufora.

## 0.17.3
- Bufor: drugi, lekki strumień (sub) do szybkiego szukania najlepszej chwili; pełna klatka i klip nadal z głównego strumienia.

## 0.17.4
- Wybór klatki: wstępny wybór po ruchu względem tła, detekcja tylko na kilku klatkach (zamiast wszystkich); log czasu składania klipu.

## 0.17.5
- Polecenie `zloz_bench` (pomiar składania klipu na Pi).

## 0.17.6
- Składanie klipu i klatek: sklejanie kawałków bajtami zamiast demuxera concat (z ~7 s do ~1 s na Pi); kawałki z ciągłymi znacznikami czasu.

## 0.17.7
- Panel: lista zdarzeń bez przycisków Wideo / Fałsz. / Usuń i bez oznaczania aut (kompaktowo). W arkuszu zdarzenia: Fałszywy alarm, Usuń i rozwinięte oznaczanie aut.

## 0.17.8
- Panel: z listy zdarzeń zniknęły znacznik „powiadomiono” i lista rozpoznanych aut (są w arkuszu zdarzenia).

## 0.17.9
- Panel: oznaczanie aut w arkuszu zdarzenia domyślnie zwinięte.

## 0.17.10
- Miniatury osób w całości (bez ucinania), ten sam rozmiar.
- Podgląd na żywo: naprawione gubienie zoomu szczypaniem (czyszczenie zawieszonych wskaźników dotyku).

## 0.17.11
- Najlepsza klatka przejazdu: okno szukania zaczyna się 8 s PRZED zdarzeniem (kamera zgłasza zdarzenie z opóźnieniem, auto bywało już poza kadrem), do 5 kandydatów zamiast 4.

## 0.17.12
- Komenda `przelicz`: ponowny wybór najlepszej klatki dla zdarzenia z bufora (do 45 min wstecz).

## 0.17.13
- Komenda `diag` (diagnostyka klatek zdarzenia).

## 0.17.14
- Pełna klatka dopasowywana obrazem do wybranej klatki lekkiego strumienia (strumienie mają różne zera czasu - szybkie auto trafiało w inny kadr).

## 0.17.15
- Dopasowanie pełnej klatki po mapie ruchu (różnica od tła) zamiast surowych pikseli, okno ±3 s.

## 0.17.16
- diag2 (diagnostyka głównego strumienia).

## 0.17.17
- diag2: niższy próg.

## 0.17.18
- diag2: parametry okna.

## 0.17.19
- diag2: statystyki klatek.

## 0.17.20
- diag2: tylko klatki z ruchem.

## 0.17.21
- Snap przejazdu z klatki lekkiego strumienia (główny bufor bywa rozjechany/bez auta), po nagraniu z kamery podmiana na pełną jakość (dopasowanie po ruchu w jej własnej osi czasu).
