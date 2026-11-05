## 0.1.0
- Pierwsza wersja: co N minut zapisuje kontekst (jasnosc, deszcz, pogoda, swiatlo, czulosci) i metryki obrazu, loguje zdarzenia z kontekstem i etykiety "falszywy alarm" z panelu. Niczego nie zmienia w kamerze.

## 0.2.0
- Sensor sensor.kamera_strojenie dla panelu (kontekst, serie metryk, zdarzenia 7 dni, falszywe).

## 0.3.0
- Auto-presety (input_boolean.kamera_auto_presety): noc/dzien wg jasnosci zewnetrznej (histereza 15/30 lx) + deszcz (i 20 min po), zmiany tylko przy przejsciu stanu.

## 0.3.1
- Preset Noc jak w panelu (jaśniej).

## 0.4.0
- Bezposredni dostep do kamery (Isp) z danymi logowania z integracji Reolink: GET/POST /isp.

## 0.4.1
- Log odpowiedzi kamery (diagnostyka).

## 0.4.2
- GET /isp zwraca tez zakresy parametrow.

## 0.5.0
- Płynne auto-strojenie: czułości AI/ruchu i obraz interpolowane z jasności zewnętrznej (log), deszcz/śnieg jako płynny współczynnik, DRC kamery nocą, ręczna zmiana wstrzymuje suwak na 3 h, dobowe uczenie poprawek z oznaczeń Fałszywy.

## 0.5.1
- GET /odczyt?c=Get... (tylko odczyt, diagnostyka ustawien kamery).

## 0.5.2
- Naprawa: brakowala funkcja ustaw() - auto-presety nie dzialaly (opis "liczę…").

## 0.5.3
- GET /szukaj?min=N - lista nagran z karty SD kamery (odczyt).
