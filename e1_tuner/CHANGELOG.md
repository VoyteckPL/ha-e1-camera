## 0.1.0
- Pierwsza wersja: co N minut zapisuje kontekst (jasnosc, deszcz, pogoda, swiatlo, czulosci) i metryki obrazu, loguje zdarzenia z kontekstem i etykiety "falszywy alarm" z panelu. Niczego nie zmienia w kamerze.

## 0.2.0
- Sensor sensor.kamera_strojenie dla panelu (kontekst, serie metryk, zdarzenia 7 dni, falszywe).

## 0.3.0
- Auto-presety (input_boolean.kamera_auto_presety): noc/dzien wg jasnosci zewnetrznej (histereza 15/30 lx) + deszcz (i 20 min po), zmiany tylko przy przejsciu stanu.
