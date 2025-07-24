SuperResolution Aktivieren oder deaktivieren:
 use_super_resolution = True/false in der main.py Datei

Perspektiv-Kalibrierung (Pflicht für Geschwindigkeitsmessung)
Um die Geschwindigkeit korrekt in km/h zu berechnen, musst du zuerst das Kamerabild perspektivisch entzerren. Das funktioniert über eine Kalibrierung.

Schritt-für-Schritt Anleitung
Öffne die Datei
speed/calibrate_perspective.py (oder wie deine Kalibrierungsdatei heißt).

Ändere den Pfad zum Video
Bearbeite am Anfang der Datei die Zeile:

python
speed/calibrate.py
VIDEO_PATH = ""
→ Gib hier den Pfad zu deinem gewünschten Video an.

Starte die Kalibrierung

Führe die Datei aus:
python speed/calibrate_perspective.py
Wähle Kalibrierungsmethode in der Konsole
Es erscheinen zwei Optionen:

Wähle Kalibrierungsmethode:
1 = Automatisch (Segformer)
2 = Manuell (4 Punkte klicken)
👉 Eingabe (1 oder 2):
Gib 1 ein für automatische Kalibrierung (nutzt KI zur Straßenerkennung).

Gib 2 ein für manuelle Kalibrierung:

Du klickst vier Punkte im Bild an, die ein Rechteck der Straße darstellen (z. B. Spurlinien vorne/hinten links/rechts).

Matrix wird gespeichert
Die Perspektivmatrix wird automatisch unter KAL/MATFILE.npy gespeichert.
Diese Datei wird später automatisch beim Start der Analyse geladen.

Wichtig:
Führe diesen Schritt vor der ersten Analyse durch.

Ohne Kalibrierung können keine realen Geschwindigkeiten (km/h) berechnet werden.


# Anleitung zur Installation und Ausführung:

1-Installiere Python 3.10.11 und füge es zum Systempfad hinzu.
2-Öffne ein Terminal und navigiere in den Projektordner.
3-Erstelle eine neue virtuelle Umgebung mit der richtigen Python-Version:
    py -3.10 -m venv venv310
4-Aktiviere die virtuelle Umgebung:
    .\venv310\Scripts\Activate
5-Installiere die benötigten Pakete aus der requirements.txt:
    pip install -r requirements.txt
6-Überprüfe den Pfad zum Video, das analysiert werden soll, in der Datei main.py.
7-Lösche die Datei lanes.pkl im Verzeichnis data, wenn noch keine Fahrspuren für den gewählten Videopfad definiert wurden. Die Datei lanes.pkl speichert ausschließlich die zuletzt definierten Fahrspuren.
8-Starte das Programm:
    python main.py
# Fahrspuren-Definitionsanleitung:
Schritt 1: Auswahl der Fahrbahnlinien:
Beim Start wird der erste Frame des Videos oder Livestreams angezeigt – mit grünen Linien überlagert, die automatisch erkannt wurden, und die bei Auswahl rot werden.

-Wähle die korrekten Linien in Reihenfolge aus, beginnend entweder von rechts oder von links.

-Wähle nur eine Linie pro Spur, da dieselbe Spur mehrfach oder fälschlicherweise erkannt worden sein kann.

-Wenn sich zwischen zwei Spuren Straßenhindernisse (z.B. Betonblöcke) befinden, drücke vor dem Auswählen der nächsten Linie die Taste N, um den „Neue-Gruppe“-Modus zu aktivieren.
Dadurch wird verhindert, dass Linien automatisch verbunden werden und ein falsches Polygon entsteht.

-Nach dem Drücken von N wähle die nächste Linie wie gewohnt aus. Der „Neue-Gruppe“-Modus wird danach automatisch deaktiviert.

-Fahre mit der Auswahl weiterer Linien fort.

Wenn alle gewünschten Linien ausgewählt wurden, drücke S, um die Auswahl zu speichern.

Schritt 2: Anpassung der Linien (Bezier-Kurven)
Nach dem Speichern öffnet sich automatisch das Fenster zur Anpassung der Linien als Bezier-Kurven.

-Jede Linie wird nun als Kurve mit zwei grünen Punkten (Anfang und Ende) angezeigt.

    Ziehe die grünen Punkte, um die Länge der Linie anzupassen.

-Ein roter Punkt auf der Kurve ermöglicht das Anpassen der Krümmung.

    Ziehe diesen Punkt nach oben/unten oder links/rechts, um die Form an die tatsächliche Fahrspur anzupassen.

-Wenn die Bearbeitung abgeschlossen ist, drücke Q, um den Kalibrierungsmodus zu verlassen.
    Das System wechselt automatisch zum Fenster für die Verkehrsanalyse, wobei die gewählten Spuren als farbige Polygone dargestellt werden.

* Wenn nicht alle Linien erkannt wurden
Falls im ersten Schritt nicht alle gewünschten Fahrbahnlinien erkannt wurden, gibt es zwei Optionen:

-Manuelle Definition sofort starten:
    Drücke S, ohne Linien auszuwählen, um direkt zur manuellen Definition zu wechseln.

-Teilweise Auswahl:
    Wenn z.B. nur die Linie ganz rechts oder links erkannt wurde:

        Wähle diese Linie und weitere angrenzende Linien aus.

        Sobald eine gewünschte Linie fehlt oder nicht auswählbar ist, drücke S, um den Rest manuell zu definieren.

Schritt 3: Manuelle Definition von Linien
Im manuellen Modus klickst du drei Punkte entlang der gewünschten Fahrbahnlinie:

-Erster Punkt: oben im Bild (entferntester Punkt)

-Zweiter Punkt: in der Mitte

-Dritter Punkt: unten im Bild (näher zur Kamera)

Diese Punkte werden zu einer Bezier-Kurve verbunden, die du anschließend genauso anpassen kannst wie zuvor.

Diese Reihenfolge sorgt für eine saubere Verbindung mit bestehenden Linien und ermöglicht die korrekte Bildung von zusammenhängenden Polygonen.

Neue Gruppe im manuellen Modus:
-Wenn du eine Linie nicht mit der vorherigen verbinden möchtest (z.B. wegen Straßenhindernissen):

-Drücke N, um den Neue-Gruppe-Modus zu aktivieren.

-Klicke drei Punkte zur Definition der Linie.

-Nach dem Erstellen der Linie wird der Modus automatisch deaktiviert, und du kannst mit der normalen Definition weitermachen.


