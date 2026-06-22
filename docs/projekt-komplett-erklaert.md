# Das Projekt komplett erklärt — verstehen und verteidigen

> Dieses Dokument ist für genau eine Aufgabe geschrieben: dass jemand, der dieses Projekt **vorher
> nicht kannte** und auch **keine Computer-Vision-Methoden** kennt, am Ende trotzdem genau versteht,
> **was** wir gemacht haben, **warum**, **wie** (auch grob technisch), welche **Probleme** auftauchten
> und **welche Ergebnisse** herauskamen — und das Ganze in einer Prüfung als **eigene Arbeit
> verteidigen** kann.
>
> Es ist bewusst lang und in einfacher Sprache. Du brauchst keine Formeln. Du brauchst keine
> Vorkenntnisse außer „ein Bild ist ein Raster aus Helligkeitswerten". Alles andere bauen wir Schritt
> für Schritt auf. Lies es einmal ganz durch wie eine Geschichte; danach reichen die
> Zusammenfassungen und das Q&A am Ende zum Wiederholen.
>
> **Wichtig zur Ehrlichkeit:** Jede Zahl in diesem Dokument ist aus unseren Ergebnisdateien
> nachgerechnet (das Projekt hat ein eigenes Prüfskript, `verify_claims.py`, das genau das tut). Wir
> reden nichts schön. Gerade die ehrliche, selbstkritische Haltung ist Teil dessen, was die Arbeit
> stark macht — und sie ist leicht zu verteidigen, weil sie stimmt.

---

# INHALT

- Teil 0 — Die Kurzfassung in 60 Sekunden
- Teil 1 — Das Problem: worum geht es überhaupt?
- Teil 2 — Die Ausgangslage: was wir hatten und was nicht
- Teil 3 — Die Schlüssel-Idee: Tracking liefert die Wahrheit
- Teil 4 — Der Gesamtansatz: die Pipeline im Überblick
- Teil 5 — Die Methoden im Detail (der Kern, jede Methode von Grund auf)
- Teil 6 — Die Probleme und wie wir sie gelöst haben
- Teil 7 — Die technische Umsetzung (Sprache, Libraries, Aufwand, Werkzeuge)
- Teil 8 — Die Ergebnisse, ehrlich
- Teil 9 — Die Validierung mit echten menschlichen Labels
- Teil 10 — Grenzen und offene Punkte
- Teil 11 — Verteidigung: typische Prüfungsfragen mit Antworten
- Teil 12 — Glossar: jeder Fachbegriff in zwei Sätzen

---

# TEIL 0 — DIE KURZFASSUNG IN 60 SEKUNDEN

Stell dir eine Drohne vor, die nachts mit einer **Wärmebildkamera** über ein Feld oder einen Wald
fliegt und nach Wildtieren sucht (Rehe, Hirsche, Wildschweine). Die Tiere erscheinen als kleine,
**warme, verschwommene Flecken**. Unsere Aufgabe in diesem Projekt war: **In welche Richtung bewegt
sich so ein Tier?**

Das klingt einfach, ist aber schwer, aus einem Grund: Man kann auf so einem Fleck **den Kopf fast nie
erkennen**. Und ohne Kopf weiß man nicht, ob das Tier nach links oder nach rechts läuft. Es gibt also
keine fertige „richtige Antwort" (keine *Ground Truth*) für die Richtung, gegen die man ein Verfahren
messen könnte.

Unsere zentrale Idee war deshalb: **Wir nehmen die Wahrheit nicht aus einem einzelnen Bild, sondern
aus der Bewegung über mehrere Bilder.** Wir verfolgen das Tier über die Einzelbilder eines Fluges
(„Tracking"), rechnen die Eigenbewegung der Drohne heraus, und was vom Tier an Bewegung übrig bleibt,
**ist** seine echte Laufrichtung. Diese aus dem Tracking gewonnene Richtung ist unsere Wahrheit.

Darauf aufbauend haben wir, wie die Aufgabe es verlangt, **verschiedene Computer-Vision-Methoden
verglichen**: klassische Bildverarbeitung, klassisches maschinelles Lernen, ein selbst gebautes
neuronales Netz und große vorgelernte „Foundation Models". Alles haben wir **ehrlich und ohne
Schummeln** ausgewertet. Zum Schluss hat ein Teamkollege (Andreas) 1.500 Bilder von Hand annotiert —
nicht als Wahrheit, sondern als **unabhängige Kontrolle**. Das Ergebnis dieser Kontrolle: **Der Mensch
stimmt mit unserer Tracking-Richtung überein (~20 Grad Abweichung), und der Kopf ist tatsächlich nur in
14 % der Fälle erkennbar.** Damit ist unser ganzer Ansatz bestätigt.

Das ist die Geschichte. Jetzt im Detail.

---

# TEIL 1 — DAS PROBLEM: WORUM GEHT ES ÜBERHAUPT?

## 1.1 Der Kontext: BAMBI und Wärmebild-Drohnen

Das Projekt gehört zum Umfeld von **BAMBI** an der FH Hagenberg — das ist ein reales Forschungsthema
zur Überwachung von Wildtieren aus der Luft. Hintergrund: Wildtiere aus dem Flugzeug oder per Drohne zu
zählen und zu beobachten ist nützlich für Naturschutz, für die Jagd-Statistik, und ganz praktisch zum
Beispiel, um Rehkitze vor dem Mähtod zu retten (ein Mähdrescher erkennt ein im Gras liegendes Kitz
nicht — eine Wärmebild-Drohne schon).

Aufgenommen wird mit einer **Wärmebildkamera** (Thermalkamera). Die misst nicht Farbe oder normales
Licht, sondern **Wärme**. Ein lebendes Tier ist wärmer als der Boden und leuchtet deshalb hell auf.
Das ist der große Vorteil: Es funktioniert nachts, und das Tier hebt sich klar vom kühleren
Hintergrund ab. Der Nachteil: Ein Wärmebild ist **grob und unscharf**. Man sieht keine Details, keine
Beine, kein Geweih, oft nicht mal eine klare Körperform — nur einen **hellen Klecks**.

## 1.2 Eine Besonderheit der Aufnahme: AOS-Lichtfeld

Unsere Bilder sind keine normalen Einzelfotos. Sie kommen aus einem Verfahren namens **AOS — Airborne
Optical Sectioning** (auf Deutsch etwa „luftgestützte optische Schnittbildgebung"). Das muss man
verstehen, weil es später wichtig wird.

Die Idee von AOS ist ein Trick gegen Verdeckung durch Pflanzen: Die Drohne nimmt **dieselbe Stelle am
Boden aus vielen leicht verschiedenen Positionen** auf, während sie fliegt. Dann werden all diese
Aufnahmen rechnerisch so verschoben, dass sie auf den **Boden** ausgerichtet sind, und übereinander
gemittelt. Das Ergebnis ist ein einziges „Integralbild".

**Analogie:** Stell dir vor, du fotografierst durch einen Gartenzaun. Von einer Position siehst du
hinter jedem Latten-Schatten nichts. Wenn du aber seitlich entlanggehst und viele Fotos machst und
diese passend übereinanderlegst, dann „mittelt sich der Zaun weg" — er ist auf jedem Foto woanders —
und du siehst plötzlich durch. Genauso mittelt AOS Büsche und Äste weg und legt den (warmen) Boden und
die Tiere darunter frei.

Konkret: Ein solches Lichtfeld-Bild besteht bei uns aus rund **9 Einzelaufnahmen**, die über knapp
**eine Sekunde** verteilt aufgenommen wurden (etwa 3 Bilder pro Sekunde aus einem 30-Bilder-pro-Sekunde
Video). Jedes fertige Bild ist **2048 × 2048 Pixel** groß.

## 1.3 Warum gerade das ein Geschenk für die Bewegungsrichtung ist

Jetzt kommt der für uns entscheidende Punkt, und der ist eigentlich schön: Was passiert mit einem Tier,
das sich während dieser einen Sekunde **bewegt**?

- Ein Tier, das **still steht**, ist in allen 9 Aufnahmen am selben Ort. Beim Mitteln bleibt es **scharf**.
- Ein Tier, das **läuft**, ist in jeder der 9 Aufnahmen ein Stück weiter. Beim Mitteln entsteht ein
  **Schmierstreifen** in genau die Richtung, in die es läuft.

**Analogie:** Das ist wie ein Foto mit langer Belichtungszeit von einer Straße bei Nacht. Stehende
Laternen sind scharfe Punkte; fahrende Autos werden zu langen Lichtspuren. Die Richtung der Lichtspur
verrät die Fahrtrichtung.

Diese Bewegungsunschärfe (englisch *motion blur*) ist also ein echtes Signal: **Die Richtung des
Schmiers ist die Bewegungsrichtung.** Das war die Grundidee, mit der wir gestartet sind. (Spoiler: Sie
funktioniert, aber nur begrenzt — siehe später; das eigentliche Arbeitspferd wurde das Tracking.)

## 1.4 Warum das Problem trotzdem schwer ist

Vier Dinge machen es hart, und die muss man in der Verteidigung nennen können:

1. **Die Tiere sind winzig.** Die längste Seite des Tier-Kästchens ist im Median nur rund **65 Pixel**
   auf einem 2048-Pixel-Bild. Das ist ein kleiner Klecks. Auf so wenig Pixeln ist jede Aussage wackelig.

2. **Es gibt fast immer mehrere Tiere im Bild.** In **67 %** der Bilder ist mehr als ein Tier zu sehen,
   im Schnitt etwa 3–4, manchmal über 20. Das macht das Verfolgen schwer: Wenn zwei gleichartige
   Flecken nah beieinander sind, kann ein Tracker sie verwechseln.

3. **Der Kopf ist meist unsichtbar.** Das ist der Kern des Problems. Auf einem warmen, verschwommenen
   Klecks erkennt man die **Achse** (die Linie, entlang der das Tier liegt oder schmiert) oft noch — aber
   nicht, **welches Ende der Kopf ist**. Das nennt man die **180-Grad-Mehrdeutigkeit**: Man weiß die
   Linie, aber nicht das Vorzeichen (vorne/hinten).

4. **Es gibt keine fertige Wahrheit für die Richtung.** Bei normalen Aufgaben hat man gelabelte Daten:
   Jemand hat „richtig" hingeschrieben, und man misst dagegen. Hier konnte das niemand zuverlässig tun,
   weil man den Kopf eben nicht sieht. Hand-Labeln der Richtung wäre nur **„geraten gegen geraten"**.

Punkt 4 ist das eigentliche wissenschaftliche Problem. Der Rest des Projekts ist im Grunde **eine
saubere Antwort auf die Frage: Woher nehmen wir eine verlässliche Wahrheit, wenn niemand sie hinschreiben
kann?**

---

# TEIL 2 — DIE AUSGANGSLAGE: WAS WIR HATTEN UND WAS NICHT

## 2.1 Der Auftrag

Die Kurssituation: Das Projekt zählt zu 50 % zur Note (die andere Hälfte ist eine schriftliche
Prüfung). Erlaubt war, sich aus den BAMBI-Datensätzen **eine Aufgabe selbst zu wählen** (Segmentierung,
Detektion, Pose, Architektur-Vergleich oder etwas Eigenes), die **Vorverarbeitung und Annotation**
gehören ausdrücklich dazu, und man soll **Computer-Vision-Methoden aus der Vorlesung anwenden,
auswerten und vergleichen**. Wir sind ein Team von zwei (statt der üblichen 3–4), der erwartete Aufwand
liegt bei etwa **40–50 Stunden pro Person**.

Unsere selbst gewählte und im Proposal festgelegte Aufgabe war: **Bewegungsrichtung von Wildtieren aus
thermischen AOS-Lichtfeld-Drohnenbildern schätzen.** Die *Methodik* durften und mussten wir selbst
entwerfen — und genau das ist der Teil, den man verteidigen können muss.

## 2.2 Der Datensatz

Der Datensatz liegt auf einer Plattform namens **Roboflow** und ist über einen API-Schlüssel
abrufbar. Wichtige, nachgeprüfte Eckdaten (alle aus den echten Dateien gezählt):

- **12.655 Bilder** insgesamt, davon **12.514 mit mindestens einem Tier**.
- **46.046 markierte Tiere** (Bounding Boxes).
- Diese verteilen sich auf **3 Klassen** mit den IDs 0, 1, 2 — mit **21.787 / 17.403 / 6.856** Tieren.
- **223 verschiedene Flüge** (Flug-IDs); 221 davon haben tatsächlich markierte Tiere.
- Bilder sind 2048 × 2048, jedes ist ein AOS-Integralbild.

Ein **wichtiges Ehrlichkeits-Detail** zu den Klassen: In der Datei, die den Datensatz beschreibt
(`data.yaml`), heißen die Klassen schlicht **„2", „3", „4"** — also nur Nummern, keine Tierarten. Wir
**vermuten** (aus einem älteren Projekt übernommen), dass 0 = Rotwild (Rothirsch), 1 = Rehwild (Reh),
2 = Schwarzwild (Wildschwein) ist. **Diese Zuordnung ist nicht bestätigt.** Wir schreiben das überall
ehrlich dazu und behandeln die Klassen im Zweifel als „Klasse 0/1/2". Wer in der Prüfung fragt: „Welche
Tierart ist Klasse 2?" — die ehrliche Antwort ist: „Vermutlich Wildschwein, aber der Datensatz speichert
nur die ID; die Artnamen sind eine unbestätigte Annahme, die man mit dem BAMBI-Team verifizieren müsste."
Das ist kein Schwächezeichen, sondern saubere Wissenschaft.

## 2.3 Was schon gegeben war — und was das für die Ehrlichkeit bedeutet

Sehr wichtig zu wissen, weil es eine typische Fangfrage betrifft: **Die Tier-Kästchen (Bounding Boxes)
waren bereits im Datensatz vorhanden**, im sogenannten **YOLO-Format**.

- „YOLO" ist eigentlich ein bekannter **Objektdetektor** (ein neuronales Netz, das Objekte in Bildern
  findet). Das **Format**, in dem die Kästchen gespeichert sind, heißt auch YOLO-Format: pro Tier eine
  Zeile mit Klassennummer, Mittelpunkt und Größe des Kästchens (als Anteil der Bildbreite/-höhe).
- **Wir haben selbst KEINEN Detektor / kein YOLO trainiert.** Detektion war nicht unsere Aufgabe. Wir
  haben auf den **bereits vorhandenen** Kästchen aufgesetzt.

Das ist eine unserer **fünf Ehrlichkeits-Säulen**, die wir konsequent durchziehen. Wenn jemand fragt
„Habt ihr ein neuronales Netz trainiert, um die Tiere zu finden?", lautet die korrekte Antwort: „Nein,
die Boxen kamen fertig mit dem Datensatz im YOLO-Format. Wir haben sie nur weiterverarbeitet."

Was es **nicht** gab: irgendeine Information über die **Bewegungsrichtung**, und kein GPS / keine
Positionsdaten der Drohne im Export. Das heißt, selbst die Bewegung der Drohne mussten wir **aus den
Bildern selbst** schätzen, nicht aus Sensordaten.

## 2.4 Erste Pflicht: die Daten verstehen (EDA)

Bevor man Methoden baut, muss man die Daten anschauen. Das nennt man **EDA** (Exploratory Data
Analysis, „erkundende Datenanalyse"). Wir haben pro Klasse Statistiken berechnet. Die interessantesten
Erkenntnisse:

- **Tiergröße:** Median der längsten Box-Seite rund 65 Pixel (pro Klasse 69 / 61 / 61). Sehr klein.
- **Mehrtier-Szenen:** 67 % der Bilder mit mehr als einem Tier — der oben genannte Tracking-Stolperstein.
- **Helligkeit/Wärme pro Klasse:** Klasse 1 ist am wärmsten (mittlere Helligkeit ~116), Klasse 0
  mittel (~102), **Klasse 2 am kühlsten (~79)**. Das wird gleich wichtig.
- **Eine ehrliche Erkenntnis zu Klasse 2 (vermutlich Wildschwein):** Wenn wir das Tier aus dem Kästchen
  „freischneiden" (siehe Segmentierung, Teil 5), erwischen wir bei Klasse 2 nur einen sehr kleinen
  Kern — die Fläche schrumpft auf etwa **37 %** des Kästchens, gegenüber ~81 % / 115 % bei den anderen
  Klassen. Der Grund: Wildschweine sind am kühlsten und kontrastärmsten, also „greift" unser
  Helligkeits-Schwellwert nur den heißesten Kern (rund 7 % der Bildfläche). Das ist **keine schlechte
  Box**, sondern eine **Unter-Segmentierung** wegen geringen Kontrasts. Dass wir diesen Unterschied
  erkennen und richtig erklären, zeigt, dass wir die Daten wirklich verstanden haben — ein guter Punkt
  für die Verteidigung.
- **Datenqualität:** Manche AOS-Exporte haben am Rand **Geister-Artefakte** (verschobene Doppelbilder,
  englisch *ghosting*), weil die Ausrichtung dort nicht perfekt war. Eine einfache Heuristik markiert
  rund 0,9 % der Bilder als verdächtig. Das ist nur eine Triage zum Aussortieren, kein geprüfter Detektor.

---

# TEIL 3 — DIE SCHLÜSSEL-IDEE: TRACKING LIEFERT DIE WAHRHEIT

## 3.1 Die Sackgasse, die uns zur Idee zwang

Unser ursprünglicher Plan (aus dem Proposal) war zweigleisig: (1) für **stehende** Tiere die
Körper-**Orientierung** schätzen, (2) für **bewegte** Tiere die **Schmier-Richtung** (motion blur)
schätzen, und das **Tracking nur als Gegen-Check** verwenden.

Beim Ausprobieren merkten wir aber schnell das Kernproblem aus Teil 1.4: Wir hatten **keine
verlässliche Wahrheit**, gegen die wir diese Einzelbild-Schätzer hätten messen können. Selbst ein
Mensch (auch Andreas) kann auf den meisten Bildern den Kopf nicht erkennen. Eine Methode gegen
unsichere Handlabels zu messen ist sinnlos — man weiß nie, ob ein Fehler an der Methode oder am
unsicheren Label liegt.

## 3.2 Der Pivot (die wichtigste Entscheidung des Projekts)

Hier kam unsere zentrale, eigenständige Überlegung — der **Pivot**: Wir drehen die Rollen um. **Nicht
das Einzelbild liefert die Wahrheit, sondern die Bewegung über die Zeit.**

Der Gedanke: Wenn wir ein Tier über die aufeinanderfolgenden Frames eines Fluges **verfolgen**, dann
sehen wir, wohin es tatsächlich wandert. Das ist reine **Geometrie über die Zeit** — keine Meinung
eines Modells, kein Raten am Klecks. Die einzige Schwierigkeit: Die Drohne bewegt sich auch, also
verschiebt sich das **ganze** Bild. Diese Eigenbewegung der Drohne (englisch *ego-motion*) müssen wir
zuerst herausrechnen. Was dann an Bewegung des Tiers **übrig bleibt**, ist seine echte Laufrichtung.

Diese aus dem Tracking gewonnene Richtung wird unsere **Ground Truth** (Wahrheit). Menschliche Labels
brauchen wir danach nur noch zur **Validierung** — also um zu prüfen, ob die Tracking-Wahrheit mit dem
übereinstimmt, was ein Mensch dort sieht, wo er etwas sieht.

**Warum ist das legitim und nicht „auch nur geraten"?** Weil es auf einer anderen, härteren
Informationsquelle beruht: nicht auf dem Aussehen eines einzelnen unscharfen Klecks, sondern auf der
**tatsächlichen Ortsveränderung** desselben Tiers über mehrere Bilder. Ein Tier, das sich über eine
Sekunde nach Nordosten bewegt hat, **ist** nach Nordosten gelaufen — egal wie unscharf der einzelne
Fleck ist. Das ist die ganze Stärke der Idee.

In einem Satz für die Prüfung: **„Wir hatten keine Richtungs-Wahrheit, weil man den Kopf nicht sieht.
Also haben wir die Wahrheit aus dem Tracking gewonnen: Tier über die Frames verfolgen, Drohnenbewegung
herausrechnen, Restbewegung = echte Richtung. Menschliche Labels dienen nur zur Bestätigung."**

---

# TEIL 4 — DER GESAMTANSATZ: DIE PIPELINE IM ÜBERBLICK

Bevor wir in jede Methode hineinzoomen, hier die **Pipeline** — die Kette der Verarbeitungsschritte —
als Landkarte. „Pipeline" heißt einfach: Daten fließen von Schritt zu Schritt, jeder Schritt nimmt das
Ergebnis des vorherigen.

1. **Daten** — AOS-Wärmebilder mit den fertigen Tier-Kästchen.
2. **Vorverarbeitung** — aus jedem groben Kästchen das eigentliche warme Tier „freischneiden"
   (klassische Segmentierung), damit wir z. B. die Körperachse messen können.
3. **Tracking als Wahrheit** — Tiere über die Frames verfolgen, Drohnenbewegung herausrechnen, pro Tier
   eine zuverlässige Richtung mit Konfidenz gewinnen. **Das ist das Herzstück.**
4. **Vergleich der Methoden** — verschiedene Verfahren gegen diese Wahrheit messen: klassische
   Einzelbild-Schätzer, klassisches ML, ein eigenes neuronales Netz, vorgelernte Foundation Models.
5. **Evaluation** — alles **ehrlich und ohne Datenleck** auswerten (faire Vergleichsmaßstäbe,
   getrennte Flüge für Training und Test, Selbst-Audit, am Ende Validierung mit menschlichen Labels).

Die ganze **Philosophie** dahinter ist die der Aufgabe: nicht ein einziges Verfahren bauen, sondern
**vergleichen** — klassisch vs. lernend, einfach vs. komplex — und ehrlich sagen, was wie gut
funktioniert und warum.

## 4.1 Wie das Projekt zeitlich ablief (die Stages)

Damit die Geschichte rund ist, hier die **Reihenfolge**, in der wirklich gearbeitet wurde — das ist
auch die ehrliche Antwort auf „Wie seid ihr vorgegangen?":

1. **Daten verstehen (EDA).** Bilder anschauen, pro Klasse Größen/Helligkeiten/Mehrtier-Anteil zählen,
   Qualität (Ghosting) prüfen. Erkenntnis: Tiere winzig, viele Mehrtier-Szenen, Klasse 2 kontrastarm.
2. **Erster Plan & Sackgasse.** Einzelbild-Richtung aus Orientierung/Schmier schätzen — scheitert an der
   fehlenden Wahrheit, weil der Kopf unsichtbar ist.
3. **Der Pivot.** Tracking als Wahrheitsquelle: Registrierung, Tracking, Kreisstatistik,
   Vertrauens-Tor bauen.
4. **Registrierung verbessern.** CLAHE + Lowe-Test → 164 auf 190 verwertbare Tracks, Inlier 0,72 → 0,86.
5. **Vergleichsmethoden.** Klassische Einzelbild-Schätzer; dann bewegt/steht mit klassisch / CNN /
   Foundation Models, alles flug-getrennt.
6. **Selbst-Audit.** Eigene Zahlen hart prüfen (190 → belastbar 138, Einheitsfehler korrigiert),
   `verify_claims.py` bauen.
7. **Annotations-Tool & Validierung.** Tool bauen, Andreas labelt 1.500 Bilder, Tracking-Wahrheit
   bestätigen, Bewegungsstufen datengetrieben auf binär reduzieren.
8. **Aufbereitung.** Präsentation, Dokumentation, Reproduzierbarkeit.

Das Wichtigste an dieser Reihenfolge: Der Pivot kam **als Reaktion auf ein echtes Problem**, nicht als
fertiger Plan von Anfang an. Genau das macht es zu eigener Denkarbeit, die man verteidigen kann.

---

# TEIL 5 — DIE METHODEN IM DETAIL

Das ist der längste und wichtigste Teil. Für **jede** Methode erklären wir nach demselben Muster:
**Was** ist das (in einfachen Worten + Analogie), **Wie** funktioniert es Schritt für Schritt, **Warum
gerade so** (welche Alternative gab es, warum unsere Wahl), und das **Zwischenergebnis** (was es bei uns
gebracht hat, mit echter Zahl).

## 5.1 Vorverarbeitung: vom groben Kästchen zum echten Tier (klassische Segmentierung)

**Was.** Ein Bounding-Box-Kästchen ist nur ein grobes Rechteck um das Tier — meist zu groß, mit viel
Hintergrund. Für Richtungs- und Größenmessungen wollen wir aber das **Tier selbst**, nicht das
Rechteck. „Segmentierung" heißt: jedes Pixel als **Tier** oder **Hintergrund** markieren. Wir machen das
**klassisch**, also mit klassischer Bildverarbeitung, **ohne** neuronales Netz. Das ist eine unserer
Ehrlichkeits-Säulen.

**Wie (Schritt für Schritt).**

1. **Glätten.** Zuerst ein leichtes Weichzeichnen (Gauß-Filter), damit einzelne Pixel-Ausreißer nicht
   stören. (Ein Gauß-Filter ersetzt jeden Pixel durch einen gewichteten Mittelwert seiner Nachbarn —
   wie ein leichtes „Verschmieren", das Rauschen reduziert.)
2. **Schwellwert mit Otsu.** Jetzt müssen wir entscheiden: ab welcher Helligkeit ist ein Pixel „warm
   genug" = Tier? Ein fester Wert wäre schlecht, weil Bilder unterschiedlich hell sind. **Otsus
   Methode** findet den Schwellwert **automatisch**: Sie schaut sich die Verteilung der Helligkeiten an
   (das Histogramm) und sucht die **Trennlinie, die dunkle und helle Pixel am saubersten in zwei
   Gruppen teilt** (so, dass die zwei Gruppen in sich möglichst gleichmäßig sind). Bei einem warmen
   Tier auf kühlem Grund gibt es zwei „Berge" im Histogramm — Otsu legt den Schnitt ins Tal dazwischen.
   *Analogie:* Es ist wie das automatische Finden der Wasserlinie zwischen „dunklem Meer" und „heller
   Insel".
3. **Aufräumen mit Morphologie.** Nach dem Schwellwert hat man oft Löcher im Tier oder kleine helle
   Punkte daneben. **Morphologische Operationen** räumen das auf: „Öffnen" entfernt kleine isolierte
   Flecken, „Schließen" füllt kleine Löcher. *Analogie:* wie Radieren von Krümeln und Auffüllen von
   Kratzern.
4. **Zusammenhängende Komponenten.** Jetzt zählen wir die **zusammenhängenden** weißen Flächen
   („connected components"). Wir nehmen die **größte Fläche, deren Schwerpunkt in der Bildmitte liegt**
   — denn das Tier ist mittig im Kästchen, Hintergrund-Reste liegen am Rand.
5. **Bildmomente für die Achse.** Aus dieser Tier-Fläche berechnen wir **Bildmomente**. Das sind
   einfache statistische Kennzahlen über die Pixelverteilung. Daraus bekommt man den **Schwerpunkt**,
   die **Hauptachse** (die Richtung, in der das Tier am längsten ist) und die **Exzentrizität** (wie
   länglich es ist, 0 = rund, nahe 1 = lang gestreckt). *Analogie:* Wenn man eine Wolke von Punkten hat,
   sagt einem die Hauptachse, in welche Richtung die Wolke „zeigt" — wie die Längsrichtung einer Gurke.

**Warum gerade so (Alternative).** Man hätte ein neuronales Netz zur Segmentierung nehmen können
(z. B. ein U-Net). Aber: (a) das hätte Trainingsdaten gebraucht, die wir nicht hatten; (b) bei so
einfachen „heller Klecks auf dunklem Grund"-Bildern ist klassisch völlig ausreichend und viel
schneller und transparenter; (c) es passt zur Aufgabe, klassische Methoden zu zeigen. Klassisch ist
hier die **richtige**, nicht die faule Wahl.

**Zwischenergebnis.** Funktioniert sauber auf echten Bildern. Die ehrliche Ausnahme ist Klasse 2
(Wildschwein, am kühlsten): Dort greift Otsu nur den heißesten Kern, die Tier-Fläche schrumpft auf
~37 % des Kästchens. Wir haben das **erkannt und korrekt als Kontrast-Problem erklärt**, nicht als
„schlechte Box".

**Zahlenbeispiel zum Mitdenken.** Stell dir die Helligkeiten im Kästchen in zwei Häufungen vor: viele
dunkle Pixel um Wert 40 (Boden) und einige helle um Wert 180 (Tier). Im Histogramm sind das zwei Berge
mit einem Tal dazwischen, etwa bei 110. Otsu probiert jede mögliche Grenze durch und nimmt die, bei der
die zwei entstehenden Gruppen in sich am gleichmäßigsten sind — hier genau das Tal bei ~110, alles
darüber wird „Tier". Bei einem kühlen Wildschwein liegen die beiden Berge enger beieinander, das Tal ist
flach, und Otsu schneidet höher an — deshalb bleibt nur der heißeste Kern übrig (die ~37 %-Geschichte).

## 5.2 Bildregistrierung: die Drohnenbewegung herausrechnen

Das ist der erste Schritt des Trackings und technisch der anspruchsvollste. Bitte gut verstehen, das
ist eine beliebte Prüfungsfrage.

**Was.** „Registrierung" heißt: zwei Bilder so **deckungsgleich übereinanderlegen**, dass derselbe
Punkt am Boden in beiden Bildern an derselben Stelle liegt. Weil sich die Drohne zwischen zwei Frames
bewegt, ist nämlich das **ganze** Bild verschoben (und leicht gedreht/skaliert). Wir wollen
herausfinden: **„Um welche Verschiebung/Drehung muss ich Bild A schieben, damit es zu Bild B passt?"**
Diese Transformation **ist** die Drohnenbewegung — und genau die wollen wir abziehen.

**Wie (Schritt für Schritt).**

1. **Merkmalspunkte finden mit ORB.** In beiden Bildern suchen wir markante Punkte — „Ecken",
   wiedererkennbare Stellen am Boden (eine Geländekante, ein Stein, ein heller Fleck). Dafür benutzen
   wir einen Algorithmus namens **ORB**. ORB findet solche Punkte und beschreibt ihre Umgebung mit einem
   kurzen „Fingerabdruck" (einem Merkmalsvektor), sodass man denselben Punkt im anderen Bild
   wiedererkennt. *Analogie:* Wie wenn du in zwei Fotos desselben Platzes dieselben Wahrzeichen suchst,
   um sie übereinanderzulegen.
2. **Kontrast anheben mit CLAHE.** Wärmebilder sind kontrastarm, da findet ORB schlecht Punkte.
   Deshalb heben wir vorher lokal den Kontrast an (**CLAHE** — eine lokale Histogramm-Spreizung, die in
   kleinen Kacheln dunkle/helle Bereiche auseinanderzieht, ohne zu überstrahlen). Das hat die Zahl
   brauchbarer Punkte deutlich erhöht.
3. **Punkte zuordnen mit dem Lowe-Ratio-Test.** Wir paaren Punkte aus Bild A mit den ähnlichsten in
   Bild B. Aber nicht jeder „beste Treffer" ist gut. Der **Lowe-Ratio-Test** behält ein Paar nur, wenn
   der beste Treffer **deutlich besser** ist als der zweitbeste (Verhältnis < 0,75). *Analogie:* Wenn
   zwei Stellen fast gleich gut passen, ist die Zuordnung unsicher — dann lieber wegwerfen. Das filtert
   Fehlpaare aggressiv heraus.
4. **Die Transformation robust schätzen mit RANSAC.** Selbst nach dem Filtern sind einige Paare falsch
   (Ausreißer). Würde man einfach alle mitteln, ruinieren wenige Ausreißer das Ergebnis. **RANSAC**
   löst das so: Es probiert viele Male, **rät** aus wenigen zufälligen Paaren eine Transformation, und
   zählt, **wie viele der anderen Paare dazu passen** (die „Inlier", also die Treuen). Am Ende behält es
   die Transformation, zu der die **meisten** Paare passen. *Analogie:* Statt den Durchschnitt aller
   Meinungen zu nehmen (inklusive Spinner), sucht man die Meinung, der die **größte Mehrheit** zustimmt.
   Das macht es robust gegen Ausreißer.
5. **Welche Art Transformation?** Wir erlauben eine **Ähnlichkeitstransformation** (Verschiebung +
   Drehung + gleichmäßige Skalierung), englisch *partial affine*. Keine wilde Verzerrung — die Drohne
   schiebt und dreht das Bild halt, mehr braucht es nicht.
6. **Tempo-Trick.** Wir rechnen das Ganze auf einer **halb so großen** Kopie der Bilder (schneller) und
   skalieren am Ende nur den Verschiebungsanteil wieder hoch. (Das ist mathematisch korrekt: Dreh- und
   Skalierungsanteil ändern sich beim Verkleinern nicht, nur die Verschiebung muss zurückskaliert
   werden — wir haben das geprüft.)

**Warum gerade so (Alternativen).** Man hätte das Bild auch über GPS/Drohnen-Telemetrie ausrichten
können — aber die gab es im Export nicht. Man hätte „dichten optischen Fluss" nehmen können — aber das
ist langsamer und unnötig genau für eine globale Verschiebung. ORB + Lowe + RANSAC ist der
**Standard-Werkzeugkasten** für „zwei Bilder zueinander ausrichten", robust und schnell.

**Zwischenergebnis.** Auf einem echten, aufeinanderfolgenden Bildpaar liefert die Registrierung eine
fast perfekte Ausrichtung (Inlier-Anteil ~0,97–1,0 bei klaren Paaren, im Schnitt über alle Tracks
**0,86**). Als wir CLAHE und den Lowe-Test eingebaut haben, stieg dieser Inlier-Schnitt von 0,72 auf
0,86 — und die Zahl der verwertbaren Tracks von 164 auf 190. Auf einem schönen Einzeltier-Flug
entfernt die Registrierung rund **800 Pixel** aufsummierte Drohnenbewegung und lässt etwa **205 Pixel**
echte Tierbewegung übrig. Wenn man das ausgerichtete Bild vom nächsten abzieht, **verschwindet der
Hintergrund nahezu vollständig (wird schwarz) und nur das Tier bleibt sichtbar** — das ist der visuelle
Beweis, dass das Prinzip funktioniert.

**Zahlenbeispiel zum Mitdenken.** Nimm an, ORB findet 500 Punktpaare zwischen zwei Frames. 430 davon
sind echte Bodenpunkte, die sich alle um „12 Pixel nach rechts, 3 nach unten" verschoben haben; 70 sind
Fehlpaare, die kreuz und quer zeigen — auch das bewegte Tier selbst ist so ein „Ausreißer". RANSAC rät
aus wenigen Paaren eine Verschiebung, zählt die Zustimmer und findet schnell die „12 rechts, 3 runter"-
Lösung, der 430 Paare zustimmen; die 70 Ausreißer werden ignoriert. Genau deshalb verfälscht das
bewegte Tier die Drohnen-Schätzung nicht — es ist in der Minderheit und fällt als Ausreißer heraus.

## 5.3 Tracking: das Tier über die Frames verfolgen

**Was.** „Tracking" heißt: erkennen, dass der Fleck in Bild 1 **dasselbe Tier** ist wie ein bestimmter
Fleck in Bild 2, in Bild 3 usw. So entsteht eine **Spur** (englisch *tracklet*) — die Folge von
Positionen eines einzelnen Tiers über die Zeit.

**Wie.** Wir machen das bewusst **einfach**: ein **gieriger Nächster-Schwerpunkt-Zuordner** (greedy
nearest-centroid). Im Klartext: Wir nehmen den Schwerpunkt jedes Tiers in Frame *t* und verbinden ihn
mit dem **nächstgelegenen** Schwerpunkt derselben Klasse in Frame *t+1* — solange der Abstand unter
einer Grenze liegt (sonst beginnt eine neue Spur). „Gierig" heißt: einfach immer den nächsten nehmen,
ohne kompliziert global zu optimieren.

**Warum gerade so (Alternative).** Es gibt ausgefeiltere Tracker, z. B. **SORT** mit einem
**Kalman-Filter** (der die nächste Position vorhersagt) und ungarischer Zuordnung. Wir haben das
**bewusst weggelassen**. Begründung: (a) Unsere Spuren sind kurz und die Tiere bewegen sich langsam
zwischen den Frames — da reicht „nimm den nächsten"; (b) ein Kalman-Filter müsste man tunen und würde
eine Bewegungsannahme einbauen, die das Ergebnis verfälschen könnte; (c) einfach = transparent und
leicht zu verteidigen. Das ist eine **Design-Entscheidung**, kein Versäumnis. (In der Verteidigung:
„Wir haben Kalman/SORT bewusst nicht genommen — auf diesen kurzen, langsamen Tracks ist der einfache
Nächster-Nachbar-Zuordner ausreichend und ehrlicher.")

**Zwischenergebnis.** Über alle Flüge entstehen **2.697 Tracklets**. Nicht alle sind brauchbar (viele
sind kurz oder von stehenden Tieren) — die Auswahl der vertrauenswürdigen kommt im nächsten Schritt.

## 5.4 Richtung und Konfidenz aus dem Tracking (Kreisstatistik)

**Was.** Aus einer Spur wollen wir **eine** Richtung machen — und ehrlich sagen, **wie sicher** wir
sind. Dafür brauchen wir Statistik, aber eine besondere Art: **Kreisstatistik**, weil Winkel
**rundherum** gehen (359 Grad und 1 Grad sind nah beieinander, nicht weit auseinander).

**Wie (Schritt für Schritt).**

1. **Pro Schritt eine Richtung.** Für jeden Schritt der Spur (von Frame *t* zu *t+1*): Wir nehmen den
   Schwerpunkt des Tiers in *t*, **schieben ihn mit der Drohnen-Transformation vorwärts** (so, wie sich
   der Hintergrund bewegt hat), und vergleichen mit der echten Position in *t+1*. Die Differenz ist die
   **Restbewegung** des Tiers — bereinigt um die Drohne. Der Winkel dieser Restbewegung ist die
   Schritt-Richtung. (Konvention: 0° = Osten, 90° = Süden, weil die Bild-y-Achse nach unten zeigt — im
   Uhrzeigersinn auf dem Bildschirm.)
2. **Schritte mitteln (Kreismittel).** Die einzelnen Schritt-Richtungen mitteln wir **zirkulär** zu
   einer mittleren Richtung. (Normales Mitteln wäre falsch: Der Mittelwert von 350° und 10° ist nicht
   180°, sondern 0°.)
3. **Wie einig sind sich die Schritte? (R, die „Konzentration").** Wir messen die **Resultierende
   Länge R** — eine Zahl zwischen 0 und 1. R nahe 1 heißt: Alle Schritte zeigen in fast dieselbe
   Richtung (das Tier läuft geradlinig — vertrauenswürdig). R nahe 0 heißt: Die Schritte zeigen kreuz
   und quer (Zappeln eines stehenden Tiers oder Rauschen — nicht vertrauenswürdig). *Analogie:* Wenn
   alle Pfeile einer Gruppe in dieselbe Richtung zeigen, ist R hoch; zeigen sie zufällig herum, ist R
   niedrig.
4. **Ist die Richtung echt oder Zufall? (Rayleigh-Test.)** Der **Rayleigh-Test** gibt einen p-Wert: die
   Wahrscheinlichkeit, dass eine so einheitliche Richtung **rein zufällig** entsteht. Kleiner p-Wert =
   die Richtung ist statistisch echt, nicht Zufall.
5. **Achse vs. Richtung (der 180-Grad-Trick).** Wenn wir nur die **Achse** (Linie ohne Vorne/Hinten)
   betrachten, rechnen wir „modulo 180" — dafür **verdoppelt** man die Winkel, bevor man sie auf den
   Kreis legt. So gelten 10° und 190° als **dieselbe Achse**. Bei der **vollen Richtung** (mit
   Vorne/Hinten) rechnet man „modulo 360".
6. **Das Vertrauens-Tor (Trust Gate).** Eine Richtung gilt nur dann als **vertrauenswürdig**, wenn
   **alle** folgenden Bedingungen erfüllt sind: genug Schritte (≥ 5), gute Registrierung
   (Inlier-Median ≥ 0,5), einheitliche Richtung (R ≥ 0,5), statistisch nicht zufällig (Rayleigh-p
   < 0,05) **und** eine Mindest-Wegstrecke (Netto-Versatz ≥ 5 Pixel, damit Mess-Rauschen nicht als
   Bewegung zählt). Ein stehendes Tier fällt durch dieses Tor — und das ist **richtig so**: Es hat eben
   keine Richtung.

**Warum gerade so.** Winkel **müssen** zirkulär behandelt werden, sonst rechnet man Unsinn (siehe
350°/10°-Beispiel). Den Rayleigh-Test und R nehmen wir, weil sie die **Standard-Werkzeuge** für „gibt
es eine bevorzugte Richtung in winkligen Daten?" sind. Das Vertrauens-Tor ist unsere Versicherung gegen
Fehlalarme: lieber wenige, aber **sichere** Richtungen behalten.

**Zwischenergebnis.** Von den 2.697 Spuren passieren **190** das Vertrauens-Tor. Beim genauen Hinsehen
ist dieses Tor aber **etwas zu großzügig** (z. B. hohe Einigkeit R bei nur wenigen Schritten ist
teilweise trivial; und hohes R bei winziger Wegstrecke kann ein kleiner konstanter Registrierungs-Fehler
vortäuschen). Deshalb definieren wir einen **verlässlichen Kern von 138** Spuren: zusätzlich
mindestens 8 Schritte **und** mindestens 50 Pixel Wegstrecke. Wir berichten ehrlich **beide** Zahlen
und nennen 138 die belastbare Zahl. (Pro Klasse: 190 = 83/68/39, Kern 138 = 58/49/31; mittlere
Wegstrecke im Kern ~327 Pixel.) **Diese Selbstkorrektur — „190 ist zu optimistisch, belastbar sind
138" — ist genau die Art Ehrlichkeit, die man verteidigen will.**

**Zahlenbeispiel zum Mitdenken.** Spur A hat zehn Schritte, die alle ungefähr nach Nordosten zeigen
(40–50°): Das Kreismittel ist ~45°, R liegt nahe 1 (sehr einig), der Rayleigh-p ist winzig — also
verlässlich. Spur B ist ein stehendes Tier, dessen Schwerpunkt nur durch Messrauschen mal hierhin, mal
dorthin zappelt (Schritte über den ganzen Kreis verteilt): R liegt nahe 0, der Rayleigh-p ist groß, der
Netto-Versatz fast null — das Vertrauens-Tor lehnt es korrekt ab. So trennen R und Rayleigh „echte
Bewegung" sauber von „Zappeln".

## 5.5 Einzelbild-Schätzer: Richtung aus EINEM Bild (GST und Frequenz-Methoden)

Jetzt die andere Seite des Vergleichs: Kann man die Richtung auch aus **einem einzelnen Bild** lesen
(aus der Bewegungsunschärfe), ohne Tracking? Das ist nützlich, wenn man nur ein Bild hat. Wir haben
mehrere klassische Schätzer verglichen.

**GST — Gradient Structure Tensor (unser bester Einzelbild-Schätzer).**

- **Was.** GST findet die **dominante Achse** eines Musters aus seinen **Kanten**. Ein
  Bewegungs-Schmierstreifen hat Kanten **entlang** der Schmierrichtung (an den Längsseiten der Spur).
- **Wie (intuitiv).** Man schaut sich an jedem Pixel an, in welche Richtung sich die Helligkeit am
  stärksten ändert (das ist der **Gradient** — er steht **senkrecht** zur Kante). Sammelt man die
  Gradienten der ganzen Region, ergibt sich eine bevorzugte Gradient-Richtung. Die **Längsachse** des
  Schmiers steht **senkrecht** dazu. Zusätzlich liefert GST eine **Kohärenz** (0–1): wie eindeutig die
  Achse ist (ein klarer Streifen hat hohe Kohärenz, ein runder Klecks niedrige). *Analogie:* Bei einem
  gekämmten Haarschopf zeigen alle „Helligkeitssprünge" quer zur Kämmrichtung — daraus liest man die
  Kämmrichtung ab.
- **Wichtige Einschränkung:** GST liefert nur die **Achse**, nicht das **Vorzeichen** (vorne/hinten) —
  die 180-Grad-Mehrdeutigkeit bleibt.

**Frequenz-basierte Alternativen, die wir auch getestet haben:**

- **Spektrum (FFT/Radon).** Eine **Fourier-Transformation** zerlegt ein Bild in Wellenmuster
  verschiedener Richtungen und Größen. Ein gerichteter Schmier zeigt sich als Streifenmuster im
  Frequenzbild; dessen Ausrichtung verrät die Achse.
- **Cepstrum.** Eine „Fourier der Fourier" — gut, um die **Länge** eines wiederholten Schmiers zu
  finden (ein Schmier ist im Grunde dasselbe Bild zweimal, leicht versetzt).
- **Gradient-Histogramm** und **Bildmomente/PCA** — einfachere Achs-Schätzer aus Kantenrichtungen bzw.
  aus der Form.

**Warum mehrere?** Weil die Aufgabe **Vergleich** verlangt: Wir wollten wissen, welcher klassische
Ansatz auf diesen kleinen, kontrastarmen Wärme-Klecks am besten ist — Ortsraum (GST) oder Frequenzraum
(FFT/Cepstrum).

**Zwischenergebnis (ehrlich).** Gegen die Tracking-Bewegungsrichtung gemessen (nur Achse, modulo 180)
ist **GST der Beste** mit einem mittleren Fehler von **29,1 Grad** (Treffer innerhalb von 45 Grad:
68 %). Danach Spektrum 32,8, Cepstrum 33,1, Gradient 33,5, Momente 35,2; reines Raten läge bei ~44.
**Aber** — und das ist die ehrliche Pointe — der faire Vergleichsmaßstab ist nicht „Raten", sondern
„rate einfach die **mittlere Richtung der jeweiligen Klasse**". Dieser einfache Trick ist erstaunlich
stark (25,0 Grad gepoolt), weil die Richtungen pro Klasse nicht gleichverteilt sind. Pro Klasse: Bei
**Klasse 0 verliert GST sogar gegen diesen Trick** (39,4 vs. 16,1 — die Richtung ist hier schon aus der
Klasse vorhersagbar), bei Klasse 1 und 2 **gewinnt** GST (25,4 vs. 32,6 bzw. 19,3 vs. 36,2). **Fazit:
Einzelbild-Richtung ist ein echtes, aber schwaches, klassenabhängiges Signal — kein Ersatz fürs
Tracking.** Genau diese nüchterne Einordnung ist Gold für die Verteidigung.

**Zahlenbeispiel zum Mitdenken.** Stell dir einen senkrechten hellen Streifen vor (ein nach oben/unten
schmierendes Tier). Geht man quer (links/rechts) über den Streifen, ändert sich die Helligkeit stark —
die Gradienten zeigen also nach links/rechts. GST sammelt das und schließt: dominante Gradient-Richtung
horizontal, also Längsachse senkrecht (90°), mit hoher Kohärenz, weil fast alle Gradienten gleich
zeigen. Bei einem runden Klecks zeigen die Gradienten in alle Richtungen gleichmäßig — Kohärenz nahe 0,
keine klare Achse. Genau deshalb ist GST gut bei länglichen Schmieren und unsicher bei runden Tieren.

## 5.6 Bewegt vs. steht: klassisch, CNN und Foundation Models im Vergleich

**Was.** Eine einfachere, aber wichtige Teilfrage: Kann man aus einem **einzelnen** Tier-Ausschnitt
sagen, ob das Tier **sich bewegt oder steht**? Hier vergleichen wir vier Familien von Methoden — das
ist das Kern-„Bake-off" der Aufgabe.

**Familie 1 — Klassisch (Hand-Features + einfaches ML).** Wir berechnen von Hand ein paar
**aussagekräftige Zahlen** pro Ausschnitt (englisch *features*): wie scharf/unscharf, wie länglich
(GST-Kohärenz), wie hell, wie stark der Kontrast zum Rand usw. Diese Zahlen geben wir einem einfachen
Lernverfahren:

- **Logistische Regression (LogReg):** zieht eine gewichtete Trennlinie zwischen „bewegt" und „steht".
- **Random Forest:** viele kleine **Entscheidungsbäume**, die per Mehrheit abstimmen. Ein
  Entscheidungsbaum ist eine Kette von Ja/Nein-Fragen („Ist die Schärfe < X? Ist die Länglichkeit
  > Y?"). Viele leicht verschiedene Bäume zusammen sind robuster als einer. *Analogie:* ein Gremium aus
  vielen Faustregel-Gutachtern statt eines einzelnen.

**Familie 2 — CNN von Grund auf.** Ein **CNN** (Convolutional Neural Network, „faltendes neuronales
Netz") lernt **selbst**, welche Bildmuster wichtig sind — man gibt ihm rohe Pixel, keine Hand-Features.
*Wie es grob funktioniert:* Kleine Filter fahren über das Bild und reagieren auf Muster (Kanten,
Streifen); übereinandergestapelte Schichten erkennen daraus immer komplexere Muster. „Von Grund auf"
heißt: Wir trainieren es nur auf unseren Daten, ohne Vorwissen.

**Familie 3 — Foundation Models (eingefroren).** Das sind **riesige, auf Millionen Bildern
vortrainierte** Netze — wir haben **DINOv2, CLIP und BioCLIP** genommen (BioCLIP ist speziell auf
Naturfotos/Arten vortrainiert). Wir benutzen sie **eingefroren** (englisch *frozen*): Das heißt, wir
**trainieren ihre Gewichte NICHT**, sondern lassen das Netz nur jeden Ausschnitt in einen
**Merkmalsvektor** (eine Zahlenliste, ein „Steckbrief" des Bildes) umwandeln und trainieren darauf nur
einen winzigen Klassifikator obendrauf. *Analogie:* Wir leihen uns das geübte „Auge" eines Experten,
ohne ihn umzuschulen, und setzen nur einen einfachen Schalter dahinter. (Wichtig für die Ehrlichkeit:
Wir haben **keine** großen Modelle selbst trainiert — sie sind eingefroren.)

**Familie 4 — Baselines (Vergleichsmaßstäbe).** Damit Zahlen Sinn ergeben, braucht man **Nulllinien**:
Zufall (50 %), „immer die häufigere Klasse raten" (Mehrheit) und — am wichtigsten — die **Szenen-Decke**
(dazu gleich).

**Warum dieser Vergleich.** Genau das ist der Kern der Aufgabe: **klassisch vs. lernend, einfach vs.
groß**. Wir wollten zeigen, was wirklich hilft — und ob die teuren großen Modelle ihr Geld wert sind.

**Zwischenergebnis (ehrlich).** Gemessen als **balancierte Genauigkeit** (fair auch bei
ungleichen Klassen): Foundation Models leicht vorne (BioCLIP 0,64, DINOv2 0,63, CLIP 0,62), dann
Random Forest 0,58, das selbst trainierte CNN bei 0,50 (es ist „kollabiert" — auf so wenig kleinen,
schwachen Daten lernt ein Netz von Grund auf kaum etwas). **Aber:** Die Unterschiede sind klein und
statistisch kaum trennbar — „BioCLIP ist Bester" ist im Rauschen. Und der **ehrliche Maßstab ist nicht
50 %, sondern die Szenen-Decke ~0,84** (siehe nächster Abschnitt) — und **alle** Modelle liegen
**darunter**. Heißt: Was die Modelle teilweise lernen, ist nicht „bewegt das Tier sich", sondern „aus
welcher Szene/welchem Flug stammt der Ausschnitt". Mehr dazu gleich.

## 5.7 Evaluation: warum unsere Zahlen ehrlich sind

Das ist der unscheinbarste, aber für die Note vielleicht wichtigste Teil: **fair messen**. Vier Säulen.

1. **Flug-getrennte Aufteilung (gegen Datenleck).** Beim maschinellen Lernen teilt man Daten in
   **Training** (zum Lernen) und **Test** (zum fairen Prüfen). Wenn Bilder **desselben Fluges** in
   beiden landen, kann das Modell schummeln: Es merkt sich die **Szene** (denselben Hintergrund,
   dieselbe Tiergruppe) statt der eigentlichen Frage. Das nennt man **Datenleck** (englisch *leakage*).
   Wir verhindern es mit **flug-getrennten Splits** (GroupKFold nach Flug-ID): Ein ganzer Flug ist
   **entweder** im Training **oder** im Test, nie in beiden. *Analogie:* Man prüft den Schüler mit
   Aufgaben, die er noch nie gesehen hat — nicht mit denen aus der Übungsstunde.
2. **Faire Baselines.** Man muss die **starke** Nulllinie schlagen, nicht die schwache. Bei der
   Einzelbild-Richtung ist das „rate die mittlere Richtung der Klasse" (25 Grad), nicht „rate komplett
   zufällig" (44 Grad). Bei bewegt/steht ist es die Szenen-Decke (~0,84), nicht 50 %.
3. **Die Szenen-Decke.** Innerhalb eines Fluges sind oft fast alle Tiere im selben Zustand (alle
   bewegen sich, oder alle stehen). Ein „Modell", das einfach pro Flug die Mehrheit rät, erreicht
   schon ~0,84. Das ist die **ehrliche Obergrenze**, die durch Szenen-Struktur erklärbar ist — und
   unsere Modelle liegen darunter. Diese Decke **offen anzugeben** ist Ehrlichkeit.
4. **Selbst-Audit.** Wir haben unsere eigenen Zahlen hart hinterfragt und Fehler **selbst korrigiert**
   — z. B. „190 ist zu optimistisch, belastbar sind 138" und ein früherer Einheitsfehler bei einer
   Signal-Rausch-Angabe (wir hatten kurz „~290×" stehen, korrekt sind ~30–40× netto). So etwas zu
   finden und zu berichtigen ist ein Qualitätsmerkmal, kein Makel.

## 5.8 Das Annotations-Tool: warum und wie wir es selbst gebaut haben

**Was/Warum.** Für die Validierung (Teil 9) musste ein Mensch (Andreas) Bilder von Hand labeln. Dafür
haben wir ein **eigenes, kleines Annotations-Programm** gebaut. Warum selbst bauen statt fertiger
Tools? Weil unsere Frage speziell ist (bewegt/steht + Achse + Kopf-Ende, mit der Möglichkeit, ehrlich
„weiß nicht" zu sagen), und weil es **offline und ohne Installation** laufen musste — Andreas sollte es
einfach starten können.

**Wie.** Es ist ein winziges Python-Programm, das nur die Bibliothek **Pillow** (zum Bilder anzeigen)
braucht. Es zeigt einen Tier-Ausschnitt mit grünem Kästchen. Der Annotierende drückt Tasten:
**bewegt / steht / unsicher**, dann optional eine **Achse** (mit Taste 5 in 8 Stufen à 22,5 Grad
drehen) und nur falls erkennbar einen **Pfeil** ans Kopf-Ende (Tasten 1/2), sonst „kein Pfeil" (3) oder
„nichts erkennbar" (0). Alles wird laufend in eine Datei `labels.csv` gespeichert, sodass man
unterbrechen und weitermachen kann. Jede Antwort ist über eine **crop_id** eindeutig mit dem richtigen
Bild verknüpft.

**Eine ehrliche Design-Lehre.** Anfangs hatten wir **vier** Bewegungsstärken (keine / leicht / mittel /
stark). Aus den echten Daten haben wir später gelernt, dass diese Feinheit **kein verlässliches Signal**
trägt (siehe Teil 9) und auf **steht / bewegt / unsicher** reduziert. Bestehende Labels mussten dafür
**nicht** neu gemacht werden — wir fassen die alten Stufen automatisch zusammen.

---

# TEIL 6 — DIE PROBLEME UND WIE WIR SIE GELÖST HABEN

Ein Projekt verteidigt man am besten über seine **Schwierigkeiten** — denn die zeigen, dass man wirklich
gearbeitet und nachgedacht hat. Hier die wichtigsten „Panne → Fix"-Geschichten.

**1. Hand-Labeln der Richtung war unzuverlässig → Pivot zum Tracking.** Das war die größte Wendung.
Weil der Kopf fast nie sichtbar ist, konnten wir keine verlässliche Richtungs-Wahrheit von Hand
erzeugen. Lösung: die Wahrheit aus der **Bewegung über Zeit** (Tracking) nehmen, Handlabels nur zur
Validierung. (Siehe Teil 3.)

**2. Die Registrierung war anfangs zu schwach → CLAHE + Lowe-Test.** Auf den kontrastarmen Wärmebildern
fand ORB zu wenige gute Punkte, viele Tracks waren unbrauchbar. Lösung: vorher den Kontrast lokal
anheben (CLAHE) und Fehlpaare mit dem Lowe-Ratio-Test wegwerfen. Effekt: Inlier-Schnitt **0,72 → 0,86**,
verwertbare Tracks **164 → 190**.

**3. ID-Switch in dichten Szenen (unsere ehrlichste Schwäche).** Weil 67 % der Bilder mehrere
gleichartige Tiere zeigen und unser einfacher Tracker **kein Aussehensmodell** hat, kann er in einem
Gedränge zwei Tiere **verwechseln** (englisch *ID-Switch*). Wir können das nicht völlig ausschließen.
Unsere Verteidigung dagegen ist das **Vertrauens-Tor**: Nur Spuren mit hoher Einigkeit (R), statistisch
echter Richtung (Rayleigh) und genug Wegstrecke zählen — ein einmaliger Verwechsler reißt R nach unten
und fliegt meist raus. Wir nennen diese Schwäche **offen** und sagen, dass sie den ganzen dichten Teil
betrifft, nicht nur ein paar Ausreißer.

**4. Schwarzwild unter-segmentiert → erkannt und korrekt erklärt.** Klasse 2 ist am kühlsten, deshalb
greift Otsu nur den heißen Kern (~37 % Fläche). Wir haben das nicht „repariert" (man könnte den
Schwellwert pro Klasse anpassen), sondern als **ehrliche Erkenntnis** dokumentiert: Es ist ein
Kontrast-/Segmentierungs-Effekt, keine schlechte Box.

**5. Vier Bewegungsstufen → datengetrieben auf zwei reduziert.** Andreas' Labels zeigten: „stark" kam
nur **2×** vor, „mittel" 40×, und „leicht vs. stärker" ist aus einem Einzelbild **nicht** unterscheidbar
(Genauigkeit 0,52 = Zufall; eine 3-Stufen-Skala 0,43 = unter Zufall). Lösung: auf **steht / bewegt**
zusammenfassen. Entscheidung **aus den Daten**, nicht aus dem Bauch.

**6. Datenleck-Falle → flug-getrennte Auswertung.** Siehe Teil 5.7. Ohne flug-getrennte Splits hätten
unsere Klassifikations-Zahlen viel zu gut ausgesehen (das Modell hätte die Szene auswendig gelernt). Wir
haben das von Anfang an verhindert und die Szenen-Decke offen als ehrlichen Maßstab angegeben.

**7. Eigene Zahlen korrigiert (Selbst-Audit).** „190 trusted" → belastbar **138**; eine
Signal-Rausch-Angabe war ein Einheitsfehler („~290×" → real ~30–40× netto). Selbst gefunden, selbst
berichtigt, alles dokumentiert.

---

# TEIL 7 — DIE TECHNISCHE UMSETZUNG (für High-Level-Technikfragen)

Dieser Teil ist dafür da, dass du auf Fragen wie „**In welcher Sprache? Mit welcher Library? Wie lange?
Welches Framework?**" souverän und konkret antworten kannst — ohne in jede Code-Zeile gehen zu müssen.

## 7.1 Sprache und Grundbausteine

- **Programmiersprache: Python.** Standard in Computer Vision und Data Science, riesiges Ökosystem.
- **NumPy** — die Grundbibliothek für Zahlen-Arrays (Bilder sind bei uns NumPy-Arrays). Alles Rechnen
  mit Pixeln und Vektoren läuft darüber.
- **OpenCV** (`cv2`, Version 4.x) — die zentrale Bildverarbeitungs-Bibliothek. Von ihr nutzen wir
  Otsu-Schwellwert, Morphologie, zusammenhängende Komponenten, ORB-Merkmale, den Matcher, RANSAC
  (`estimateAffinePartial2D`), CLAHE und das Warpen/Ausrichten von Bildern. **Die klassische
  CV-Maschine.**
- **pandas** — Tabellen-Bibliothek; alle Ergebnisse (Tracks, Fehler, Labels) liegen als Tabellen
  (CSV-Dateien) vor und werden damit ausgewertet.
- **scikit-learn** — klassisches maschinelles Lernen: logistische Regression, Random Forest, die
  flug-getrennte Kreuzvalidierung (GroupKFold), balancierte Genauigkeit / AUC.
- **Matplotlib** — alle Grafiken/Abbildungen.

## 7.2 Der Deep-Learning-Teil

- **PyTorch** (Version 2.x, mit CUDA für die Grafikkarte) — das Deep-Learning-Framework für unser
  selbst trainiertes CNN und um die Foundation Models laufen zu lassen.
- **open_clip / torchvision** — liefern die vortrainierten Modelle **DINOv2, CLIP, BioCLIP**. Wir nutzen
  sie **eingefroren** (keine Gewichts-Updates) und cachen ihre Merkmalsvektoren auf der Platte, damit
  man sie nicht bei jedem Lauf neu berechnen muss. Darauf trainiert nur ein winziger Klassifikator.
- Wichtig zum Mitnehmen: **Das einzige selbst trainierte neuronale Netz ist das kleine CNN von Grund
  auf** (und es ist kollabiert). Alle großen Modelle sind nur „Augen zum Ausleihen", nicht trainiert.
  Und **YOLO/Detektor: gar nicht trainiert** (Boxen kamen fertig).

## 7.3 Das Annotations-Tool

- Nur **Python + Pillow** (Bildanzeige). Bewusst **minimal**, damit es überall offline ohne komplizierte
  Installation läuft. Andreas startet es per Doppelklick; Antworten landen in `labels.csv`.

## 7.4 Wie wir gearbeitet haben (Qualität & Reproduzierbarkeit)

- **Test-getriebene Entwicklung (TDD).** Für die Kern-Logik haben wir **zuerst Tests** geschrieben, die
  prüfen, ob eine Funktion das Richtige tut, und dann den Code. Es gibt aktuell **136 Tests, alle grün**.
  Sie prüfen echtes Verhalten an bekannten Eingaben (z. B.: Registrierung findet eine bekannte
  Verschiebung wieder; die Kreisstatistik behandelt die 350°/10°-Falle korrekt; das Vertrauens-Tor
  lehnt jeden einzelnen Verletzungsfall ab).
- **Selbst-Prüfskript `verify_claims.py`.** Ein Skript, das **jede Kennzahl** aus diesem Dokument aus
  den gespeicherten Ergebnis-Tabellen **neu nachrechnet** und gegen den dokumentierten Wert prüft (138
  Kern, GST 29,1°, die Bewegungs-Tabelle, die Validierungs-Zahlen, …). Es läuft **ohne die rohen Bilder**
  (nur die kleinen Ergebnis-CSVs liegen im Repository). Aktuell: **alle Prüfungen bestehen.** Das macht
  die Behauptungen **überprüfbar**, ohne dass jemand die 1,9 GB Bilder braucht — ein starkes Argument
  für „das ist real und nicht erfunden".
- **Versionskontrolle mit Git/GitHub** (privates Repository) für die Arbeit über zwei Rechner. Die
  rohen Bilder sind zu groß für Git und liegen separat auf der Platte; das Programm findet sie
  automatisch über einen konfigurierten Pfad.

## 7.5 Aufwand, Daten- und Rechen-Dimensionen (für „wie lange / wie groß"-Fragen)

- **Aufwand:** im Rahmen des Kurses rund **40–50 Stunden pro Person** über mehrere Wochen, verteilt auf
  Daten verstehen, Methoden bauen und testen, auswerten, annotieren, validieren und dokumentieren.
- **Datenmenge:** ~12.655 Bilder à 2048×2048, ~1,9 GB; 46.046 Tier-Boxen; 223 Flüge.
- **Rechenzeit:** Die klassische Pipeline (Segmentierung, Registrierung, Tracking über alle Flüge) ist
  der schwerste Teil und läuft in der Größenordnung von Minuten bis wenige Stunden auf einem normalen
  Rechner (jede Registrierung ist ein ORB+RANSAC auf einem 2048er-Bild). Die Foundation-Features rechnet
  man **einmal** auf der Grafikkarte und cacht sie. Die eigentlichen Klassifikatoren (LogReg, Random
  Forest) trainieren in Sekunden.
- **Eine technische Eigenheit dieses Rechners:** Eine Sicherheits-Richtlinie (Application Control)
  blockierte das Laden bestimmter kompilierter Bibliotheken aus dem Desktop-Ordner, deshalb läuft alles
  über einen dedizierten Python-Interpreter. Auf einem normalen Rechner genügt eine frische virtuelle
  Umgebung (`requirements.txt` reproduziert sie). Das ist ein gutes Beispiel für eine kleine, ehrliche
  „echtes Leben"-Hürde, die man im Gespräch erwähnen kann.

---

# TEIL 8 — DIE ERGEBNISSE, EHRLICH

Hier alle wichtigen Zahlen gebündelt — alle aus den Ergebnisdateien nachgerechnet.

**Tracking (die Wahrheit selbst).**
- 2.697 Spuren insgesamt; **190** passieren das Vertrauens-Tor; **138** bilden den belastbaren Kern.
- Mittlerer Registrierungs-Inlier-Anteil über alle Spuren: **0,86** (nach Verbesserung von 0,72).
- Beispiel-Beweis: ~800 Pixel Drohnenbewegung entfernt, ~205 Pixel echte Tierbewegung übrig; Hintergrund
  fällt im Differenzbild auf nahe Schwarz.

**Einzelbild-Richtung (klassische Schätzer, gegen Tracking gemessen, nur Achse).**
- GST am besten: **29,1 Grad** mittlerer Fehler, 68 % innerhalb 45 Grad.
- Reihenfolge: GST 29,1 < Spektrum 32,8 < Cepstrum 33,1 < Gradient 33,5 < Momente 35,2; Zufall ~44.
- Ehrlicher Maßstab (mittlere Klassen-Richtung): gepoolt 25,0 — **schlägt GST insgesamt knapp**. Pro
  Klasse: GST verliert bei Klasse 0 (39,4 vs. 16,1), gewinnt bei Klasse 1 (25,4 vs. 32,6) und Klasse 2
  (19,3 vs. 36,2). **→ schwaches, klassenabhängiges Signal.**

**Bewegt vs. steht (Methoden-Vergleich, flug-getrennt, balancierte Genauigkeit).**
- Foundation Models: BioCLIP 0,64, DINOv2 0,63, CLIP 0,62 (untereinander statistisch nicht trennbar).
- Klassisch: Random Forest 0,58, LogReg ~0,56. CNN von Grund auf: 0,50 (kollabiert). Mehrheit: 0,50.
- **Ehrliche Decke: Szenen-Struktur ~0,84** — alle Modelle liegen darunter, also teils Szenen-Korrelation.

**Selbstkritik im Ergebnis.** Kein „Wow-Score", aber das war nie das Ziel. Der **Beitrag** ist die
**Methodik** (Tracking als Wahrheit) und die **ehrliche, jetzt validierte Analyse** — und genau das ist
das, was man in einer Bachelor-Arbeit zeigen will.

---

# TEIL 9 — DIE VALIDIERUNG MIT ECHTEN MENSCHLICHEN LABELS

Das ist die unabhängige Kontrolle, auf die das ganze Projekt gewartet hat. Andreas hat **alle 1.500
ausgewählten Ausschnitte** von Hand annotiert. Damit konnten wir prüfen, ob unsere Tracking-Wahrheit
mit menschlicher Wahrnehmung übereinstimmt.

## 9.1 Eine wichtige technische Hürde: die Winkel-Konvention

Bevor man Mensch und Tracking vergleicht, muss man ihre **Winkel-Konventionen** angleichen — sonst
vergleicht man Äpfel mit Birnen.

- Das **Annotations-Tool** speichert: 0° = nach oben, im Uhrzeigersinn.
- Das **Tracking** rechnet: 0° = nach Osten (rechts), und weil die Bild-y-Achse nach unten zeigt,
  ebenfalls im Uhrzeigersinn.

Beide drehen also gleich herum, der Nullpunkt liegt nur unterschiedlich (oben vs. rechts = 90 Grad
Unterschied). Man muss den menschlichen Winkel deshalb um **−90 Grad** umrechnen. Dass das **richtig**
ist, sieht man am Ergebnis selbst: **Ohne** die Umrechnung beträgt die Abweichung 67 Grad (fast der
schlechtestmögliche Wert), **mit** der Umrechnung 22,7 Grad. Diese kleine, oft übersehene Sache richtig
gemacht zu haben, ist ein gutes Detail für die Verteidigung.

## 9.2 Die zentrale Bestätigung (der „Linchpin")

Auf den Ausschnitten, wo das Tracking eine Richtung **vertraut** und der Mensch eine **Linie** gezeichnet
hat, vergleichen wir die Achsen:

- Über alle solchen Fälle (n = 56): **22,7 Grad** mittlere Abweichung, 79 % innerhalb 45 Grad.
- Wo der Mensch zusätzlich „bewegt" sagt (n = 22): **19,1 Grad**, 86 % innerhalb 45 Grad.
- Zum Vergleich Zufall: ~50 Grad.
- Wo der Mensch ein Kopf-Ende markiert hat (n = 7), stimmte das **Vorzeichen** (vorne/hinten) zu
  **100 %** mit dem Tracking überein.

**→ Die Tracking-Wahrheit ist durch unabhängige menschliche Wahrnehmung bestätigt** (~20 Grad, weit
über Zufall). Ehrliche Einschränkung: Die Überlappung ist klein (n = 56), also „bestätigt, aber auf
kleiner, konsistenter Stichprobe" — wir sagen das genau so.

## 9.3 Drei weitere Ergebnisse aus den Labels

- **GST ist ein guter Orientierungs-Schätzer:** Gegen die **menschliche Achse** (alle 963
  Linien-Ausschnitte) liegt GST bei nur **10,7 Grad** (90 % innerhalb 45 Grad). Das ist viel besser als
  die 29 Grad gegen die **Bewegungs**-Richtung. Die Erklärung ist schön: **Orientierung** (wie liegt der
  Körper) ist leicht; die **signierte Bewegungsrichtung** (wohin mit Kopf voran) ist das eigentlich
  Schwere. GST liest dieselbe Achse wie ein Mensch — nur das Vorne/Hinten bleibt offen.
- **Der Kopf ist fast nie erkennbar:** Insgesamt gab der Mensch nur in **14 %** der Fälle einen Kopf an
  (bei klar bewegten Tieren 27 %). **Das bestätigt quantitativ unsere ganze Ausgangsentscheidung** —
  genau weil man den Kopf nicht sieht, nehmen wir die Richtung aus dem Tracking, nicht aus Handlabels.
- **Bewegt/steht auf echten Labels:** LogReg **0,62** balancierte Genauigkeit (Random Forest 0,58,
  Mehrheit 0,50, Szenen-Decke 0,78) — **dasselbe Bild wie mit den Hilfslabels.** Eine ehrliche
  Feinheit: Der Mensch nennt viele Tiere „bewegt" (meist „leicht"), die unser strenges Tracking-Tor
  **nicht** als bewegt erkennt — das ist **kein Widerspruch**, sondern Absicht: Das Tor ist
  **hochpräzise, aber niedrig-empfindlich** (es fängt nicht jeden Bewegten, aber bei denen, die es
  behält, hat es bei der Richtung recht).

## 9.4 Warum nur zwei Bewegungsstufen (aus den Daten entschieden)

Die feineren Stufen tragen kein Signal: „stark" kam nur 2× vor, „mittel" 40×; „leicht vs. stärker" aus
einem Einzelbild ist **0,52 = Zufall**, eine 3-Stufen-Skala **0,43** (unter dem Zufall von 0,33). Nur
die Trennung **steht vs. bewegt** ist lernbar (0,62). Deshalb: auf **binär** reduziert. Übrigens ist
auch interessant, dass der Mensch in **43 %** der Fälle „unsicher" wählte — selbst „bewegt oder steht?"
ist auf diesen Bildern oft nicht entscheidbar. Das ist selbst ein Ergebnis.

## 9.5 Das Fazit der Validierung

Die menschlichen Labels **bestätigen das Projekt**: Tracking findet eine echte Bewegungsrichtung
(Mensch stimmt ~20 Grad zu), GST liest die Orientierung gut (~11 Grad), aber nicht die signierte
Richtung, und der Kopf ist selten sichtbar (14 %) — genau deshalb ist Tracking, nicht Handlabeln, die
Wahrheit. **Kein Teil der Pipeline musste neu gebaut werden.** Stärker kann eine Validierung kaum
ausfallen.

---

# TEIL 10 — GRENZEN UND OFFENE PUNKTE (offen genannt)

Eine gute Verteidigung nennt die Grenzen **selbst**, bevor jemand fragt:

- **ID-Switch** in dichten Szenen ist die Hauptschwäche des einfachen Trackers (kein Aussehensmodell).
  Abgefedert durch das Vertrauens-Tor, aber nicht eliminiert.
- **Einzelbild gibt nur die Achse**, nicht die Kopfrichtung — die 180-Grad-Mehrdeutigkeit bleibt (Kopf
  nur 14 % sichtbar).
- **Die Artnamen sind unbestätigt** (nur Klassen-IDs 0/1/2).
- **Die Validierungs-Überlappung ist klein** (Mensch × vertrauenswürdiges Tracking, n = 56): konsistent
  und klar über Zufall, aber keine große Stichprobe.
- **Das selbst trainierte CNN ist kollabiert** — zu wenige, zu kleine, zu schwache Bilder, um von Grund
  auf zu lernen; das ist ein ehrliches Negativ-Ergebnis und kein Skandal.
- **Offen für die Zukunft:** ein überwachtes Deep-Learning-Verfahren, das aus der Tracking-Richtung
  direkt die Richtung regressiert (datenlimitiert), und der schriftliche Bericht.

---

# TEIL 11 — VERTEIDIGUNG: TYPISCHE PRÜFUNGSFRAGEN MIT ANTWORTEN

Hier in „so würde ich antworten"-Form. Kurz, sicher, ehrlich.

**„Was war eure Aufgabe in einem Satz?"**
„Die Bewegungsrichtung von Wildtieren aus thermischen AOS-Lichtfeld-Drohnenbildern zu schätzen — und
verschiedene Computer-Vision-Methoden dafür zu vergleichen."

**„Was ist an dem Problem das Schwierige?"**
„Man sieht auf den unscharfen Wärme-Klecks den Kopf fast nie, also gibt es keine fertige Wahrheit für
die Richtung. Außerdem sind die Tiere winzig (~65 px) und es sind meist mehrere im Bild (67 %)."

**„Woher nehmt ihr dann eure ‚richtige Antwort'?"**
„Aus dem Tracking. Wir verfolgen jedes Tier über die Frames eines Fluges, rechnen die Drohnenbewegung
heraus, und die Restbewegung ist die echte Richtung. Das ist Geometrie über die Zeit, kein Raten am
Einzelbild. Menschliche Labels dienen nur der Validierung."

**„Habt ihr ein neuronales Netz für die Tier-Erkennung trainiert / YOLO?"**
„Nein. Die Bounding Boxes kamen fertig mit dem Datensatz im YOLO-Format. Detektion war nicht unsere
Aufgabe; wir haben darauf aufgesetzt."

**„Wie funktioniert eure Segmentierung — ist das Deep Learning?"**
„Nein, klassisch: Gauß-Glättung, Otsu-Schwellwert, morphologisches Aufräumen, größte zentrale
zusammenhängende Komponente, dann Bildmomente für Achse und Länglichkeit. Bei so einfachen
‚heller Klecks auf dunkel'-Bildern reicht das und ist transparenter als ein Netz."

**„Erklär RANSAC.“**
„RANSAC schätzt eine Transformation robust gegen Ausreißer: Es rät viele Male aus wenigen zufälligen
Punktpaaren eine Transformation und behält die, zu der die meisten anderen Paare passen. So ruinieren
einzelne Fehlpaare das Ergebnis nicht."

**„Warum kein Kalman-Filter / SORT beim Tracking?"**
„Bewusste Entscheidung. Unsere Tracks sind kurz und die Tiere bewegen sich langsam zwischen Frames — da
reicht der einfache Nächster-Schwerpunkt-Zuordner. Ein Kalman-Filter müsste man tunen und würde eine
Bewegungsannahme einbauen, die das Ergebnis verzerren könnte. Einfach ist hier ehrlicher."

**„Was ist R und der Rayleigh-Test?"**
„R misst, wie einig sich die Schritt-Richtungen einer Spur sind (0 = chaotisch, 1 = perfekt geradlinig).
Der Rayleigh-Test sagt, ob diese Einheitlichkeit statistisch echt ist oder Zufall sein könnte. Beides
nutzen wir als Filter, um nur verlässliche Richtungen zu behalten."

**„Warum braucht man Kreisstatistik?"**
„Weil Winkel rundherum gehen: Der Mittelwert von 350° und 10° ist 0°, nicht 180°. Normale Statistik
würde hier Unsinn rechnen. Für die Achse ohne Vorne/Hinten verdoppeln wir zusätzlich die Winkel
(modulo 180)."

**„Welche Methoden habt ihr verglichen und was war das Ergebnis?"**
„Klassische Einzelbild-Schätzer (GST am besten, ~29° Achsfehler), klassisches ML (Random Forest), ein
CNN von Grund auf (kollabiert) und eingefrorene Foundation Models (DINOv2/CLIP/BioCLIP, ~0,62–0,64
bei bewegt/steht). Kein klarer Sieger, und alle unter der ehrlichen Szenen-Decke ~0,84 — teils
Szenen-Korrelation."

**„Was heißt ‚Foundation Model eingefroren'?"**
„Wir nehmen ein riesiges, vortrainiertes Netz und ändern seine Gewichte nicht. Es verwandelt jeden
Ausschnitt nur in einen Merkmalsvektor; darauf trainieren wir einen winzigen Klassifikator. Wir leihen
sein geübtes Auge, schulen es aber nicht um."

**„Wie habt ihr Datenleck vermieden?"**
„Flug-getrennte Aufteilung: Ein ganzer Flug ist entweder im Training oder im Test, nie in beiden. Sonst
würde das Modell die Szene auswendig lernen statt die eigentliche Frage."

**„Was ist die Szenen-Decke?"**
„Innerhalb eines Fluges sind oft fast alle Tiere im selben Zustand. Wer einfach pro Flug die Mehrheit
rät, erreicht schon ~0,84. Das ist die ehrliche Obergrenze, an der man Modelle messen muss — nicht die
50 %."

**„Wie habt ihr validiert, dass eure Tracking-Richtung stimmt?"**
„Ein Teamkollege hat 1.500 Bilder unabhängig von Hand gelabelt. Wo Tracking und Mensch beide eine Linie
haben, stimmen die Achsen auf ~22,7° im Median überein (19,1° bei klar Bewegten), gegen ~50° bei Zufall;
und wo ein Kopf markiert war, stimmte das Vorzeichen zu 100 %."

**„Was, wenn die menschlichen Labels widersprochen hätten?"**
„Dann wäre das ein **Befund** gewesen — ein Hinweis auf ID-Switches oder darauf, dass die Bewegung doch
nicht erkennbar ist — und kein Beinbruch. Sie haben aber bestätigt, und nichts in der Pipeline musste
neu gebaut werden."

**„Warum nur zwei Bewegungsstufen statt vier?"**
„Aus den Daten: ‚stark' kam nur 2× vor, ‚leicht vs. stärker' ist aus einem Bild Zufall (0,52), eine
3-Stufen-Skala sogar unter Zufall. Nur steht-vs-bewegt ist lernbar. Also haben wir datengetrieben auf
binär reduziert."

**„In welcher Sprache / mit welchen Libraries habt ihr das gebaut?"**
„Python. Klassische CV mit OpenCV und NumPy, Auswertung mit pandas, klassisches ML mit scikit-learn,
Deep Learning mit PyTorch und open_clip, Grafiken mit Matplotlib, das Annotations-Tool nur mit Pillow."

**„Wie stellt ihr sicher, dass die Zahlen stimmen / reproduzierbar sind?"**
„Test-getriebene Entwicklung mit 136 grünen Tests für die Kern-Logik, und ein Skript `verify_claims.py`,
das jede Kennzahl aus den gespeicherten Ergebnis-Tabellen neu nachrechnet und gegen die Doku prüft —
alle Prüfungen bestehen, und das sogar ohne die rohen Bilder."

**„Was ist die größte Schwäche eurer Arbeit?"**
„Der ID-Switch in dichten Szenen, weil unser einfacher Tracker kein Aussehensmodell hat. Wir federn ihn
mit dem Vertrauens-Tor ab und nennen ihn offen. Und die Validierungs-Überlappung ist klein (n=56) —
konsistent, aber keine große Stichprobe."

**„Was würdet ihr als Nächstes machen?"**
„Ein überwachtes Deep-Learning-Modell, das aus der Tracking-Richtung direkt die Richtung lernt
(flug-getrennt), und den schriftlichen Bericht. Und die Artnamen mit dem BAMBI-Team verifizieren."

**„Ist das nicht zirkulär — ihr nehmt Tracking als Wahrheit und messt dann damit?"**
„Für die Einzelbild-Richtung (GST) nicht: GST ist ein völlig unabhängiger Schätzer, der nur das Bild
ansieht; wir messen ihn gegen die geometrische Tracking-Wahrheit — das ist ganz normal ‚Schätzer gegen
Ground Truth'. Zirkulär war nur die erste bewegt/steht-Variante, deren Hilfslabels aus dem Tracking
kamen; das haben wir offen gesagt und mit Andreas' unabhängigen Handlabels gegengeprüft (gleiche
~0,62) — damit ist die Zirkularität aufgelöst."

**„Wie viele Bilder hat so eine Spur?"**
„Sehr unterschiedlich; unser Beispiel-Tracklet 809 hat 65 Beobachtungen. Für den verlässlichen Kern
verlangen wir mindestens 8 Schritte, damit eine Richtung überhaupt belastbar ist."

**„Warum sind die Bilder 2048×2048?"**
„So liefert der AOS-Export die Integralbilder. Die Registrierung rechnen wir aus Tempogründen auf halber
Größe und skalieren nur den Verschiebungsanteil korrekt wieder hoch."

**„Was, wenn die Drohne sich dreht und nicht nur verschiebt?"**
„Deshalb erlauben wir eine Ähnlichkeitstransformation — Verschiebung **plus** Drehung **plus**
gleichmäßige Skalierung —, nicht nur eine reine Verschiebung. RANSAC schätzt Drehung und Skalierung mit."

**„Warum balancierte Genauigkeit statt normaler Genauigkeit?"**
„Weil die Klassen ungleich groß sind (mehr stehende als bewegte Tiere). Normale Genauigkeit könnte man
durch ‚immer steht raten' künstlich aufblähen; die balancierte Genauigkeit mittelt die Trefferrate je
Klasse und ist deshalb fair."

**„Wie verhindert ihr Überanpassung beim CNN?"**
„Vor allem durch die flug-getrennte Auswertung. Das CNN ist auf so wenig schwachen Daten ohnehin
kollabiert (0,50) — das Problem war nicht Über-, sondern Unterlernen, und das berichten wir ehrlich."

**„Warum habt ihr nur 1.500 Bilder gelabelt, nicht alle?"**
„Es ist eine **Validierungs**-Stichprobe, keine Trainingsmenge — die Wahrheit kommt aus dem Tracking.
1.500 gezielt ausgewählte, klare Einzeltiere reichen, um die Übereinstimmung mit dem Menschen
statistisch zu zeigen."

**„Warum ist BioCLIP nicht klar besser, obwohl es auf Tieren vortrainiert ist?"**
„Weil unsere Bilder thermische, winzige, unscharfe Klecks sind — ganz anders als die scharfen Naturfotos,
auf denen BioCLIP gelernt hat. Sein Vorwissen passt kaum; der Vorsprung vor DINOv2/CLIP liegt im
Rauschen."

**„Was würde ein:e Reviewer:in zuerst kritisieren?"**
„Die kleine Validierungs-Überlappung (n=56) und den ID-Switch. Beides nennen wir selbst; n ist klein,
aber das Signal ist klar über Zufall und in jeder Teilmenge konsistent."

**„Was ist euer eigentlicher wissenschaftlicher Beitrag?"**
„Die Methodik: zu zeigen, dass man bei fehlender Richtungs-Wahrheit die Wahrheit sauber aus dem Tracking
gewinnen und damit Einzelbild-Methoden ehrlich bewerten kann — plus die quantitative Bestätigung, dass
der Kopf praktisch unsichtbar ist (nur 14 %)."

**„Wie ist der Code organisiert / wie groß ist er?"**
„Eine kompakte Bibliothek in `src/` (Datenladen, Segmentierung, Registrierung, Tracking, Richtung,
Kreisstatistik, GST/Blur, Klassifikation, Annotation), ausführbare Skripte in `scripts/`, Tests in
`tests/` (136 grün) und Ergebnis-Tabellen in `output/`."

**„Könnt ihr die Ergebnisse ohne die Originalbilder belegen?"**
„Ja — die kleinen Ergebnis-CSVs liegen im Repository, und `verify_claims.py` rechnet jede Kennzahl daraus
nach und prüft sie gegen die Doku. Die 1,9 GB Bilder braucht man dafür nicht."

---

# TEIL 12 — GLOSSAR: JEDER BEGRIFF IN ZWEI SÄTZEN

- **AOS (Airborne Optical Sectioning):** Aufnahmeverfahren, das viele leicht versetzte Drohnen-Bilder
  auf den Boden ausrichtet und mittelt, um Verdeckungen (Pflanzen) wegzurechnen. Ergebnis ist ein
  „Integralbild" über ~9 Frames / ~1 Sekunde.
- **Wärmebild / Thermal:** Bild der abgestrahlten Wärme statt des Lichts; warme Tiere leuchten hell auf
  kühlem Grund, aber grob und unscharf.
- **Bounding Box:** Rechteck um ein Objekt; bei uns fertig im Datensatz vorhanden (YOLO-Format).
- **YOLO-Format:** Speicherart für Boxen (Klasse, Mittelpunkt, Größe als Anteil). „YOLO" ist auch ein
  Detektor — den haben wir aber nicht trainiert.
- **EDA (Exploratory Data Analysis):** das systematische Anschauen und Statistik-Machen der Daten, bevor
  man Methoden baut.
- **Segmentierung:** jedes Pixel als Objekt oder Hintergrund markieren; bei uns klassisch.
- **Otsu-Schwellwert:** findet automatisch die Helligkeitsgrenze zwischen dunkel und hell, indem er die
  beiden Gruppen optimal trennt.
- **Morphologie (Öffnen/Schließen):** Aufräum-Operationen auf der Schwarz-Weiß-Maske; entfernt kleine
  Flecken bzw. füllt kleine Löcher.
- **Zusammenhängende Komponente:** eine durchgehend verbundene weiße Fläche in der Maske; wir nehmen die
  größte zentrale als das Tier.
- **Bildmomente:** statistische Kennzahlen der Pixelverteilung; liefern Schwerpunkt, Hauptachse und
  Länglichkeit (Exzentrizität).
- **Registrierung:** zwei Bilder deckungsgleich übereinanderlegen; bei uns, um die Drohnenbewegung zu
  bestimmen und abzuziehen.
- **ORB:** Algorithmus, der markante Punkte in einem Bild findet und mit einem „Fingerabdruck"
  beschreibt, damit man sie im anderen Bild wiederfindet.
- **CLAHE:** lokale Kontrastspreizung; macht kontrastarme Wärmebilder so, dass ORB mehr Punkte findet.
- **Lowe-Ratio-Test:** behält eine Punktzuordnung nur, wenn der beste Treffer deutlich besser als der
  zweitbeste ist; filtert unsichere Paare.
- **RANSAC:** robuste Schätzung gegen Ausreißer; nimmt die Lösung, der die meisten Datenpunkte zustimmen.
- **Affine / Ähnlichkeitstransformation:** Verschiebung + Drehung + (gleichmäßige) Skalierung — das, was
  die Drohnenbewegung mit dem Bild macht.
- **Tracking / Tracklet:** ein Tier über mehrere Frames verfolgen; ein Tracklet ist seine Positionsspur.
- **Greedy nearest-centroid:** einfacher Tracker, der jeden Schwerpunkt mit dem nächstgelegenen im
  nächsten Frame verbindet.
- **Ego-Motion:** die Eigenbewegung der Kamera/Drohne, die das ganze Bild verschiebt.
- **Kreisstatistik:** Statistik für Winkel, die berücksichtigt, dass 359° und 1° benachbart sind.
- **Resultierende Länge R:** Maß für die Einheitlichkeit von Richtungen (0 = chaotisch, 1 = geradlinig).
- **Rayleigh-Test:** prüft, ob eine bevorzugte Richtung statistisch echt ist oder Zufall sein könnte.
- **Axial (mod 180):** Achse ohne Vorne/Hinten; man verdoppelt die Winkel, damit 10° und 190° dasselbe
  sind.
- **180-Grad-Mehrdeutigkeit:** man kennt die Achse, aber nicht das Vorzeichen (Kopf vorne oder hinten).
- **Trust Gate (Vertrauens-Tor):** Bedingungen, die eine Tracking-Richtung erfüllen muss, um als
  verlässlich zu gelten (genug Schritte, gute Registrierung, hohes R, signifikant, genug Wegstrecke).
- **GST (Gradient Structure Tensor):** schätzt die dominante Achse eines Musters aus seinen
  Kantenrichtungen; bei uns der beste Einzelbild-Achsschätzer.
- **Gradient:** Richtung des stärksten Helligkeitswechsels an einem Pixel; steht senkrecht zur Kante.
- **FFT / Spektrum:** zerlegt ein Bild in Wellenmuster; gerichteter Schmier zeigt sich als gerichtetes
  Frequenzmuster.
- **Cepstrum:** „Fourier der Fourier"; gut, um die Länge eines wiederholten/versetzten Musters zu finden.
- **Feature (Merkmal):** eine aussagekräftige Zahl, die man aus einem Bild berechnet (z. B. Schärfe,
  Länglichkeit) und einem Lernverfahren gibt.
- **Logistische Regression:** einfaches Lernverfahren, das eine gewichtete Trennlinie zwischen zwei
  Klassen zieht.
- **Random Forest:** viele Entscheidungsbäume, die per Mehrheit abstimmen; robust und einfach.
- **Entscheidungsbaum:** eine Kette von Ja/Nein-Fragen, die zu einer Entscheidung führt.
- **CNN (Convolutional Neural Network):** neuronales Netz, das mit kleinen Filtern selbst lernt, welche
  Bildmuster wichtig sind.
- **Foundation Model:** sehr großes, auf riesigen Datenmengen vortrainiertes Netz (DINOv2/CLIP/BioCLIP),
  das wir eingefroren als „Augen" benutzen.
- **Eingefroren (frozen):** das Modell wird nicht weiter trainiert; nur seine fertigen Merkmalsvektoren
  werden genutzt.
- **Merkmalsvektor / Embedding:** eine Zahlenliste, die ein Bild zusammenfasst, damit ein Klassifikator
  damit rechnen kann.
- **Balancierte Genauigkeit:** Genauigkeit, die ungleich große Klassen fair behandelt (Mittel der
  Trefferraten je Klasse).
- **AUC:** Maß für die Trennschärfe eines Klassifikators über alle Schwellen (0,5 = Zufall, 1 = perfekt).
- **Flug-getrennte Aufteilung (GroupKFold):** Training/Test so trennen, dass ein ganzer Flug nur auf
  einer Seite ist — gegen Datenleck.
- **Datenleck (Leakage):** wenn Test-Information ins Training sickert und Ergebnisse zu gut aussehen
  lässt.
- **Szenen-Decke (Scene ceiling):** die ehrliche Obergrenze, die allein aus Szenen-Struktur erreichbar
  ist (~0,84); der faire Maßstab statt 50 %.
- **Baseline:** einfacher Vergleichsmaßstab (Zufall, Mehrheit, mittlere Klassen-Richtung), den eine
  Methode schlagen muss, um etwas wert zu sein.
- **ID-Switch:** wenn der Tracker zwei nahe, gleichartige Tiere verwechselt; unsere Hauptschwäche.
- **TDD (test-getriebene Entwicklung):** zuerst den Test schreiben, dann den Code; sichert echtes
  Verhalten ab (136 Tests grün).
- **Reproduzierbarkeit / verify_claims:** ein Skript rechnet alle Kennzahlen aus den Ergebnis-Tabellen
  nach und prüft sie gegen die Doku — alles besteht, sogar ohne die rohen Bilder.
- **Ground Truth (Wahrheit):** die „richtige Antwort", gegen die man misst; bei uns aus dem Tracking,
  nicht aus Handlabeln.

---

# SCHLUSSWORT

Wenn du nur drei Dinge mitnimmst, dann diese:

1. **Das Problem:** Richtung aus unscharfen Wärme-Klecks, ohne sichtbaren Kopf und ohne fertige Wahrheit.
2. **Die Idee:** Wahrheit aus der **Bewegung über Zeit** (Tracking, Drohne herausgerechnet), nicht aus
   dem Einzelbild — und alles ehrlich, leckagefrei vergleichen.
3. **Das Ergebnis:** Tracking liefert eine echte Richtung, **von Menschen bestätigt (~20 Grad)**;
   Einzelbild gibt nur die Achse; der Wert der Arbeit ist die saubere, selbstkritische **Methodik**.

Das ist eine Arbeit, die man **gut verteidigen** kann — nicht weil sie einen Rekord-Score hat, sondern
weil sie ein echtes Problem **ehrlich und durchdacht** löst und jede Behauptung belegen kann.
