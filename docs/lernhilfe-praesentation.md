# BAMBI — Projekt verstehen & präsentieren

> Für **dich und deinen Kollegen** (ohne Vorwissen). Genug Tiefe, um das Projekt zu **verstehen** und als
> **eure eigene Arbeit** zu verteidigen — aber kompakt. Jede Methode: **was sie tut, wie sie funktioniert
> (mit Mini-Beispiel), warum wir sie wählten (+ Alternative), was rauskam.** Alltagssprache, **keine
> Formeln**, nur projektrelevante Methoden.
> Lesen: Teil A (verstehen) → B/C (Ergebnisse/Pannen) → D/E (Entscheidungen/Folien) → F (Fragen üben) →
> G (Glossar).

## Roter Faden (auswendig)
Aus thermischen Drohnen-Lichtfeldbildern (AOS) die **Bewegungsrichtung** von Wildtieren schätzen. Problem:
dafür gibt es **keine Ground Truth** (keine vorgegebene richtige Antwort) — auf den verschwommenen
Wärmeflecken sieht man den Kopf meist nicht. Idee: die Wahrheit aus **Tracking** holen (Drohnenbewegung
rausrechnen → Restbewegung = echte Tierbewegung) und darauf **mehrere CV-Methoden** ehrlich vergleichen.
Botschaft: Tracking funktioniert, Einzelbild-Richtung ist schwach, Wert = sauberer Vergleich + ehrliche
Analyse, nicht ein Top-Score.

## Pipeline auf einen Blick
**Daten (AOS-Thermal)** → **Vorverarbeitung** (Kästen aufs Tier ziehen) → **Tracking als Ground-Truth**
(Ego-Motion raus → Richtung) → **Methodenvergleich** (Einzelbild + bewegt/steht: klassisch vs CNN vs
Foundation) → **ehrliche Evaluation**.

## Das Projekt in 6 Sätzen (Schnellabruf)
1. Wir schätzen die **Bewegungsrichtung** von Wild aus thermischen **AOS-Lichtfeldern**.
2. Bewegte Tiere **verschmieren** im Integralbild — die Schmierrichtung ist unser Signal.
3. Für die Richtung gibt es **keine Ground Truth**, also holen wir sie aus **Tracking** (Drohne rausrechnen → Restbewegung).
4. Darauf **vergleichen** wir klassische, CNN- und Foundation-Methoden — leckagefrei und mit fairen Baselines.
5. Ergebnis: Tracking **funktioniert** (138 verlässliche Richtungen), Einzelbild-Richtung ist **schwach**, kein Modell klar überlegen.
6. Der Wert ist die **Methodik + ehrliche Analyse**, nicht ein Top-Score; menschliche Labels **validieren** am Ende.

---

# TEIL A — Projekt & Methoden verstehen

## A1. Aufgabe
BAMBI (FH Hagenberg) überwacht Wild aus der Luft mit Wärmebild-Drohnen. Unsere Frage: **in welche Richtung
läuft ein Tier?** Nützlich, um Tiere zu zählen und Verhalten zu verstehen. Schwer, weil Tiere kleine,
verschwommene Wärmeflecken sind (~65 px), oft mehrere pro Bild, Kopf meist unsichtbar.

## A2. Daten — Thermal, Lichtfeld, AOS
**Was:** thermische Lichtfeld-Aufnahmen aus der Drohne.
**Wie es funktioniert:**
- **Thermal:** Wärmebild, ein Kanal pro Pixel (Wert = wie warm). Ein Tier ist wärmer als der Boden → **heller
  Fleck** auf dunklem Grund. Genau diese Annahme („Tier = die helle Region") nutzt später die Segmentierung.
- **AOS / Lichtfeld:** Die Drohne nimmt aus leicht verschiedenen Positionen auf; ein Computer richtet alle
  Aufnahmen **auf die Bodenebene** aus und **mittelt** sie zu einem **Integralbild**. *Analogie:* durch einen
  Gartenzaun fotografieren und beim Seitwärtsgehen viele Fotos so überlagern, dass der **Boden** deckungsgleich
  liegt → der Zaun (in jedem Foto woanders) mittelt sich weg, man „sieht durchs Laub". Bei uns ~**9 Frames
  über ~1 s**.
- **Warum bewegte Tiere verschmieren (DAS Signal):** Beim Mitteln über ~1 s bleibt ein **stehendes** Tier
  scharf (immer am selben Ort), ein **laufendes** wird zu einem **Schmierstreifen in Laufrichtung** — wie die
  Lichtspur eines fahrenden Autos bei Langzeitbelichtung. *Mini-Beispiel:* wir haben in einem Track gemessen,
  dass ein Tier über ~8 Schritte ~205 px wandert (~26 px/Schritt) → ein deutlicher Streifen, dessen **Richtung
  = die Bewegungsrichtung**.
**Genauer (für Fragen):** „Auf die Bodenebene ausrichten" heißt: Weil man ungefähr weiß, wie die Drohne
zwischen den Aufnahmen geflogen ist, kann man jedes Foto so verschieben, dass ein gedachter flacher Boden in
allen Fotos deckungsgleich liegt (das nennt man **synthetische Apertur** — man simuliert eine riesige
Kameralinse aus vielen kleinen Blickwinkeln). Was auf Bodenhöhe ist (Tier), bleibt dann scharf; was darüber
ist (Blätter), liegt in jedem Foto woanders und mittelt sich weg. *Mini-Beispiel:* 9 Halbtransparent-Folien
desselben Waldstücks exakt aufeinanderlegen → das Laub wird milchig-blass, der warme Tierfleck bleibt kräftig.
**Warum so / Alternative:** Einzel-Frames statt Integralbilder wären viel verrauschter und verdeckter
(Literatur der Betreuer: ~42 % vs. ~97 % Treffer) — das Integralbild ist die ganze Stärke von AOS.
**Zahlen (zum Zitieren):** 2048×2048; **12.514 Bilder mit Tieren / 46.046 Kästen**; **223 Flüge**; 3 Klassen
nur als **IDs 0/1/2** (Artnamen Rot-/Reh-/Schwarzwild **unbestätigt**, im Datensatz stehen nur Nummern);
67 % Bilder mehrtierig; Tier-Median 65 px; kein GPS/keine Drohnen-Position.

## A3. Woher die Kästen kommen — YOLO-Format
**Was:** Zu jedem Bild ist schon eingezeichnet, **wo** Tiere sind — als **Bounding Box** (Rechteck) im
**YOLO-Format**.
**Wie es funktioniert:** Pro Tier eine Textzeile `Klasse x_mittelpunkt y_mittelpunkt breite höhe`, alles auf
0–1 normiert. *Mini-Beispiel:* `2 0.45 0.30 0.05 0.06` → Klasse 2, Mitte bei 45 %/30 % des Bildes, Box 5 %×6 %
groß; mal 2048 ergibt das die Pixel-Koordinaten. YOLO selbst ist ein **Objektdetektor** (legt ein Gitter übers
Bild und sagt pro Zelle voraus, ob/wo ein Objekt ist); das **Format** ist nur die Speicherart.
**Genauer (wie ein Detektor arbeitet, falls gefragt):** YOLO („You Only Look Once") schaut das Bild **einmal**
an, legt gedanklich ein Gitter darüber und sagt pro Zelle: „ist hier ein Objekt, wo genau, welche Klasse?".
Trainiert wird es mit vielen Bildern, bei denen die Kästen bekannt sind. Auf **Thermalbildern** ist Detektion
zusätzlich schwer (winzige, kontrastarme Flecken) — aber das mussten wir nicht lösen, weil die Kästen schon
da waren.
**Warum so / Alternative:** Das **Finden** der Tiere war **nicht unsere Frage** (die Richtung war's), und die
Kästen lagen fertig vor → **wir haben kein YOLO/keinen Detektor trainiert**. Ein eigenes Training hätte eigene
Labels + eigene Auswertung gebraucht und vom Thema abgelenkt. Aus dem Dateinamen (`flug_frame_...`) lesen wir
Flug- und Frame-Nummer — die Brücke zum Tracking.

## A4. Vorverarbeitung — Kästen aufs Tier ziehen (klassische Segmentierung)
**Was:** Die manuellen Kästen sind oft zu groß; wir ziehen sie auf den warmen Fleck zusammen — **klassisch,
ohne neuronales Netz**, und messen nebenbei Größe/Kontrast pro Klasse (EDA = explorative Datenanalyse).
**Wie es funktioniert, Schritt für Schritt:**
1. **Otsu-Schwellwert:** wählt **automatisch** die Helligkeitsgrenze, die helle und dunkle Pixel **am besten
   trennt** (zwei Gruppen mit möglichst wenig Überlappung) → Schwarz-Weiß-Bild, Tier weiß.
2. **Morphologie:** *Öffnen* (erst schrumpfen, dann wachsen) entfernt einzelne Störpixel; *Schließen*
   (umgekehrt) füllt kleine Löcher im Tier.
3. **Connected Components:** alle **berührenden** weißen Pixel bekommen dieselbe „Klecks-Nummer"; wir nehmen
   den größten Klecks, dessen Schwerpunkt in der **mittleren 60 %** liegt (helle Ränder ignorieren).
4. **Bildmomente:** man behandelt die weißen Pixel wie eine Massenverteilung und berechnet Schwerpunkt + die
   **Richtung der größten Ausdehnung** = lange Achse, plus „wie länglich" (= eine eingepasste Ellipse).
**Genauer (mit Mini-Beispiel):**
- *Otsu:* Man zählt, wie viele Pixel jede Helligkeit haben (ein **Histogramm**). Bei einem Thermalcrop gibt es
  meist **zwei Hügel** — ein großer dunkler (Boden) und ein kleiner heller (Tier). Otsu legt die Grenze genau
  ins **Tal** dazwischen, automatisch, ohne dass wir eine Zahl festlegen.
- *Öffnen/Schließen:* „Schrumpfen dann Wachsen" frisst zuerst einzelne weiße Störpunkte weg und stellt das
  Tier danach wieder auf Originalgröße; „Wachsen dann Schrumpfen" stopft umgekehrt kleine schwarze Löcher im
  Tier.
- *Connected Components:* Stell dir aus, Wasser in einen weißen Klecks zu gießen — alles, was zusammenhängt,
  ist ein Tier-Kandidat. Wir nehmen den größten, dessen Mitte nicht am Rand klebt.
- *Momente → Ellipse:* aus den Pixel-Positionen folgt automatisch eine Ellipse; ihre lange Achse ist die
  Körperachse, das Verhältnis lang/kurz sagt „Strich" (länglich) vs. „Kreis" (rund).
**Warum so / Alternative:** Kein U-Net/SAM (große Segmentierungs-Netze), weil wir **keine Pixel-Masken** zum
Trainieren haben, solche Netze auf Thermalbildern eine **Domain-Lücke** haben und klassisch interpretierbar ist
— für winzige warme Flecken reicht das.
**Ergebnis:** verfeinerte/Box-Fläche **0,81 / 1,15 / 0,37**; Intensität **102 / 116 / 79**. Der 0,37-Wert bei
Klasse 2 ist **keine** lose Box, sondern **Untersegmentierung**: Schwarzwild ist am kältesten/kontrastärmsten,
der Schwellwert erwischt nur den warmen Kern (~7 % des Crops). → eine eurer ehrlichen Erkenntnisse.

## A5. Das Kernproblem & unsere Idee (der Pivot)
**Problem:** Für die **Richtung** gibt es keine prüfbare menschliche Wahrheit — Kopf unsichtbar, 67 %
mehrtierig, kein Gegencheck möglich. Hand-Labeln wäre nur „zweites Raten".
**Idee (Pivot):** Richtung aus **Tracking** statt aus Labels — reine Geometrie über die Zeit. Menschliche
Labels nur noch für eine **kleine Validierung**: stimmt die Tracking-Richtung mit dem überein, was ein Mensch
auf **klaren** Einzeltier-Fällen sieht, plus das bewegt/steht-Urteil und die **Kopf-Erkennbarkeitsrate** (wie
oft ist der Kopf überhaupt sichtbar — selbst ein Ergebnis).
**Genauer (was die menschliche Validierung leistet):** Andreas labelt nur **klar sichtbare Einzeltiere** —
dort, wo ein Mensch die Richtung wirklich sehen kann. Damit prüfen wir drei Dinge: (1) stimmt die
Tracking-Richtung mit der menschlichen überein? (2) liegt er bei bewegt/steht richtig? (3) **wie oft ist der
Kopf überhaupt erkennbar** (die „Erkennbarkeitsrate") — das ist selbst ein ehrliches Ergebnis und begründet,
warum wir nicht einfach alles von Hand labeln. So bleibt der Mensch eine **unabhängige Kontrolle**, nicht die
Quelle der Wahrheit.
**„Ist das nicht Zirkelschluss?" (Antwort):** Nein — das Tracking nutzt **zeitliche** Information (Verschiebung
über mehrere Frames), die in einem **Einzelbild gar nicht steckt**; es ist Geometrie, keine Modell-Meinung; und
der Mensch ist die dritte, unabhängige Stimme. Das ist die zentrale, eigenständige Designidee.

## A6. Ego-Motion entfernen — Bildregistrierung (Kernstück 1)
**Problem:** Die Drohne bewegt sich → das **ganze Bild** verschiebt sich. *Mini-Beispiel:* über einen Track
summiert sich die Drohnenbewegung auf **~800 px**, die echte Tierbewegung nur **~205 px** — ohne Rausrechnen
ertrinkt das Tier im Drohnenwackeln.
**Was Registrierung tut:** zwei aufeinanderfolgende Frames so „übereinanderlegen", dass der **unbewegte
Hintergrund deckungsgleich** ist; die nötige Schiebe-Dreh-Anweisung ist die **Drohnenbewegung**.
**Wie es funktioniert, Schritt für Schritt:**
- **Verkleinern (halbe Größe) + CLAHE:** schneller, und CLAHE hebt **lokal den Kontrast** an, weil
  Thermalbilder flau sind — sonst findet man zu wenige Merkmale. (Die im Kleinbild gemessene Verschiebung
  wird am Ende wieder verdoppelt, damit sie zur vollen Größe passt.)
- **ORB-Keypoints:** sucht bis zu ~2000 **markante Punkte** (Ecken, Astgabeln — wie Landmarken auf einer
  Karte) und gibt jedem einen „Steckbrief" zum Wiederfinden.
- **Matching + Lowe-Ratio-Test:** dieselben Punkte im nächsten Frame finden; ein Paar wird nur behalten, wenn
  der beste Treffer **deutlich** besser ist als der zweitbeste (unsichere Paare raus).
- **RANSAC (Kernidee):** nimm ein paar **zufällige** Punkt-Paare, berechne die Verschiebung, die sie
  implizieren, und **zähle, wie viele andere Paare dazu passen** (Inlier); wiederhole das oft und behalte die
  Verschiebung mit den **meisten** Inliern. So gewinnen die 90 % Hintergrund-Punkte, und Ausreißer (z. B.
  Punkte auf dem **bewegten Tier**) werden ignoriert.
- Geschätzt wird eine **partial-affine** Transformation (Drehung + Skalierung + Verschiebung, 4 Stellschrauben)
  — bewusst weniger als eine volle perspektivische (8 Stellschrauben), das ist über kurze Zeit **stabiler**.
- Der **Inlier-Anteil** (z. B. 0,86 = 86 % der Paare passen) ist unser **Qualitätsmaß** für die Registrierung.
**Genauer (wie RANSAC und der Match konkret laufen):**
- *Matching:* Jeder ORB-Punkt hat einen kurzen Binär-„Steckbrief"; man sucht im zweiten Frame den Punkt mit
  dem ähnlichsten Steckbrief. Der Lowe-Test akzeptiert nur, wenn der **beste** Steckbrief klar besser passt
  als der **zweitbeste** — sonst ist es wahrscheinlich eine Verwechslung.
- *RANSAC, ein Durchlauf:* nimm **zufällig** ein paar Punkt-Paare, berechne daraus die Schiebe-Dreh-Anweisung,
  und schau, **wie viele** der übrigen Paare sich damit (fast) decken (= Inlier). Das wiederholt man hunderte
  Male und behält die Anweisung mit den **meisten** Inliern. *Mini-Beispiel:* von 500 Paaren sagen 430 „alle
  ~50 px nach links + 2° gedreht", 70 sagen Unsinn (Punkte auf dem Tier/Fehlmatches) → RANSAC nimmt die 430,
  ignoriert die 70; Inlier-Anteil = 430/500 = 0,86.
- *Die „Anweisung":* eine kleine Zahlentabelle, mit der man jeden Bildpunkt umrechnen kann; wendet man sie auf
  die alte Tierposition an, weiß man, **wo das Tier wäre, wenn es sich nicht bewegt hätte** — die Differenz
  zur echten neuen Position ist die Tierbewegung.
**Warum Frame-zu-Frame (Alternative):** Registrierung direkt auf einen **weit entfernten** Frame scheitert,
weil die Szene zu verschieden ist und die Punkte nicht zusammenfinden — das haben wir gemessen.
**Failure→Fix (Zwischenergebnis):** anfangs zu schwach (zu wenige Punkte) → mit CLAHE + Lowe-Ratio stieg die
Qualität **0,72 → 0,86** und die verlässlichen Tracks **164 → 190**, ohne das Gate aufzuweichen.

## A7. Vom Track zur Richtung — Verfolgung & Konfidenz (Kernstück 2)
**Tracklet bilden:** das Tier über Frames verketten per **Greedy-Nearest-Centroid** — im nächsten Frame das
**nächstgelegene Tier gleicher Klasse** als „dasselbe" nehmen, Distanz-Grenze wächst mit der Frame-Lücke.
Bewusst **kein Kalman/SORT** (aufwändigere Tracker), weil pro Schritt nur wenige Tiere im Bild sind.
**Richtung pro Schritt:** vorige Tierposition mit der Drohnenbewegung korrigieren, dann „neue Position minus
korrigierte alte" = echte Tier-Verschiebung; ihr Winkel ist die Schritt-Richtung.
**Warum Kreisstatistik (wie es funktioniert):** Winkel kann man nicht normal mitteln — der Mittelwert von 359°
und 1° wäre 180° (falsch), obwohl beide fast gleich sind. Trick: jeden Winkel als kleinen **Pfeil**
(Einheitsvektor) darstellen, die Pfeile **addieren** — die Richtung der Summe ist der echte Mittelwert, und
die **Länge der Summe (R, 0…1)** sagt, wie **einig** die Schritte waren (1 = alle gleich, 0 = wild durcheinander).
Ein zweiter Wert (Rayleigh) sagt, ob diese Einigkeit **mehr als Zufall** ist.
**Vertrauens-Gate:** „verlässlich" nur bei genug Schritten (≥5), guter Registrierung, hoher Einigkeit (R≥0,5),
Signifikanz (nicht Zufall) und genug Bewegung. Stehende Tiere → keine Richtung (korrekt verworfen).
**Genauer (was R und das Gate konkret heißen):**
- *R (Einigkeit), Mini-Beispiel:* Zeigen 10 Schritt-Pfeile fast alle nach Nordost, ist R nahe 1 → klare
  Richtung. Zeigen sie kreuz und quer, mitteln sie sich fast weg, R nahe 0 → keine echte Richtung. Wir
  verlangen R ≥ 0,5.
- *Warum jede Gate-Bedingung:* **≥5 Schritte**, damit ein Zufall nicht „klare Richtung" vortäuscht;
  **gute Registrierung**, sonst ist die Drohnenkorrektur unzuverlässig; **R ≥ 0,5 + Signifikanz**, damit die
  Einigkeit nicht Zufall ist; **genug Bewegung**, damit ein stehendes Tier mit Mess-Zittern nicht als „Läufer"
  zählt.
- *Streng vs. großzügig:* Mit dem Gate sind 190 Tracks „verlässlich"; wenn wir strenger werten (≥8 Schritte
  **und** spürbare Strecke), bleibt ein **harter Kern von 138** — den nennen wir die belastbare Zahl.
**Beweis, dass es das Tier ist (nicht die Drohne):** ~800 px Drohne raus → **~205 px kohärente** Tierbewegung
übrig; im **Differenzbild** (zweites Bild minus registriertes erstes) verschwindet der Hintergrund (wird
schwarz), nur das bewegte Tier bleibt sichtbar; und **stehende** Tiere im selben Flug teilen **keine**
gemeinsame Driftrichtung (sonst wäre es Drohnenbewegung).
**Ergebnis:** 2.697 Tracklets → **190** verlässlich → **138** hochkonfidenter Kern (58/49/31 pro Klasse).

## A8. Richtung aus EINEM Bild? — GST & klassische Schätzer
**Hypothese:** vielleicht reicht die **Bewegungsunschärfe** in einem Crop, um die Achse zu lesen (wäre billiger
als Tracking).
**Wie GST funktioniert (Gradient Structure Tensor):** Ein **Gradient** ist die Richtung, in die die Helligkeit
am schnellsten zunimmt (= zeigt quer über eine Kante). Bei einem länglichen Fleck zeigen alle Kanten-Gradienten
**quer** zur langen Achse. Der GST **sammelt die Gradientrichtungen** über den ganzen Crop (so, dass sich
entgegengesetzte Richtungen nicht aufheben) und nimmt die **dominante** davon — die lange Achse steht
senkrecht darauf. Ein Zusatzwert (Coherence, 0…1) sagt, wie eindeutig die Achse ist (0 = runder Klecks, 1 =
klare Linie). *Analogie:* aus den Querfasern eines Holzbretts die Längsrichtung ablesen. (Weitere getestete
Schätzer werten das Frequenz-/Unschärfe-Muster aus — auf kleinen Crops aber schwächer.)
**Genauer (warum das die Achse trifft):** An einer Kante zeigt der Gradient immer **senkrecht** zur Kante
(vom Dunklen ins Helle). Die Ränder eines länglichen Flecks verlaufen entlang der Körperachse → ihre
Gradienten stehen quer dazu. Mittelt man alle Gradientrichtungen geschickt (so dass „oben" und „unten" sich
**nicht** gegenseitig auslöschen), bleibt eine dominante Querrichtung übrig — und die Körperachse ist die
Senkrechte darauf. Die **Coherence** sagt, wie stark diese eine Richtung dominiert (klare Linie → nahe 1,
runder Klecks → nahe 0). *Mini-Beispiel:* bei einem klaren diagonalen Schmierstreifen kommt GST verlässlich
auf „diagonal"; bei einem fast runden Fleck ist die Coherence niedrig und die Achse kaum aussagekräftig.
**180°-Mehrdeutigkeit:** ein Bild gibt nur die **Achse** (z. B. „diagonal"), nicht Kopf vs. Schwanz — das
Vorzeichen liefert nur das Tracking.
**Ergebnis (ehrlich):** GST ~**29°** mittlerer Fehler, Treffer innerhalb 45°: **68 %**; schlägt eine
„immer-die-mittlere-Richtung"-Baseline nur knapp und bei der **häufigsten** Klasse sogar **schlechter** (39°
vs. 16°). → **Einzelbild-Richtung ist schwach**, Tracking bleibt die bessere Wahrheit.

## A9. Methoden vergleichen — bewegt vs. steht aus einem Crop
**Frage/Zweck:** erkennt ein Modell aus **einem** Crop bewegt (verschmiert) vs. steht (scharf)? Das ist der
„CV-Methoden anwenden und **vergleichen**"-Teil der Kursaufgabe.
**Labels:** **Proxy-Labels** aus dem Tracking (bewegt = verlässlicher Track, steht = kaum verschoben). „Proxy"
= Ersatz, teils zirkulär mit dem Tracker → später mit menschlichen Labels gegenprüfen.
**Vier Familien (und wie sie funktionieren):**
1. **Klassische Hand-Features + ML.** „Hand-Features" = 6 selbst berechnete Zahlen pro Crop (GST-Coherence,
   Länglichkeit, Schärfe innen vs. außen, Unschärfe-Länge, Kontrast zum Umfeld, Größe). Darauf:
   - *Logistische Regression:* gewichtete Summe der Zahlen → eine **Trenngrenze** zwischen „bewegt"/„steht".
   - *Random Forest:* viele kleine **Entscheidungsbäume** (jeder stellt Ja/Nein-Fragen an die Zahlen), die
     **abstimmen**.
2. **CNN „from scratch":** ein **neuronales Netz**, das **eigene Bildfilter** aus den Beispielen lernt — kleine
   Filter gleiten übers Bild und erkennen Muster, von einfachen Kanten bis zu Formen; trainiert wird, indem die
   Filter Schritt für Schritt angepasst werden, bis weniger Fehler passieren.
3. **Foundation-Models (DINOv2, CLIP, BioCLIP), eingefroren:** sehr große, auf **Millionen Fotos** vortrainierte
   Netze. „Eingefroren" = wir trainieren sie nicht, sondern schicken jeden Crop durch und nehmen nur ihren
   **Ausgabe-Vektor** (das **Embedding**, ein paar hundert Zahlen, die das Bild beschreiben) als Eingabe für
   einen kleinen Klassifizierer.
**Genauer (wie die Modelle entscheiden):**
- *Entscheidungsbaum (im Random Forest):* eine Kette von Ja/Nein-Fragen an die Zahlen, z. B. „Coherence > 0,4?
  → ja → Unschärfe-Länge > 5? → ja → **bewegt**". Ein Baum allein ist wackelig; **viele** Bäume mit leicht
  anderen Daten stimmen ab → stabiler (das ist der „Forest").
- *CNN-Filter:* ein kleiner 3×3-Stempel gleitet übers Bild und reagiert auf ein Muster (z. B. eine
  Schräg-Kante); mehrere Schichten bauen von Kanten zu Formen auf. Beim Training werden die Stempel so lange
  nachjustiert, bis die Vorhersage öfter stimmt. Bei uns zu wenige Beispiele → es lernt nichts Stabiles.
- *Embedding (Foundation-Model):* wir schicken den Crop einmal durch das große, **unveränderte** Netz und
  greifen seinen Ausgabe-Vektor ab (ein paar hundert Zahlen, die das Bild „zusammenfassen") — diese Zahlen
  sind dann die Eingabe für eine simple logistische Regression. Wir trainieren also nur den kleinen Aufsatz,
  nicht das Riesennetz.
**Warum Foundation-Models testen (statt annehmen):** sie kennen nur normale Farbfotos, **nicht** Thermal →
**Domain-Lücke**; **BioCLIP** (auf Tierfotos trainiert) als **Negativ-Kontrolle**.
**Ergebnis (balanced accuracy):** BioCLIP 0,64 · DINOv2 0,63 · CLIP 0,62 · Random Forest 0,58 · LogReg 0,56 ·
CNN 0,50 · Mehrheit 0,50. → Foundation leicht vorne, aber **untereinander praktisch gleich** und **nicht klar
besser** als klassisch; CNN (datenarm) kollabiert.

## A10. Faire Auswertung — warum die Zahlen ehrlich sind
**train/test:** Modell auf einem Teil lernen, auf einem **anderen** testen.
**Leakage (Datenleck):** *Mini-Beispiel:* wären Bilder **desselben Flugs** in Training **und** Test, könnte das
Modell die **Szene** wiedererkennen und richtig raten, ohne das Tier zu verstehen → Scores zu gut. Lösung:
**flight-disjoint** (GroupKFold) — ganze Flüge bleiben zusammen, entweder Training **oder** Test.
**Baselines (Hürden, die man schlagen muss):** Zufall, „häufigste Klasse", „mittlere Richtung". Die mittlere
Richtung ist **stark** (Richtungen sind nicht gleichverteilt) — die muss man schlagen, nicht nur den Zufall.
**Balanced Accuracy:** Mittel aus „wie gut erkenne ich bewegt" und „wie gut erkenne ich steht" **getrennt** —
lässt sich nicht von ungleichen Klassengrößen täuschen.
**Cross-Validation (wie genau):** Wir teilen die Flüge in 5 Gruppen; trainieren auf 4, testen auf der 5.,
und rotieren durch, bis jede Gruppe einmal Test war — und das mit mehreren Zufalls-Seeds. So hängt die Zahl
nicht an einem glücklichen Split. Berichtet wird der Mittelwert ± Streuung.
**ROC-AUC (kurz):** ein zweites Gütemaß zwischen 0,5 (Zufall) und 1,0 (perfekt), das misst, wie gut das
Modell „bewegt" generell höher einstuft als „steht" — unabhängig von einer festen Entscheidungsschwelle.
**Pseudo-Replikation (Caveat):** 945 Crops stammen aus nur **189 unabhängigen Tracks** (5 Crops teilen sich
eine Track-Wahrheit) → wir rechnen Konfidenz **pro Track**, nicht pro Crop, sonst wirken die Zahlen sicherer
als sie sind.
**Wichtige Ehrlichkeit:** Die **Szenen-Decke** (nur über die Flug-Zugehörigkeit erreichbar) liegt bei **~0,84**
— unsere Modelle liegen **darunter**, also ist das Ergebnis teils Szenen-Korrelation. Sagen wir offen.

---

# TEIL B — Ergebnisse (ehrlich)
- **Tracking funktioniert:** 138 verteidigbare Richtungen, mehrfach geprüft (Hintergrund cancelt; stehende
  Tiere ohne Scheinrichtung).
- **Einzelbild-Richtung schwach** (GST ~29°), hilft nur wo Richtungen streuen.
- **bewegt/steht:** Foundation leicht vorne (~0,63), aber nicht klar besser als klassisch, teils Szenen-Effekt.
- **Kernbeitrag:** neue Ground-Truth-Idee (Tracking) + fairer Methodenvergleich + ehrliche Auswertung.

# TEIL C — Was schiefging und wie wir es lösten (Failures → Fixes)
*Erzähl 2–3 davon aktiv — das klingt nach echter eigener Arbeit.*
1. **Hand-Labeln der Richtung unzuverlässig** → **Pivot zu Tracking**.
2. **Direktregistrierung auf ferne Frames scheitert** → **Frame-zu-Frame**.
3. **Registrierung zu schwach** → **CLAHE + Lowe-Ratio** → Tracks 164→190, Qualität 0,72→0,86.
4. **CNN from scratch kollabiert** (zu wenig Daten) → ehrlich so berichtet.
5. **Schwarzwild-Box schien „zu groß"** → eigentlich **Untersegmentierung** (Kontrast) → korrigiert.
6. **Ergebnis überschätzt** („bewegte stimmen besser überein") → als **Confound** erkannt → verworfen.

# TEIL D — Designentscheidungen (für „warum so?"-Fragen)
| Entscheidung | Warum so | Verworfene Alternative |
|---|---|---|
| Richtung aus Tracking | objektiv, ohne unsichere Labels | Hand-Labeln: nicht verifizierbar |
| Kein Detektor trainiert | nicht unser Thema, Boxen lagen vor | eigenes YOLO: eigene Labels nötig |
| Klassische Segmentierung | label-frei, reicht für kleine Flecken | U-Net/SAM: keine Masken, Domain-Lücke |
| partial-affine Registrierung | robust, wenige Parameter; kein GPS | volle perspektivische Transformation |
| Frame-zu-Frame | kleiner Versatz = stabil | ferner Frame: scheitert (gemessen) |
| Greedy-Tracker (kein Kalman) | wenige Tiere/Schritt → reicht | SORT/Kalman: Overkill, nächster Schritt |
| Konfidenz-Gate | filtert Rausch-Richtungen | alles behalten: stehende Tiere = Lärm |
| Foundation-Models testen | Domain-Lücke messen | blind annehmen, dass sie besser sind |
| flight-disjoint Split | verhindert Szenen-Leckage | Zufalls-Split: Scores zu hoch |

# TEIL E — Folien-Fahrplan (~20 min, ~13 Folien)
| # | ~min | Folie | Kernsatz |
|---|---|---|---|
| 1 | 0:30 | Titel + Frage | „Bewegungsrichtung von Wild aus Thermal-Lichtfeldern?" |
| 2 | 1:30 | Aufgabe & Motivation | Was raus soll, wozu, warum schwer. |
| 3 | 2:00 | Daten / AOS | Lichtfeld (Gartenzaun), Integralbild, warum bewegte Tiere verschmieren. |
| 4 | 1:00 | Kernproblem | Keine Ground Truth für Richtung. |
| 5 | 2:00 | Idee (Pivot) | Richtung aus Tracking. |
| 6 | 1:00 | Pipeline | Überblicksdiagramm. |
| 7 | 1:30 | Vorverarbeitung | klassische Segmentierung + EDA (Boar-Untersegmentierung). |
| 8 | 2:30 | Tracking → Richtung | **Kern**: Registrierung + Verfolgung + Beweis-Bild (800→205 px). |
| 9 | 2:00 | Einzelbild-Richtung | GST → ~29°, schwach. |
| 10 | 2:00 | Methodenvergleich | klassisch vs CNN vs Foundation → kein klarer Sieger. |
| 11 | 1:00 | Evaluation | flight-disjoint + Baselines + Szenen-Decke 0,84. |
| 12 | 1:30 | Ergebnisse + Grenzen | 138 Richtungen; ID-Switch als Hauptcaveat. |
| 13 | 1:00 | Fazit + Ausblick | Beitrag + nächster Schritt (menschliche Labels). |

# TEIL F — Fragen & Antworten (Ich-Form, ohne Formeln)
- **Welche Tools?** Python, OpenCV (klassische CV), scikit-learn (ML + Splits), PyTorch/timm/open_clip (DL).
- **YOLO trainiert?** Nein — Boxen kamen fertig im YOLO-Format; Detektion war nicht unser Thema.
- **Welche Segmentierung?** Klassisch: Otsu + Morphologie + größte zentrale Region + Momente. Kein Netz.
- **Wie ohne Ground Truth an die Richtung?** Aus dem Tracking: Drohnenbewegung rausrechnen, Rest = Richtung.
- **Was macht der GST?** Findet die dominante Kanten-/Achsenrichtung im Crop = lange Achse des Tiers.
- **Wie funktioniert die Registrierung?** ORB-Landmarken finden, matchen, mit RANSAC Ausreißer raus → das ist
  die Drohnenbewegung, die ich abziehe.
- **Was ist RANSAC?** Zufällige Punkt-Paare ausprobieren, das Modell mit den meisten passenden Punkten gewinnt
  → Ausreißer (Punkte aufs Tier) stören nicht.
- **Wann ist eine Richtung „verlässlich"?** Genug Schritte, gute Registrierung, hohe Einigkeit (R), Signifikanz,
  genug Bewegung — sonst gilt das Tier als stehend.
- **Wie wisst ihr, dass es nicht die Drohne ist?** Restbewegung kohärent; Differenzbild cancelt Hintergrund;
  stehende Tiere ohne gemeinsame Drift; 800→205 px.
- **180°-Mehrdeutigkeit?** Einzelbild gibt nur die Achse, nicht Kopf/Schwanz; Vorzeichen aus Tracking.
- **Warum klassisch statt Deep Learning?** Wenig Labels + Fokus „vergleichen"; klassisch interpretierbar/label-frei.
- **Warum Foundation-Models nicht klar besser?** Domain-Lücke + schwache Proxy-Labels + Szenen-Verzerrung.
- **Warum CNN kollabiert?** Zu wenig Trainingsdaten für ein von Null trainiertes Netz.
- **train/test-Split?** Flight-disjoint — sonst Szenen-Leckage.
- **Stärkste Baseline?** „Immer die mittlere Richtung" — stark, weil Richtungen nicht gleichverteilt sind.
- **Was ist die Szenen-Decke?** ~0,84 — so gut wäre man allein über die Flug-Zugehörigkeit; die echte Hürde.
- **Größte Schwäche?** ID-Switch in dichten Szenen (Tracker ohne Aussehensmodell) — offen benannt.
- **Was verbessern?** Tracker mit Aussehensmodell + Validierung gegen menschliche Labels.
- **Wie Selbsttäuschung vermieden?** Leckage-freier Split, faire Baselines, kritischer Self-Audit (eigene Zahl korrigiert).
- **Euer Beitrag?** Neue GT-Strategie (Tracking), fairer Methodenvergleich, ehrliche Auswertung.
- **Warum AOS/Lichtfeld statt normaler Fotos?** Es sieht durch Laub (Verdeckung weg) UND lässt bewegte Tiere
  verschmieren — beides brauchen wir.
- **Was ist ein Integralbild?** Das gemittelte Bild aus den ~9 ausgerichteten Frames.
- **Wie viel habt ihr gelabelt vs. getrackt?** Die GT kommt aus dem Tracking über alle Flüge (138 Kern);
  Menschen labeln nur eine **kleine Validierungsmenge**.
- **Ist Tracking-GT + Klassifikation nicht zirkulär?** Die Tracking-GT ist Geometrie; die Klassifikation nutzt
  nur Bild-Features; und am Ende validiert der Mensch unabhängig. Die Proxy-Label-Zirkularität benennen wir offen.
- **Was, wenn die menschlichen Labels dem Tracking widersprechen?** Das ist ein **Befund** (ID-Switch /
  Erkennbarkeit), kein Neustart — wir prüfen zuerst die sauberen Einzeltier-Fälle; stimmen die, ist die GT solide.

# TEIL G — Mini-Glossar (Spickzettel)
- **AOS / Lichtfeld / Integralbild:** viele ausgerichtete Drohnen-Frames überlagert; „sieht durch Laub".
- **Thermal:** Wärmebild; Tier = heller Fleck.
- **Bounding Box / YOLO-Format:** Rechteck ums Tier; YOLO = Detektor, Format = Speicherart.
- **Otsu:** automatische Hell/Dunkel-Grenze. **Morphologie:** Maske aufräumen. **Connected Components:**
  berührende Pixel zu Klecksen. **Bildmomente:** Achse + Länglichkeit (Ellipse).
- **Gradient:** Richtung schnellster Helligkeitsänderung (quer zur Kante). **GST:** mittelt Gradienten →
  dominante Achse + Eindeutigkeit (Coherence).
- **ORB / Matching / Lowe-Ratio:** Landmarken finden / wiederfinden / unsichere Paare verwerfen.
- **RANSAC:** robustes Schätzen, meiste passende Punkte gewinnen. **CLAHE:** lokale Kontrastverstärkung.
- **Registrierung / Ego-Motion:** Frames am Hintergrund ausrichten → Drohnenbewegung.
- **partial-affine:** Drehung+Skalierung+Verschiebung. **Inlier-Anteil:** Qualitätsmaß der Registrierung.
- **Tracklet / Greedy-Assoziation:** Tier über Frames verketten (nächstes gleichartiges).
- **Kreisstatistik / R:** richtig mit Winkeln rechnen; R = wie einig die Schritte sind (0…1).
- **180°-Mehrdeutigkeit:** Einzelbild = nur Achse, nicht Kopf/Schwanz.
- **Logistische Regression / Random Forest:** einfache ML-Modelle (Trenngrenze / abstimmende Bäume).
- **CNN:** lernt eigene Bildfilter aus Beispielen.
- **Foundation-Model / frozen / Embedding:** großes vortrainiertes Netz, eingefroren; Embedding = seine Zahlen
  pro Bild. **Domain-Lücke:** neue Daten sehen anders aus (Thermal vs. Foto).
- **Proxy-Label:** Ersatz-Label (aus Tracking). **Leakage / flight-disjoint:** Szenen-Schummeln vermeiden.
- **Baseline / Balanced Accuracy / Szenen-Decke (~0,84):** Hürde / faire Genauigkeit / was allein die Szene bringt.
- **ID-Switch:** Tracker verwechselt zwei Tiere. **Confound:** versteckte Störgröße, die ein Ergebnis vortäuscht.

# TEIL H — Die 5 Sätze, die nie falsch sein dürfen + Tools
1. **Kein YOLO/Detektor trainiert** — Boxen kamen fertig.
2. **Segmentierung klassisch** (Otsu + Morphologie), kein Netz.
3. **Tracking = einfacher Greedy-Tracker** (kein Kalman/SORT), bewusst so.
4. **Spezies-Zuordnung unbestätigt** — nur IDs 0/1/2.
5. **Richtungs-Wahrheit aus Tracking**, menschliche Labels nur zur Validierung.

**Tech-Stack:** Python + OpenCV (klassische CV), scikit-learn (ML + Evaluation), PyTorch/timm/open_clip
(CNN + Foundation-Models); Daten aus Roboflow; eigenes kleines Annotations-Tool für die Validierung.
