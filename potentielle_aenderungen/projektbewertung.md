# Projektbewertung und potentielle Aenderungen

Stand: 2026-06-21

Diese Datei ist absichtlich nur eine externe Bewertung. Ich habe die hier beschriebenen
Aenderungen nicht im bestehenden Projekt umgesetzt. Eine andere KI oder Person kann damit
pruefen, ob die Vorschlaege sinnvoll sind.

## Kurzfazit

Das Projekt ist insgesamt deutlich besser als ein typisches schnell zusammengebautes
Course-Project: Die zentrale Methodik ist nachvollziehbar, die Ergebnisse sind aus CSVs
reproduzierbar, die wichtigsten Caveats sind in `docs/audit.md` bereits ehrlich benannt,
und die Tests laufen. Besonders stark ist, dass die Richtung nicht aus unsicheren
Human-Labels erzwungen wird, sondern ueber registriertes Tracking hergeleitet wird.

Der groesste verbleibende Schwachpunkt ist nicht ein kaputter Algorithmus, sondern die
Konsistenz der Darstellung: Das Projekt weiss in `docs/PROJECT.md`, dass die
ID-zu-Spezies-Zuordnung unbestaetigt ist, verwendet in Code, EDA-Dateien und Plots aber
teilweise trotzdem feste Namen wie Rotwild/Rehwild/Schwarzwild. Das kann in einer
Abgabe wie eine unbelegte biologische Behauptung wirken.

## Was meiner Pruefung nach passt

- Die Arbeitskopie war vor der Pruefung sauber.
- Tests laufen: `128 passed`.
- Dataset-Zahlen sind reproduzierbar:
  - 12,655 Bilddateien auf Platte.
  - 12,514 Bilder mit mindestens einer Box.
  - 46,046 Boxen.
  - Klassen 0/1/2 mit 21,787 / 17,403 / 6,856 Boxen.
  - 223 Flight-IDs in den Bilddateien, 221 Flight-IDs mit Boxen.
- Tracking-Zahlen sind reproduzierbar:
  - 2,697 Tracklets.
  - 190 Tracklets passieren den aktuellen trusted gate.
  - 138 Tracklets bilden den defensiblen High-Confidence-Core
    (`n_steps >= 8` und `disp_px >= 50`).
  - Trusted by class: 83 / 68 / 39.
  - Core by class: 58 / 49 / 31.
  - Median registration inlier ratio: ca. 0.8646.
- Blur-Evaluation ist reproduzierbar:
  - 945 Mover-Crops aus 189 Tracklets.
  - GST: Median axial error 29.1 deg, Acc@45 0.68.
  - Spectrum: 32.8 deg, Acc@45 0.61.
  - Cepstrum: 33.1 deg, Acc@45 0.60.
  - Gradient: 33.5 deg, Acc@45 0.59.
  - Moments: 35.2 deg, Acc@45 0.59.
  - Random baseline mit Seed 42: 44.2 deg, Acc@45 0.50.
- Movement-Experiment ist inhaltlich vorsichtig genug beschrieben:
  - BioCLIP/DINOv2/CLIP liegen numerisch vorne, aber der Text behauptet keinen klaren
    Sieger.
  - Proxy-Label- und Scene-Leakage-Caveats sind benannt.
- Ghosting-Beispielbild wirkt plausibel: Die Montage zeigt tatsaechliche Rand- und
  Scherartefakte.
- Annotation-Crops wirken echt und menschlich beurteilbar: Thermal-Crops, gruener
  Zielkasten, keine kuenstlich erfundenen Bildinhalte.

## Kritische Punkte

### 1. Unbestaetigte Speziesnamen werden noch als harte Labels verwendet

`docs/PROJECT.md` sagt korrekt, dass die Dataset-Exportdateien nur Klassen-IDs 0/1/2
enthalten und dass die Zuordnung zu Rotwild/Rehwild/Schwarzwild unbestaetigt ist.
Trotzdem stehen diese Namen noch in:

- `config.py` (`CLASS_NAMES`, `CLASS_EN`)
- `src/data_loader.py` fallback mapping
- `scripts/eda.py` und dadurch `output/eda_stats.csv` sowie EDA-Plots
- `scripts/build_annotation_package.py`
- `dist/bambi_annotation/manifest.csv` ueber die Spalte `species`

Warum das relevant ist:
Eine Prueferin koennte das als unbelegte Tatsachenbehauptung lesen. Die eigentlichen
CV-Ergebnisse haengen nicht daran, aber wissenschaftlich ist es sauberer, nur
`class 0/1/2` zu berichten, bis das BAMBI-Team die Zuordnung bestaetigt.

Potentielle Aenderung:

- In `config.py`: `CLASS_NAMES = {0: "class 0", 1: "class 1", 2: "class 2"}`.
- In EDA/Plots: Legenden und CSV-Spalten auf Klassen-IDs umstellen.
- Im Annotation-Manifest: Spalte `species` in `class_id` umbenennen.
- Optional: `LabelStore.species()` als Rueckwaertskompatibilitaets-Alias behalten, damit
  alte Manifeste nicht brechen.

### 2. Plan-Dokument hat noch historische Aussagen als aktuelle Entscheidungen

Der Masterplan wurde offenbar vor und nach dem Methodik-Pivot weitergeschrieben. Er enthaelt
beides: erst die korrekte Aussage, dass Direction-GT aus Tracking kommt, spaeter aber noch
"Gold human labels = primary" und einen alten Spezies-Fokus.

Warum das relevant ist:
Die Projektgeschichte ist in `docs/PROJECT.md` viel klarer als im Plan. Wenn eine andere
Person zuerst den Plan liest, wirkt es widerspruechlich.

Potentielle Aenderung:

- Den Plan als "historical master plan, updated after feasibility pivot" markieren.
- "Gold human labels = primary" ersetzen durch:
  - Tracking displacement = primary direction GT for moving animals.
  - Human labels = validation / visibility / moving-stationary check.
- "species focus" entfernen oder als alten verworfenen Plan markieren.
- Annotation-Plan auf Andreas und ca. 1,500 Crops synchronisieren.

### 3. Random-Baseline in `docs/results.md`

In der Tabelle stand bei meiner Reproduktion fuer die Random-Blur-Baseline nicht der aktuelle
reproduzierbare Wert. Recompute aus `output/blur_eval.csv` mit Seed 42 ergibt:

- Median axial error: 44.2 deg
- Acc@45: 0.50

Warum das relevant ist:
Nicht tragend fuer die Story, aber kleine Zahlenabweichungen untergraben Vertrauen.

Potentielle Aenderung:

- `docs/results.md` Tabelle auf `44.2 / 0.50` setzen.
- `docs/audit.md` kann den alten Wert weiter als Korrekturhistorie nennen.

### 4. "Class 2 = boar" wird in Interpretationen stellenweise noch vorausgesetzt

Einige Ergebnis-Texte deuten class 2 als boar/wild boar. Das kann stimmen, ist aber laut
eigener Projektdoku nicht belegt.

Warum das relevant ist:
Die Aussage "class 2 is cooler/lower contrast" ist belegt. Die Aussage "boars are cooler"
ist ohne bestaetigte Zuordnung nicht belegt.

Potentielle Aenderung:

- In Ergebnis- und Audit-Texten "boar/boars" durch "class-2 animals" ersetzen, ausser in
  der Quell-/Aufgabenbeschreibung, wo die Spezies allgemein als Dataset-Kontext genannt sind.

### 5. Tracking ist stark, aber ID-Switch bleibt der echte wissenschaftliche Caveat

Das Projekt benennt diesen Punkt bereits gut. Ich wuerde ihn nicht weichspuelen.

Belegte Zahlen:

- Median trusted/core Tracklet sitzt in Frames mit ca. 8 same-class animals.
- Nur 6/190 trusted und 3/138 core Tracklets sind wirklich single-animal tracks.
- Der Tracker nutzt nearest-centroid association nach Klasse, ohne Appearance-Modell.

Potentielle Aenderung:

- Kein grosses Refactor vor der Abgabe, wenn Zeit knapp ist.
- Fuer die finale Praesentation klar sagen:
  - Tracking zeigt echten Residual-Motion-Signal nach Ego-Motion-Canceling.
  - Der High-Confidence-Core ist defensibler als die vollen 190.
  - Crowd-ID-switching bleibt die zentrale Unsicherheit.
- Wenn noch Zeit ist: kleine qualitative Abbildung mit 2-3 Tracklets:
  - ein single-animal/switch-immunes Beispiel,
  - ein crowded Beispiel,
  - ein abgelehntes/stationaeres Beispiel.

### 6. Annotation-Paket: Inhalt wirkt gut, aber Spaltenname sollte neutral sein

Die Crops sehen plausibel aus. Das Tool verlangt bewusst "unsure" statt Raten. Das ist gut.
Der einzige fachliche Haken ist der Manifest-Spaltenname `species`, weil tatsaechlich nur
eine Klassen-ID gespeichert wird.

Potentielle Aenderung:

- Manifest-Spalte `species` -> `class_id`.
- UI weiter als "class X" anzeigen.
- README des Annotation-Pakets nicht "deer or wild boar" sagen lassen, sondern neutral
  "one animal from the BAMBI dataset".

## Was ich nicht aendern wuerde

- Ich wuerde die Tracking-Methodik nicht kurz vor knapp ersetzen. Sie ist nachvollziehbar
  und testbar.
- Ich wuerde keine neuen Deep-Learning-Resultate erfinden oder nachschieben, solange die
  Human-Labels fehlen.
- Ich wuerde die Caveats nicht kleiner machen. Gerade die ehrliche Audit-Datei laesst das
  Projekt menschlicher und glaubwuerdiger wirken.
- Ich wuerde die alten Proposal-Fakten nicht loeschen. Es ist sinnvoll zu zeigen, dass sich
  die Methodik nach echter Datenpruefung veraendert hat.

## Vorschlag fuer eine andere KI

Wenn eine andere KI das bewerten soll, wuerde ich sie gezielt bitten:

1. Pruefe, ob die Speziesnamen wirklich aus dem Dataset belegbar sind.
2. Pruefe, ob der Plan mit `docs/PROJECT.md`, `docs/results.md` und `docs/audit.md`
   widerspruchsfrei ist.
3. Pruefe, ob die Random-Baseline in `docs/results.md` zum aktuellen `blur_eval.csv` passt.
4. Pruefe, ob das Annotation-Paket fuer Andreas keine unbestaetigten Artennamen vorgibt.
5. Pruefe, ob die finale Story genug betont:
   - Tracking-GT ist sinnvoll, aber crowding/ID-switch bleibt Caveat.
   - Single-image direction is weak.
   - Foundation models sind proxy-limitiert und nicht klar unterscheidbar.

## Priorisierte potentielle Aenderungsliste

### P1: Wissenschaftliche Sauberkeit

- Klassen neutral als `class 0/1/2` berichten.
- Speziesnamen nur als unbestaetigte externe Dataset-Kontextinformation nennen.
- Ergebnisinterpretationen fuer class 2 neutral formulieren.

### P1: Dokument-Konsistenz

- Masterplan mit dem Tracking-Pivot synchronisieren.
- "Gold labels" in "manual validation labels" umbenennen, wo es um Andreas' Labels geht.

### P2: Kleine Zahlenkorrektur

- Random baseline in `docs/results.md` auf den reproduzierten Wert setzen:
  `44.2` Medianfehler und `0.50` Acc@45.

### P2: Annotation-Paket

- `species`-Spalte in `class_id` umbenennen.
- README neutralisieren.

### P3: Presentation/Report polish

- Eine kleine Abbildung oder Tabelle fuer den High-Confidence-Core einbauen.
- ID-switch-Caveat klar, aber nicht panisch darstellen.
- Keine ueberstarken Formulierungen wie "proves" fuer population-level tracking verwenden.

## Befund in einem Satz

Das Projekt wirkt echt und belastbar, solange es seine Klassen als unbestaetigte Dataset-IDs
behandelt und die Tracking-Ergebnisse als starkes, aber crowding-begrenztes Pseudo-GT
praesentiert.
