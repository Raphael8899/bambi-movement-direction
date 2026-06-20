# Assignment and source facts

The original inputs the project is built on. Kept verbatim-in-substance so a fresh reader has the
real brief. The decoded proposal slides are in `proposal_decoded.txt`.

## Course assignment (Computer Vision, FH Hagenberg)
- Grade = 50% group project + 50% written exam (~1 h, after the final session).
- Project: based on the BAMBI datasets (real aerial wildlife monitoring). Teams of 3-4 students;
  **we are 2 (Raphael + Andreas).** Choose a dataset and a task (segmentation, detection, pose,
  architecture comparison, or your own). **Pre-processing and annotation are part of the project.**
  Apply, evaluate and **compare** computer-vision methods from the lectures. Present the proposal
  early and the final results in the last session. Expect ~40-50 h per student.
- Our chosen task is **fixed by our proposal**: *Movement Direction Estimation of Wildlife from
  Thermal Light-Field (AOS) Drone Imagery*. The methodology is ours to design.

## Dataset facts from the supervisor (Christoph Praschl, FH Hagenberg) - these are "set"
- The dataset lives on Roboflow. The email pointed to workspace `dseducation`, project
  `bambi-alfs-20250520-upload04`; the copy we actually use (API key from the old project) is
  workspace **`bambi-overview`**, project **`bambi-alfs-20250520-upload04-sdakr`**, **version 2**.
- It is a **thermal light-field dataset of 225 drone flights** with labels for **red deer
  (Rotwild), roe deer (Rehwild) and wild boar (Wildschwein/Schwarzwild)**.
- Each light field = one central video frame with **4 frames before and 4 after combined** (so ~9
  frames), sampled around ~3 FPS, i.e. one light field spans just under 1 second of the 30 FPS video.
- Some exports have **ghosting artefacts in the border region** (the cause is fixed but those
  frames were not re-exported yet - ignore them for now).
- Christoph's *original* suggestion (different from our course topic, kept for context): first
  improve the data quality (auto-tighten the manual bounding boxes, which tend to be too large),
  evaluate it statistically (size per species, colour distributions), then try to classify moving
  vs. stationary animals from the boxes, possibly with simple unsupervised clustering. Useful work
  there could later be funded. (Our course deliverable is the movement-direction topic above.)

## What the supervisor names tell us
- Supervisors: **FH-Prof. Christoph Praschl** and **FH-Prof. Dr. David Schedl** (AIST group,
  FH Hagenberg; related to JKU Linz / Oliver Bimber's lab). Earlier notes saying "Dreiseitl" are wrong.
- The imaging method is **Airborne Optical Sectioning (AOS)** - the same group's published work
  (see `../references.md`, especially IAOS: Nathan, Kurmi, Bimber, Drones 2022).
