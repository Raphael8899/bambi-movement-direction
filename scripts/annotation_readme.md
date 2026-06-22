# BAMBI animal annotation

## What this is about
We are testing whether the movement direction of wild animals can be read from thermal
drone images. To check our automatic methods, we need a set of images labelled by a
person - that is what you are doing here. Each image shows one animal (deer or wild boar)
filmed from above with a thermal camera.

## What to label (the focus)
For every image the main question is: **is the animal moving or standing still?**
After that, set the body line's angle, and add a head arrow only if you can clearly see
which end is the head. The head is often impossible to tell in these images - that is fine
and expected; just leave the line without an arrow. Do not guess.

## Setup (once)
1. Install Python 3 from python.org. On Windows, tick "Add python.exe to PATH".
2. Open this folder and run:  pip install -r requirements.txt
   (Windows tip: in the folder, click the address bar, type cmd, press Enter, then run that line.)

## Start
Double-click run.bat, or run:  python annotate.py
Answers are saved to labels.csv in this folder (after every image). You can stop and
reopen any time - it continues where you left off. Send labels.csv back when you are done.

## What you see
A large image with a GREEN box. Judge only the animal inside the green box; ignore the rest.
The animal is a bright (warm) blob on a darker background.

## Per image, set two things
1) Moving or standing?
   s = stationary (crisp)   d = moving (smeared / streaked)   u = unsure

2) The body/blur LINE, and the HEAD only if you can tell.
   First press 5 to show a line, and press 5 again to rotate it (8 steps of 22.5 deg)
   until it lies along the animal's body / smear.
   Then add an arrow ONLY if you can see which end is the head:
       1 = arrow at one end       2 = arrow at the other end
   If you cannot tell the head, leave the plain line:
       3 = no arrow (axis only)
   0 = nothing usable (you cannot make out the animal at all)

## How to tell the head (usually you cannot - that is ok)
- Moving smear: the front (where it is now) is brighter / sharper; the blur behind is the
  trail, so the head points to the sharper end.
- Body shape: the thinner end is the neck / head, the bulkier end is the rump.
- Visible legs or antlers.
Set the line with 5; if you cannot tell the head, press 3 (no arrow). If nothing is
visible at all -> press u, then 0.

## One rule
Be honest. "no arrow" (3) or "unsure" (u) helps us more than a guess - we even measure how
often the head is recognisable. Accuracy over speed, and you can stop and continue any time.

## Keys
Enter / Space = next     Backspace = previous (to fix)     Esc = save & quit
