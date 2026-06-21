# BAMBI animal annotation

## What this is about
We are testing whether the movement direction of wild animals can be read from thermal
drone images. To check our automatic methods, we need a set of images labelled by a
person - that is what you are doing here. Each image shows one animal (deer or wild boar)
filmed from above with a thermal camera.

## What to label (the focus)
For every image the main question is: **is the animal moving or standing still?**
After that, only if you can clearly see which way it faces, give its direction. The
direction is often impossible to tell in these images - that is fine and expected, just
mark "axis" or "unsure". Do not guess.

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
   s = standing (crisp blob)    d = moving (smeared / streaked)    u = unsure

2) If you can tell, which way is the HEAD pointing? The number keys are a compass,
   up = top of the image:

       7 NW    8 N     9 NE
       4 W     5 axis  6 E
       1 SW    2 S     3 SE

   1-9 (not 5) = the head points that way
   5 = axis only: you see the body/blur LINE but not which end is the head.
       Press 5 again to rotate the line through the four orientations
       (vertical -> diagonal / -> horizontal -> diagonal \) until it matches the animal.
   0 = nothing usable (you cannot make out the animal)

## How to tell the head (usually you cannot - that is ok)
- Moving smear: the front (where it is now) is brighter / sharper; the blur behind is the
  trail, so the head points to the sharper end.
- Body shape: the thinner end is the neck / head, the bulkier end is the rump.
- Visible legs or antlers.
If you see the line but not the head -> press 5, then press 5 again until the orange line
lies along the animal's body. If nothing -> press u, then 0.

## One rule
Be honest. "axis" (5) or "unsure" (u) helps us more than a guess - we even measure how
often the head is recognisable. Accuracy over speed, and you can stop and continue any time.

## Keys
Enter / Space = next     Backspace = previous (to fix)     Esc = save & quit
