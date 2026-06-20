# References

Background and method sources for the project, grouped by topic. The most important one
for us is IAOS (the supervisors' own group): it already frames a moving target as causing
directional motion blur in AOS integrals and gives a recipe to recover its direction.

## Airborne Optical Sectioning (the imaging method behind our data)
- Nathan, Kurmi, Bimber. **Inverse Airborne Optical Sectioning (IAOS).** Drones 6(9):231, 2022.
  arXiv:2207.13344. Moving targets blur in AOS integrals; recovers motion direction (theta) and
  speed (s) by registering frames along the motion and maximising gray-level variance; the Radon
  transform of the integral encodes the blur angle. Closest prior work to our task.
- Nathan, Kurmi, Schedl, Bimber. **Through-Foliage Tracking with AOS.** J. Remote Sensing, 2022.
  arXiv:2111.06959. Detecting/tracking moving targets in AOS integrals beats single frames
  (97% vs 42% precision); integration both enables detection and stabilises tracking.
- Kurmi, Schedl, Bimber. **Thermal Airborne Optical Sectioning.** Remote Sensing 11(14):1668, 2019.
  Establishes far-infrared AOS: integrating thermal frames removes occlusion and yields sharp warm
  blobs - the imaging principle behind our dataset.
- Schedl, Kurmi, Bimber. **An autonomous drone for search and rescue in forests using AOS.**
  Science Robotics 6(55), 2021. arXiv:2105.04328. Onboard thermal AOS + learned classification of
  warm blobs; establishes the warm-blob target model.
- BAMBI project. bambi.eco and aist.fh-hagenberg.at (Praschl, Schedl). Light-field/AOS occlusion
  removal + AI classification for area-wide wildlife counts.

## Motion-blur direction from a single image (moving-animal branch)
- Gradient structure tensor (orientation + coherence): OpenCV anisotropic-segmentation tutorial.
  Spatial-domain, holds up on small low-contrast patches; our primary estimator (src/gst.py).
- Radon transform of the power spectrum (blur angle is perpendicular to the spectral ripples).
  PLOS One 2020, 10.1371/journal.pone.0238259 (coarse->fine refinement, sub-degree on synthetic;
  beats cepstrum under noise). EURASIP J. Adv. Signal Process. 2007 (noise limits ~22 dB SNR).
- Cepstrum / log-power-spectrum peak: twin peaks at +-blur-length along the motion direction.
  Baseline; weakest for short blur / small patches.
- Vidal et al. **Estimation of motion-blur kernel parameters using regression CNNs.** arXiv:2308.01381.
  Regresses (length, angle); angle is only defined on a 180-degree interval; short blurs are
  inherently ambiguous (matters for slow animals).
- The 180-degree ambiguity: a symmetric blur gives an axis, not a signed heading; resolve the sign
  with the cross-frame tracking displacement. "Image as an IMU", arXiv:2503.17358.

## Orientation / direction estimation with neural nets (stationary-animal branch)
- Beyer, Hermans, Leibe. **Biternion Nets.** GCPR 2015. Output (cos, sin); cosine loss; validated on
  low-resolution crops - our regime. Use the doubled angle (cos 2theta, sin 2theta) for an axis.
- Prokudin, Gehler, Nowozin. **Deep Directional Statistics.** ECCV 2018. von Mises NLL with a
  learned concentration (uncertainty); mixtures for the ambiguous head/tail case.
- Yang, Yan. **Arbitrary-Oriented Object Detection with Circular Smooth Label.** ECCV 2020.
  Angle-as-classification with a circular Gaussian label; forgiving with noisy labels.
- Yang et al. **Rotated detection with Gaussian Wasserstein Distance loss.** ICML 2021. Removes the
  180-degree and box-definition ambiguities; angle-insensitive on near-square (round) blobs.
- Classical baselines: PCA / image moments / fitEllipse for the body axis (src/bb_refinement.py).

## Evaluation (circular statistics)
- Wrapped shortest-arc distance for angular error; mean/median/RMSE + percentiles; Acc@k tolerance;
  flip-corrected variant to separate axis error from 180-degree sign error. arXiv:2603.25351.
- Axial data via angle doubling; mean resultant length, circular variance, Rayleigh uniformity test.
- Jammalamadaka-Sarma circular correlation for predicted-vs-tracking agreement; permutation test for
  significance; block-bootstrap CIs at the flight level (the boxes are correlated within flights).
- von Mises as the standard heading model in movement ecology (PLOS One 2012, PMC3511459).

## Tracking and ego-motion (the ground-truth source)
- SORT (Bewley et al. 2016): constant-velocity Kalman + Hungarian/IoU. The velocity components give
  direction directly. For small fast blobs, associate on centroid distance, not IoU.
- Global motion compensation: keypoints + RANSAC homography/affine to remove camera motion, then the
  residual is target motion. YOLOMG, arXiv:2503.07115 (three-frame differencing). Sensors 15(4):8214.

## Domain gap (why we test, not assume, foundation models)
- An empirical study of drone thermal wildlife detection. arXiv:2310.11257. ImageNet/COCO backbones
  fine-tuned on ~850 thermal images; small-object scale is the bottleneck (~32 px) - our regime.
- IRSAM (ECCV 2024, arXiv:2407.07520): pretrained SAM falls short on infrared due to the domain gap.
- "Caught in the Lens: Zero-Shot BioCLIP Struggles with Camera Trap Wildlife." Springer 2025.
  Zero-shot accuracy ~18-42% on camera traps - so BioCLIP is a negative control here, not a backbone.
- DINOv2 (arXiv:2304.07193): frozen features + a small head curb overfitting in low-data settings.
- Comparative study of custom CNNs vs transfer learning, arXiv:2601.02246: transfer does not reliably
  beat from-scratch when the target domain is far from ImageNet - so we compare both.
