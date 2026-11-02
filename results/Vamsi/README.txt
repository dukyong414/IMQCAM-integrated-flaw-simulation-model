Raptor 5 x 5 x 14 mm velocity sweep
====================================

Geometry: 5 mm x 5 mm x 14 mm pillar.
Velocities: 1300 to 1700 mm/s, step 50.
Power 370 W, spot 100 µm.
Hatch spacing: 140 µm on every velocity.
Layer thickness: 30 µm.
Melt pool width: flucWsMTs mean width at that velocity.
One melt pool per velocity: the cosine spectrum is the average of the eight single-track orientation files. λ=1.04 shifts the mean width, depth, and cap only, matching the 0.5 mm cube. Voxel size is 5 µm. Scan paths come from Raptor's ScanPathBuilder.
No edge pores and no scan-vector file.

Velocity (mm/s)   Hatch (µm)   Melt pool width (µm)   Width used in Raptor (µm)
    1300            140           172.7                    179.6
    1350            140           171.0                    177.9
    1400            140           164.0                    170.6
    1450            140           154.3                    160.5
    1500            140           153.3                    159.5
    1550            140           151.4                    157.5
    1600            140           147.8                    153.8
    1650            140           142.4                    148.1
    1700            140           139.5                    145.1

Pore number density, pores/mm³. Each column is a different source.

Paper — Subraveti et al., Comput. Mater. Sci. 275 (2026) 115083.
  Written in the paper. Fig. 6 is a log plot of volume fraction and does not print a number at each speed.
  Fatigue window 1300–1500 mm/s, pores of 20–30 µm: 1–3 pores per 200 mm³, which is 0.005–0.015 pores/mm³.
  That rate is for the window, not a separate measurement at each speed.
  λ=1.14: no stochastic LoF at 1300, 1350, and 1400 mm/s (0 pores/mm³, 0% volume fraction).
  λ=1.04: volume fraction on the order of 10^-5 %. At 1300 mm/s they describe the material as 99.99999% dense.
  1550 mm/s is the calibration speed (scanned volume 336 mm³), not a validation point.
  Largest calibration pores: 65–83 µm at λ=1.04 and 33–39 µm at λ=1.14.
  1600–1700 mm/s are outside the paper.

Integrated model, his geometry — 0.5 mm cube, 5 µm voxels, λ on the mean only,
  tracks extended by l/2. Pores smaller than 20 µm equivalent diameter are left out,
  the same cutoff he used against experiment.
  1000 cubes (125 mm³) at 1300–1550 mm/s, 250 cubes (31.25 mm³) at 1600–1700 mm/s.
  The pillar's 0.5 mm surface skin is not applied: the cube is only 0.5 mm across.

Integrated model, this 5×5×14 mm pillar — 5 µm, λ on the mean only.
  Raptor LoF replaces CMU LoF. Gas and keyhole stay CMU.
  Raptor LoF keeps pores whose center is more than 0.5 mm inside the pillar.
  λ=1.04 is in this folder. λ=1.14 is in Desktop\Vamsi_lambda114.

Velocity   Paper, stated                         Integrated model, his 0.5 mm cube, ≥20 µm    Pillar 5×5×14
(mm/s)     pores/mm³                              λ=1.04    λ=1.14                       λ=1.04    λ=1.14
   1300    λ=1.14 is 0; window 0.005–0.015        0         0                 0         0
   1350    λ=1.14 is 0; window 0.005–0.015        0         0                 0         0
   1400    λ=1.14 is 0; window 0.005–0.015        0         0                 0         0
   1450    window 0.005–0.015                     0         0                 0.051     0
   1500    window 0.005–0.015                     0.016     0                 0.011     0
   1550    calibration, not a validation density  0.048     0                 0.363     0
   1600    not reported                           0.160     0                 0.814     0
   1650    not reported                           1.920     0                 5.771     0.014
   1700    not reported                           6.272     0.032             14.346    0.166

Paper volume fraction, a different unit from the columns above:
  λ=1.04 in the window: order of 10^-5 %.
  λ=1.14 at 1300, 1350, 1400: 0%.
His 0.5 mm cube, pores ≥ 20 µm, counts:
  λ=1.04: 0, 0, 0, 0, 2, 6, 5, 60, 196
  λ=1.14: 0, 0, 0, 0, 0, 0, 0, 0, 1

Pillar counts, λ=1.04: 0, 0, 0, 18, 4, 127, 285, 2020, 5021.
Pillar counts, λ=1.14: 0, 0, 0, 0, 0, 0, 0, 5, 58.

CMU source. Scanned volume is the coupon volume the model matches, in mm³.
Pore counts are the pores measured on those coupons. LoF ≥ 20 µm keeps pores
whose equivalent diameter is at least 20 µm, in that same scanned volume.
Keyhole matches no coupon at these speeds. Gas uses the same coupons as LoF
from 1300 to 1450 mm/s, and no coupon from 1500 mm/s up.
At 1300 mm/s the gas density counts those 89 pores twice (partition weight 2),
so the density is 0.328 /mm³ rather than 89 / 542.2.

Velocity   LoF volume   LoF pores   LoF ≥20 µm pores   Keyhole pores   Gas volume   Gas pores
(mm/s)     (mm³)                                                                      (mm³)
   1300       542.2          89              9               0            542.2         89
   1350       253.0         357              4               0            253.0        357
   1400       317.9         284              1               0            317.9        284
   1450       317.4         714            102               0            317.4        714
   1500       154.3        6672           1929               0              0            0
   1550       541.5       34733           8645               0              0            0
   1600       214.5        9199           1269               0              0            0
   1650       327.0        3121            855               0              0            0
   1700       232.5       21816           9580               0              0            0

Coupons: 1300 CMU02-05-1 295.3 and CMU02-32-1 246.8; 1350 CMU02-22-1;
1400 CMU02-11-2; 1450 CMU02-03-2; 1500 CMU02-04-2; 1550 CMU02-06-2 335.7
and CMU02-25-1 205.7; 1600 CMU03-01-2; 1650 CMU03-17-2; 1700 CMU03-02-2.
CMU LoF density (pores/mm³): 0.164, 1.411, 0.893, 2.249, 43.246, 64.148, 42.889, 9.545, 93.845.
CMU LoF ≥ 20 µm density: 0.017, 0.016, 0.003, 0.321, 12.503, 15.967, 5.917, 2.616, 41.210.
CMU gas density: 0.328, 1.411, 0.893, 2.249, 0, 0, 0, 0, 0.
CMU keyhole density is 0 at every speed.


Each V_#### folder has only:
  melted_build.png       outer faces (grey melted, black boundary, yellow overlap)
  volume_sweep.gif       isometric view; a surface sweeps along Y and builds the solid
  all_pore_sources.png  see-through build with every pore type (LoF, keyhole, gas)
  pore_size_shape.png   ESD, sphericity, aspect ratio
  pores.csv             every pore: location (plate mm) and shape
  view3d.npz            3D viewer data
  process.json          hatch, melt-pool width, and the pores/mm³ comparison

The folder also has conditions.csv (hatch and width) and predictions.csv
(CMU and Raptor, both in pores/mm³). Raptor LoF is the value that replaces CMU LoF.

View the results
----------------
Unzip the whole folder, then double-click "Open results GUI.bat".

The python folder next to the bat file is a portable Windows 64-bit Python.
The other computer does not need Python installed.
The GUI opens in the browser at http://127.0.0.1:8055
Leave the black command window open while you use the GUI. Close it to stop the GUI.

If Windows says the file is protected, choose More info, then Run anyway.

X and Y are plate millimeters. Z is height above the plate.
Keyhole and gas pores are spheres (sphericity 1, aspect ratio 1).
LoF sphericity is measured only for pores with enough voxels.
