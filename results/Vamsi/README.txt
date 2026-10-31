Raptor 5 x 5 x 14 mm velocity sweep
====================================

Geometry: 5 mm x 5 mm x 14 mm pillar.
Velocities: 1300 to 1700 mm/s, step 50.
Power 370 W, spot 100 µm.
Hatch spacing: 140 µm on every velocity.
Layer thickness: 30 µm.
Melt pool width: flucWsMTs mean width at that velocity.
One melt pool per velocity: the cosine spectrum is the average of the eight single-track orientation files. The mean width, depth, and cap spectra are multiplied by λ=1.04, which is Raptor's scale on every Fourier amplitude. Scan paths come from Raptor's ScanPathBuilder.
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

Pore number density. CMU and Raptor are both in pores/mm³.
Raptor LoF replaces the CMU LoF prediction. Keyhole and gas stay the CMU prediction.
Raptor LoF counts pores more than 0.5 mm inside the pillar. LoF closer to the surface is left out.
Counts are that density times the 5 x 5 x 14 mm pillar (350 mm³).

Velocity   CMU LoF   Raptor LoF   CMU keyhole   CMU gas    (pores/mm³)
   1300      0.164      0.000        0.000       0.328
   1350      1.411      0.000        0.000       1.411
   1400      0.893      0.000        0.000       0.893
   1450      2.249      0.000        0.000       2.249
   1500     43.246      0.000        0.000       0.000
   1550     64.148      0.000        0.000       0.000
   1600     42.889      0.000        0.000       0.000
   1650      9.545      0.017        0.000       0.000
   1700     93.845      0.163        0.000       0.000


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
