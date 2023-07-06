# QR-Code-Security

This code can be used to superimpose textures on QR Code in order to to make counterfeiting of QR Codes difficult.

---
First Stage: Generation of anti-counterfeiting texture
1. Gaussian Texture
2. Bilinear interpolation texture
3. Halftone texture
---
Second Stage: Refine QR Code  
Original QR Code -> Refined QR Code

---
Third Stage: fusion of the First and Second Stages  
Texture hidden anti-counterfeiting QR Code

---
Now, to differentiate between the printed QR(original) and photocopy QR(counterfeited) the intensity of the QR will be calculated.
A flask app was made where user can upload the QR and Pyimagej was used to calculate the mean intensity of QR.
PyImageJ provides a set of wrapper functions for integration between ImageJ2 and Python. It also supports the original ImageJ API and data structures.
A major advantage of this approach is the ability to combine ImageJ and ImageJ2 with other tools available from the Python software ecosystem, including NumPy, SciPy, scikit-image, CellProfiler, OpenCV, ITK and many more.
