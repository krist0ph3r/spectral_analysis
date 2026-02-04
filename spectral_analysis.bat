@echo off
echo Run below line if packages are not installed:
echo python -m pip install numpy matplotlib opencv-python

cd images

FOR %%i IN (*.*) DO python ..\spectral_analysis.py "%%i"

cd ..