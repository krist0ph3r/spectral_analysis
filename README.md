# motivation
the motivation of this script is to break down the wavelengths in coloured light sources. this is probably not a very reliable method but I used this script on photos taken on a DSLR with exposure stepped down to only catch the light source (and some direct reflections). this is out of curiosity to ascertain if the LED lights I use at home emit undesirable blue light.

## spectral_analysis.py
Load one or more images and visualize the histograms for RGB channels.
Visualize wavelength vs intensity.
Display a thumbnail showing which parts of the image was actually used for the graphs (near-black and near-white/any likely gamut clipping are ignored as they will not produce reliable results)

## spectral_analysis.bat
.bat file is a handy shortcut as Windows doesn't expand wildcards on commandline arguments (it scans for all files in an images subdirectory)
