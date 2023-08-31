# FLATW'RM

FLATW'RM (FLAre deTection With Ransac Method) is a code that uses machine learning method to detect flares. For details, see the <a href="https://ui.adsabs.harvard.edu/abs/2018A%26A...616A.163V/abstract">A&A paper</a>.

FLATW'RM detects stellar flares in light curves using a classical machine-learning method. The code tries to find a rotation period in the light curve and splits the data to detection windows. By default it uses a window size of 1.5*P<sub>rot</sub>, which is typically longer than the time scale of the flares, but not too long to fit with a polynomial or a reasonable order. The light curve sections are fit with the <a href="https://en.wikipedia.org/wiki/Random_sample_consensus">RANSAC (Random sample consensus) method</a> – a robust fitting algorithm – and outlier points (flare candidates) above the pre-set detection level are marked for each section. For the given detection window only those flare candidates are kept (given a vote) that have at least a given number of consecutive points (three by default). After each window is analyzed, those flare candidates are marked as flares that have a given number of votes. 

<b>NOTE:</b> when using FLATW’RM, always check the output, the default settings (detection sensitivity, number of consecutive points) probably should be changed depending on the light curve noise, data sampling frequency, and scientific needs.

The code depends on the following packages:
* matplotlib 
* scipy
* sklearn (the machine learning toolkit)
* gatspy (for period search)

These can be installed by the system package manager, e.g.: `sudo aptitude install python-matplotlib` or `sudo port install py-matplotlib`; 
or using Python package manager: `pip install gatspy`.

To use the code, simply run `./flatwrm.py <input file(s)>`, for a detailed list of command-line options use `./flatwrm.py --help`

<!--<img src="flatworm.png" width="250">-->
<p align="center">
  <br><br>
<img src="flatworm-rect.png" width="300">
</p>


