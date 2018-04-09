# FLATW'RM

FLATW'RM (FLAre deTection With Ransac Method) is a code that uses machine learning method to detect flares. 

The code depends on the following packages:
* matplotlib 
* scipy
* sklearn (the machine learning toolkit)
* gatspy (for period search)

These can be installed by the system package manager, e.g.: `sudo aptitde install python-matplotlib` or `sudo port install py-matplotlib`; 
or using Python package manager: `pip install gatspy`.

To use the code, simply run `./flatwrm.py <input file(s)>`, for a detailed list of command-line options use `./flatwrm.py --help`

<img src="flatworm.png" width="250">
