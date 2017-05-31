Tracking
========

A package for performing storm cell tracking in Python. The package uses tools
from the Python-ARM Radar Toolkit (Py-ART) and SciPy to track, forecast, and
compute statistics of individual storm cells throughout series of radar scans. 

Dependencies
------------
- `Py-ART <http://arm-doe.github.io/pyart/>`_

Install
-------
To install tracking, first clone::

	git clone https://github.com/mhpicel/tracking.git

then,::

	cd tracking
	python setup.py install

Special Thanks
--------------
This package is based on a set of R codes written by Bhupendra Raut.
The R codes can be found here:
	https://github.com/RBhupi/Darwin-Rscripts

References
----------
Dixon, M. and G. Wiener, 1993: TITAN: Thunderstorm Identification, Tracking,
Analysis, and Nowcasting—A Radar-based Methodology. J. Atmos. Oceanic
Technol., 10, 785–797, doi: 10.1175/1520-0426(1993)010<0785:TTITAA>2.0.CO;2.

Leese, J.A., C.S. Novak, and B.B. Clark, 1971: An Automated Technique for Obtaining Cloud Motion from Geosynchronous Satellite Data Using Cross Correlation. J. Appl. Meteor., 10, 118–132, doi: 10.1175/1520-0450(1971)010<0118:AATFOC>2.0.CO;2.
