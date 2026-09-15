case.py holds every physical and numerical parameter as a set of dataclasses
(geometry, mesh, fluids, gas source, turbulence, numerics, solid pressure,
inlet, post-processing). A single instance, CASE, is created at import time
and shared by the whole solver.

To change a parameter permanently, edit the default value here.
To change it for one run without editing this file, use the constants at the
top of run.py instead.
