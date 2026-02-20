"""
author: N. Abrate.

file: makeinp.py

description: Make NJOY input for neutrons and photo-atomic evaluations.
"""

import sys
import pathlib

pwd = pathlib.Path.cwd()
src = (pwd.parent).joinpath("source")
sys.path.append(str(src))
import ProcessNDL as ndl

# define "tutorial" path
pwd = pathlib.Path.cwd()
tutorialpath = pwd.joinpath("endf")
# define path
n_datapath = tutorialpath.joinpath("neutrons")
pa_datapath = tutorialpath.joinpath("photoat")
ar_datapath = tutorialpath.joinpath("atomic_relax")

# define additional arguments for NJOY input creation
outpath = tutorialpath
pattern_n = "S-A.endf"  # pattern of the ENDF-6 file. S is the nuclide symbol, A the mass number
pattern_pa = "S.endf"
libname = "ENDF-B/VIII.0"
njoyver = "2016"
broad_temp = [300]

# make NJOY input for neutron evaluations
ndl.make_input(n_datapath, pattern_n, "n", libname, broad_temp, outpath=outpath)
# make NJOY input for photo-atomic evaluations
ndl.make_input(pa_datapath, pattern_pa, "pa", libname, broad_temp=None,
              outpath=outpath, atomrelax_datapath=ar_datapath)
