
"""
author: N. Abrate.

file: makeinp.py

description: Make NJOY input for neutrons and photo-atomic evaluations.
"""

import sys
import pathlib

pwd = pathlib.Path.cwd()
src = pwd.joinpath("NDL")
sys.path.append(str(src))
import ProcessNDL as ndl

# define "tutorial" path
tutorialpath = pwd.joinpath("tests").joinpath("endf")
# define path
dec_datapath = tutorialpath.joinpath("decay")
nfy_datapath = tutorialpath.joinpath("nfy")
out_dir = tutorialpath.joinpath("out")
libname = "endfb8"

out_dec = ndl.build_burnup_data(in_dir=dec_datapath, out_dir=out_dir, lib="endfb8", suffix="dec", 
                            pattern="*.endf", recursive=False,)

out_nfy = ndl.build_burnup_data(in_dir=nfy_datapath,out_dir=out_dir,lib="endfb8",suffix="nfy",
                            pattern="*.endf", recursive=False,)