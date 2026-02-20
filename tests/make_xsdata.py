"""
author: N. Abrate.

file: makexsdata.py

description: Convert xsdir files to a single .xsdata file for Serpent-2.
"""
import sys
import pathlib

pwd = pathlib.Path.cwd()
src = (pwd.parent).joinpath("source")
sys.path.append(str(src))
import ProcessNDL as ndl

# define "tutorial" path
pwd = pathlib.Path.cwd()
tutorialpath = pwd.joinpath(pwd, "endf")

# define path
out_datapath = tutorialpath.joinpath("out")

proj = ["n", "pa"]
libname = "endf8"
# nuclear data library path final location
ndlpath = "/opt/serpent/xsdata/endfb8/acedir"
# convert xsdir into xsdata single file
ndl.convert_xsdir(str(out_datapath), proj, libname, ndlpath, currpath=None)
