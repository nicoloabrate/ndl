"""
Authors: N. Abrate - NEMO Group, Energy Department, Politecnico di Torino.
email: nicolo.abrate@polito.it

File: ProcessNDL.py
Description: Module for generating NJOY input files and processing ENDF-6
             files in order to generate ACE files for Serpent and MCNP

"""
import os
import re
import gzip
import glob
import chardet
import tempfile
import fileinput
import time as t
import numpy as np
import shutil as sh
import multiprocessing as mp

from itertools import groupby
from datetime import datetime
from socket import gethostname
from operator import itemgetter
from os.path import isfile, join
from subprocess import Popen, PIPE


# define particles synonims dict
partdict = {"n": ["neutron", "neutrons", "neutronic", "n"],
            "pa": ["photon", "photons", "photo-atomic", "photoatomic",
                   "pa"],
            "pn": ["gamma", "photo-nuclear", "photonuclear", "pn"]}

replace_dict = {"Z": "\\d+", 
                "S": "\\w+",
                "A": "\\d+",
                "N": "\\d+",
                "L": "\\w+",
                "I": "\\d+"
               }

# define particles ACE type
pdict = {"n": "c", "pa": "p", "pn": "g"}

def get_njoy():
    """
    Extract NJOY executable from system environment variable `NJOY`.

    Returns
    -------
    `string`
        NE executable

    Raises
    ------
    `NDLError`
        if environment variable `NJOY` is not assigned
    """
    if "NJOY" in os.environ:
        exe = join(os.environ["NJOY"])
    else:
        raise ValueError("environment variable 'NJOY' is not assigned")
    return exe


def get_njoy_ver():
    if "2016" in get_njoy():
        njoyver = "2016"
    elif "2021" in get_njoy():
        njoyver = "21"
    else:
        raise OSError("Cannot recognise NJOY version!")
    return njoyver


def build_ace_lib(inpath, libpath, data, libext, particles,
                atom_relax=None, np=None, copyflag=True, binary=True,
                njoypath=None, kerma=True):
    """
    Build the ACE library (with NJOY stream, VIEWR, XSDIR PENDF output files).

    Parameters
    ----------
    inpath : string
        absolute path of NJOY input files location
    libpath : string
        absolute path of nuclear data library
    data : string
        sub-dir name inside "libpath" containing ENDF-6 files to be processed
    libext : string
        ENDF-6 file extension (e.g. ".jeff", ".tendl", ".endf")
    particles : string
        kind of incident-particle data defined in "partdict"
        (e.g. "n", "photo-atomic")
    njoyver : string, optional
        NJOY version number (only 2016 and 2021 are supported).
        The default is 2016.
    atom_relax : string, optional
        Name of atomic relaxation data sub-dir inside "libpath".
        The default is None.
    np : int, optional
        number of CPUs for the parallel calculation.
        The default is None. For default values, the code takes all CPUs-2
    copyflag : bool, optional
        flag to copy ENDF-6 files inside temporary directories created for
        the parallel calculation. The default is False, so the files are moved
        each time in order to reduce the I/O burden.

    Returns
    -------
    None.
    """
    njoyver = get_njoy_ver()
    # define number of CPUs
    if np is None:
        np = mp.cpu_count()-2  # leave 2 free CPUs

    # start parallel pool
    pool = mp.Pool(np)

    # define "out" tree
    baseoutpath = os.path.join(libpath, "out")

    # make input type consistent
    if type(particles) is str:
        particles = [particles]

    # loop over particles
    for part in particles:
        # define particle key
        for key, names in partdict.items():
            if part in names:
                proj = key

        # define particle-wise directory paths
        outpath = os.path.join(baseoutpath, proj)
        # create "out" sub-dirs
        outdirs = {"n": ["pendfdir_bin", "acedir", "xsdir", "njoyout",
                         "njoyinp", "viewheatdir", "viewacedir"],
                   "pa": ["acedir", "xsdir", "njoyout", "njoyinp"],
                   "pn": ["pendfdir_bin", "acedir", "xsdir", "njoyout",
                          "njoyinp", "viewacedir"]}

        [mkdir(name, outpath) for name in outdirs[proj]]

        # gather input files
        allfiles = sorted(os.listdir(os.path.join(inpath, proj)))
        inpfiles = [f for f in allfiles if isfile(join(inpath, proj, f))
                    if f.endswith(".njoyinp")]

        # create KERMA input files list consistently
        inpfilesK = [None]*len(inpfiles)
        if kerma:
            for ipos, f in enumerate(inpfiles):
                name, ext = f.split(".")
                kname = f"{name}.njoyinpK"

                if isfile(join(inpath, proj, kname)):
                    inpfilesK[ipos] = kname

        # print warning for the user
        if inpfiles == []:
            raise OSError(f"Warning: {os.path.join(inpath, proj)} is empty!")

        if None in inpfilesK and proj == "n":
            print(f"Warning: some KERMA files are missing for {os.path.join(inpath, proj)}!")
        # process library with NJOY

        # define list of arguments for parallel function
        args = [(inpath, outpath, proj, libext, libpath, data, atom_relax,
                 copyflag, njoyver, binary, njoypath, kerma)]
        args = list(zip(inpfiles, inpfilesK, args*len(inpfiles)))

        # process input with NJOY on np cores
        pool.map(par_ace_lib, args)

        print(f"The processed library is stored in {outpath}")


def par_ace_lib(args):
    """
    Define the instructions performed independently by each processor.

    Parameters
    ----------
    args : list
        list of input file names to be processed and a tuple of arguments
        that allow to pass to this function the input argument of function
        "build_ace_lib" (for more information, see args of this function)

    Returns
    -------
    None.

    """
    # unpack input arguments
    f, fK, tup = args
    inpath, outpath, proj, libext, libpath, data, atom_relax, copyflag, njoyver, binary, njoypath, kerma = tup

    # create temporary directories in outpath to avoid mixing NJOY output
    with tempfile.TemporaryDirectory(dir=outpath) as tmpath:

        os.chdir(tmpath)
        sh.copyfile(os.path.join(inpath, proj, f), os.path.join(tmpath, f))

        if fK is not None:
            sh.copyfile(os.path.join(inpath, proj, fK), os.path.join(tmpath, fK))

        fname_tmp, ext = os.path.splitext(f)
        fname, tmp = fname_tmp.split("_")

        if libext is not None:
            endfname = f"{fname}.{libext}"
        else:
            endfname = fname

        # move ENDF-6 files in input dir
        try:

            if copyflag is False:
                # FIXME: add symlink option
                sh.copyfile(os.path.join(libpath, data, endfname),
                            os.path.join(tmpath, "tape20"))

            else:
                sh.copyfile(os.path.join(libpath, data, endfname),
                            os.path.join(tmpath, "tape20"))

        except FileNotFoundError:
            print(f"File {os.path.join(libpath, data, endfname)} does not exist!")

        # move atomic relaxation ENDF-6 files in input dir
        if proj == "pa":
            sh.move(os.path.join(libpath, atom_relax, endfname),
                    os.path.join(tmpath, "tape21"))

        # run NJOY, then clean directory
        start_time = t.time()

        outstream = run_njoy(os.path.join(inpath, proj, f))

        outstreamK = None

        if fK is not None:
            outstreamK = run_njoy(os.path.join(inpath, proj, fK))

        success, KERMA_warn = move_and_clean(f, fK, outpath, tmpath, libpath, data,
                                             endfname, proj, atom_relax, binary, kerma=kerma)

        # completed and failed
        if success:
            with open(os.path.join(outpath, 'COMPLETED.txt'), 'a') as compl:
                arg1 = f.split(".")[0]
                compl.write(f"{arg1} processing COMPLETED. {printime(start_time)} \n")
        else:
            with open(os.path.join(outpath, 'FAILED.txt'), 'a') as fail:
                arg1 = f.split(".")[0]
                fail.write(f"{arg1} processing FAILED. {printime(start_time)} \n")

        # NJOY warning and errors
        if outstream is not None:
            if outstream == "warning":
                with open(os.path.join(outpath, 'WARNINGS_NJOY.txt'), 'a') as warn:
                    warn.write(f"-------------- {f} -------------- \n")
                    warn.write(f"Consistency problems found in acer running {f} \n")
            else:
                with open(os.path.join(outpath, 'ERRORS_NJOY.txt'), 'a') as fail:
                    fail.write(f"-------------- {f} -------------- \n")
                    fail.write(outstream+'\n')

        # KERMA NJOY warning and errors
        if outstreamK is not None:
            if outstreamK == "warning":
                with open(os.path.join(outpath, 'WARNINGS_NJOY_KERMA.txt'), 'a') as warn:
                    warn.write(f"-------------- {f} -------------- \n")
                    warn.write(f"Consistency problems found in acer running {f} \n")
            else:
                with open(os.path.join(outpath, 'ERRORS_NJOY_KERMA.txt'), 'a') as fail:
                    fail.write(f"-------------- {f} -------------- \n")
                    fail.write(outstream+'\n')

        # KERMA ENDF-6 warning
        if KERMA_warn is not None:
            with open(os.path.join(outpath, 'WARNINGS_ENDF_KERMA.txt'), 'a') as warn:
                warn.write(f"-------------- {f} -------------- \n")
                warn.write("\n".join(KERMA_warn))
                warn.write("\n")


def move_and_clean(inp, inpK, path, tmpath, libpath, data, endfname, proj,
                   atom_relax=None, binary=True, kerma=False):
    """
    Move and clean output files in temporary directory.

    Parameters
    ----------
    inp : string
        NJOY input file name.
    inpK : string
        NJOY KERMA input file name.
    path : string
        output directory path.
    tmpath : string
        path of the temporary working directory.
    libpath : string
        ENDF-6 file library path.
    data : string
        sub-dir name inside "libpath" containing ENDF-6 files to be processed.
    endfname : string
        name of the ENDF-6 file to be moved in the library directory.
    proj : string
        kind of incident particle (e.g. "n", "pa", "pn").
    atom_relax : string, optional
        Name of atomic relaxation data sub-dir inside "libpath".
        The default is None.

    Returns
    -------
    success : bool
        True if processing and cleaning is successful, False otherwise.
    """
    success = True

    ASAIDT, ext = os.path.splitext(inp)

    ZAIDT0, MAT, natflag, iso = parseENDF6("tape20")
    ZAIDT1 = str(int(ZAIDT0)+100*iso)

    # attach temperature and ACE type extension
    temp = ASAIDT.split('_')[1]
    ZAIDT0 = "%s%s" % ('.'.join([ZAIDT0, temp]), pdict[proj])
    ZAIDT1 = "%s%s" % ('.'.join([ZAIDT1, temp]), pdict[proj])

    datapath = os.path.join(libpath, data)

    if atom_relax is not None:
        atom_relax_datapath = os.path.join(libpath, atom_relax)

    else:
        atom_relax_datapath = None

    pendfnames = ["tape26"] if binary else ["tape96"]

    # neutrons dict with names of files to be kept when cleaning files
    dir_names = {"n": {datapath: "tape20", "pendfdir_bin": pendfnames,
                       "acedir": "tape29", "xsdir": "tape30_1", "njoyout": "out.gz",
                       "viewheatdir": ["tape35"], "viewacedir": "tape34",
                       "njoyinp": [inp]},
                 "pa": {datapath: "tape20", atom_relax_datapath: "tape21",
                        "acedir": "tape29", "xsdir": "tape30_1",
                        "njoyout": "out.gz", "njoyinp": inp},
                 "pn": {datapath: "tape20", "pendfdir_bin": "tape22", "acedir":
                        "tape29", "xsdir": "tape30_1", "njoyout": "out.gz",
                        "viewacedir": "tape34", "njoyinp": inp}}

    if kerma:
        pendfnames = ["tape26", "tape56"] if binary else ["tape96", "tape67"]
        dir_names["n"]["viewheatdir"] = pendfnames
        dir_names["n"]["viewheatdir"] = ["tape35", "tape60"]
        dir_names["n"]["njoyinp"] = [inp, inpK]

    # neutrons dict with new names when moving files
    ext_names = {"n": {"tape20": endfname, "tape21": endfname,
                       "tape26": ".pendf", "tape96": ".pendf", "tape29": ".ace",
                       "tape30_1": ".xsdir", "out.gz": ".out.gz",
                       "tape34": ".eps", "tape35": ".eps", inp: ".njoyinp",
                       },
                 "pa": {"tape20": endfname, "tape21": endfname,
                        "tape29": ".ace", "tape30_1": ".xsdir",
                        "out.gz": ".out.gz", inp: ".njoyinp"},
                 "pn": {"tape20": endfname, "tape22": ".pendf",
                        "tape29": ".ace", "tape30_1": ".xsdir",
                        "out.gz": ".out.gz", "tape34": ".eps",
                        inp: ".njoyinp"}}
    if kerma:
        ext_names["n"][inpK] =  ".njoyinpK"
        ext_names["n"]["tape56"] = "_KERMA.pendf"
        ext_names["n"]["tape60"] = "_KERMA.eps"
        ext_names["n"]["tape67"] = "_KERMA.pendf"

    try:

        if kerma and (proj == 'n' or proj == 'pn'):
            fiss, ures = False, False
            warn = []
            # fix tape29 (ACE)
            with fileinput.FileInput("tape29", inplace=True) as tape29:
                for iline, line in enumerate(tape29):

                    if iline == 0:
                        print(line.replace(ZAIDT0, ZAIDT1), end='')

                    else:
                        print(line, end='')

                        if iline == 8:
                            cols = line.split()

                            if int(cols[1]) > 0:
                                fiss = True

                        elif iline == 10:
                            cols = line.split()

                            if int(cols[6]) > 0:
                                ures = True

            # additional MT data
            if fiss:
                # write MT458 data
                count = 0
                with open('tape29', 'a') as tape29:
                    with open('tape20') as tape20:
                        for iline, line in enumerate(tape20):
                            if line[71:75] == "1458":
                                tape29.write(line)
                                count = count + 1

                if count == 0:
                    warn.append("Warning: no MT458 data for %s" % ZAIDT1)

                # write local fission data
                count = 0
                with open('tape29', 'a') as tape29:
                    with open('tape66') as tape66:
                        for iline, line in enumerate(tape66):
                            if line[71:75] == "3318":
                                line = line[:71]+'3319'+line[75:]
                                tape29.write(line)
                                count = count + 1

                if count == 0:
                    warn.append("Warning: no local fission KERMA data for %s" % ZAIDT1)

            # write non-local KERMA data
            count = 0
            with open('tape29', 'a') as tape29:
                with open('tape67') as tape67:
                    for iline, line in enumerate(tape67):
                        if line[71:75] == "3301":
                            tape29.write(line)
                            count = count + 1

            if count == 0:
                warn.append("Warning: no non-local KERMA data for %s" % ZAIDT1)

            if fiss:
                # write non-local fission KERMA data
                count = 0
                with open('tape29', 'a') as tape29:
                    with open('tape67') as tape67:
                        for iline, line in enumerate(tape67):
                            if line[71:75] == "3318":
                                tape29.write(line)
                                count = count + 1

                if count == 0:
                    warn.append("Warning: no non-local fission KERMA data for %s" % ZAIDT1)

            # write ures data
            if ures:
                count = 0
                with open('tape29', 'a') as tape29:
                    with open('tape67') as tape67:
                        for iline, line in enumerate(tape67):
                            if line[71:75] == "2153":
                                tape29.write(line)
                                count = count + 1

                if count == 0:
                    warn.append("Warning: no additional ures data for KERMA" +
                                f" data for {ZAIDT1}")
            if warn == []:
                warn = None
        else:
            warn = []

    except FileNotFoundError as fnf:
        print(str(fnf))
        success = False

    try:
        # fix tape30 (xsdir)
        find_replace = {ZAIDT0: ZAIDT1, "filename": ASAIDT+".ace", "route": "0"}

        with open("tape30") as fold:
            with open("tape30_1", "w") as fnew:
                for line in fold:
                    for key in find_replace:
                        if key in line:
                            line = line.replace(key, find_replace[key])

                    # write new line in new file
                    fnew.write(line)

    except FileNotFoundError as fnf:
        print(str(fnf))
        success = False

    # loop over dictionaries
    for dirname, fnamelst in dir_names[proj].items():

        if type(fnamelst) is not list:
            fnamelst = [fnamelst]

        for fname in fnamelst:

            # other files in "out" tree
            if fname == "out.gz":
                # read output file
                with open('output', 'rb') as f_in:

                    with gzip.open('out.gz', 'wb') as f_out:
                        sh.copyfileobj(f_in, f_out)

                # define I/O path
                ipath = os.path.join(tmpath, fname)
                opath = os.path.join(path, dirname, ASAIDT+ext_names[proj][fname])

            elif fname != 'tape20' and fname != 'tape21':
                # define I/O path
                if fname is not None:
                    ipath = os.path.join(tmpath, fname)
                    opath = os.path.join(path, dirname, ASAIDT+ext_names[proj][fname])

            else:
                # ENDF-6 back to data dir
                ipath = os.path.join(tmpath, fname)
                opath = os.path.join(dirname, ext_names[proj][fname])

            try:
                # move and rename file
                sh.move(ipath, opath)

            except FileNotFoundError as fnf:
                print(str(fnf))
                success = False

    # clean base directory from other NJOY tapes
    f_del = glob.glob(os.path.join(tmpath, "tape*"))

    for f in f_del:
        os.remove(f)

    # remove output file
    os.remove("output")

    return success, warn


def run_njoy(inp):
    """
    Execute the Linux system command to run NJOY.

    NJOY must be correctly installed and should properly work on the machine OS
    Parameters
    ----------
    inp : string
        NJOY input file name with absolute path
        (e.g. "/home/user/njoyinput/H-001.njoyinp")

    Raises
    ------
    OSError
        -The specified NJOY version is not available.

    Returns
    -------
    outstream: string or None
        NJOY output stream. If no error occurs, outstream is None.
    """
    # split input name
    fname, ext = os.path.splitext(inp)

    # define NJOY command
    cmd = get_njoy()
    if "2016" in cmd:
        cmd = f"{cmd} < {inp}"
    elif "2021" in cmd:
        cmd = f"{cmd} -i {inp}"
    else:
        print("The specified NJOY version cannot be recognised!")
        raise OSError

    # execute NJOY in the shell
    p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE, close_fds=True)
    stream, stream_stderr = p.communicate()
    # decode in UTF-8
    stream, stream_stderr = stream.decode('utf-8'), stream_stderr.decode('utf-8')

    # check presence of warnings in stream
    warn_msg = "---message from consis---consistency problems found"

    # define outstream
    if stream_stderr != '':
        outstream = stream_stderr
    else:
        if re.search(warn_msg, stream) is not None:
            outstream = "warning"
        else:
            outstream = None

    return outstream


def make_input(datapath, pattern, part, libname, broad_temp=None, kerma=True,
              outpath=None, atomrelax_datapath=None, random=False,
              binary=True, newlibext=None):
    """
    Write the NJOY input to files for future processing.

    Parameters
    ----------
    datapath : string
        ENDF-6 files absolute path.
    pattern : string
        string describing the file name pattern (e.g. n-Z_AS_A.endf, where Z
        is a keyword for atomic number, AS a keyword for the atomic symbol
        and A a keyword for the mass number).
        List of keywords:
            -A : mass number
            -S : atomic symbol
            -Z : atomic number
            -N : file number (for perturbed files)
            -L : letters to be skipped
            -I : digits to be skipped
    part : string
        incident particle for the desired nuclear data (e.g. "n" or "neutron",
        "pa" or "photo-atomic", and so on...)
    libname : string
        complete library name (e.g. "ENDF-B/VII.1")
    broad_temp : list of int, optional
        List with integer values of temperatures [K] for Doppler broadening.
        The default is None.
    outpath : string, optional
        path where output files produced by this function (i.e. NJOY input
        files) are stored. The default is None. In this case, the output is
        stored in the working directory.
    atomrelax_datapath : string, optional
        path with atomic relaxation data for photo-atomic data.
        The default is None.

    Raises
    ------
    OSError
        -Atomic relaxation data path not provided for photo-atomic processing
        -ENDF-6 file names does not contain explicit Z or atomic symbol
        -ENDF-6 file names does not contain explicit mass number
    Returns
    -------
    None.
    """
    # get NJOY version
    cmd = get_njoy()
    njoyver = get_njoy_ver()
    # find pattern separators
    pattern, lib_extension = os.path.splitext(pattern)
    filesep = re.split(r"[ A Z S N L I]+", pattern)

    # join for using re.split later
    filesep = "|".join(filesep)

    # add escape character "\" in front of special character "-", if any
    filesep = filesep.replace("-", r"\-")

    # split according to separators
    keys = re.split(r"["+filesep+"]+", pattern)

    # squeeze out empty strings, if any
    keys = list(filter(None, keys))

    # check minimum split has been done or act otherwise
    len_keys = [len(k) for k in keys]

    if max(len_keys) > 1:
        # split each key checking its length
        newkeys = []
        for k in keys:
            if len(k) > 1:
                k = list(filter(None, re.split("", k)))
                newkeys.extend(k)
            else:
                newkeys.append(k)

        # define new keys
        keys = newkeys

    # make dictionary to identify keys position in filename
    patterndict = {s: ipos for ipos, s in enumerate(keys)}

    # replace A, AS and Z (if present) with \w+ for regex later use
    str_iterator = re.finditer(r"[A Z S N L I]+", pattern)
    str_pos = [val.span() for val in str_iterator]
    min_pos = min(str_pos, key=itemgetter(0))[0]
    max_pos = max(str_pos, key=itemgetter(1))[1]
    pattern = pattern[min_pos:max_pos]

    # store separator between AS, Z and A
    filesep = list(filter(None, re.split(r"[ A Z S N L I]+", pattern)))

    # join for using re.split later
    x = "|".join(filesep)
    mystr = re.search("([a-z]+)", x)
    if mystr is not None:
        for s in mystr.groups():
            x = x.replace(s, '|{}|'.format(s))
    # replace A, AS, Z and N (if present) with \w+ for regex later use
    filesep = x
    for what, which in replace_dict.items():
        pattern = pattern.replace(what, which)

    # define general pattern
    pattern = re.compile(pattern)

    # gather all files in datapath
    endfiles = [f for f in sorted(os.listdir(datapath))
                if isfile(join(datapath, f))
                if f.endswith(lib_extension)]

    # print warning for the user
    if endfiles == []:
        print("Warning: %s is empty!" % datapath)

    # define particle key
    for key, names in partdict.items():
        if part in names:
            proj = key

    # gather atomic relaxation ENDF-6 files in datapath
    if atomrelax_datapath is None and proj == "pa":

        raise OSError("Atomic relaxation data path not provided!")

    if atomrelax_datapath is not None:

        ar_endfiles = [f for f in sorted(os.listdir(atomrelax_datapath))
                       if isfile(join(atomrelax_datapath, f))
                       if f.endswith(lib_extension)]

    outpath = mkdir("njoyinp", outpath)

    # make projectile-wise dir
    outpath = mkdir(proj, outpath)

    # generates input only for files with AS or Z inside elementdict
    for ifile, endf in enumerate(endfiles):

        # split extension
        nuclname, lib_extension = os.path.splitext(endf)

        if "." in lib_extension:
            lib_extension = lib_extension.split(".")[-1]

        if lib_extension == '':
            lib_extension = 'endf6'
        if newlibext is not None:
            lib_extension = newlibext
        # 1st check if nuclide is metastable
        if re.search("([0-9]+)m", nuclname) is not None:
            metaflag = 1  # initial value of flag for metastable elements

        else:
            metaflag = None  # initial value of flag for metastable elements

        # split according to separators
        try:
            iS, iE = re.search(pattern, endf).span()
        except AttributeError:
            raise NDLError(f"Pattern '{pattern.pattern}' not found in file name '{endf}'!")

        nuclname = nuclname[iS:iE]
        if filesep != '':
            if 'endf' in filesep:
                keys = re.split(filesep, nuclname)
            else:
                keys = re.split(r"["+filesep+"]+", nuclname)
            # check keys consistency
            if len(keys) != len(patterndict.keys()):
                tmpkeys = []
                for k in keys:
                    _ = list(filter(None, re.split("(\\d+)", k)))
                    tmpkeys.extend(_)
                if len(tmpkeys) != len(patterndict.keys()):
                    raise OSError('Pattern name not recognised!')
                else:
                    keys = tmpkeys
        else:
            keys = list(filter(None, re.split("(\\d+)", nuclname)))

        # get atomic number Z and atomic symbol AS
        try:  # name contains atomic symbol explicitly

            AS = keys[patterndict["S"]].capitalize()  # get position with dict val
            try:
                N = keys[patterndict["N"]]
            except KeyError:
                N = None
            try:
                Z = ASZ_periodic_table()[AS]

            except KeyError:
                Z = -1  # if AS is not inside the dict, dummy value for Z

        except KeyError:  # name contains atomic number
            # get atomic number
            try:
                Z = keys[patterndict["Z"]]

            except KeyError:
                raise OSError("ENDF-6 file name does not contain neither" +
                              " Z nor the atomic symbol!")
            # get atomic symbol
            try:
                AS = ZAS_periodic_table()[Z]

            except KeyError:
                AS = ""  # if AS is not inside the dict, dummy value for AS
                Z = -1  # if AS is not inside the dict, dummy value for AS

        # neutron or photo-nuclear data
        if Z != -1 and (proj == "n" or proj == "pn"):

            # get mass number A
            try:
                A = keys[patterndict["A"]]
                # check if nuclide is metastable
                try:
                    int(A)

                except ValueError:
                    A = A.split("m")[0]

            except KeyError:
                raise OSError("ENDF-6 filename should contain mass number!")

            # define ASA (atomic symbol and mass number)
            if metaflag is None:
                ASA = AS+"-"+A

            else:
                ASA = AS+"-"+A+"m"

            # define basic ZAID (atomic and mass number)
            ZAID = f"{Z}{int(A):03d}"

            # parse MAT number from library file
            endf = os.path.join(datapath, endf)

            # parse ENDF to get MAT number and to check if natural isotope or isomer
            ZA, MAT, natflag, iso = parseENDF6(endf)

            if metaflag == 1 and iso == 0:
                raise OSError(f"Name and isomer state conflict in {endf}")

            if ZA != ZAID and natflag != 1:
                raise OSError(f"Name and file ZAID conflict in {endf}")

            # rename ENDF-6 file with PoliTo nomenclature
            endfnewname = '{}-{}'.format(ASA, N) if N else '{}'.format(ASA)
            endfnewname = '{}.{}'.format(endfnewname, lib_extension) if lib_extension else endfnewname
            endfnewname = os.path.join(datapath, endfnewname)
            if endf != endfnewname:
                sh.move(endf, endfnewname)

            # generate njoy input file
            for tmp in broad_temp:  # loop over temperatures
                # print message for the user
                print("Building input file for %s at T=%s K..." % (endf, tmp))

                # define file content
                njoyinp = build_njoy_deck(MAT, ASA, proj, libname, njoyver, tmp=tmp, binary=binary)
                T = int(tmp/100)
                basename = "{}-{}_{:02d}".format(ASA, N, T) if N else "{}_{:02d}".format(ASA, T)

                if outpath is not None:
                    fname = os.path.join(outpath, basename)

                f = open(fname+".njoyinp", "w")
                f.write(njoyinp)
                f.close()
                print("DONE \n")

                if kerma:  # generate additional input
                    # print message for the user
                    print("Building additional input file for %s at T=%s K..."
                          % (endf, tmp))

                    # define file content
                    njoyinp = build_njoy_deck(MAT, ASA, proj, libname, njoyver,
                                              tmp=tmp, kerma=kerma)

                    if outpath is not None:
                        fname = os.path.join(outpath, basename)

                    f = open(fname+".njoyinpK", "w")
                    f.write(njoyinp)
                    f.close()
                    print("DONE \n")

        # photo-atomic data
        elif Z != -1 and proj == "pa":

            endf = os.path.join(datapath, endf)

            # parse ENDF to get MAT number and to check if natural isotope or isomer
            ZA, MAT, natflag, iso = parseENDF6(endf)

            # rename ENDF-6 file with PoliTo nomenclature
            sh.move(endf, os.path.join(datapath, AS+lib_extension))

            # rename also atomic relaxation ENDF-6 files
            ar_endf = os.path.join(atomrelax_datapath, ar_endfiles[ifile])
            sh.move(ar_endf, os.path.join(atomrelax_datapath, AS+lib_extension))

            print("Building input file for %s..." % (endf))

            # define file content
            njoyinp = build_njoy_deck(MAT, AS, proj, libname, njoyver)

            fname = AS+"_00"
            # define complete path
            if outpath is not None:
                fname = os.path.join(outpath, fname)

            f = open(fname+".njoyinp", "w")
            f.write(njoyinp)
            f.close()

            print("DONE \n")

        else:  # dummy value of Z means fake element

            print("Skipping %s-%s. It is not an element." % (AS, Z))


def build_njoy_deck(MAT, ASA, proj, libname, vers, tmp=None, kerma=None, binary=True):
    """
    Build the NJOY deck for processing.

    Parameters
    ----------
    MAT : string
        MAT number describing the nuclide inside the ENDF-6 file.
        It is read from the file itself (user should not provide it).
        The default is None.
    ASA : string
        Atomic symbol.
    proj : string
        incident particle for the desired data (e.g. "n", "pa", "pn").
    libname : string
        complete library name (e.g. "ENDF-B/VII.1")
    vers : string, optional
        NJOY version number (only 2016 and 2021 are supported).
        The default is 2016.
    tmp : int
        Temperature for Doppler broadening. The default is None.
    binary : bool, optional
        Flag to convert PENDF in ASCII. Default is ``False``.

    Returns
    -------
    outstr : string
        string that is passed to master function in order to write the NJOY
        input file
    """
    lst = []
    lstapp = lst.append

    # define temperature suffix
    if tmp is not None:
        tmpsuff = "{:02d}".format(int(tmp/100))

    now = datetime.now()
    now = now.strftime("%d/%m/%Y, %H:%M:%S")

    hostname = gethostname()

    if proj == "n" and kerma is None:  # fast (continuous-energy) neutron data

        # MODER module
        lstapp("moder")
        lstapp("1 -21/")  # convert tape20 in block-binary mode for efficiency
        lstapp(f"'{ASA}'/")
        lstapp(f"20 {MAT}/")
        lstapp("0/")

        # RECONR module
        lstapp("reconr")
        lstapp("-21 -22/")
        lstapp(f"'{ASA} PENDF'/")
        lstapp(f"{MAT} 2/")
        lstapp("0.001 0.0 0.01 5.0e-7/")
        lstapp("''/")
        lstapp("''/")
        lstapp("0/")

        # BROADR module
        lstapp("broadr")
        lstapp("-21 -22 -23/")
        lstapp(f"{MAT} 1/")
        lstapp("0.001 2.0e6 0.01 5.0e-7/")
        lstapp(f"{tmp:12.5e}/")
        lstapp("0/")

        # HEATR module (local deposition)
        lstapp("heatr")
        lstapp("-21 -23 -24 40/")
        lstapp(f"{MAT} 7 0 0 1 2/")
        lstapp("302 303 304 318 401 443 444/")

        # GASPR module
        lstapp("gaspr")
        lstapp("-21 -24 -25/")

        # PURR module
        lstapp("purr")
        lstapp("-21 -25 -26/")
        lstapp(f"{MAT} 1 1 20 64/")
        lstapp(f"{tmp:12.5e}/")
        lstapp("1.0e10/")
        lstapp("0/")

        # MODER module
        if binary is False:
            lstapp("moder")
            lstapp("-26 96/")
        # ACER module
        lstapp("acer")
        lstapp("-21 -26 0 27 28/")
        lstapp(f"1 1 1 .{tmpsuff}/")
        lstapp(f"'{ASA}, {libname}, NJOY{vers}, {hostname} {now}'/")
        lstapp(f"{MAT} {tmp:12.5e}/")
        lstapp("1 1/")
        lstapp("/")
        lstapp("acer")  # re-run for QA checks
        lstapp("0 27 33 29 30/")
        lstapp(f"7 1 1 .{tmpsuff}/")
        lstapp("''/")

        # VIEWR module
        lstapp("viewr")
        lstapp("33 34/")  # plot ACER output
        lstapp("viewr")
        lstapp("40 35/")  # plot HEATR output

    elif proj == "n" and kerma:

        # HEATR module (gamma transport)
        lstapp("heatr")
        lstapp("-21 -23 -54 60/")
        lstapp(f"{MAT} 7 0 0 0 2/")
        lstapp("302 303 304 318 401 443 444/")

        # PURR module
        lstapp("purr")
        lstapp("-21 -54 -56/")
        lstapp(f"{MAT} 1 7 20 64/")
        lstapp(f"{tmp:12.5e}/")
        lstapp("1.0e10 1.0e5 1.0e4 1.0e3 1.0e2 1.0e1 1/")
        lstapp("0/")

        # MODER module
        lstapp("moder")
        lstapp("-26 66/")
        lstapp("moder/")
        lstapp("-56 67/")

    elif proj == "pa":  # photo-atomic data
        # ACER module
        lstapp("acer")
        lstapp("20 21 0 29 30/")
        lstapp("4 1 1 .00/")
        lstapp(f"'{ASA}, {libname}, NJOY{vers}, {hostname} {now}'/")
        lstapp(f" {MAT}/")
        # no QA checks, no ACER plot available for this kind of data (2020)

    elif proj == "pn":  # photo-nuclear data
        # FIXME: NJOY docs do not contain any pn complete example

        # MODER module
        lstapp("moder")
        lstapp("20 -21/")  # convert tape20 in block-binary mode for efficiency

        # RECONR module
        lstapp("reconr")
        lstapp("-21 -22/")
        lstapp("/")
        lstapp(f"{MAT} 1 0/")
        lstapp("0.001 0./")
        lstapp("/")
        lstapp("0/")

        # ACER module
        lstapp("acer")
        lstapp("-21 -22 0 27 28/")
        lstapp(f"5 0 1 .{tmpsuff}/")
        lstapp(f"'{ASA}, {libname}, NJOY{vers}, {hostname} {now}'/")
        lstapp(f"{MAT}/")
        lstapp("acer")  # re-run for QA checks
        lstapp("0 27 33 29 30/")
        lstapp(f"7 1 1 .{tmpsuff}/")
        lstapp("/")

        # VIEWR module
        lstapp("viewr")
        lstapp("33 34/")  # plot ACER output

    # add STOP card in whatever input deck
    lstapp("stop")
    # concatenate list with "\n" into a string
    outstr = "\n".join(lst)
    return outstr


def convert_xsdir(datapath, proj, libname, ndlpath, currpath=None):
    '''
    Merge all .xsdir files and convert to .xsdata file.

    Parameters
    ----------
    datapath : str
        Path where .xsdir are located.
    proj : list[str]
        Incident particle.
    libname : str
        Library name.
    ndlpath : str
        Library path.
    currpath : str, optional
        Current working directory path. The default is None.

    Returns
    -------
    None.

    '''
    if currpath is None:
        pwd = os.getcwd()

    else:
        pwd = currpath

    # loop over projectiles
    for p in proj:
        # define paths
        datastr = "datapath=%s" % "/".join([ndlpath, p])
        fname = f'sss2_{libname}_{p}.xsdir'
        fname = os.path.join(pwd, fname)
        xsdirpath = os.path.join(datapath, p, "xsdir")

        # insert datapath in xsdir_header
        headname = os.path.join(pwd, 'xsdir_header')
        header = _insert_top_lines_str(datastr, headname)

        # collect .xsdir files
        xsdirfiles = [os.path.join(xsdirpath, f) for f in
                      sorted(os.listdir(xsdirpath))
                      if isfile(os.path.join(xsdirpath, f))
                      if f.endswith(".xsdir")]

        # merge .xsdir files
        _mergefiles(fname, xsdirfiles)

        # insert header on top of merged .xsdir files
        xsdirlines = _insert_top_lines_str(header, fname)
        with open(fname, 'w') as f:
            f.write(xsdirlines)

        # run xsdirconvert.pl utility (by VTT)
        xsdataname = f'sss2_{libname}_{p}.xsdata'
        cmd = f"./xsdirconvert.pl {fname} > {xsdataname}"
        p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE, close_fds=True)
        stream, stream_stderr = p.communicate()
        # decode in UTF-8
        stream, stream_stderr = stream.decode('utf-8'), stream_stderr.decode('utf-8')
        # print shell output
        print("/n".join([stream, stream_stderr]))

    # merge projectiles files
    xsdatalist = [os.path.join(pwd, f) for f in sorted(os.listdir(pwd))
                  if isfile(os.path.join(pwd, f))
                  if f.endswith(".xsdata")]

    xsdirlist = [os.path.join(pwd, f) for f in sorted(os.listdir(pwd))
                 if isfile(os.path.join(pwd, f))
                 if f.endswith(".xsdir")]

    xsdatafname = f'sss2_{libname}.xsdata'
    xsdirfname = f'sss2_{libname}.xsdir'

    _mergefiles(xsdatafname, xsdatalist)
    _mergefiles(xsdirfname, xsdirlist)

    p1 = os.path.join(pwd, xsdatafname)
    p2 = os.path.join(ndlpath, xsdatafname)

    sh.move(p1, p2)

    p1 = os.path.join(pwd, xsdirfname)
    p2 = os.path.join(ndlpath, xsdirfname)

    sh.move(p1, p2)

    # move and clean files
    f_del = glob.glob(os.path.join(pwd, "*.xs*"))
    for f in f_del:
        os.remove(f)


def _insert_top_lines_str(datastr, fname):
    # read header file content
    with open(fname, 'r') as fh:
        temp = fh.read()
    # append file lines to string
    newstr = "\n".join([datastr, temp])
    return newstr


def _mergefiles(fname, filelist):
    with open(fname, 'w') as wfd:
        for f in filelist:
            with open(f, 'r') as fd:
                sh.copyfileobj(fd, wfd)


def ASZ_periodic_table():
    """
    Define a dictionary with atomic symbol and atomic number.

    Returns
    -------
    periodictable : dict
        key: atomic symbol AS.
        value: atomic number Z.
    """
    # define atomic symbols list
    AS = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg',
          'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr',
          'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
          'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',
          'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La',
          'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er',
          'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',
          'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
          'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
          'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn',
          'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
    # atomic number
    Z = np.arange(1, 119)
    # define dict
    periodictable = dict(zip(AS, Z))
    # return element dict
    return periodictable


def parseENDF6(fname):
    """Parse ENDF-6 format file to check Z, A, MAT, natural and isomeric flags."""
    linepos = -10
    # determine enconding (not all files are UTF-8)
    with open(fname, "rb") as fenc:
        encoding = chardet.detect(fenc.read())["encoding"]

    with open(fname, encoding=encoding, errors='ignore') as f:

        for iline, line in enumerate(f):

            if line[71:75] == "1451" and linepos == -10:
                # read MAT number inside ENDF-6 format file
                MAT = line[66:70]

                # parse Z and A
                line1 = line[0:11]

                # remove exponential symbol, if any
                line1 = line1.replace('e', '').replace('E', '')
                line1 = line1.split("+")

                if len(line1) == 1:
                    line1 = line1[0].split(".")

                ZA = int(float("%sE+%s" % (line1[0], line1[1])))
                Z = int(ZA/1e3)
                A = int(ZA-int(Z*1e3))

                # check if natural element
                if A == 0:
                    natflag = 1

                else:
                    natflag = 0

                linepos = iline

            elif iline == linepos+1:

                # parse exitation energy and isomeric state
                line = line.split()
                ext_en = float(line[0].replace('+', 'E+'))
                iso_st = int(line[3])

                # check consistency
                if ext_en > 0 and iso_st == 0:
                    raise ValueError("Exitation energy > 0 for" +
                                     "non-isomeric state in %s." % fname)

                # set isomer multiplier
                if iso_st > 0:
                    if A > 200:
                        iso = 1

                    elif A > 100:
                        iso = 2

                    else:
                        iso = 3

                else:  # if iso_st = 0:
                    iso = 0

                break

            else:
                continue

    return str(ZA), MAT, natflag, iso


def ZAS_periodic_table():
    """
    Define a dictionary with atomic number and atomic symbol.

    Returns
    -------
    periodictable : dict
        key: atomic number Z.
        value: atomic symbol AS.
    """
    # define atomic symbols list
    AS = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg',
          'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr',
          'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
          'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',
          'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La',
          'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er',
          'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',
          'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
          'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
          'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn',
          'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
    # atomic number
    Z = np.arange(1, 119)
    # define dict
    periodictable = dict(zip(Z, AS))
    # return element dict
    return periodictable


def mkdir(dirname, path):
    """
    Make a new directory named dirname inside path.

    Parameters
    ----------
    dirname : string
        directory name
    path : string
        path where the new directory is created.

    Returns
    -------
    None.
    """
    if path is None:
        path = dirname
    else:
        path = os.path.join(path, dirname)
    os.makedirs((path), exist_ok=True)
    return path


def printime(start_time):
    """
    Print the elapsed time in s/m/h.

    Parameters
    ----------
    start_time : float
        initial time.

    Returns
    -------
    elaps : string
        Elapsed time
    """
    dt = t.time() - start_time
    if dt < 60:
        elaps = "Elapsed time %f s." % dt
    elif dt >= 60:
        elaps = "Elapsed time %f m." % (dt/60)
    elif dt >= 3600:
        elaps = "Elapsed time %f h." % (dt/3600)

    return elaps


def _parse_endf_ctl_flexible(line):
    """
    Parse ENDF control fields from tail.
    Supports:
      - 80-col lines: MAT/MF/MT/NS in cols 67-80 (1-based) => [66:80]
      - 75-col lines: MAT/MF/MT in cols 67-75 (1-based) => [66:75], NS missing

    Returns (mat, mf, mt, ns_or_None) or None if not parseable.
    """
    n = len(line)

    if n >= 80:
        tail = line[66:80]
        try:
            mat = int(tail[0:4])
            mf = int(tail[4:6])
            mt = int(tail[6:9])
            ns = int(tail[9:14])
            return mat, mf, mt, ns
        except Exception:
            return None

    if n >= 75:
        tail = line[66:75]
        try:
            mat = int(tail[0:4])
            mf = int(tail[4:6])
            mt = int(tail[6:9])
            return mat, mf, mt, None
        except Exception:
            return None

    return None


def _fmt_endf_line(data66, mat, mf, mt, ns):
    """
    Create a canonical 80-column ENDF line (plus newline):
      cols 1-66:  data/comment (padded)
      cols 67-70: MAT (4)
      cols 71-72: MF  (2)
      cols 73-75: MT  (3)
      cols 76-80: NS  (5)
    """
    data = (data66[:66]).ljust(66)
    tail = f"{mat:4d}{mf:2d}{mt:3d}{ns:5d}"
    return f"{data}{tail}\n"


def build_burnup_data(in_dir, out_dir=None, lib="endfb8", suffix="dec", 
                        pattern="*.endf", recursive=False, encoding=None, skip_rev_lines=True, drop_unparsed=True,):
    """
    Build Serpent burnup auxiliary data tapes (.dec/.nfy/.sfy) by merging ENDF-6
    single-material files.

    Parameters
    ----------
    in_dir : str
        Directory containing ENDF files to merge.
    out_dir : str, optional
        Directory where the merged file is written. If None, uses in_dir.
    lib : str, optional
        Output library tag used in file name, e.g. "endfb8".
    suffix : str, optional
        Output suffix: "dec", "nfy" or "sfy".
    pattern : str, optional
        Glob pattern to match input files, default "*.endf".
    recursive : bool, optional
        If True, search recursively for pattern.
    encoding : str, optional
        File encoding for reading. If None, tries utf-8 then latin-1.
    skip_rev_lines : bool, optional
        If True, skip lines whose first non-space character is '$' (e.g. $Rev::).
    drop_unparsed : bool, optional
        If True, drop lines that cannot be parsed for MAT/MF/MT.
        (Recommended, because output must be consistent ENDF control-wise.)

    Returns
    -------
    out_path : str
        Absolute path of merged output file.

    Raises
    ------
    NDLError
        If no input files are found or suffix is invalid.
    """
    suffix = str(suffix).lower().strip()
    if suffix not in ("dec", "nfy", "sfy"):
        raise NDLError("suffix must be one of: 'dec', 'nfy', 'sfy'")

    in_dir = os.path.abspath(in_dir)
    if out_dir is None:
        out_dir = in_dir
    else:
        out_dir = os.path.abspath(out_dir)
        os.makedirs(out_dir, exist_ok=True)

    if recursive:
        files = sorted(glob.glob(os.path.join(in_dir, "**", pattern), recursive=True))
    else:
        files = sorted(glob.glob(os.path.join(in_dir, pattern), recursive=False))

    files = [f for f in files if os.path.isfile(f)]
    if len(files) == 0:
        raise NDLError(f"No input files found in '{in_dir}' matching pattern '{pattern}'")

    ZERO66 = " 0.000000+0 0.000000+0          0          0          0          0"

    def is_send(mat, mf, mt): return (mat > 0) and (mf > 0) and (mt == 0)
    def is_fend(mat, mf, mt): return (mat > 0) and (mf == 0) and (mt == 0)
    def is_mend(mat, mf, mt): return (mat == 0) and (mf == 0) and (mt == 0)
    def is_tend(mat, mf, mt): return (mat == -1) and (mf == 0) and (mt == 0)

    ns_counter = {}  # key=(MAT, MF, MT) => running count
    out_lines = []

    # simple encoding fallback consistent with typical ENDF text
    encodings_to_try = [encoding] if encoding else ["utf-8", "latin-1"]

    dropped_tend = 0
    skipped_rev = 0
    unparsed = 0
    kept = 0

    for fp in files:
        text = None
        last_err = None
        for enc in encodings_to_try:
            try:
                with open(fp, "r", encoding=enc, errors="strict") as fh:
                    text = fh.read()
                break
            except Exception as e:
                last_err = e
        if text is None:
            raise NDLError(f"Cannot read '{fp}' with encodings {encodings_to_try}: {last_err}")

        for raw in text.splitlines(True):
            line = raw.rstrip("\n")

            if not line.strip():
                continue

            if skip_rev_lines and line.lstrip().startswith("$"):
                skipped_rev += 1
                continue

            ctl = _parse_endf_ctl_flexible(line)
            if ctl is None:
                unparsed += 1
                if drop_unparsed:
                    continue
                else:
                    # keeping unparsed lines would break control structure; by default we drop
                    continue

            mat, mf, mt, _ns = ctl

            # drop per-file TEND
            if is_tend(mat, mf, mt):
                dropped_tend += 1
                continue

            # normalize termination records
            if is_send(mat, mf, mt):
                out_lines.append(_fmt_endf_line(ZERO66, mat, mf, 0, 99999))
                kept += 1
                continue
            if is_fend(mat, mf, mt):
                out_lines.append(_fmt_endf_line(ZERO66, mat, 0, 0, 0))
                kept += 1
                continue
            if is_mend(mat, mf, mt):
                out_lines.append(_fmt_endf_line(ZERO66, 0, 0, 0, 0))
                kept += 1
                continue

            # normal record: renumber NS 1..N per (MAT,MF,MT)
            key = (mat, mf, mt)
            ns_counter[key] = ns_counter.get(key, 0) + 1
            new_ns = ns_counter[key]

            out_lines.append(_fmt_endf_line(line[:66], mat, mf, mt, new_ns))
            kept += 1

    # final single TEND
    out_lines.append(_fmt_endf_line(ZERO66, -1, 0, 0, 0))

    out_path = os.path.join(out_dir, f"{lib}.{suffix}")
    with open(out_path, "w", encoding="utf-8", errors="strict") as fh:
        fh.write("".join(out_lines))

    # optional: lightweight info for the caller (no print by default)
    return out_path


class NDLError(Exception):
    pass
