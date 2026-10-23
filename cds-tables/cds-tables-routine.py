"""Generate CDS tables for VELOCE-III.

Routine that generates the CDS tables for the VELOCE-III paper complete with
the readme file.
"""  # noqa: INP001

import os
import re
from pathlib import Path

import cdspyreadme
import numpy as np
from astropy.io import fits
from astropy.table import MaskedColumn, Table

__PATH_TO_RESULTS_VELOCEIII__ = (
    Path(os.environ.get("RESULTS_DIR", "")).expanduser() / "veloce_III"
)


# README ------------------------------------------------------------------------------
title = "VELOCE-III. Reconstructing radial velocity curves of classical Cepheids"
author = "Viviani+"
authors = ["Giordano Viviani", "Richard I. Anderson"]
authors_str = ", ".join(authors)
date = "2026"
abstract = """We present a novel framework for accurately reconstructing radial
velocity (RV) curves of classical Cepheids (Cepheids) from sparsely sampled
time-series data suitable for application in large spectroscopic surveys. The
framework provides a prior for the principal components of RV curves
established based on high-precision measurements from the VELOcities of
CEpheids (VELOCE) project; template RV curves of Cepheids can be readily
extracted from our results. We demonstrate the ability of our framework to
estimate unbiased pulsation average velocities, v_ gamma, to within 20-30 m/s,
and peak-to-peak amplitudes, P2P, to within ~1.1%. Subsampling the initial data
set, we show that v_ gamma and P2P can be determined to within ~500 m/s and 4%,
respectively, from as few as three observations drawn at random pulsation
phases, with the phase of minimum radius fitted rather than assumed. As
expected, the precision on both v_gamma and P2P improves markedly with the
number of observations, N_RV; the precision on P2P stabilizes beyond
N_RV~10-15, whereas that on v_gamma continues to improve. We fitted existing
time-series RV data of Cepheids in the Large and Small Magellanic Clouds (LMC, SMC)
using this framework and obtained typical root mean square errors of 0.5-2.0
km/s. The typical total uncertainty on v_gamma achieved for the SMC Cepheids is
~0.84 km/s, providing sensitivity to spectroscopic binaries (SB). We identified
8 SB1 systems; two and one of which are new detections in the LMC and SMC,
respectively. This yields a single-lined SB fraction of ~25% and 27% in the two
galaxies, similar to the Milky Way's SB fraction of 29%, established as part of
VELOCE. Despite their relatively small number, LMC Cepheids reproduce the known
line-of-sight component of the LMC's large-scale rotation, which differs in the
extremes by more than 80km/s. The kinematics of the SMC are more complex and
not sufficiently sampled by the available Cepheids. Our framework is designed
to yield accurate v_gamma and P2P of Cepheids observed by large spectroscopic
surveys, such as 4MOST, SDSS-V, and others, and will unlock new insights into
the kinematics and multiplicity of evolved intermediate-mass stellar
populations.
"""
keywords = """Stars: oscillations - Stars: variables: Cepheids -
Techniques: radial velocities"""

cds_catalogue = "J/A+A/"

references = {
    "Anderson et al. (2024) (VELOCE-I)": "2024A&A...686A.177A",
    "Storm et al. (2011)": "2011A&A...534A..95S",
    "Gieren et al. (2018)": "2018A&A...620A..99G",
}

description = """A principal component analysis (PCA) was applied to the
Fourier-series models of the radial velocity (RV) curves of 219 Galactic
classical Cepheids from VELOCE DR1 (Anderson et al., 2024A&A...686A.177A),
sampled on a uniform grid of 1000 phase points in [0,1), with phase 0 at
minimum radius. The PCA was computed on a randomly selected training set 
(85% of the stars) after subtracting the mean training-set curve, Vmean; the
remaining stars form the test set. The first six principal components (PCs),
which explain ~99.2% of the variance, were retained. An RV curve is then
modelled as V(phi) = vgamma + Vmean(phi+dphi) + sum_{i=1}^{6} p_i
PC_i(phi+dphi). File pca.dat lists Vmean and the six PCs on the phase grid.
File coeffs.dat lists, for each Cepheid, log P, the training/test membership,
the PC coefficients p_i obtained by fitting its VELOCE-I model, and the phase
offset dphi between the Gaia DR3 G-band reference epoch and minimum radius.
File kde.dat lists the bandwidths and domains of the kernel density estimates
(KDE) that defines the priors P(p_1,...,p_6|log P) and P(dphi|log P). The
priors can be rebuilt exactly from the training-set rows of coeffs.dat and
kde.dat. File tableb1.dat (Table B.1) lists the results of applying the
framework to literature RV time series of 22 LMC Cepheids (Storm et al.,2011A&A...534A..95S)
and 24 SMC Cepheids (Gieren et al., 2018A&A...620A..99G)."""


# Initialize object
tablemaker = cdspyreadme.CDSTablesMaker()

# Add info to the readme file
tablemaker.title = title
tablemaker.author = author
tablemaker.authors = authors_str
tablemaker.date = date
tablemaker.catalogue = cds_catalogue
tablemaker.abstract = abstract
tablemaker.keywords = keywords
tablemaker.more_description = description

for ref, bibcode in references.items():
    tablemaker.putRef(bibcode, title=ref)

# Read Fits file ----------------------------------------------------------------------
path_to_file = __PATH_TO_RESULTS_VELOCEIII__ / "veloce_III_framework.fits"
NPC = 6
with fits.open(path_to_file) as hdul:
    hdul.info()

    # Header
    # pyrefly: ignore [missing-attribute]
    n_components = hdul["PRIMARY"].header["NCOMP"]
    # pyrefly: ignore [missing-attribute]
    pca = np.array(hdul["PCA"].data, dtype=float)  # (7, 1000)
    # pyrefly: ignore [missing-attribute]
    evr = [hdul["PCA"].header[f"EVAR{i:02d}"] for i in range(1, NPC + 1)]
    # pyrefly: ignore [missing-attribute]
    phase = np.array(hdul["PHASE_GRID"].data, dtype=float)  # (1000,)
data = Table.read(path_to_file, hdu="DATA")
kde_pc = Table.read(path_to_file, hdu="KDE_PC")
kde_ph = Table.read(path_to_file, hdu="KDE_PHASE")

# sanity checks: catch a wrong row/axis convention before writing
assert pca.shape == (NPC + 1, phase.size)
gram = pca[1:] @ pca[1:].T
assert np.allclose(gram, np.eye(NPC), atol=1e-10), "PCs not orthonormal"
dphase = np.diff(phase)
assert phase[0] == 0 and np.allclose(dphase, 1 / phase.size), (
    "expected np.linspace(0, 1, N, endpoint=False)"
)

# pca.dat -----------------------------------------------------------------------------
t_pca = Table()
t_pca["Phase"] = np.round(phase, 3) + 0.0  # +0.0 removes -0.0
t_pca["Vmean"] = np.round(pca[0], 6) + 0.0
t_pca["Vmean"].unit = "km/s"
for i in range(1, NPC + 1):
    t_pca[f"PC{i}"] = np.round(pca[i], 9) + 0.0  # dimensionless

# coeffs.dat --------------------------------------------------------------------------
t_co = Table()
t_co["Name"] = [str(n).strip() for n in data["NAME"]]
t_co["Set"] = [str(s).strip() for s in data["SET"]]
t_co["logP"] = np.round(np.asarray(data["LOG_P"], float), 6)
for i in range(1, NPC + 1):
    t_co[f"p{i}"] = np.round(np.asarray(data[f"PC{i}"], float), 4)
    t_co[f"p{i}"].unit = "km/s"

v = np.asarray(
    data["DELTA_PHASE"].filled(np.nan)
    if hasattr(data["DELTA_PHASE"], "filled")
    else data["DELTA_PHASE"],
    float,
)
t_co["dphi"] = MaskedColumn(np.round(v, 6) + 0.0, mask=~np.isfinite(v))

t_co.sort("Name")

# kde.dat -----------------------------------------------------------------------------
lab = {"LOG_P": "logP", "DELTA_PHASE": "dphi"}
lab.update({f"PC{i}": f"p{i}" for i in range(1, NPC + 1)})
unit = {"logP": "[d]", "dphi": "-"}
unit.update({f"p{i}": "km/s" for i in range(1, NPC + 1)})
rows = [("PC", k) for k in kde_pc] + [("dphi", k) for k in kde_ph]
t_kde = Table()
t_kde["Prior"] = [p for p, _ in rows]
t_kde["Var"] = [lab[str(k["VAR_NAME"]).strip()] for _, k in rows]
t_kde["Unit"] = [unit[v] for v in t_kde["Var"]]
for src, dst in (("BANDWIDTH", "bw"), ("DOMAIN_LO", "Min"), ("DOMAIN_HI", "Max")):
    t_kde[dst] = [float(k[src]) for _, k in rows]


# =====================================================================================
# Creating CDS files
# =====================================================================================
def describe(table, text):
    for name, desc in text.items():
        table.get_column(name).description = desc


# 1) pca.dat
c1 = tablemaker.addTable(
    t_pca,
    name="pca.dat",
    description="Mean RV curve and principal components",
)
describe(
    c1,
    {
        "Phase": "Pulsation phase (phase 0 at minimum radius) (1)",
        "Vmean": "Mean RV curve of the training set (2)",
        **{f"PC{i}": f"Principal component {i} (3)" for i in range(1, 7)},
    },
)
c1.get_column("Phase").set_format("F5.3")
c1.get_column("Vmean").set_format("F10.6")
for i in range(1, NPC + 1):
    c1.get_column(f"PC{i}").set_format("F12.9")

# 2) coeffs.dat
c2 = tablemaker.addTable(
    t_co,
    name="coeffs.dat",
    description="PC coefficients of the VELOCE-I Cepheids",
)
describe(
    c2,
    {
        "Name": "Cepheid name, as in VELOCE-I",
        "Set": "[test/train] PCA training or test set (4)",
        "logP": "Log of the pulsation period (VELOCE-I)",
        **{f"p{i}": f"Coefficient of PC{i} (5)" for i in range(1, 7)},
    },
)
describe(
    c2,
    {"dphi": "Phase offset between Gaia DR3 G-band epoch and minimum radius (6)"},
)
c2.get_column("logP").unit = "[d]"
c2.get_column("logP").set_format("F8.6")
for i in range(1, NPC + 1):
    c2.get_column(f"p{i}").set_format("F9.4")
c2.get_column("dphi").set_format("F9.6")
# 3) kde.dat
c3 = tablemaker.addTable(
    t_kde,
    name="kde.dat",
    description="Bandwidths and domains of the KDE priors",
)
describe(
    c3,
    {
        "Prior": "[PC/dphi] Prior to which the row applies (7)",
        "Var": "Variable of the KDE (column label in coeffs.dat)",
        "Unit": "Unit of bw, Min and Max (9)",
        "bw": "Gaussian kernel bandwidth (cv_ml)",
        "Min": "Lower bound of the prior domain (8)",
        "Max": "Upper bound of the prior domain (8)",
    },
)
for c in ("bw", "Min", "Max"):
    c3.get_column(c).set_format("E15.8")

# 4) table-b1
tb = Table.read(__PATH_TO_RESULTS_VELOCEIII__ / "tableb1.csv", format="ascii.csv")
# CSV column (tableb1_export.py) -> CDS label, unit, decimals, format
spec = {
    "ID": ("Name", None, None, None),
    "Gal": ("Gal", None, None, None),
    "Ref": ("Ref", None, None, None),
    "mean_rv": ("vgamma", "km/s", 2, "F6.2"),
    "mean_rv_err": ("e_vgamma", "km/s", 2, "F4.2"),
    "mean_rv_p2p": ("P2Pvgcl", "km/s", 2, "F5.2"),
    "P2P": ("P2P", "km/s", 2, "F5.2"),
    "RMSE": ("RMSE", "km/s", 2, "F4.2"),
    "RMSE_rel": ("RMSE/P2P", "%", 1, "F3.1"),
    "SB1": ("SB1", None, None, None),
}
missing = set(spec) - set(tb.colnames)
assert not missing, f"tableb1.csv lacks columns {missing}"
tb = tb[list(spec)]
for col, (label, unit_, ndec, _) in spec.items():
    if ndec is not None:
        v = np.asarray(
            tb[col].filled(np.nan) if hasattr(tb[col], "mask") else tb[col],
            float,
        )
        tb[col] = MaskedColumn(np.round(v, ndec) + 0.0, mask=~np.isfinite(v))
        tb[col].unit = unit_
    else:
        tb[col] = [str(x).strip() for x in tb[col]]
    tb.rename_column(col, label)
assert set(tb["SB1"]) <= {"Y", "N"} and set(tb["Gal"]) <= {"LMC", "SMC"}
# P2Pvgcl must be given exactly for the SB1 stars
assert np.array_equal(~tb["P2Pvgcl"].mask, tb["SB1"] == "Y")

c4 = tablemaker.addTable(
    tb,
    name="tableb1.dat",
    description="RV curve properties of LMC and SMC Cepheids (Table B.1)",
)
describe(
    c4,
    {
        "Name": "Cepheid identifier (10)",
        "Gal": "[LMC/SMC] Host galaxy",
        "Ref": "Source of the RV time series (10)",
        "vgamma": "Pulsation-averaged (systemic) velocity (11)",
        "e_vgamma": "Uncertainty on vgamma",
        "P2Pvgcl": "Peak-to-peak variation of the cluster-averaged vgamma (11)",
        "P2P": "Peak-to-peak pulsation amplitude of the model",
        "RMSE": "Root-mean-square error of the fit",
        "RMSE/P2P": "Ratio of RMSE to P2P",
        "SB1": "[YN] Evidence of orbital motion (11)",
    },
)
for _col, (label, _, ndec, fmt) in spec.items():
    if fmt:
        # widen automatically if a value needs more integer digits
        w = max(
            int(fmt[1:].split(".")[0]),
            *(len(f"{x:.{ndec}f}") for x in tb[label].compressed()),
        )
        c4.get_column(label).set_format(f"F{w}.{ndec}")

tablemaker.writeCDSTables()

with open("ReadMe", "w") as fd:
    tablemaker.makeReadMe(out=fd)
# =====================================================================
# 4. Insert the notes after the byte-by-byte block they refer to
# =====================================================================
evr_txt = ", ".join(f"{100 * e:.2f}" for e in evr)
NOTES = {
    "pca.dat": f"""\
Note (1): 1000 equally spaced points, 0 <= phase < 1 (step 0.001).
    The curves are periodic: phase 1 is equivalent to phase 0.
Note (2): Mean of the training-set curves, with vgamma removed.
Note (3): Unit-norm eigenvectors (sum of squares over the grid = 1),
    ordered by explained variance ratio, in %:
    {evr_txt}""",
    "coeffs.dat": """\
Note (4): The PCs and the KDE priors were derived from the training set
    only; the test set was used for validation.
Note (5): Coefficients p_i of the PC model fitted to the star's VELOCE-I
    Fourier-series model (Eq. 1 of the paper).
Note (6): dphi = [((E_GaiaDR3 - E_VELOCE)/P_VELOCE + 0.5) mod 1] - 0.5
    (Eq. 2 of the paper). Missing when the star is absent from the Gaia
    DR3 vari_cepheid table or the phase drift exceeds 0.1.""",
    "kde.dat": """\
Note (7): PC = 7-D KDE of (logP, p1,...,p6); dphi = 2-D KDE of
    (logP, dphi). Both use Gaussian product kernels (statsmodels
    KDEMultivariate, var_type='c'*n, bw='cv_ml') on the training set.
    The prior at a given period is the KDE conditioned on logP.
Note (8): Training-set minimum and maximum, extended by 10% of the
    training-set range on each side.
Note (9): [d] = decimal logarithm of a period in days (CDS convention).
    For logP, bw is a width in log P, i.e. in dex.""",
    "tableb1.dat": """\
Note (10): LMC: names and RV time series from Storm et al.
    (2011A&A...534A..95S); SMC: OGLE identifiers and RV time series from
    Gieren et al. (2018A&A...620A..99G).
Note (11): For SB1=Y stars, results of the hierarchical fit with
    cluster-specific vgamma (M_RV^MULTI) and P2P of the cluster-averaged
    vgamma; otherwise results of the single-vgamma fit (M_RV^SOLO).
    SMC-CEP-0524 formally exceeds the f_sigma,cluster threshold but is
    considered probably single (Sect. 4.4 of the paper).""",
}
lines = open("ReadMe").read().split("\n")
out, current, ndash = [], None, 0
for line in lines:
    out.append(line)
    m = re.match(r"Byte-by-byte Description of file: (\S+)", line)
    if m:
        current, ndash = m.group(1), 0
    elif current and re.fullmatch(r"-{80}", line):
        ndash += 1
        if ndash == 3:  # closing rule of the block
            out.extend(NOTES.get(current, "").split("\n"))
            out.append("-" * 80)
            current = None
text = "\n".join(out)
text = re.sub(r"\n-{80}\n-{80}\n", "\n" + "-" * 80 + "\n", text)
open("ReadMe", "w").write(text)


bad = [l for l in text.split("\n") if len(l) > 80 or not l.isascii()]
print("ReadMe written;", "lines violating 80-char/ASCII rule:", len(bad))
for l in bad:
    print("  ", repr(l))
