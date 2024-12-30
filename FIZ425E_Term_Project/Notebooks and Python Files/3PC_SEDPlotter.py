"""
Plot SEDs as computed for 4FGL-DR3 along with PLEC4 models, 
in particular the custom fits done for the Fermi LAT 3rd Pulsar Catalog ("3PC"). 
mtk = Matthew T. Kerr, May 2023.

Minor changes by das = David A. Smith, June 2023:
oo build_database reads a 3PC "little bigfile" FITS file.
   Spectral information is in hdu[3], pulsar properties are in hdu[1],
   read by get_sedplotter_info .
 
oo To make the 3PC "little bigfile" FITS file:
i) shrink_mtk_fits.py removes all but the pulsar lines from mtk_20230218_102657.fits,
which is a LAT collaboration internal version of the 4FGL-DR3 point source catalog,
to which we have added the PLEC4 fits specific to 3PC ("bfr" and "b23" described in 3PC Section 6).
The output is called new_table.fits .

ii) shrinkMe new_table.fits
uses the ftool fdelcol to remove 88 columns.

iii) fappend new_table.fits 3PClittleBigfileWithSEDhdu_20230607.fits
adds hdu[3] to the 3PC "little bigfile" FITS file.

The "bigfile" software package was created & maintained by Denis Dumora beginning in 2010,
and maintained by das since 2018. 
Starting from information in the ATNF pulsar catalog (https://www.atnf.csiro.au/research/pulsar/psrcat/expert.html),
it collates additional information acquired from several sources.

David A. Smith, July 2023: Fixed bugs 
i) to prevent crash when there is no 4FGL source.
ii) ditto, when e.g. Edot is "redacted".

"""

import argparse
import glob
import os

from astropy.io import fits
import pylab as pl
import numpy as np
from astropy import log


def set_plot_style(doit=True, serif=True, use_tex=True):
    """Here are some notes that might help you get this to work on Ubuntu:
    # sudo apt install texlive-latex-base
    # sudo apt install msttcorefonts -qq
    # sudo apt-get install dvipng texlive-latex-extra texlive-fonts-recommended cm-super
    # rm ~/.cache/matplotlib -rf
    """

    from matplotlib import cycler

    new_rcparams = {
        # Set color cycle: blue, green, yellow, red, violet, gray
        # Set color cycle: blue, orange, green, red, violet, gray
        "axes.prop_cycle": cycler(
            "color",
            ["0C5DA5", "FF9500", "00B945", "FF2C00", "845B97", "474747", "9e9e9e"],
        ),
        # Set default figure size
        "figure.figsize": (4 * 1.5, 3 * 1.5),
        # Set x axis
        "xtick.direction": "in",
        "xtick.major.size": 3 * 2,
        "xtick.major.width": 0.5,
        "xtick.minor.size": 1.5 * 2,
        "xtick.minor.width": 0.5,
        "xtick.minor.visible": True,
        "xtick.top": True,
        # Set y axis
        "ytick.direction": "in",
        "ytick.major.size": 3 * 2,
        "ytick.major.width": 0.5,
        "ytick.minor.size": 1.5 * 2,
        "ytick.minor.width": 0.5,
        "ytick.minor.visible": True,
        "ytick.right": True,
        # Set line widths
        "axes.linewidth": 0.5,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.0,
        # Remove legend' frame
        "legend.frameon": False,
        # Always save as 'tight'
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
    if use_tex:
        new_rcparams.update(
            {
                # Use LaTeX for math formatting
                "text.usetex": True,
                "text.latex.preamble": r"\usepackage{amsmath}",
            }
        )
    if serif:
        new_rcparams.update(
            {
                # Use serif fonts
                "font.serif": "Times New Roman",
                "font.family": "serif",
                "mathtext.fontset": "dejavuserif",
                "legend.fontsize": "large",
                "xtick.labelsize": "x-large",
                "ytick.labelsize": "x-large",
                "axes.labelsize": "xx-large",
            }
        )

    if doit:
        pl.rcParams.update(new_rcparams)
    else:
        return new_rcparams


class FGLParams(object):
    """Collate information about a source, including the best-fit params,
    covariance matrix, JNAME (if applicable, etc.).
    """

    @staticmethod
    def param_order(spec_type):
        """Return the "standard" order of params for given model."""
        if spec_type == "LogParabola":
            return ["norm", "alpha", "beta", "Eb"]
        if spec_type == "PLSuperExpCutoff4":
            return ["Prefactor", "IndexS", "Scale", "ExpfactorS", "Index2"]
        if spec_type == "PowerLaw":
            return ["Prefactor", "Index", "Scale"]
        raise ValueError("Unrecognized spectral type: %s" % spec_type)

    def __init__(
        self, spec_type, param_names, param_vals, param_free, param_cov, src_name
    ):
        """param_names: ordered list with parameters as appearing in XML
        param_vals : ordered list/array with parameter values
        param_cov  : array with covariance in order as in param_names
        """
        param_order = FGLParams.param_order(spec_type)
        # check parameter completeness / spectral type
        param_map = [param_names.index(param) for param in param_order]
        # re-order parameters
        param_vals = np.asarray([param_vals[i] for i in param_map])
        # re-order covariance matrix
        cov = np.empty_like(param_cov)
        for inew, iold in enumerate(param_map):
            for jnew, jold in enumerate(param_map):
                cov[inew, jnew] = param_cov[iold, jold]
        self._spec_type = spec_type
        self._param = param_order
        self._param_val = param_vals
        self._param_cov = cov
        self._param_free = param_free
        self._src_name = src_name
        self._jname = None
        self._pivot_energy = None

    def bfree(self, val=2.0 / 3):
        # return (self._spec_type=='PLSuperExpCutoff4') and (abs(self._param_val[-1]-val)<1e-5)
        return (self._spec_type == "PLSuperExpCutoff4") and (self._param_free[-1])

    def get_plec(self):
        if self._spec_type != "PLSuperExpCutoff4":
            return None
        # Need to swap parameter order
        # n0,gamma,d,b,e0 = p vs.
        # 'Prefactor','IndexS','Scale','ExpfactorS','Index2'
        new_order = np.asarray([0, 1, 3, 4, 2], dtype=int)
        new_p = self._param_val[new_order]
        new_cov = np.empty_like(self._param_cov)
        for inew, iold in enumerate(new_order):
            for jnew, jold in enumerate(new_order):
                new_cov[inew, jnew] = self._param_cov[iold, jold]
        # change index convention
        signs = np.asarray([1, -1, 1, 1, 1], dtype=int)
        msigns = np.outer(signs, signs)
        new_p *= signs
        new_cov *= msigns
        return PLEC4(
            new_p,
            cov=new_cov,
            srcname=self._src_name,
            jname=self._jname,
            pivot=self._pivot_energy,
        )

    def get_param(self, pname):
        return self._param_val[self._param.index(pname)]

    @staticmethod
    def get_fits_names(spec_type):
        """Return names as used in a FITS "catalog"."""
        # TODO -- check
        if spec_type == "PLSuperExpCutoff4":
            return [
                "PLEC_Flux_Density",
                "PLEC_IndexS",
                "Pivot_Energy",
                "PLEC_ExpfactorS",
                "PLEC_Exp_Index",
            ]
        raise NotImplementedError()

    @staticmethod
    def get_fits_units(spec_type):
        if spec_type == "PLSuperExpCutoff4":
            return ["ph cm-2 MeV-1 s-1", None, "MeV", None, None]
        raise NotImplementedError()

    @staticmethod
    def get_fits_disp(spec_type):
        if spec_type == "PLSuperExpCutoff4":
            return ["E10.4", "F8.4", "F10.2", "F8.5", "F8.4"]

    def get_sparse_cov(self):
        X, Y = np.triu_indices(self._param_cov.shape[0])
        return self._param_cov[X, Y]


class PLEC4(object):
    """Encapsulate a PLEC4 with useful methods.

    NB parameter names/order are different from 4FGL convention.
    """

    def __init__(self, p, cov=None, srcname=None, jname=None, pivot=None):
        self._p = np.asarray(p)
        self._cov = np.asarray(cov)
        self._srcname = srcname
        self._jname = jname
        self._pivot = pivot

    def __call__(self, e):
        """Evaluate dN/dE at the provided energies (in MeV)."""
        n0, gamma, d, b, e0 = self._p
        x = e / e0
        return n0 * x ** (-gamma + d / b) * np.exp((d / b**2) * (1 - x**b))

    def sed(self, e):
        return self(e) * e**2

    def dsed_dlam(self, e):
        """Return the SED derivative with respect to the parameters."""
        n0, gamma, d, b, e0 = self._p
        x = e / e0
        lnx = np.log(x)
        xb = x**b
        dn0 = 1.0 / n0
        dgamma = -lnx
        dd = lnx / b + (1 - xb) / b**2
        db = -d / b**2 * lnx - 2 * d / b**3 * (1 - xb) - d / b**2 * lnx * xb
        sed = self.sed(e)
        return np.asarray([sed * dn0, sed * dgamma, sed * dd, sed * db])

    def unc_sed(self, e):
        D = self.dsed_dlam(e)
        C = self._cov[0:4, 0:4]
        return np.einsum("ij,jk,ki->i", D.T, C, D) ** 0.5

    def d2de(self, e):
        """Return the curvature (d^2(self)dE^2)."""
        n0, gamma, d, b, e0 = self._p
        x = e / e0
        f = x ** (-gamma + d / b) * np.exp((d / b**2) * (1 - x**b))
        dlnf = (2 - gamma + d / b) / x - (d / b) * x ** (b - 1)
        d2lnf = -(2 - gamma + d / b) / x**2 - (d / b) * (b - 1) * x ** (b - 2)
        return f * (d2lnf + dlnf**2) / (e0**2)

    def epeak(self, default=np.nan):
        n0, gamma, d, b, e0 = self._p
        arg = 1 + (b / d) * (2 - gamma)
        if (arg < 0) or (b < 0):
            # print(arg,b,arg**(1./b)*e0)
            # return default
            pass
        epeak = arg ** (1.0 / b) * e0
        # check that it's a local maximum
        if self.d2de(epeak) >= 0:
            return default
        return epeak

    def unc_epeak(self, default_epeak=np.nan):
        """
        Return uncertainty in epeak obtained by gaussian prop. of errors.

        arg =1+(b/d)*(2-gamma)
        Epeak/E0 = arg**(1./b)
        log(epeak/e0) = (1./b)*log(arg)
        dlog(epeak/e0)/dg = 1./b/arg*(-1)*b/d = -1./d/arg
        dlog(epeak/e0)/dd = 1./b/arg*b*(2-gamma)(-1/d^2) = -(2-g)/d^2/arg
        dlog(epeak/e0)/db = -log(arg)/b^2 + 1./b/arg*(2-gamma)/d
        """
        epeak = self.epeak(default=default_epeak)
        if np.isnan(epeak):
            return np.nan, np.nan
        n0, gamma, d, b, e0 = self._p
        arg = 1 + (b / d) * (2 - gamma)
        ddg = -1.0 / (arg * d)
        ddd = -(2 - gamma) / (arg * d**2)
        ddb = -np.log(arg) / b**2 + (2 - gamma) / (arg * b * d)
        derivs = np.asarray([ddg, ddd, ddb])
        # make derivative matrix
        D = np.outer(derivs, derivs)
        C = self._cov[1:4, 1:4]
        frac_err = np.sum(D * C) ** 0.5
        return epeak, frac_err * epeak

    def unc_dpeak(self, default=np.nan):
        # TODO -- haven't checked yet
        # TODO -- check for gamma -ve or just return -ve value?
        n0, gamma, d, b, e0 = self._p
        epeak = self.epeak()
        if np.isnan(epeak):
            return default, default
        dpeak = d + b * (2 - gamma)
        ddd = 1
        ddb = 2 - gamma
        ddg = -b
        derivs = np.asarray([ddg, ddd, ddb])
        # make derivative matrix
        D = np.outer(derivs, derivs)
        C = self._cov[1:4, 1:4]
        err = np.sum(D * C) ** 0.5
        return dpeak, err

    def epsN(
        self, N=10, default_epeak=np.nan, epeak=None, emin=10, emax=50000, upper=True
    ):
        """Return peak energy and "eps_N", i.e. the energy by which the
            the SED has fallen to N% of the SED peak.  Can be the energy
            *above* or *below* Epeak depending on keyword arguments.

        Params
        ------
        p     Optional set of PLEC4 parameters to evaluate SED with.
        N     Target fraction in 10% of SED peak, i.e. 37 <--> 1/e.
        default_epeak Value to use if epeak is not well defined.
        epeak Optional value of epeak (e.g. if already calculated.)
        emax  Numerical value (MeV) to use as bound on search for epsN.
        emin  Numerical value (MeV) to use as bound on search for epsN.
        upper If True, return epsN *above* Epeak, else below.
        """
        if epeak is None:
            ep = self.epeak(default=default_epeak)
        else:
            ep = epeak
        if np.isnan(ep):
            return np.nan, np.nan
        if (ep > emax) or (ep < emin):
            return np.nan, np.nan
        v0 = self.sed(ep)
        target = v0 * N * (1.0 / 100)
        func = lambda e: np.abs(self.sed(np.exp(e)) - target)
        if upper:
            # check that solution lies within bounds
            if self.sed(emax) > target:
                return ep, np.nan
            epsN = np.exp(fminbound(func, np.log(ep), np.log(emax), xtol=1e-3))
            if abs(epsN - emax) < 1:
                epsN = emax
        else:
            # check that solution lies within bounds
            if self.sed(emin) > target:
                return ep, np.nan
            epsN = np.exp(fminbound(func, np.log(emin), np.log(ep), xtol=1e-3))
            if abs(epsN - emin) < 1:
                epsN = emin
        return ep, epsN

    def unc_epsN(self, N=10, upper=True):
        """From Jean:

        -- E10^2 dN/dE(E10) = 0.1 Ep^2 dN/dE(Ep)
                with (Ep/E0)^b = 1 + b/d (2 – GammaS).
        -- X10 = ln(E10/E0)
        -- Y10 = b (X10 - Xp) = b ln(E10/Ep)
        -- exp(Y10) - Y10 = 1 + b^2 ln10 / (d + b (2 - GammaS))
                          = 1 + b^2 ln10 / q

        -- This defines q = d + b (2 - GammaS) = d (Ep/E0)^b


        -- Deriving the equation results in terms on both sides.
           Barring any mistakes, I get the following partial derivatives:

        dX10/dGammaS = - 1/q + b^2 ln10 / (exp(Y10) - 1) / q^2
        dX10/dd = - (2 - GammaS) / d / q - b ln10 / (exp(Y10) - 1) / q^2
        dX10/db = ((2 - GammaS)/q - X10) / b + ln10 (d + q) / (exp(Y10) - 1) / q^2

        MK note: Jean's use of "10" isn't consistent with the use of N in
        the kwarg. For Jean, is related to the value by which the SED has
        fallen by 1/N, whereas the definition I am using is the energy for
        which the SED has is equal to a certain decrement.  Concretely,
        suppose we want eps_50, i.e. when the SED has fallen by 50%.  For
        Jean, that would be "N" = 1./0.5 = 2, while for me it would be N=50,
        i.e. it has fallen to 50%.  They just happen to be equal for N=10.
        """
        epeak, epsN = self.epsN(N=N, upper=upper)
        if np.isnan(epeak) or np.isnan(epsN):
            return np.nan, np.nan
        # TODO -- sanity check on epeak/epsN
        n0, gamma, d, b, e0 = self._p
        q = d + b * (2 - gamma)
        lnN = np.log(100.0 / N)
        XN = np.log(epsN / e0)
        YN = b * np.log(epsN / epeak)
        x = lnN / (np.exp(YN) - 1) / q**2
        ddg = -1.0 / q + b**2 * x
        ddd = -(2 - gamma) / (d * q) - b * x
        ddb = ((2 - gamma) / q - XN) / b + lnN * (d + q) / (np.exp(YN) - 1) / q**2
        derivs = np.asarray([ddg, ddd, ddb])
        # make derivative matrix
        D = np.outer(derivs, derivs)
        C = self._cov[1:4, 1:4]
        frac_err = np.sum(D * C) ** 0.5
        return epsN, frac_err * epsN

    def unc_epsN_less_epeak(self, N=10, upper=True):
        """Quick and dirty error calculation on the quantity epsN-epeak."""

        epeak, epsN = self.epsN(N=N, upper=upper)
        if np.isnan(epeak) or np.isnan(epsN):
            return np.nan, np.nan
        n0, gamma, d, b, e0 = self._p
        arg = 1 + (b / d) * (2 - gamma)
        ddg = -1.0 / (arg * d)
        ddd = -(2 - gamma) / (arg * d**2)
        ddb = -np.log(arg) / b**2 + (2 - gamma) / (arg * b * d)
        derivs1 = np.asarray([ddg, ddd, ddb]) * epeak

        q = d + b * (2 - gamma)
        lnN = np.log(100.0 / N)
        XN = np.log(epsN / e0)
        YN = b * np.log(epsN / epeak)
        x = lnN / (np.exp(YN) - 1) / q**2
        ddg = -1.0 / q + b**2 * x
        ddd = -(2 - gamma) / (d * q) - b * x
        ddb = ((2 - gamma) / q - XN) / b + lnN * (d + q) / (np.exp(YN) - 1)
        derivs2 = np.asarray([ddg, ddd, ddb]) * epsN
        if upper:
            derivs = derivs2 - derivs1
        else:
            derivs = derivs1 - derivs2

        # make derivative matrix
        D = np.outer(derivs, derivs)
        C = self._cov[1:4, 1:4]
        err = np.sum(D * C) ** 0.5
        if upper:
            return epsN - epeak, err
        else:
            return epeak - epsN, err

    def unc_epsN_less_epsN(self, N=10):
        """Quick and dirty error calculation on the quantity epsN-espN.
        Specifically, this would be an "upper" epsN less a "lower" one, e.g.
        for computing a FWHM.
        """

        epeak, epsN_hi = self.epsN(N=N, upper=True)
        if np.isnan(epeak) or np.isnan(epsN_hi):
            return np.nan, np.nan
        epeak, epsN_lo = self.epsN(N=N, upper=False)
        if np.isnan(epeak) or np.isnan(epsN_lo):
            return np.nan, np.nan

        n0, gamma, d, b, e0 = self._p
        q = d + b * (2 - gamma)
        lnN = np.log(100.0 / N)

        XN = np.log(epsN_hi / e0)
        YN = b * np.log(epsN_hi / epeak)
        x = lnN / (np.exp(YN) - 1) / q**2
        ddg = -1.0 / q + b**2 * x
        ddd = -(2 - gamma) / (d * q) - b * x
        ddb = ((2 - gamma) / q - XN) / b + lnN * (d + q) / (np.exp(YN) - 1)
        derivs_hi = np.asarray([ddg, ddd, ddb]) * epsN_hi

        XN = np.log(epsN_lo / e0)
        YN = b * np.log(epsN_lo / epeak)
        x = lnN / (np.exp(YN) - 1) / q**2
        ddg = -1.0 / q + b**2 * x
        ddd = -(2 - gamma) / (d * q) - b * x
        ddb = ((2 - gamma) / q - XN) / b + lnN * (d + q) / (np.exp(YN) - 1)
        derivs_lo = np.asarray([ddg, ddd, ddb]) * epsN_lo

        derivs = derivs_hi - derivs_lo

        # make derivative matrix
        D = np.outer(derivs, derivs)
        C = self._cov[1:4, 1:4]
        err = np.sum(D * C) ** 0.5
        return epsN_hi - epsN_lo, err

    def unc_epsN_div_epeak(self, N=50, upper=True):
        """Quick and dirty error calculation on the quantity epsN/epeak."""

        epeak, epsN = self.epsN(N=N, upper=upper)
        if np.isnan(epeak) or np.isnan(epsN):
            return np.nan, np.nan

        n0, gamma, d, b, e0 = self._p
        q = d + b * (2 - gamma)
        lnN = np.log(100.0 / N)

        XN = np.log(epsN / e0)
        YN = b * np.log(epsN / epeak)
        x = lnN / (np.exp(YN) - 1) / q**2
        ddg = -1.0 / q + b**2 * x
        ddd = -(2 - gamma) / (d * q) - b * x
        ddb = ((2 - gamma) / q - XN) / b + lnN * (d + q) / (np.exp(YN) - 1)
        derivs_epsN = np.asarray([ddg, ddd, ddb])

        arg = 1 + (b / d) * (2 - gamma)
        ddg = -1.0 / (arg * d)
        ddd = -(2 - gamma) / (arg * d**2)
        ddb = -np.log(arg) / b**2 + (2 - gamma) / (arg * b * d)
        derivs_epeak = np.asarray([ddg, ddd, ddb])

        # NB don't scale derivs yet because working in log quantities
        derivs = (derivs_epsN - derivs_epeak) * (1 if upper else -1)

        # make derivative matrix
        D = np.outer(derivs, derivs)
        C = self._cov[1:4, 1:4]
        err = np.sum(D * C) ** 0.5
        if upper:
            return epsN / epeak, err * (epsN / epeak)
        else:
            return epeak / epsN, err * (epeak / epsN)

    def unc_epsN_div_epsN(self, N=50):
        """Quick and dirty error calculation on the quantity epsN/epsN.
        Specifically, this would be an "upper" epsN less a "lower" one, e.g.
        for computing a FWHM.
        """

        epeak, epsN_hi = self.epsN(N=N, upper=True)
        if np.isnan(epeak) or np.isnan(epsN_hi):
            return np.nan, np.nan
        epeak, epsN_lo = self.epsN(N=N, upper=False)
        if np.isnan(epeak) or np.isnan(epsN_lo):
            return np.nan, np.nan

        n0, gamma, d, b, e0 = self._p
        q = d + b * (2 - gamma)
        lnN = np.log(100.0 / N)

        XN = np.log(epsN_hi / e0)
        YN = b * np.log(epsN_hi / epeak)
        x = lnN / (np.exp(YN) - 1) / q**2
        ddg = -1.0 / q + b**2 * x
        ddd = -(2 - gamma) / (d * q) - b * x
        ddb = ((2 - gamma) / q - XN) / b + lnN * (d + q) / (np.exp(YN) - 1)
        derivs_hi = np.asarray([ddg, ddd, ddb])

        XN = np.log(epsN_lo / e0)
        YN = b * np.log(epsN_lo / epeak)
        x = lnN / (np.exp(YN) - 1) / q**2
        ddg = -1.0 / q + b**2 * x
        ddd = -(2 - gamma) / (d * q) - b * x
        ddb = ((2 - gamma) / q - XN) / b + lnN * (d + q) / (np.exp(YN) - 1)
        derivs_lo = np.asarray([ddg, ddd, ddb])

        # NB we didn't multiply derivs_hi/lo by epsN, because we're doing
        # the natural log
        derivs = derivs_hi - derivs_lo

        # make derivative matrix
        D = np.outer(derivs, derivs)
        C = self._cov[1:4, 1:4]
        err = np.sum(D * C) ** 0.5
        return epsN_hi / epsN_lo, err * (epsN_hi / epsN_lo)

    def slope(self, e):
        """Return -d ln n(e) /d ln(e), viz the spectral index, at e (MeV).

        ln dn/de = ln n0 +(-gamma+d/b)*ln(x) + (d/b^2)*(1-(exp(ln x))^b)
        d/dln(x) = (-gamma+d/b) +d/b^2*(-b)*(exp(ln x))^(b-1)*exp(ln x)
                 = (-gamma+d/b) -d/b*x^b
        """
        n0, gamma, d, b, e0 = self._p
        x = e / e0
        return gamma - d / b * (1 - x**b)

    def unc_slope(self, e):
        """Return the slope and uncertainty on the local spectral idx.
        slope = gamma-d/b + d/b*(e/e0)^b
              = gamma-d/b + d/b*exp(b*ln x)
        ddg = 1
        ddd = -1/b +1/b*x^b = -1/b*(1-x^b)
        ddb = d/b^2 -d/b^2*x^b +d/b *ln x*x^b
            = d/b^2(1-x^b) + d/b*lnx*x^b
        """
        n0, gamma, d, b, e0 = self._p
        x = e / e0
        slope = gamma - d / b + d / b * x**b
        ddg = 1.0
        ddd = -1.0 / b * (1 - x**b)
        ddb = d / b**2 * (1 - x**b) + d / b * np.log(x) * x**b
        D = np.asarray([ddg, ddd, ddb])
        C = self._cov[1:4, 1:4]
        err = (D.T @ C @ D) ** 0.5
        return slope, err

    def leslope(self):
        """Return the asymptotic spectral index, gamma-d/b."""
        n0, gamma, d, b, e0 = self._p
        return gamma - d / b

    def unc_leslope(self):
        """Return the asymptotic spectral index, -gamma+d/b, and unc."""
        n0, gamma, d, b, e0 = self._p
        ddg = 1
        ddd = -1.0 / b
        ddb = d / b**2
        derivs = np.asarray([ddg, ddd, ddb])
        D = np.outer(derivs, derivs)
        C = self._cov[1:4, 1:4]
        err = np.sum(D * C) ** 0.5
        leslope = gamma - d / b
        return leslope, err

    def get_randp(self, p=None, cov=None, n=1, posb=False):
        if p is None:
            p = self._p
        if cov is None:
            cov = self._cov
        if cov is None:
            raise ValueError("Must have covariance matrix!")
        free_indices = np.abs(cov) > 0
        m = free_indices[0]
        c = np.reshape(cov[free_indices], (m.sum(), m.sum()))
        randp = multivariate_normal.rvs(mean=p[m], cov=c, size=n)
        # NB -- hardcoded to structure, not the best...
        if posb and m[3]:
            for i in range(10):
                mask = randp[:, 3] <= 0
                if np.any(mask):
                    randp[mask, :] = multivariate_normal.rvs(
                        mean=p[m], cov=c, size=mask.sum()
                    )
                else:
                    break
        rvals = np.empty([n, p.shape[0]])
        rvals[:, m] = randp
        rvals[:, ~m] = p[~m]
        return rvals

    def plot(self):
        """Low effort first stab at plotting."""
        import pylab as pl

        dom = np.logspace(np.log10(50), np.log10(15000), 201)
        pl.loglog(dom, self.sed(dom))
        epeak, e10 = self.epsN()
        pl.axvline(epeak)
        pl.axvline(e10)

    def bfree(self):
        return not abs(self._p[-2] - (2.0 / 3)) < 1e-5

    def k22_analog(self):
        """Return a cutoff energy that agrees with K22 in most cases."""
        n0, gamma, d, b, e0 = self._p
        return 1.0 / (d / b * (10000 / e0) ** (b - 1) / e0)

    def eflux(self, emin=100, emax=1e6, to_ergs=True):
        dom = np.logspace(np.log10(emin), np.log10(emax), 501)
        flux = simps(dom * self(dom), x=dom)
        if to_ergs:
            return flux * 1.60218e-6
        return flux

    def unc_eflux(self, emin=100, emax=1e6, to_ergs=True):
        """Propagate error on free parameters.  Norm is trivial."""
        n0, gamma, d, b, e0 = self._p
        p0 = self._p.copy()
        scales = 1e-3 * np.diag(self._cov) ** 0.5
        derivs = np.zeros(4)
        for i in range(4):
            if scales[i] == 0:
                continue
            self._p[i] = p0[i] + scales[i]
            eflux_hi = self.eflux(emin=emin, emax=emax, to_ergs=to_ergs)
            self._p[i] = p0[i] - scales[i]
            eflux_lo = self.eflux(emin=emin, emax=emax, to_ergs=to_ergs)
            derivs[i] = 0.5 * (eflux_hi - eflux_lo) / (scales[i])
            self._p[:] = p0
        D = np.outer(derivs, derivs)
        C = self._cov[:4, :4]
        err = np.sum(D * C) ** 0.5
        return self.eflux(emin=emin, emax=emax, to_ergs=to_ergs), err


def get_psr_fgl_names(fname):
    with fits.open(fname) as f:
        hdu = f["pulsars_bigfile"]
        codes = hdu.data["psr_code"]
        pmask = ~np.asarray([len(x) == 0 for x in codes], dtype=bool)
        jnames = hdu.data["psrj"][pmask]
        fgl_names = hdu.data["fglname"][pmask]
        return jnames, fgl_names


def get_sedplotter_info(fname):
    with fits.open(fname) as f:
        codes = f[1].data["psr_code"]
        psr_mask = ~np.asarray([len(x) == 0 for x in codes], dtype=bool)
        edot = f[1].data["edot"][psr_mask]
        p0 = f[1].data["p0"][psr_mask]
        psr_code = f[1].data["psr_code"][psr_mask]
        char_code = f[1].data["char_code"][psr_mask]
        return [p0, edot, psr_code, char_code]


def get_src_cols(fname, src_names, columns, hdu=3, name_col="source_name"):
    """Get values for specified sources and columns from a FITS file.

    Parameters
    ----------
    fname : the FITS file
    src_names : sources to match on (4FGL names)
    columns : names of the columns to get
    hdu : which hdu to use
    name_col : [source_name] specify column which has source names
    """

    with fits.open(fname) as f:
        data = f[hdu].data
        names = data[name_col] # 4FGL Name from Catalog HDU 3
        a = np.argsort(names)
        indices = np.searchsorted(names[a], src_names)
        # indices not found are set to N, which causes an index error with the mask line below
        # instead, change those to -1 (last element in the array)
        indices[indices>=len(names[a])] = -1
        '''
        for ii in range(len(indices)):
            if indices[ii]<len(src_names):
                log.info(f"{ii} {indices[ii]} {src_names[ii]} {names[a][indices[ii]]}")
            else:
                log.info(f"{ii} {indices[ii]} {src_names[ii]} {295}")
        '''
        mask = names[a][indices] == src_names
        cols = [data[col][a][indices] for col in columns]
        return mask, cols


def build_database(spec_fits):
    """
    Use the 4FGL-DR3 custom 3PC spectral analysis to join together the sets of SEDs (4FGL) and models (3PC).
    """
    jnames, fgl_names = get_psr_fgl_names(spec_fits)
    fgl_cols = ["flux_band", "unc_flux_band", "sqrt_ts_band", "nuFnu_Band"]
    fgl_mask, fgl_col_vals = get_src_cols(spec_fits, fgl_names, fgl_cols)

    plec_cols = FGLParams.get_fits_names("PLSuperExpCutoff4")
    plec_cols += ["Cov_PLEC"]
    plec_cols_b23 = [col + "_b23" for col in plec_cols]
    plec_cols_bfr = [col + "_bfr" for col in plec_cols]

    # Load up the FITS name columns, but existing code uses the XML convention,
    # so just translate the names before instantiating
    plec_mask_b23, plec_vals_b23 = get_src_cols(spec_fits, fgl_names, plec_cols_b23)
    plec_mask_bfr, plec_vals_bfr = get_src_cols(spec_fits, fgl_names, plec_cols_bfr)
    assert np.all(plec_mask_b23 == plec_mask_bfr)

    plecs_b23 = [None for i in range(len(plec_mask_b23))]
    plecs_bfr = [None for i in range(len(plec_mask_b23))]
    plecs = [plecs_b23, plecs_bfr]

    plec_names = FGLParams.param_order("PLSuperExpCutoff4")
    Xcov, Ycov = np.triu_indices(5)
    pfrees = [
        np.asarray([True, True, False, True, False]),
        np.asarray([True, True, False, True, True]),
    ]
    for isrc, src in enumerate(fgl_names):
        if not plec_mask_b23[isrc]:
            continue
        for imod in range(2):
            pfree = pfrees[imod]
            vals = [plec_vals_b23, plec_vals_bfr][imod]
            pvals = [col[isrc] for col in vals]
            if np.isnan(pvals[0]):
                plecs[imod][isrc] = None
                continue
            pcov = np.empty((5, 5))
            pcov[Xcov, Ycov] = vals[-1][isrc]
            pcov[Ycov, Xcov] = vals[-1][isrc]
            pfgl = FGLParams("PLSuperExpCutoff4", plec_names, pvals, pfree, pcov, src)
            plecs[imod][isrc] = pfgl.get_plec()

    # populate some metadata
    bf_data = get_sedplotter_info(spec_fits)

    db = (
        jnames,
        fgl_names,
        fgl_mask,
        fgl_cols,
        fgl_col_vals,
        plecs_b23,
        plecs_bfr,
        bf_data,
    )
    return db


def plot_sed(db, isrc):
    """
    Parameters
    ----------
    db : the output of "build_database"
    isrc : integer index of source within the list
    """
    (
        jnames,
        fgl_names,
        fgl_mask,
        fgl_cols,
        fgl_col_vals,
        plecs_b23,
        plecs_bfr,
        bf_data,
    ) = db

    pl.figure(1)
    pl.clf()
    ax = pl.gca()
    ax.set_xscale("log")
    ax.set_yscale("log")
    dom = np.logspace(np.log10(40), np.log10(1.1e6), 201)
    mev_in_erg = 624151
    have_model = False
    sed = None
    if plecs_bfr[isrc] is not None:
        sed = plecs_bfr[isrc].sed(dom) / mev_in_erg * 1e12
        sede = plecs_bfr[isrc].unc_sed(dom) / mev_in_erg * 1e12
        ax.plot(dom * 1e-3, sed, ls="-", color="k", label="$b$\ free")
        ax.fill_between(dom * 1e-3, sed + sede, y2=sed - sede, color="k", alpha=0.3)
        have_model = True
    if plecs_b23[isrc] is not None:
        sed = plecs_b23[isrc].sed(dom) / mev_in_erg * 1e12
        ax.plot(dom * 1e-3, sed, ls="--", color="k", label=r"$b=\frac{2}{3}$")
        if not have_model:
            sede = plecs_b23[isrc].unc_sed(dom) / mev_in_erg * 1e12
            ax.fill_between(dom * 1e-3, sed + sede, y2=sed - sede, color="k", alpha=0.3)
    if fgl_mask[isrc]:
        flux, fluxe, sig, nuFnu = [col[isrc] for col in fgl_col_vals]
        ul = np.isnan(fluxe[:, 0]) | (sig < 2)
        edges = np.asarray([50, 100, 300, 1000, 3000, 10000, 30000, 1e5, 1e6])
        centers = (edges[:-1] * edges[1:]) ** 0.5
        # The catalog convention is to store nuFnu directly, so all that is
        # needed is to estimate the relative error from the flux values
        rel_err = fluxe / flux[:, None]
        nuFnu = nuFnu / mev_in_erg
        flux = nuFnu * 1e12
        fluxe = flux[:, None] * rel_err
        xerr_lo = (centers - edges[:-1]) * 1e-3
        xerr_hi = (edges[1:] - centers) * 1e-3
        # as for error bars, it is transparent as to whether or not the
        # source has only an upper limit present (err = [Nan, val]) or if
        # it is low-significance (err = [-val1, val2]).  Simply use
        # flux val + 2 * upper error to get a 2-sigma upper limit.
        ax.errorbar(
            x=centers[~ul] * 1e-3,
            y=flux[~ul],
            xerr=[xerr_lo[~ul], xerr_hi[~ul]],
            yerr=[-fluxe[~ul, 0], fluxe[~ul, 1]],
            marker=None,
            ls=" ",
            color="k",
            lw=1,
        )
        ax.errorbar(
            x=centers[ul] * 1e-3,
            y=flux[ul] + 2 * fluxe[ul, 1],
            xerr=[xerr_lo[ul], xerr_hi[ul]],
            yerr=(flux[ul] + 2 * fluxe[ul, 1]) * 0.6,
            uplims=True,
            marker="v",
            mfc="white",
            ms=10,
            ls=" ",
            color="k",
            barsabove=True,
        )

    ax.set_xticks([0.1, 1, 10, 100, 1000])
    ax.set_xticklabels(["0.1", "1", "10", "100", "1000"])
    ax.set_xlabel("Energy (GeV)")
    ax.set_yticks([0.001, 0.01, 0.1, 1, 10, 100, 1000])
    ax.set_yticklabels(["0.001", "0.01", "0.1", "1", "10", "100", "1000"])
    ax.set_ylabel("Energy (GeV)")
    ax.set_ylabel("$\\nu F_{\\nu}$ (10$^{-12}$ erg cm$^{-2}$ s$^{-1}$)")
    ax.axis([0.9 * 50e-3, 1.1e6 * 1e-3, 1e-3, 5e3])
    ax.legend(loc="lower left")

    p0, edot, psr_code, char_code = [x[isrc] for x in bf_data]
    pl.figtext(0.16, 0.9525, f"PSR~{jnames[isrc]} --- ", size="x-large")
    pl.figtext(0.46, 0.9525, f"{fgl_names[isrc]}", size="x-large")
    print(" Plotting... ", jnames[isrc], fgl_mask[isrc], psr_code, char_code)
    pl.figtext(0.77, 0.9525, f" --- {psr_code}\ {char_code}", size="x-large")
    text2 = f"P\,\,=\,{p0*1e3:.1f}ms"
    if edot < 1.0e22:
        edot = 1.0e-9  # Redacted pulsars like J0955-3947 have zero edot when they get here, and crashed the code (das July '23).
    edot_exp = int(np.log10(edot))
    edot_num = edot / 10**edot_exp
    bob = f"\n$\dot{{\mathrm{{E}}}}$\,\,=\,{edot_num:.1f}$\\times10^{{{edot_exp}}}$\,erg\,s$^{{-1}}$"
    if edot < 1.0:
        bob = f"\n$\dot{{\mathrm{{E}}}}$\,\,=\,redacted"  # Redacted pulsars like J0955-3947 have zero edot when they get here, and crashed the code (das July '23).
    text2 += bob
    if plecs_bfr[isrc] is not None:
        model = plecs_bfr[isrc]
    else:
        model = plecs_b23[isrc]
    epeak_s = "No Value"
    if model is not None:
        epeak = model.epeak()
        if not np.isnan(epeak):
            epeak_s = f"{epeak*1e-3:.1f}\,GeV"
    text2 += f"\nE$_p$=\,{epeak_s}"
    pl.subplots_adjust(top=0.94, bottom=0.12, left=0.14, right=0.97)
    pl.figtext(0.67, 0.785, text2, size="x-large")

    writeSEDascii = open(f"{args.outdir}/{jnames[idx]}.txt", "w")
    writeSEDascii.write(
        "# %s %s %s %.1f  %.1e \n"
        % (jnames[isrc], char_code, psr_code, p0 * 1000.0, edot)
    )
    writeSEDascii.write(
        "# Energy values (MeV) for data points, lower error bars, upper error bars, Measured nuFnu values (1e-12 erg/cm2/s), lower error bars, upper error bars, Upper Limit Flag:  \n"
    )
    writeSEDascii.write(
        "# For Upper Limit Flag == True, the 2 sigma upper limit is (nuFnu value) + 2.*(upper error bar)  \n"
    )
    if (
        fgl_mask[isrc] and edot > 1.0 and sed is not None
    ):  # Redacted pulsars like J0955-3947 have zero edot when they get here, and crashed the code (das July '23).
        for i, value in enumerate(centers):
            writeSEDascii.write(
                " %.3f  %.3f  %.3f  %.3e  %.3e  %.3e  %s\n"
                % (
                    centers[i] / 1000.0,
                    xerr_lo[i],
                    xerr_hi[i],
                    flux[i],
                    fluxe[i][0],
                    fluxe[i][1],
                    ul[i],
                )
            )
        writeSEDascii.write(
            "# Energy values (GeV) for SED curve, SED curve values (1e-12 erg/cm2/s):  \n"
        )
        for i, value in enumerate(dom):
            writeSEDascii.write(" %.3f  %.3e \n" % (dom[i] / 1000.0, sed[i]))
    else:
        writeSEDascii.write(
            "# No 4FGL-DR3 counterpart for this pulsar, and so no SED = Spectral Energy Distribution. \n"
        )
    writeSEDascii.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot one, some, or all pulsar SEDs. Generates .png, .pdf, and .txt files."
    )
    parser.add_argument(
        "jnames",
        nargs="*",
        action="store",
        default=None,
        help="Specify particular pulsars to produce SEDs for by jname, e.g. J0030+0451.",
    )
    parser.add_argument(
        "--catalog",
        type=str,
        default=None,
        help="Specify the BigFile + SED FITS file to use.  Otherwise, default to the most recent version.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="3PC",
        help="Override the default output directory (./3PC).",
    )
    args, u = parser.parse_known_args()

    if args.catalog is None:
        cats = glob.glob("3PC_Catalog+SEDs*fits")
        if len(cats) < 1:
            raise ValueError("Could not find 3PC_Catalog+SEDs_YYYYMMDD file!")
        args.catalog = sorted(cats)[-1]

    print(args.catalog)
    db = build_database(args.catalog)
    (
        jnames,
        fgl_names,
        fgl_mask,
        fgl_cols,
        fgl_col_vals,
        plecs_b23,
        plecs_bfr,
        bf_data,
    ) = db
    if len(args.jnames) > 0:
        proc_idx = []
        for jname in args.jnames:
            idx = np.flatnonzero(jnames == jname)
            if len(idx) == 0:
                print(f"{jname} not available in list of pulsars.")
            else:
                proc_idx.append(idx[0])
    else:
        proc_idx = np.arange(len(jnames))

    set_plot_style()
    os.system(f"mkdir -p {args.outdir}")
    for idx in proc_idx:
        plot_sed(db, idx)
        pl.savefig(f"{args.outdir}/{jnames[idx]}_3PC_SED.png")
        pl.savefig(f"{args.outdir}/{jnames[idx]}_3PC_SED.pdf")
