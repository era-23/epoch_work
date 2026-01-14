import glob
from pathlib import Path
import epydeck
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import netCDF4 as nc
import astropy.units as u
import xrft  # noqa: E402
import copy
import os
import re
from scipy import stats
from scipy.interpolate import make_smoothing_spline, BSpline
from plasmapy.formulary import frequencies as ppf
from plasmapy.formulary import lengths as ppl
from plasmapy.formulary import speeds as pps
from dataclasses import dataclass

from collections.abc import Sequence
from warnings import warn
from inference.pdf.hdi import sample_hdi
from inference.pdf.kde import GaussianKDE, KDE2D

from scipy.stats import linregress

from matplotlib import colormaps

from scipy import constants
import astropy.units as u

@dataclass
class LinearGrowthRate:
    gamma : float
    timeStartIndex : float = None
    timeEndIndex : float = None
    timeMidpointIndex : float = None
    timeMidpoint: float = None
    yIntercept : float = None
    rSquared : float = None
    stdErr : float = None
    pValue : float = None
    rawRSquared : float = None
    rawStdErr : float = None
    rawPValue : float = None
    wavenumber : float = None
    maxPowerFrequency : float = None
    peakPower : float = None
    totalPower : float = None
    smoothingFunction : BSpline = None

    def to_string(self) -> str:
        return f"Wavenumber: {float(self.wavenumber)}, Peak power: {float(self.peakPower)}, Total power: {float(self.totalPower)}, Time (midpoint): {float(self.timeMidpoint)}, Growth rate: {float(self.gamma)}, SoS residual: {float(self.rSquared)}"

@dataclass
class LinearGrowthRateByT:
    time: float
    gamma: float
    yIntercept: float
    residual: float

@dataclass
class MaxGrowthRate:
    wavenumber: float
    time: float
    gamma: float
    yIntercept: float

E_TRACE_SPECIES_COLOUR_MAP = {
    "backgroundIon" : "orange",
    "electron" : "blue",
    "magneticField" : "purple",
    "electricField" : "green",
    "fastIon" : "red"
}

SPECIES_NAME_MAP = {
    "backgroundIonMeanEnergyDensity" : "Bkgd ion",
    "electronMeanEnergyDensity" : "Bkgd electron",
    "magneticFieldMeanEnergyDensity" : "B-field",
    "electricFieldMeanEnergyDensity" : "E-field",
    "fastIonMeanEnergyDensity" : "Fast ion"
}

ICE_METRICS = [
    "ICEmetric_fundamentalPower",
    "ICEmetric_fundamentalPower_pct",
    "ICEmetric_harmonicPower",
    "ICEmetric_harmonicPower_pct",
    "ICEmetric_c22Acf",
    "ICEmetric_c22AcfFirstMin",
    "ICEmetric_c22HighFluct",
    "ICEmetric_fundamentalPeakFloorRatio",
    "ICEmetric_fundamentalPeakMeanRatio",
    "ICEmetric_harmonicPeakFloorRatio",
    "ICEmetric_harmonicPeakMeanRatio",
]

fieldNameToText_dict = {
    "Energy/electricFieldEnergyDensity_delta" : "E_deltaE",
    "Energy/magneticFieldEnergyDensity_delta" : "B_deltaE", 
    "Energy/backgroundIonEnergyDensity_delta" : "background\ndelta KE", 
    "Energy/electronEnergyDensity_delta" : "e_deltaKE",
    "Energy/fastIonEnergyDensity_max" : "fast ion\nmax KE", 
    "Energy/fastIonEnergyDensity_timeMax" : "FI_timeMaxKE", 
    "Energy/fastIonEnergyDensity_min" : "fast ion\nmin KE", 
    "Energy/fastIonEnergyDensity_timeMin" : "fast ion\nmin time", 
    "Energy/fastIonEnergyDensity_delta" : "fast ion\ndelta KE",
    "Energy/backgroundIonEnergyDensity_max" : "background\nmax KE",
    "Energy/backgroundIonEnergyDensity_timeMax" : "background\npeak time",
    "Energy/electronEnergyDensity_timeMax" : "electron_timeMaxKE",
    "Energy/electricFieldEnergyDensity_timeMax" : "Ex_timeMaxE",
    "Energy/magneticFieldEnergyDensity_timeMax" : "Bz_timeMaxE",
    "Energy/backgroundIonEnergyDensity_timeMin" : "bkgdIon_timeMinKE",
    "Energy/electronEnergyDensity_timeMin" : "electron_timeMinKE",
    "Energy/electricFieldEnergyDensity_timeMin" : "Ex_timeMinE",
    "Energy/magneticFieldEnergyDensity_timeMin" : "Bz_timeMinE",
    "Energy/ICEmetric_energyTransfer" : "energy transfer",
    "Magnetic_Field_Bz/totalMagnitude" : "Bz_totalPower", 
    "Magnetic_Field_Bz/meanMagnitude" : "Bz_meanPower", 
    "Magnetic_Field_Bz/totalDelta" : "Bz_deltaTotalPower", 
    "Magnetic_Field_Bz/meanDelta" : "Bz_deltaMeanPower", 
    "Magnetic_Field_Bz/peakTkSpectralPower" : "Bz (peak) [T]", 
    "Magnetic_Field_Bz/meanTkSpectralPower" : "Bz (mean)", 
    "Magnetic_Field_Bz/peakTkSpectralPowerRatio" : "Bz_tkPowerRatio", 
    "Magnetic_Field_Bz/power/powerByFrequency" : "Bz", 
    "/Magnetic_Field_Bz/power/powerByFrequency" : "Power in Bz",
    "Electric_Field_Ex/power/powerByFrequency" : "Ex",
    "Electric_Field_Ey/power/powerByFrequency" : "Ey",
    "/Electric_Field_Ex/power/powerByFrequency" : "Power in Ex",
    "/Electric_Field_Ey/power/powerByFrequency" : "Power in Ey",
    "Magnetic_Field_Bz/growthRates/max/growthRate" : "Bz_maxGamma",
    "Magnetic_Field_Bz/growthRates/max/peakPower" : "Bz_maxGammaPeakPower",
    "Magnetic_Field_Bz/growthRates/max/residual" : "Bz_maxGammaFitResidual",
    "Magnetic_Field_Bz/growthRates/max/time" : "Bz_maxGammaTime",
    "Magnetic_Field_Bz/growthRates/max/totalPower" : "Bz_maxGammaTotalPower",
    "Magnetic_Field_Bz/growthRates/max/wavenumber" : "Bz_maxGammaK",
    "Magnetic_Field_Bz/growthRates/maxInHighPeakPowerK/growthRate" : "Bz_peakKmaxGamma",
    "Magnetic_Field_Bz/growthRates/maxInHighPeakPowerK/peakPower" : "Bz_peakKmaxGammaPeakPower",
    "Magnetic_Field_Bz/growthRates/maxInHighPeakPowerK/residual" : "Bz_peakKmaxGammaFitResidual",
    "Magnetic_Field_Bz/growthRates/maxInHighPeakPowerK/time" : "Bz_peakKmaxGammaTime",
    "Magnetic_Field_Bz/growthRates/maxInHighPeakPowerK/totalPower" : "Bz_peakKmaxGammaTotalPower",
    "Magnetic_Field_Bz/growthRates/maxInHighPeakPowerK/wavenumber" : "Bz_peakKmaxGammaK",
    "Magnetic_Field_Bz/growthRates/maxInHighTotalPowerK/growthRate" : "Bz_totalKmaxGamma",
    "Magnetic_Field_Bz/growthRates/maxInHighTotalPowerK/peakPower" : "Bz_totalKmaxGammaPeakPower",
    "Magnetic_Field_Bz/growthRates/maxInHighTotalPowerK/residual" : "Bz_totalKmaxGammaFitResidual",
    "Magnetic_Field_Bz/growthRates/maxInHighTotalPowerK/time" : "Bz_totalKmaxGammaTime",
    "Magnetic_Field_Bz/growthRates/maxInHighTotalPowerK/totalPower" : "Bz_totalKmaxGammaTotalPower",
    "Magnetic_Field_Bz/growthRates/maxInHighTotalPowerK/wavenumber" : "Bz_totalKmaxGammaK",
    "Magnetic_Field_Bz/growthRates/positive/bestInHighestPeakPowerK/growthRate" : "Growth rate (peak power k)",
    "Magnetic_Field_Bz/growthRates/positive/bestInHighestPeakPowerK/time" : "Time max growth (peak power k)",
    "Magnetic_Field_Bz/growthRates/positive/bestInHighestPeakPowerK/wavenumber" : "Peak k",
    "Magnetic_Field_Bz/growthRates/positive/bestInHighestTotalPowerK/growthRate" : "Growth rate (total power k)",
    "Magnetic_Field_Bz/growthRates/positive/bestInHighestTotalPowerK/time" : "Time max growth (total power k)",
    "Magnetic_Field_Bz/growthRates/positive/bestInHighestTotalPowerK/wavenumber" : "Total power k",
    
    "/Magnetic_Field_Bz/power/powerByFrequency_ICEmetric_fundamentalPower" : "Bz: fund. pwr",
    "Magnetic_Field_Bz/power/powerByFrequency_ICEmetric_fundamentalPower" : "Bz: fund. pwr",
    "/Magnetic_Field_Bz/power/powerByFrequency_ICEmetric_fundamentalPower_pct" : "Bz: fund. pwr",
    "Magnetic_Field_Bz/power/powerByFrequency_ICEmetric_fundamentalPower_pct" : "Bz: fund. pwr",
    "/Magnetic_Field_Bz/power/powerByFrequency_ICEmetric_harmonicPower" : "Bz: harm. pwr",
    "Magnetic_Field_Bz/power/powerByFrequency_ICEmetric_harmonicPower" : "Bz: harm. pwr",
    "/Magnetic_Field_Bz/power/powerByFrequency_ICEmetric_harmonicPower_pct" : "Bz: harm. pwr",
    "Magnetic_Field_Bz/power/powerByFrequency_ICEmetric_harmonicPower_pct" : "Bz: harm. pwr",

    "/Magnetic_Field_Bz/power/powerByFrequency_ICEmetric_c22Acf" : "Bz: ACF TS (C22)",
    "Magnetic_Field_Bz/power/powerByFrequency_ICEmetric_c22Acf" : "Bz: ACF TS (C22)",
    "/Magnetic_Field_Bz/power/powerByFrequency_ICEmetric_c22AcfFirstMin" : "Bz: ACF FM (C22)",
    "Magnetic_Field_Bz/power/powerByFrequency_ICEmetric_c22AcfFirstMin" : "Bz: ACF FM (C22)",
    "/Magnetic_Field_Bz/power/powerByFrequency_ICEmetric_c22HighFluct" : "Bz: high fluc. (C22)",
    "Magnetic_Field_Bz/power/powerByFrequency_ICEmetric_c22HighFluct" : "Bz: high fluc. (C22)",
    "/Magnetic_Field_Bz/power/powerByFrequency_ICEmetric_fundamentalPeakFloorRatio" : "Bz: fund. peak:floor pwr",
    "Magnetic_Field_Bz/power/powerByFrequency_ICEmetric_fundamentalPeakFloorRatio" : "Bz: fund. peak:floor pwr",
    "/Magnetic_Field_Bz/power/powerByFrequency_ICEmetric_harmonicPeakFloorRatio" : "Bz: harm. peak:floor pwr",
    "Magnetic_Field_Bz/power/powerByFrequency_ICEmetric_harmonicPeakFloorRatio" : "Bz: harm. peak:floor pwr",
    "/Magnetic_Field_Bz/power/powerByFrequency_ICEmetric_fundamentalPeakMeanRatio" : "Bz: fund. peak:mean pwr",
    "Magnetic_Field_Bz/power/powerByFrequency_ICEmetric_fundamentalPeakMeanRatio" : "Bz: fund. peak:mean pwr",
    "/Magnetic_Field_Bz/power/powerByFrequency_ICEmetric_harmonicPeakMeanRatio" : "Bz: harm. peak:mean pwr",
    "Magnetic_Field_Bz/power/powerByFrequency_ICEmetric_harmonicPeakMeanRatio" : "Bz: harm. peak:mean pwr",

    "Electric_Field_Ex/totalMagnitude" : "Ex_totalPower", 
    "Electric_Field_Ex/meanMagnitude" : "Ex_meanPower", 
    "Electric_Field_Ex/totalDelta" : "Ex_deltaTotalPower", 
    "Electric_Field_Ex/meanDelta" : "Ex_deltaMeanPower", 
    "Electric_Field_Ex/peakTkSpectralPower" : "Ex (peak)", 
    "Electric_Field_Ex/meanTkSpectralPower" : "Ex (mean)", 
    "Electric_Field_Ex/peakTkSpectralPowerRatio" : "Ex_tkPowerRatio",
    "Electric_Field_Ex/growthRates/max/growthRate" : "Ex_maxGamma",
    "Electric_Field_Ex/growthRates/max/peakPower" : "Ex_maxGammaPeakPower",
    "Electric_Field_Ex/growthRates/max/residual" : "Ex_maxGammaFitResidual",
    "Electric_Field_Ex/growthRates/max/time" : "Ex_maxGammaTime",
    "Electric_Field_Ex/growthRates/max/totalPower" : "Ex_maxGammaTotalPower",
    "Electric_Field_Ex/growthRates/max/wavenumber" : "Ex_maxGammaK",
    "Electric_Field_Ex/growthRates/maxInHighPeakPowerK/growthRate" : "Ex_peakKmaxGamma",
    "Electric_Field_Ex/growthRates/maxInHighPeakPowerK/peakPower" : "Ex_peakKmaxGammaPeakPower",
    "Electric_Field_Ex/growthRates/maxInHighPeakPowerK/residual" : "Ex_peakKmaxGammaFitResidual",
    "Electric_Field_Ex/growthRates/maxInHighPeakPowerK/time" : "Ex_peakKmaxGammaTime",
    "Electric_Field_Ex/growthRates/maxInHighPeakPowerK/totalPower" : "Ex_peakKmaxGammaTotalPower",
    "Electric_Field_Ex/growthRates/maxInHighPeakPowerK/wavenumber" : "Ex_peakKmaxGammaK",
    "Electric_Field_Ex/growthRates/maxInHighTotalPowerK/growthRate" : "Ex_totalKmaxGamma",
    "Electric_Field_Ex/growthRates/maxInHighTotalPowerK/peakPower" : "Ex_totalKmaxGammaPeakPower",
    "Electric_Field_Ex/growthRates/maxInHighTotalPowerK/residual" : "Ex_totalKmaxGammaFitResidual",
    "Electric_Field_Ex/growthRates/maxInHighTotalPowerK/time" : "Ex_totalKmaxGammaTime",
    "Electric_Field_Ex/growthRates/maxInHighTotalPowerK/totalPower" : "Ex_totalKmaxGammaTotalPower",
    "Electric_Field_Ex/growthRates/maxInHighTotalPowerK/wavenumber" : "Ex_totalKmaxGammaK",
    
    "/Electric_Field_Ex/power/powerByFrequency_ICEmetric_fundamentalPower" : "Ex: fund. pwr",
    "Electric_Field_Ex/power/powerByFrequency_ICEmetric_fundamentalPower" : "Ex: fund. pwr",
    "/Electric_Field_Ex/power/powerByFrequency_ICEmetric_fundamentalPower_pct" : "Ex: fund. pwr",
    "Electric_Field_Ex/power/powerByFrequency_ICEmetric_fundamentalPower_pct" : "Ex: fund. pwr",
    "/Electric_Field_Ex/power/powerByFrequency_ICEmetric_harmonicPower" : "Ex: harm. pwr",
    "Electric_Field_Ex/power/powerByFrequency_ICEmetric_harmonicPower" : "Ex: harm. pwr",
    "/Electric_Field_Ex/power/powerByFrequency_ICEmetric_harmonicPower_pct" : "Ex: harm. pwr",
    "Electric_Field_Ex/power/powerByFrequency_ICEmetric_harmonicPower_pct" : "Ex: harm. pwr",

    "/Electric_Field_Ey/power/powerByFrequency_ICEmetric_fundamentalPower" : "Ey: fund. pwr",
    "Electric_Field_Ey/power/powerByFrequency_ICEmetric_fundamentalPower" : "Ey: fund. pwr",
    "/Electric_Field_Ey/power/powerByFrequency_ICEmetric_fundamentalPower_pct" : "Ey: fund. pwr",
    "Electric_Field_Ey/power/powerByFrequency_ICEmetric_fundamentalPower_pct" : "Ey: fund. pwr",
    "/Electric_Field_Ey/power/powerByFrequency_ICEmetric_harmonicPower" : "Ex: harm. pwr",
    "Electric_Field_Ey/power/powerByFrequency_ICEmetric_harmonicPower" : "Ex: harm. pwr",
    "/Electric_Field_Ey/power/powerByFrequency_ICEmetric_harmonicPower_pct" : "Ex: harm. pwr",
    "Electric_Field_Ey/power/powerByFrequency_ICEmetric_harmonicPower_pct" : "Ex: harm. pwr",

    "/Electric_Field_Ex/power/powerByFrequency_ICEmetric_c22Acf" : "Ex: ACF TS (C22)",
    "Electric_Field_Ex/power/powerByFrequency_ICEmetric_c22Acf" : "Ex: ACF TS (C22)",
    "/Electric_Field_Ex/power/powerByFrequency_ICEmetric_c22AcfFirstMin" : "Ex: ACF FM (C22)",
    "Electric_Field_Ex/power/powerByFrequency_ICEmetric_c22AcfFirstMin" : "Ex: ACF FM (C22)",
    "/Electric_Field_Ex/power/powerByFrequency_ICEmetric_c22HighFluct" : "Ex: high fluc. (C22)",
    "Electric_Field_Ex/power/powerByFrequency_ICEmetric_c22HighFluct" : "Ex: high fluc. (C22)",
    "/Electric_Field_Ex/power/powerByFrequency_ICEmetric_fundamentalPeakFloorRatio" : "Ex: fund. peak:floor pwr",
    "Electric_Field_Ex/power/powerByFrequency_ICEmetric_fundamentalPeakFloorRatio" : "Ex: fund. peak:floor pwr",
    "/Electric_Field_Ex/power/powerByFrequency_ICEmetric_harmonicPeakFloorRatio" : "Ex: harm. peak:floor pwr",
    "Electric_Field_Ex/power/powerByFrequency_ICEmetric_harmonicPeakFloorRatio" : "Ex: harm. peak:floor pwr",
    "/Electric_Field_Ex/power/powerByFrequency_ICEmetric_fundamentalPeakMeanRatio" : "Ex: fund. peak:mean pwr",
    "Electric_Field_Ex/power/powerByFrequency_ICEmetric_fundamentalPeakMeanRatio" : "Ex: fund. peak:mean pwr",
    "/Electric_Field_Ex/power/powerByFrequency_ICEmetric_harmonicPeakMeanRatio" : "Ex: harm. peak:mean pwr",
    "Electric_Field_Ex/power/powerByFrequency_ICEmetric_harmonicPeakMeanRatio" : "Ex: harm. peak:mean pwr",

    "/Electric_Field_Ey/power/powerByFrequency_ICEmetric_c22Acf" : "Ey: ACF TS (C22)",
    "Electric_Field_Ey/power/powerByFrequency_ICEmetric_c22Acf" : "Ey: ACF TS (C22)",
    "/Electric_Field_Ey/power/powerByFrequency_ICEmetric_c22AcfFirstMin" : "Ey: ACF FM (C22)",
    "Electric_Field_Ey/power/powerByFrequency_ICEmetric_c22AcfFirstMin" : "Ey: ACF FM (C22)",
    "/Electric_Field_Ey/power/powerByFrequency_ICEmetric_c22HighFluct" : "Ey: high fluc. (C22)",
    "Electric_Field_Ey/power/powerByFrequency_ICEmetric_c22HighFluct" : "Ey: high fluc. (C22)",
    "/Electric_Field_Ey/power/powerByFrequency_ICEmetric_fundamentalPeakFloorRatio" : "Ey: fund. peak:floor pwr",
    "Electric_Field_Ey/power/powerByFrequency_ICEmetric_fundamentalPeakFloorRatio" : "Ey: fund. peak:floor pwr",
    "/Electric_Field_Ey/power/powerByFrequency_ICEmetric_harmonicPeakFloorRatio" : "Ey: harm. peak:floor pwr",
    "Electric_Field_Ey/power/powerByFrequency_ICEmetric_harmonicPeakFloorRatio" : "Ey: harm. peak:floor pwr",
    "/Electric_Field_Ey/power/powerByFrequency_ICEmetric_fundamentalPeakMeanRatio" : "Ey: fund. peak:mean pwr",
    "Electric_Field_Ey/power/powerByFrequency_ICEmetric_fundamentalPeakMeanRatio" : "Ey: fund. peak:mean pwr",
    "/Electric_Field_Ey/power/powerByFrequency_ICEmetric_harmonicPeakMeanRatio" : "Ey: harm. peak:mean pwr",
    "Electric_Field_Ey/power/powerByFrequency_ICEmetric_harmonicPeakMeanRatio" : "Ey: harm. peak:mean pwr",

    "Magnetic_Field_Bz" : r"$B_z$",
    "/Magnetic_Field_Bz" : r"$B_z$",
    "Electric_Field_Ex" : r"$E_x$",
    "/Electric_Field_Ex" : r"$E_x$",
    "Electric_Field_Ey" : r"$E_y$",
    "/Electric_Field_Ey" : r"$E_y$",

    "B0strength" : "B0", 
    "B0angle" : "B0 angle", 
    "backgroundDensity" : "density (log)", 
    "beamFraction" : "beam fraction (log)",
    "pitch" : "pitch"
}

fieldNameToUnit_dict = {
    "B0strength" : "T", 
    "B0angle" : r"$^\circ$", 
    "backgroundDensity" : r"$m^{-3}$", 
    "beamFraction" : "a.u.",
    "pitch" : "a.u.",

    "/Magnetic_Field_Bz" : "T",
    "Magnetic_Field_Bz" : "T",
    "/Electric_Field_Ex" : r"$\frac{V}{m}$",
    "Electric_Field_Ex" : r"$\frac{V}{m}$",
    "/Electric_Field_Ey" : r"$\frac{V}{m}$",
    "Electric_Field_Ey" : r"$\frac{V}{m}$",
    
    "/Magnetic_Field_Bz/power/powerByFrequency_ICEmetric_fundamentalPower" : r"$T \cdot \Omega_{c, \alpha}$",
    "Magnetic_Field_Bz/power/powerByFrequency_ICEmetric_fundamentalPower" : r"$T \cdot \Omega_{c, \alpha}$",
    "/Magnetic_Field_Bz/power/powerByFrequency_ICEmetric_fundamentalPower_pct" : "%",
    "Magnetic_Field_Bz/power/powerByFrequency_ICEmetric_fundamentalPower_pct" : "%",
    "/Magnetic_Field_Bz/power/powerByFrequency_ICEmetric_harmonicPower" : r"$T \cdot \Omega_{c, \alpha}$",
    "Magnetic_Field_Bz/power/powerByFrequency_ICEmetric_harmonicPower" : r"$T \cdot \Omega_{c, \alpha}$",
    "/Magnetic_Field_Bz/power/powerByFrequency_ICEmetric_harmonicPower_pct" : "%",
    "Magnetic_Field_Bz/power/powerByFrequency_ICEmetric_harmonicPower_pct" : "%",

    "/Electric_Field_Ex/power/powerByFrequency_ICEmetric_fundamentalPower" : r"$T \cdot \Omega_{c, \alpha}$",
    "Electric_Field_Ex/power/powerByFrequency_ICEmetric_fundamentalPower" : r"$T \cdot \Omega_{c, \alpha}$",
    "/Electric_Field_Ex/power/powerByFrequency_ICEmetric_fundamentalPower_pct" : "%",
    "Electric_Field_Ex/power/powerByFrequency_ICEmetric_fundamentalPower_pct" : "%",
    "/Electric_Field_Ex/power/powerByFrequency_ICEmetric_harmonicPower" : r"$T \cdot \Omega_{c, \alpha}$",
    "Electric_Field_Ex/power/powerByFrequency_ICEmetric_harmonicPower" : r"$T \cdot \Omega_{c, \alpha}$",
    "/Electric_Field_Ex/power/powerByFrequency_ICEmetric_harmonicPower_pct" : "%",
    "Electric_Field_Ex/power/powerByFrequency_ICEmetric_harmonicPower_pct" : "%",

    "/Electric_Field_Ey/power/powerByFrequency_ICEmetric_fundamentalPower" : r"$T \cdot \Omega_{c, \alpha}$",
    "Electric_Field_Ey/power/powerByFrequency_ICEmetric_fundamentalPower" : r"$T \cdot \Omega_{c, \alpha}$",
    "/Electric_Field_Ey/power/powerByFrequency_ICEmetric_fundamentalPower_pct" : "%",
    "Electric_Field_Ey/power/powerByFrequency_ICEmetric_fundamentalPower_pct" : "%",
    "/Electric_Field_Ey/power/powerByFrequency_ICEmetric_harmonicPower" : r"$T \cdot \Omega_{c, \alpha}$",
    "Electric_Field_Ey/power/powerByFrequency_ICEmetric_harmonicPower" : r"$T \cdot \Omega_{c, \alpha}$",
    "/Electric_Field_Ey/power/powerByFrequency_ICEmetric_harmonicPower_pct" : "%",
    "Electric_Field_Ey/power/powerByFrequency_ICEmetric_harmonicPower_pct" : "%",

    "Energy/ICEmetric_energyTransfer" : "% original F.I.",
}

def fieldNameToText(name : str) -> str:
    if name in fieldNameToText_dict:
        return fieldNameToText_dict[name]
    if name.strip('/') in fieldNameToText_dict:
        return fieldNameToText_dict[name.strip('/')]
    return name

def fieldNameToUnit(name : str) -> str:
    if name in fieldNameToUnit_dict:
        return fieldNameToUnit_dict[name]
    if name.strip('/') in fieldNameToUnit_dict:
        return fieldNameToUnit_dict[name.strip('/')]
    return ""

def correlate_and_plot_iciness_vs_baseline(
        iciness_by_metric : dict, 
        baseline_name : str, 
        baseline_data : list, 
        save_folder : Path, 
        unique_name : str,
        results : list, 
        title : str = None,
        doPlot : bool = False
    ):
    
    spectra = set()
    for metric in iciness_by_metric.keys():
        metric_parts = metric.split('_ICEmetric_')
        metric_short = f"ICEmetric_{metric_parts[1]}" if len(metric_parts) > 1 else metric_parts[0]
        spectrum_name = metric_parts[0]
        spectra.add(spectrum_name)
        
        iciness = np.array(iciness_by_metric[metric])
        plotBaseline = np.array(baseline_data) # Baseline
        
        # Raw data
        res = linregress(plotBaseline, iciness)

        if doPlot:
            fig = plt.figure(figsize=(12,12))
            if title:
                fig.suptitle(title)
            else:    
                fig.suptitle(f"{fieldNameToText(metric)} vs {baseline_name}")
            plt.scatter(plotBaseline, iciness, marker="x")
            plt.plot(plotBaseline, (res.slope * plotBaseline) + res.intercept, color="black", alpha=0.9, label = f"r2 = {res.rvalue:.5f}")
            plt.xlabel(f"{fieldNameToText(baseline_name)} [{fieldNameToUnit(metric)}]")
            plt.ylabel(f"{fieldNameToText(metric)} [{fieldNameToUnit(metric)}]")
            plt.legend()
            plt.tight_layout()
            # plt.show()
            figPath = save_folder / f"{unique_name}_{metric.replace('/', '_')}_vs_{baseline_name.replace(' ', '_')}.png"
            plt.savefig(figPath)

            plt.close("all")

        result = {
            "spectrum" : spectrum_name,
            "metric" : metric_short,
            "metric_log" : False,
            "baseline" : baseline_name,
            "baseline_log" : False,
            "r2" : res.rvalue,
            "figurePath" : figPath.as_uri() if doPlot else None
        }
        results.append(result)

        # Log-log
        res = linregress(np.log10(plotBaseline), np.log10(iciness))

        if doPlot:
            fig = plt.figure(figsize=(12,12))
            if title:
                fig.suptitle(title)
            else:    
                fig.suptitle(f"log {fieldNameToText(metric)} vs log {baseline_name}")
            plt.scatter(plotBaseline, iciness, marker="x")
            plt.plot(plotBaseline, 10**(res.intercept) * plotBaseline**res.slope, color="black", alpha=0.7, label = f"r2 = {res.rvalue:.5f}")
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(f"{fieldNameToText(baseline_name)} [{fieldNameToUnit(metric)}]")
            plt.ylabel(f"{fieldNameToText(metric)} [{fieldNameToUnit(metric)}]")
            plt.legend()
            plt.tight_layout()
            figPath = save_folder / f"{unique_name}_log_{metric.replace('/', '_')}_vs_log_{baseline_name.replace(' ', '_')}.png"
            plt.savefig(figPath)                      

            plt.close("all")

        result = {
            "spectrum" : spectrum_name,
            "metric" : metric_short,
            "metric_log" : True,
            "baseline" : baseline_name,
            "baseline_log" : True,
            "r2" : res.rvalue,
            "figurePath" : figPath.as_uri() if doPlot else None
        }
        results.append(result)

        # x log
        res = linregress(np.log10(plotBaseline), iciness)
        if doPlot:
            fig = plt.figure(figsize=(12,12))
            if title:
                fig.suptitle(title)
            else:    
                fig.suptitle(f"log {fieldNameToText(metric)} vs {baseline_name}")
            plt.scatter(plotBaseline, iciness, marker="x")
            plt.plot(plotBaseline, res.slope*np.log10(plotBaseline) + res.intercept, color="black", alpha=0.7, label = f"r2 = {res.rvalue:.5f}")
            plt.xscale('log')
            plt.xlabel(f"{fieldNameToText(baseline_name)} [{fieldNameToUnit(metric)}]")
            plt.ylabel(f"{fieldNameToText(metric)} [{fieldNameToUnit(metric)}]")
            plt.legend()
            plt.tight_layout()
            figPath = save_folder / f"{unique_name}_{metric.replace('/', '_')}_vs_log_{baseline_name.replace(' ', '_')}.png"
            plt.savefig(figPath)                      

            plt.close("all")
        result = {
            "spectrum" : spectrum_name,
            "metric" : metric_short,
            "metric_log" : False,
            "baseline" : baseline_name,
            "baseline_log" : True,
            "r2" : res.rvalue,
            "figurePath" : figPath.as_uri() if doPlot else None
        }
        results.append(result)
        
        # y log
        res = linregress(plotBaseline, np.log10(iciness))
        if doPlot:
            fig = plt.figure(figsize=(12,12))
            if title:
                fig.suptitle(title)
            else:    
                fig.suptitle(f"log {fieldNameToText(metric)} vs {baseline_name}")
            plt.scatter(plotBaseline, iciness, marker="x")
            plt.plot(plotBaseline, (10**(res.slope * plotBaseline)) * 10**res.intercept, color="black", alpha=0.7, label = f"r2 = {res.rvalue:.5f}")
            plt.yscale('log')
            plt.xlabel(f"{fieldNameToText(baseline_name)} [{fieldNameToUnit(metric)}]")
            plt.ylabel(f"{fieldNameToText(metric)} [{fieldNameToUnit(metric)}]")
            plt.legend()
            plt.tight_layout()
            figPath = save_folder / f"{unique_name}_log_{metric.replace('/', '_')}_vs_{baseline_name.replace(' ', '_')}.png"
            plt.savefig(figPath)                      

            plt.close("all")
        result = {
            "spectrum" : spectrum_name,
            "metric" : metric_short,
            "metric_log" : True,
            "baseline" : baseline_name,
            "baseline_log" : False,
            "r2" : res.rvalue,
            "figurePath" : figPath.as_uri() if doPlot else None
        }
        results.append(result)

    if doPlot:
        for spectrum in spectra:
            data = [[v for k, v in iciness_by_metric.items() if spectrum in k] + [baseline_data]]
            my_matrix_plot(
                data_series=data, 
                series_labels=["metric"],
                parameter_labels=[fieldNameToText(m) for m in iciness_by_metric.keys() if spectrum in m] + [baseline_name],
                show=False,
                filename=save_folder / f"{spectrum.replace('/', '_')}_{baseline_name}_contour.png",
                equalise_pdf_heights=False,
                label_size=8,
                plot_style="contour"
            )
            plt.close("all")

            my_matrix_plot(
                data_series=data, 
                series_labels=["metric"],
                parameter_labels=[fieldNameToText(m) for m in iciness_by_metric.keys() if spectrum in m] + [baseline_name],
                show=False,
                filename=save_folder / f"{spectrum.replace('/', '_')}_{baseline_name}_hdi.png",
                equalise_pdf_heights=False,
                label_size=8,
                plot_style="hdi"
            )
            plt.close("all")

            # Log-log
            data = [[np.log10(np.array(v)).tolist() for k, v in iciness_by_metric.items() if spectrum in k] + [np.log10(np.array(baseline_data)).tolist()]]
            my_matrix_plot(
                data_series=data, 
                series_labels=["metric"],
                parameter_labels=[fieldNameToText(m) for m in iciness_by_metric.keys() if spectrum in m] + [baseline_name],
                show=False,
                filename=save_folder / f"{spectrum.replace('/', '_')}_{baseline_name}_log_contour.png",
                equalise_pdf_heights=False,
                label_size=12,
                plot_style="contour"
            )
            plt.close("all")

            my_matrix_plot(
                data_series=data, 
                series_labels=["metric"],
                parameter_labels=[fieldNameToText(m) for m in iciness_by_metric.keys() if spectrum in m] + [baseline_name],
                show=False,
                filename=save_folder / f"{spectrum.replace('/', '_')}_{baseline_name}_log_hdi.png",
                equalise_pdf_heights=False,
                label_size=12,
                plot_style="hdi"
            )
            plt.close("all")

    return results

# Formerly everything below maxRes percentile, i.e. maxRes == 0.2 --> all values within bottom (best) 20th percentile
# Now everything at maxRes proportion and below, i.e. maxRes == 0.2 --> bottom (best) 20% of values
def filter_by_residuals(x, residuals, maxRes):
    x = np.array(x)

    min_res = np.nanmin(residuals)
    max_res = np.nanmax(residuals)
    range_res = max_res - min_res
    absoluteMaxResidual = min_res + (maxRes * range_res)
    
    # Filter growth rates
    x_low_residuals = np.array([i for i in x if i.residual <= absoluteMaxResidual])

    return x_low_residuals

def calculate_simulation_metadata(
        inputDeck : dict,
        dataset,
        outputNcRoot : nc.Dataset,
        fastSpecies : str = 'He-4 2+',
        bkgdSpecies : str = 'D+',
        debug = False) -> tuple[float, float]:
    
    beam = False
    if "frac_beam" in inputDeck["constant"].keys():
        beam =  inputDeck["constant"]["frac_beam"] > 0.0

    # Check parameters in SI
    B0 = inputDeck['constant']['b0_strength']
    if beam:
        ion_gyrofrequency = ppf.gyrofrequency(B0 * u.T, fastSpecies)
    else:
        ion_gyrofrequency = ppf.gyrofrequency(B0 * u.T, bkgdSpecies)
    ion_gyroperiod = 1.0 / ion_gyrofrequency
    ion_gyroperiod_s = ion_gyroperiod * 2.0 * np.pi * u.rad
    input_simtime = float(str(inputDeck['constant']['simtime']).split(" * ")[0]) * ion_gyroperiod_s.value
    nyquist_gfs = inputDeck['constant']['nyquist_omega_gfs']
    nyquist_s = nyquist_gfs * (ion_gyrofrequency / 2.0 * np.pi)
    num_t = int(input_simtime * (2.0 * nyquist_s.value))
    outputNcRoot.numTimePoints = num_t

    sim_time = float(dataset['Magnetic_Field_Bz'].coords["time"][-1]) * u.s
    outputNcRoot.simTime_s = sim_time
    outputNcRoot.timeSamplingFreq_Hz = num_t/sim_time
    outputNcRoot.timeNyquistFreq_Hz = num_t/(2.0 *sim_time)
    outputNcRoot.ionGyrofrequency_radPs = ion_gyrofrequency.value
    outputNcRoot.ionGyroperiod_sPrad = ion_gyroperiod.value
    outputNcRoot.ionGyroperiod_s = ion_gyroperiod_s.value

    # Work out num cells
    background_density = inputDeck['constant']['background_density'] / u.m**3
    background_temp = inputDeck['constant']['background_temp'] * u.K
    debye_length = ppl.Debye_length(background_temp, background_density)
    electron_gyroradius = np.sqrt(2.0 * constants.k * (u.J / u.K) * background_temp * constants.electron_mass * u.kg) / (constants.elementary_charge * u.C * inputDeck['constant']['b0_strength'] * u.T)
    beam_frac = inputDeck['constant']['frac_beam']
    number_density_bkgd = background_density * (1.0 - beam_frac)
    alfven_velocity = pps.Alfven_speed(B0 * u.T, number_density_bkgd, bkgdSpecies)
    grid_spacing = float(np.min([debye_length.value, electron_gyroradius.value]))
    est_min_cells = None
    if 'pixels_per_k' in inputDeck['constant'].keys(): 
        pixels_per_k = inputDeck['constant']['pixels_per_k']
        est_min_cells = int(np.ceil((pixels_per_k * ion_gyroperiod_s.value * alfven_velocity.value) / grid_spacing))
    num_cells = dataset["X_Grid_mid"].size
    outputNcRoot.numCells = num_cells

    sim_L = float(dataset['Magnetic_Field_Bz'].coords["X_Grid_mid"][-1]) * u.m
    outputNcRoot.simLength_m = sim_L
    outputNcRoot.spaceSamplingFreq_Pm = num_cells/sim_L
    outputNcRoot.spaceNyquistFreq_Pm = num_cells/(2.0 *sim_L)

    outputNcRoot.B0strength = B0
    
    B0_angle = inputDeck['constant']["b0_angle"]
    outputNcRoot.B0angle = B0_angle
    
    outputNcRoot.backgroundDensity = background_density.value
    outputNcRoot.beamFraction = beam_frac

    outputNcRoot.pitch = inputDeck["constant"]["pitch"]

    outputNcRoot.debyeLength_m = debye_length.value
    sim_L_dl = sim_L / debye_length
    outputNcRoot.simLength_dL = sim_L_dl
    outputNcRoot.cellWidth_dL = sim_L_dl / num_cells
    
    plasma_freq = ppf.plasma_frequency(number_density_bkgd, bkgdSpecies)
    outputNcRoot.plasmaFrequency_radPs = plasma_freq.value

    outputNcRoot.alfvenSpeed = alfven_velocity.value

    wLH_si = ppf.lower_hybrid_frequency(B0 * u.T, number_density_bkgd, bkgdSpecies)
    outputNcRoot.lhFrequency_radPs = wLH_si.value
    wLH_cyclo = wLH_si / ion_gyrofrequency
    outputNcRoot.lhFrequency_ionGyroF = wLH_cyclo.value
    
    # Lengths
    if beam:
        irb_energy = (inputDeck['constant']['ring_beam_energy'] * u.eV).to(u.J)
        irb_mass = inputDeck['constant']['fast_ion_mass_e'] * constants.electron_mass * u.kg
        irb_momentum = np.sqrt(2.0 * irb_mass * irb_energy)
        irb_momentum_perp = irb_momentum * np.sqrt(1.0 - inputDeck['constant']['pitch']**2)
        irb_v_perp = (irb_momentum_perp / irb_mass).to(u.m/u.s)
        mass_num = int(np.rint(inputDeck['constant']['fast_ion_mass_e']/1836.2))
        particle = 'He-4 2+' if mass_num == 4 else 'D+' if mass_num == 2 else "p+"
        irb_gyroradius = ppl.gyroradius(B = B0 * u.T, particle=particle, Vperp=irb_v_perp)
        outputNcRoot.fastIonGyroradius = irb_gyroradius.value
        outputNcRoot.simLength_rLfi = sim_L.value / irb_gyroradius.value
        outputNcRoot.cellWidth_rLfi = (sim_L.value / irb_gyroradius.value) / num_cells
    mass_num = int(np.rint(inputDeck['constant']['background_ion_mass_e']/1836.2))
    particle = 'He-4 2+' if mass_num == 4 else 'D+' if mass_num == 2 else "p+"
    background_gyroradius = ppl.gyroradius(B = B0 * u.T, particle=particle, T = background_temp)
    outputNcRoot.backgroundGyroradius = background_gyroradius.value
    outputNcRoot.simLength_rLp = sim_L.value / background_gyroradius.value
    outputNcRoot.cellWidth_rLp = (sim_L.value / background_gyroradius.value) / num_cells
    outputNcRoot.electronGyroradius = electron_gyroradius.value
    outputNcRoot.simLength_rLe = sim_L.value / electron_gyroradius.value
    outputNcRoot.cellWidth_rLe = (sim_L.value / electron_gyroradius.value) / num_cells
    
    sim_L_vA_Tci = sim_L / (ion_gyroperiod_s * alfven_velocity)

    # Normalised units
    simtime_Tci = sim_time / ion_gyroperiod_s
    outputNcRoot.simTime_Tci = simtime_Tci
    outputNcRoot.timeSamplingFreq_Wci = num_t/simtime_Tci
    outputNcRoot.timeNyquistFreq_Wci = num_t/(2.0 *simtime_Tci)
    outputNcRoot.simLength_VaTci = sim_L_vA_Tci
    outputNcRoot.spaceSamplingFreq_WciOverVa = num_cells/sim_L_vA_Tci
    outputNcRoot.spaceNyquistFreq_WciOverVa = num_cells/(2.0 * sim_L_vA_Tci)

    if debug:
        print(f"Num time points: {num_t}")
        print(f"Sim time in SI: {sim_time}")
        print(f"Sampling frequency: {num_t/sim_time}")
        print(f"Nyquist frequency: {num_t/(2.0 *sim_time)}")
        if est_min_cells is not None:
            print(f"Estimated minimum cells needed: {est_min_cells}")
        print(f"Num cells: {num_cells}")
        print(f"Sim length: {sim_L}")
        print(f"Sampling frequency: {num_cells/sim_L}")
        print(f"Nyquist frequency: {num_cells/(2.0 *sim_L)}")
        
        print(f"B0 = {B0}")
        print(f"B0 angle = {B0_angle}")
        print(f"B0z = {B0 * np.sin(B0_angle * (np.pi / 180.0))}")

        print(f"Debye length: {debye_length}")
        print(f"Background ion gyroradius: {background_gyroradius}")
        print(f"Background electron gyroradius: {electron_gyroradius}")
        if beam: 
            print(f"Fast ion gyroradius: {irb_gyroradius}")
        print(f"Sim length in Debye lengths: {sim_L_dl}")
        cell_width_dl = sim_L_dl /num_cells
        cell_width = cell_width_dl * debye_length
        if beam:
            print(f"Cell width: {cell_width_dl:.3f} debye lengths ({cell_width/background_gyroradius:.3f} background gyroradii, {cell_width/electron_gyroradius:.3f} electron gyroradii, {cell_width/irb_gyroradius:.3f} fast ion gyroradii)")
        else:
            print(f"Cell width: {cell_width_dl/num_cells:.3f} debye lengths ({cell_width/background_gyroradius:.3f} background gyroradii, {cell_width/electron_gyroradius:.3f} electron gyroradii)")

        print(f"Ion gyrofrequency: {ion_gyrofrequency}")
        print(f"Ion gyroperiod: {ion_gyroperiod}")
        print(f"Ion gyroperiod: {ion_gyroperiod_s}")
        print(f"Plasma frequency: {plasma_freq}")
        print(f"Alfven speed: {alfven_velocity}")

        print(f"NORMALISED: Sim time in Tci: {simtime_Tci}")
        print(f"NORMALISED: Temporal sampling frequency in Wci: {num_t/simtime_Tci}")
        print(f"NORMALISED: Temporal Nyquist frequency in Wci: {num_t/(2.0 *simtime_Tci)}")
        print(f"NORMALISED: Sim L in vA*Tci: {sim_L_vA_Tci}")
        print(f"NORMALISED: Spatial sampling frequency in Wci/vA: {num_cells/sim_L_vA_Tci}")
        print(f"NORMALISED: Nyquist frequency in Wci/vA: {num_cells/(2.0 * sim_L_vA_Tci)}")

    return ion_gyroperiod_s, alfven_velocity

def normalise_data(dataset : xr.Dataset, ion_gyroperiod : float, alfven_velocity : float) -> xr.Dataset:
    
    print("Normalising and interpolating data....")
    evenly_spaced_time = np.linspace(dataset.coords["time"][0].data, dataset.coords["time"][-1].data, len(dataset.coords["time"].data))
    dataset = dataset.interp(time=evenly_spaced_time)
    Tci = evenly_spaced_time / ion_gyroperiod
    vA_Tci = dataset.coords["X_Grid_mid"] / (ion_gyroperiod * alfven_velocity)
    dataset = dataset.drop_vars("X_Grid")
    dataset = dataset.assign_coords({"time" : Tci, "X_Grid_mid" : vA_Tci})
    dataset = dataset.rename(X_Grid_mid="x_space")
    print("Dataset normalised")

    return dataset

def create_omega_k_plots(
        fftSpectrum : xr.DataArray,
        statsFile : nc.Dataset,
        field : str,
        field_unit : str,
        saveDirectory : Path,
        runName : str,
        inputDeck : dict,
        bkgdSpecies : str,
        fastSpecies : str,
        maxK : float,
        maxW : float,
        display : bool,
        debug):

    print("Generating w-k plots....")
    field = fieldNameToText(field)
    spec = abs(fftSpectrum.load())

    # Select positive temporal frequencies
    spec = spec.sel(frequency=spec.frequency>=0.0)

    # Trim to max wavenumber and frequency, if specified
    if maxK is not None:
        spec = spec.sel(wavenumber=spec.wavenumber<=maxK)
        spec = spec.sel(wavenumber=spec.wavenumber>=-maxK)
    if maxW is not None:
        spec = spec.sel(frequency=spec.frequency<=maxW)

    # Log stats on spectrum
    spec_sum = float(spec.sum())
    statsFile.totalWkSpectralPower = spec_sum 
    squared_sum = float((np.abs(spec)**2).sum())
    parseval_wk = squared_sum * spec.coords['frequency'].spacing * spec.coords['wavenumber'].spacing * 2.0 # Double because half the energy is outside the range we care about 
    statsFile.parsevalWk = parseval_wk
    spec_peak = float(np.nanmax(spec))
    statsFile.peakWkSpectralPower = spec_peak
    spec_mean = float(spec.mean())
    statsFile.meanWkSpectralPower = spec_mean
    
    if debug:
        print(f"Sum of omega-k sqared * dw * dk: {parseval_wk}")
        print(f"Max peak in omega-k: {spec_peak}")
        print(f"Mean of omega-k: {spec_mean}")

    # Create fields for recording powers by frequency and wavenumber
    if "power" not in statsFile.groups.keys():
        powerStats = statsFile.createGroup("power")#
        powerStats.createDimension("wavenumber", spec.coords['wavenumber'].size)
        powerStats.createDimension("frequency", spec.coords['frequency'].size)
        k_var = powerStats.createVariable("wavenumber", datatype="f4", dimensions=("wavenumber",))
        k_var[:] = spec.coords["wavenumber"].data
        k_var.units = "wCI/vA"

        w_var = powerStats.createVariable("frequency", datatype="f4", dimensions=("frequency",))
        w_var[:] = spec.coords["frequency"].data
        w_var.units = "wCI"

        powerByK = powerStats.createVariable("powerByWavenumber", datatype="f4", dimensions=("wavenumber",))
        powerByOmega = powerStats.createVariable("powerByFrequency", datatype="f4", dimensions=("frequency",))
        frequencyOfMaxPowerByK = powerStats.createVariable("frequencyOfMaxPowerByK", datatype="f4", dimensions=("wavenumber",))
    else:
        powerByK = statsFile.groups["power"].variables["powerByWavenumber"]
        powerByOmega = statsFile.groups["power"].variables["powerByFrequency"]
        frequencyOfMaxPowerByK = statsFile.groups["power"].variables["frequencyOfMaxPowerByK"]

    maxPowerFrequenciesByK = {}
    for k in spec.coords["wavenumber"]:
        try:
            maxPowerInKIndex = spec.sel(wavenumber=k).argmax()
            maxPowerFrequenciesByK[float(k.data)] = float(spec.coords["frequency"].data[maxPowerInKIndex])
        except ValueError:
            maxPowerFrequenciesByK[float(k.data)] = 0.0
    frequencyOfMaxPowerByK[:] = [v for v in maxPowerFrequenciesByK.values()]

    B0 = inputDeck['constant']['b0_strength']

    # Power in omega over all k
    fig, axs = plt.subplots(figsize=(15, 10))
    power_trace = spec.sum(dim = "wavenumber")
    powerByOmega[:] = power_trace.data
    power_trace.plot(ax=axs)
    axs.set_xticks(ticks=np.arange(np.floor(power_trace.coords['frequency'][0]), np.ceil(power_trace.coords['frequency'][-1])+1.0, 1.0), minor=True)
    axs.grid(which='both', axis='x')
    axs.set_xlabel(r"Frequency [$\Omega_{c,\alpha}$]")
    axs.set_ylabel(f"Sum of {field} power over all k [{field_unit}]")
    axs.grid()
    fig.tight_layout()
    filename = Path(f'{runName}_{field.replace("_", "").replace("$", "")}_powerByOmega_maxK-{maxK if maxK is not None else "all"}_maxW-{maxW if maxW is not None else "all"}.png')
    fig.savefig(str(saveDirectory / filename))
    if display:
        plt.show()
        plt.close("all")

    # Power in omega over all k (log proportion)
    fig, axs = plt.subplots(figsize=(15, 10))
    log_power_trace = np.log10(power_trace.data / B0)
    axs.plot(power_trace.coords['frequency'], log_power_trace)
    axs.set_xticks(ticks=np.arange(np.floor(power_trace.coords['frequency'][0]), np.ceil(power_trace.coords['frequency'][-1])+1.0, 1.0), minor=True)
    axs.grid(which='both', axis='x')
    axs.set_xlabel(r"Frequency [$\Omega_{c,\alpha}$]")
    axs.set_ylabel(f"log10({field} power over all k / {r'$B_0$'})")
    axs.grid()
    fig.tight_layout()
    filename = Path(f'{runName}_{field.replace("_", "").replace("$", "")}_powerByOmegaLog_maxK-{maxK if maxK is not None else "all"}_maxW-{maxW if maxW is not None else "all"}.png')
    fig.savefig(str(saveDirectory / filename))
    if display:
        plt.show()
        plt.close("all")

    # Power in omega over all k (log squared proportion)
    fig, axs = plt.subplots(figsize=(15, 10))
    log_power_trace = np.log10(power_trace.data**2 / B0**2)
    axs.plot(power_trace.coords['frequency'], log_power_trace)
    axs.set_xticks(ticks=np.arange(np.floor(power_trace.coords['frequency'][0]), np.ceil(power_trace.coords['frequency'][-1])+1.0, 1.0), minor=True)
    axs.grid(which='both', axis='x')
    axs.set_xlabel(r"Frequency [$\Omega_{c,\alpha}$]")
    pow2 = r'$\text{power}^{2}$'
    axs.set_ylabel(f"log10({field} {pow2} over all k / {r'$B_0^2$'}")
    axs.grid()
    fig.tight_layout()
    filename = Path(f'{runName}_{field.replace("_", "").replace("$", "")}_powerByOmegaLogSquare_maxK-{maxK if maxK is not None else "all"}_maxW-{maxW if maxW is not None else "all"}.png')
    fig.savefig(str(saveDirectory / filename))
    if display:
        plt.show()
        plt.close("all")

    # Power in omega over all k (log proportion)
    fig, axs = plt.subplots(figsize=(15, 10))
    log_power_trace = 10.0 * np.log10(power_trace.data / B0)
    axs.plot(power_trace.coords['frequency'], log_power_trace)
    axs.set_xticks(ticks=np.arange(np.floor(power_trace.coords['frequency'][0]), np.ceil(power_trace.coords['frequency'][-1])+1.0, 1.0), minor=True)
    axs.grid(which='both', axis='x')
    axs.set_xlabel(r"Frequency [$\Omega_{c,\alpha}$]")
    axs.set_ylabel(f"Normalised {field} power over all k [dB]")
    axs.grid()
    fig.tight_layout()
    filename = Path(f'{runName}_{field.replace("_", "").replace("$", "")}_powerByOmegaLog_maxK-{maxK if maxK is not None else "all"}_maxW-{maxW if maxW is not None else "all"}.png')
    fig.savefig(str(saveDirectory / filename))
    if display:
        plt.show()
        plt.close("all")

    # Power in omega over all k (decibles)
    fig, axs = plt.subplots(figsize=(15, 10))
    log_power_trace = 10.0 * np.log10(power_trace.data**2 / B0**2)
    axs.plot(power_trace.coords['frequency'], log_power_trace)
    axs.set_xticks(ticks=np.arange(np.floor(power_trace.coords['frequency'][0]), np.ceil(power_trace.coords['frequency'][-1])+1.0, 1.0), minor=True)
    axs.grid(which='both', axis='x')
    axs.set_xlabel(r"Frequency [$\Omega_{c,\alpha}$]")
    axs.set_ylabel(f"Normalised {field} {pow2} over all k [dB]")
    axs.grid()
    fig.tight_layout()
    filename = Path(f'{runName}_{field.replace("_", "").replace("$", "")}_powerByOmegaDB_maxK-{maxK if maxK is not None else "all"}_maxW-{maxW if maxW is not None else "all"}.png')
    fig.savefig(str(saveDirectory / filename))
    if display:
        plt.show()
        plt.close("all")

    # Power in k over all omega
    fig, axs = plt.subplots(figsize=(15, 10))
    power_trace = spec.sum(dim = "frequency")
    powerByK[:] = power_trace.data
    power_trace.plot(ax=axs)
    axs.set_xticks(ticks=np.arange(np.floor(power_trace.coords['wavenumber'][0]), np.ceil(power_trace.coords['wavenumber'][-1])+1.0, 1.0), minor=True)
    axs.grid(which='both', axis='x')
    axs.set_xlabel(r"Wavenumber [$\frac{\Omega_{c,\alpha}}{v_A}$]")
    omega = r'$\Omega_{c,\alpha}$'
    axs.set_ylabel(f"Sum of {field} power over all {omega} [{field_unit}]")
    axs.grid()
    fig.tight_layout()
    filename = Path(f'{runName}_{field.replace("_", "").replace("$", "")}_powerByK_maxK-{maxK if maxK is not None else "all"}_maxW-{maxW if maxW is not None else "all"}.png')
    fig.savefig(str(saveDirectory / filename))
    if display:
        plt.show()
        plt.close("all")

    # Full dispersion relation for positive omega
    fig, axs = plt.subplots(figsize=(15, 10))
    spec.plot(ax=axs, cbar_kwargs={'label': f'{field} power [{field_unit}]'}, cmap='plasma')
    axs.set_ylabel(r"Frequency [$\Omega_{c,\alpha}$]")
    axs.set_xlabel(r"Wavenumber [$\frac{\Omega_{c,\alpha}}{v_A}$]")
    axs.grid()
    fig.tight_layout()
    filename = Path(f'{runName}_{field.replace("_", "").replace("$", "")}_wk_maxK-{maxK if maxK is not None else "all"}_maxW-{maxW if maxW is not None else "all"}.png')
    fig.savefig(str(saveDirectory / filename))
    if display:
        plt.show()
        plt.close("all")

    log_spec = np.log10(spec)

    # Full dispersion relation for positive omega (log)
    fig, axs = plt.subplots(figsize=(15, 10))
    log_spec.plot(ax=axs, cbar_kwargs={'label': f'{field} power [{r"log$_{10}$"}]'}, cmap='plasma')
    axs.set_ylabel(r"Frequency [$\Omega_{c,\alpha}$]")
    axs.set_xlabel(r"Wavenumber [$\frac{\Omega_{c,\alpha}}{v_A}$]")
    axs.grid()
    fig.tight_layout()
    filename = Path(f'{runName}_{field.replace("_", "").replace("$", "")}_wk_log_maxK-{maxK if maxK is not None else "all"}_maxW-{maxW if maxW is not None else "all"}.png')
    fig.savefig(str(saveDirectory / filename))
    if display:
        plt.show()
        plt.close("all")

    # Positive omega/positive k with vA and lower hybrid frequency
    fig, axs = plt.subplots(figsize=(15, 10))
    spec = spec.sel(wavenumber=spec.wavenumber>0.0)
    spec.plot(ax=axs, cbar_kwargs={'label': f'{field} power [{field_unit}]'}, cmap='plasma')
    axs.plot(spec.coords['wavenumber'].data, spec.coords['wavenumber'].data, 'w--', linewidth = 3.0, label=r'$v_A$ branch')
    bkgd_number_density = float(inputDeck['constant']['background_density'])
    wLH_cyclo = ppf.lower_hybrid_frequency(B0 * u.T, bkgd_number_density * u.m**-3, bkgdSpecies) / ppf.gyrofrequency(B0 * u.T, fastSpecies)
    axs.axhline(y = wLH_cyclo, color='white', linestyle=':', linewidth = 3.0, label=r'lower hybrid frequency')
    axs.legend(loc='upper left')
    axs.set_ylabel(r"Frequency [$\Omega_{c,\alpha}$]")
    axs.set_xlabel(r"Wavenumber [$\frac{\Omega_{c,\alpha}}{v_A}$]")
    axs.grid()
    fig.tight_layout()
    filename = Path(f'{runName}_{field.replace("_", "").replace("$", "")}_wk_positiveK_maxK-{maxK if maxK is not None else "all"}_maxW-{maxW if maxW is not None else "all"}.png')
    fig.savefig(str(saveDirectory / filename))
    if display:
        plt.show()
        plt.close("all")

    # Positive omega/positive k with vA and lower hybrid frequency (log)
    fig, axs = plt.subplots(figsize=(15, 10))
    log_spec = log_spec.sel(wavenumber=log_spec.wavenumber>0.0)
    log_spec.plot(ax=axs, cbar_kwargs={'label': f'{field} power [{r"log$_{10}$"}]'}, cmap='plasma')
    axs.plot(log_spec.coords['wavenumber'].data, log_spec.coords['wavenumber'].data, 'k--', linewidth = 3.0, label=r'$v_A$ branch')
    axs.axhline(y = wLH_cyclo, color='black', linestyle=':', linewidth = 3.0, label=r'lower hybrid frequency')
    axs.legend(loc='upper left')
    axs.set_ylabel(r"Frequency [$\Omega_{c,\alpha}$]")
    axs.set_xlabel(r"Wavenumber [$\frac{\Omega_{c,\alpha}}{v_A}$]")
    axs.grid()
    fig.tight_layout()
    filename = Path(f'{runName}_{field.replace("_", "").replace("$", "")}_wk_positiveK_log_maxK-{maxK if maxK is not None else "all"}_maxW-{maxW if maxW is not None else "all"}.png')
    fig.savefig(str(saveDirectory / filename))
    if display:
        plt.show()
        plt.close("all")
    
    del(spec)
    del(log_spec)

    return maxPowerFrequenciesByK

def create_t_k_spectrum(
        originalFftSpectrum : xr.DataArray, 
        statsFile : nc.Dataset = None,
        maxK : float = 100.0,
        load : bool = True,
        debug : bool = False
) -> xr.DataArray :
    
    tk_spec = originalFftSpectrum.where(originalFftSpectrum.frequency>0.0, 0.0)
    original_zero_freq_amplitude = tk_spec.sel(wavenumber=0.0)
    # Double spectrum to conserve E
    tk_spec = np.sqrt(2.0) * tk_spec # <---- Should this be sqrt(2)?
    tk_spec.loc[dict(wavenumber=0.0)] = original_zero_freq_amplitude # Restore original 0-freq amplitude
    tk_spec = xrft.xrft.ifft(tk_spec, dim="frequency")
    tk_spec = tk_spec.rename(freq_frequency="time")
    abs_spec = np.abs(tk_spec)
    
    tk_sum = float(abs_spec.sum())
    tk_squared = float((abs_spec**2).sum())
    parseval_tk = tk_squared  * abs_spec.coords['wavenumber'].spacing * abs_spec.coords['time'].spacing
    tk_peak = float(np.nanmax(abs_spec))
    tk_mean = float(abs_spec.mean())

    if statsFile is not None:
        # Log stats on spectrum
        statsFile.totalTkSpectralPower = tk_sum
        statsFile.parsevalTk = parseval_tk
        statsFile.peakTkSpectralPower = tk_peak
        statsFile.meanTkSpectralPower = tk_mean
        statsFile.peakTkSpectralPowerRatio = tk_peak/tk_mean
    
    if debug:
        print(f"Sum of t-k squared * dk * dt: {parseval_tk}")
        print(f"Max peak in t-k: {tk_peak}")
        print(f"Mean of t-k: {tk_mean}")
        print(f"Ratio of peak to mean in t-k: {tk_peak/tk_mean}")

    if maxK is not None:
        tk_spec = tk_spec.sel(wavenumber=tk_spec.wavenumber<=maxK)
        tk_spec = tk_spec.sel(wavenumber=tk_spec.wavenumber>=-maxK)

    if load:
        tk_spec = tk_spec.load()

    return tk_spec

def create_t_k_plot(
        tkSpectrum : xr.DataArray,
        field : str,
        field_unit : str,
        saveDirectory : Path = None,
        runName : str = None,
        maxK : float = 100.0,
        display : bool = False):
    
    print("Generating t-k plot....")

    if maxK is not None:
        tkSpec_plot = tkSpectrum.sel(wavenumber=tkSpectrum.wavenumber<=maxK)
        tkSpec_plot = tkSpec_plot.sel(wavenumber=tkSpec_plot.wavenumber>=-maxK)
    tkSpec_plot : xr.DataArray = np.abs(tkSpec_plot) # Returns DataArray somehow (xarray magic)
    tkSpec_plot_log : xr.DataArray = np.log10(tkSpec_plot) # Returns DataArray somehow (xarray magic)
    
    # Time-wavenumber
    fig, axs = plt.subplots(figsize=(15, 10))
    field_name = fieldNameToText(field)
    tkSpec_plot.plot(ax=axs, x = "wavenumber", y = "time", cbar_kwargs={'label': f'{field_name} power [{field_unit}]'}, cmap='plasma')
    axs.grid()
    axs.set_xlabel(r"Wavenumber [$\frac{\Omega_{c,\alpha}}{v_A}$]")
    axs.set_ylabel(r"Time [$\tau_{c,\alpha}$]")
    fig.tight_layout()
    if saveDirectory is not None:
        filename = Path(f'{runName}_{field.replace("_", "")}_tk_maxK-{maxK if maxK is not None else "all"}.png')
        fig.savefig(str(saveDirectory / filename))
    if display:
        plt.show()
    plt.close('all')

    # Time-wavenumber (positive)
    fig, axs = plt.subplots(figsize=(15, 10))
    field_name = fieldNameToText(field)
    tkSpec_plot.sel(wavenumber=tkSpec_plot.wavenumber>=0.0).plot(ax=axs, x = "wavenumber", y = "time", cbar_kwargs={'label': f'{field_name} power [{field_unit}]'}, cmap='plasma')
    axs.grid()
    axs.set_xlabel(r"Wavenumber [$\frac{\Omega_{c,\alpha}}{v_A}$]")
    axs.set_ylabel(r"Time [$\tau_{c,\alpha}$]")
    fig.tight_layout()
    if saveDirectory is not None:
        filename = Path(f'{runName}_{field.replace("_", "")}_tk_maxK-{maxK if maxK is not None else "all"}_pos.png')
        fig.savefig(str(saveDirectory / filename))
    if display:
        plt.show()
    plt.close('all')

    # Time-wavenumber (log)
    fig, axs = plt.subplots(figsize=(15, 10))
    tkSpec_plot_log.plot(ax=axs, x = "wavenumber", y = "time", cbar_kwargs={'label': f'{field_name} power [{r"log$_{10}$"}]'}, cmap='plasma')
    axs.grid()
    axs.set_xlabel(r"Wavenumber [$\frac{\Omega_{c,\alpha}}{v_A}$]")
    axs.set_ylabel(r"Time [$\tau_{c,\alpha}$]")
    fig.tight_layout()
    if saveDirectory is not None:
        filename = Path(f'{runName}_{field.replace("_", "")}_tk_log_maxK-{maxK if maxK is not None else "all"}.png')
        fig.savefig(str(saveDirectory / filename))
    if display:
        plt.show()
    plt.close('all')

    # Time-wavenumber (log, positive)
    fig, axs = plt.subplots(figsize=(15, 10))
    tkSpec_plot_log.sel(wavenumber=tkSpec_plot_log.wavenumber>=0.0).plot(ax=axs, x = "wavenumber", y = "time", cbar_kwargs={'label': f'{field_name} power [{r"log$_{10}$"}]'}, cmap='plasma')
    axs.grid()
    axs.set_xlabel(r"Wavenumber [$\frac{\Omega_{c,\alpha}}{v_A}$]")
    axs.set_ylabel(r"Time [$\tau_{c,\alpha}$]")
    fig.tight_layout()
    if saveDirectory is not None:
        filename = Path(f'{runName}_{field.replace("_", "")}_tk_log_maxK-{maxK if maxK is not None else "all"}_pos.png')
        fig.savefig(str(saveDirectory / filename))
    if display:
        plt.show()
    plt.close('all')

def create_autobispectrum(
        signalWindows : list, 
        maxK = None, 
        mask : bool = True
):
    initBispec = True
    count = 0
    for window in signalWindows:
        
        # Build bispectrum by averaging FFT data around the time point
        if initBispec:
            bispec = np.zeros((window.shape[1], window.shape[1]), dtype=complex)
            initBispec = False
        
        # y = spec.mean(dim="time").to_numpy()
        y = window.to_numpy()
        nfft = window.shape[1]
        # Create all combinations of k1 and k2
        k = np.fft.fftshift(np.fft.fftfreq(nfft, 1/nfft))
        K1, K2 = np.meshgrid(k, k)
        K3 = K1 + K2
        i1 = (K1 - k.min()).astype(int)
        i2 = (K2 - k.min()).astype(int)
        i3 = (K3 - k.min()).astype(int)

        # Mask
        k_mask = np.abs(i3) < (nfft/2)
        valid_I3 = np.where(k_mask, i3, 0)
        Y3_conj = np.conj(y[:,valid_I3])

        # Use broadcasting to access X[k1], X[k2], X[k1 + k2]
        # b = y[:,K1] * y[:,K2] * Y3_conj
        bispec += np.mean(y[:,i1] * y[:,i2] * Y3_conj, axis = 0)
        bispec[np.logical_not(k_mask)] = 0.0

        count += 1

    # bispec = np.fft.fftshift(bispec) / count
    bispec = bispec / count
    waxis = np.linspace(-maxK, maxK, bispec.shape[0])

    if mask:
        # Lower triangle mask
        lower_mask = np.tri(bispec.shape[0], bispec.shape[1], k=-1)
        bispec = np.ma.array(bispec, mask = lower_mask)

    return bispec, waxis
    
def create_autobicoherence(
        signalWindows : list, 
        maxK = None, 
        mask : bool = True
):
    initBicoh = True
    count = 0
    for window in signalWindows:
        
        # Build bispectrum by averaging FFT data around the time point
        if initBicoh:
            bicoh = np.zeros((window.shape[1], window.shape[1]), dtype=complex)
            initBicoh = False
        
        # y = spec.mean(dim="time").to_numpy()
        y = window.to_numpy()
        nfft = window.shape[1]
        # Create all combinations of k1 and k2
        k = np.fft.fftshift(np.fft.fftfreq(nfft, 1/nfft))
        K1, K2 = np.meshgrid(k, k)
        K3 = K1 + K2
        i1 = (K1 - k.min()).astype(int)
        i2 = (K2 - k.min()).astype(int)
        i3 = (K3 - k.min()).astype(int)

        # Mask
        k_mask = np.abs(i3) < (nfft/2)
        valid_I3 = np.where(k_mask, i3, 0)
        Y3_conj = np.conj(y[:,valid_I3])

        # Use broadcasting to access X[k1], X[k2], X[k1 + k2]
        bicoh_top = np.abs(np.mean(y[:,i1] * y[:,i2] * Y3_conj, axis = 0))**2
        bicoh_bottom = np.mean(np.abs(y[:,i1] * y[:,i2])**2, axis=0) * np.mean(np.abs(Y3_conj)**2, axis=0)
        bicoh += bicoh_top / bicoh_bottom

        count += 1

    # bispec = np.fft.fftshift(bispec) / count
    bicoh = bicoh / count
    waxis = np.linspace(-maxK, maxK, bicoh.shape[0])

    if mask:
        # Lower triangle mask
        lower_mask = np.tri(bicoh.shape[0], bicoh.shape[1], k=-1)
        bicoh = np.ma.array(bicoh, mask = lower_mask)

    return bicoh, waxis

def bispectral_analysis(
        tkSpectrum : xr.DataArray, 
        runName : str,
        field : str,
        display : bool = False,
        saveDirectory : Path = None,
        timePoint_tci : float = None, 
        totalWindow_tci = 2.0, 
        fftWindowSize_tci = 0.25, 
        overlap = 0.5, 
        maxK = None, 
        mask : bool = True
):
    print("Doing bispectral analysis....")
    fieldName = fieldNameToText(field)
    times = np.array(tkSpectrum.coords["time"])

    timeCentre = timePoint_tci
    if timePoint_tci is None:
        timeCentre = times[len(times)//2]
        totalWindow_tci = np.floor(times[-1])
        fftWindowSize_tci = 0.25
        overlap = 0.25

    # Work out how many indices in each window and window spectrum
    windowSize_indices = int((times.size / times[-1]) * fftWindowSize_tci)
    overlap_indices = int(np.round(overlap * windowSize_indices))
    windowStart_tci = timeCentre - (0.5 * totalWindow_tci)
    windowStart_idx = np.where(abs(times-windowStart_tci)==abs(times-windowStart_tci).min())[0][0]
    windowEnd_tci = timeCentre + (0.5 * totalWindow_tci)
    windowEnd_idx = np.where(abs(times-windowEnd_tci)==abs(times-windowEnd_tci).min())[0][0]
    signalWindows = []
    startIdx = windowStart_idx
    endIdx = startIdx + windowSize_indices
    while endIdx < windowEnd_idx:
        w = tkSpectrum.isel(time=slice(startIdx, endIdx))
        signalWindows.append(w)
        startIdx = (endIdx + 1) - overlap_indices
        endIdx = startIdx + windowSize_indices

    # Autobispectrum
    bispec, waxis = create_autobispectrum(signalWindows, maxK, mask)
    maxis_waxis = waxis[-1]
    minis_waxis = waxis[0]
    fig, axs = plt.subplots(figsize=(12, 12))
    img = axs.imshow(np.abs(bispec), extent=[minis_waxis, maxis_waxis, minis_waxis, maxis_waxis], origin="lower", cmap="plasma")
    axs.set_title(f"{fieldName} autobispectrum t = {timeCentre}" if timePoint_tci is not None else f"{fieldName} autobispectrum (full time window)", pad=20.0)
    axs.set_xlabel('Wavenumber $k_1$')
    axs.set_ylabel('Wavenumber $k_2$')
    axs.grid(True)
    fig.colorbar(img, ax=axs, label='Magnitude', shrink = 0.7)
    fig.tight_layout()
    if saveDirectory is not None:
        filename = Path(f'{runName}_{field.replace("_", "")}_autobispectrum_t{timeCentre:.3f}_maxK-{maxK if maxK is not None else "all"}.png')
        fig.savefig(str(saveDirectory / filename))
    if display:
        plt.show()
    plt.close('all')

    # Log autobispectrum
    fig, axs = plt.subplots(figsize=(12, 12))
    img = axs.imshow(np.log10(np.abs(bispec)), extent=[minis_waxis, maxis_waxis, minis_waxis, maxis_waxis], origin="lower", cmap="plasma")
    axs.set_title(f"Log10 {fieldName} autobispectrum t = {timeCentre}" if timePoint_tci is not None else f"Log10 {fieldName} autobispectrum (full time window)", pad=20.0)
    axs.set_xlabel('Wavenumber $k_1$')
    axs.set_ylabel('Wavenumber $k_2$')
    axs.grid(True)
    fig.colorbar(img, ax=axs, label='Log magnitude', shrink = 0.7)
    fig.tight_layout()
    if saveDirectory is not None:
        filename = Path(f'{runName}_log_{field.replace("_", "")}_autobispectrum_t{timeCentre:.3f}_maxK-{maxK if maxK is not None else "all"}.png')
        fig.savefig(str(saveDirectory / filename))
    if display:
        plt.show()
    plt.close('all')

    # Autobicoherence
    bicoh, waxis = create_autobicoherence(signalWindows, maxK, mask)
    maxis_waxis = waxis[-1]
    minis_waxis = waxis[0]
    fig, axs = plt.subplots(figsize=(12, 12))
    img = axs.imshow(np.abs(bicoh), extent=[minis_waxis, maxis_waxis, minis_waxis, maxis_waxis], origin="lower", cmap="plasma")
    axs.set_title(f"{fieldName} auto-bicoherence{r'$^2$'} t = {timeCentre}" if timePoint_tci is not None else f"{fieldName} auto-bicoherence{r'$^2$'} (full time window)", pad=20.0)
    axs.set_xlabel('Wavenumber $k_1$')
    axs.set_ylabel('Wavenumber $k_2$')
    axs.grid(True)
    fig.colorbar(img, ax=axs, label='Magnitude', shrink = 0.7)
    fig.tight_layout()
    if saveDirectory is not None:
        filename = Path(f'{runName}_{field.replace("_", "")}_autobicoherence_t{timeCentre:.3f}_maxK-{maxK if maxK is not None else "all"}.png')
        fig.savefig(str(saveDirectory / filename))
    if display:
        plt.show()
    plt.close('all')
    
def create_netCDF_fieldGrowthRate_structure(
        fieldRoot : nc.Dataset,
        numWavenumbersToEvaluate : int
) -> nc.Dataset:
    growth_rate_group = fieldRoot.createGroup("growthRates")
    posGrowthRateGrp = growth_rate_group.createGroup("positive")
    negGrowthRateGrp = growth_rate_group.createGroup("negative")
    groups = [posGrowthRateGrp, negGrowthRateGrp]
    
    # NOTE: This will skip adding fields if the "wavenumber" dimension is already present, so this assumes no changes to the variable creation below is required.
    #       A more sophisticated diff would be better.
    for group in groups:
        if "wavenumber" in group.dimensions.keys():
            continue
        else:
            group.createDimension("wavenumber", numWavenumbersToEvaluate)

            k_var = group.createVariable("wavenumber", datatype="f4", dimensions=("wavenumber",))
            k_var.units = "wCI/vA"
            group.createVariable("frequencyOfMaxPowerInK", datatype="f4", dimensions=("wavenumber",))
            group.createVariable("peakPower", datatype="f4", dimensions=("wavenumber",))
            group.createVariable("totalPower", datatype="f4", dimensions=("wavenumber",))
            t_var = group.createVariable("time", datatype="f4", dimensions=("wavenumber",))
            t_var.units = "tCI"
            gamma_var = group.createVariable("growthRate", datatype="f8", dimensions=("wavenumber",))
            gamma_var.units = "wCI"
            gamma_var.standard_name = "linear_growth_rate"
            group.createVariable("yIntercept", datatype="f4", dimensions=("wavenumber",))
            group.createVariable("rSquared", datatype="f4", dimensions=("wavenumber",))
            group.createVariable("stdErr", datatype="f4", dimensions=("wavenumber",))
            group.createVariable("pValue", datatype="f4", dimensions=("wavenumber",))
            group.createVariable("rawRSquared", datatype="f4", dimensions=("wavenumber",))
            group.createVariable("rawStdErr", datatype="f4", dimensions=("wavenumber",))
            group.createVariable("rawPValue", datatype="f4", dimensions=("wavenumber",))

    return growth_rate_group

def plot_growth_rates(
        tkSpectrum : xr.DataArray,
        field : str,
        growthRateData : list[LinearGrowthRate],
        numToPlot : int,
        selectionMetric : str,
        save : bool,
        display : bool,
        noTitle : bool,
        saveFolder : Path,
        runName : str,
        debug : bool):

    # Short-circuit
    if numToPlot == 0:
        return
    
    spectrum = np.abs(tkSpectrum)

    signString = "positive" if growthRateData[0].gamma > 0.0 else "negative"
    
    if debug:
        print(f"Plotting best {signString} growth rates in {field} of {numToPlot} {selectionMetric} power wavenumbers....")
    
    if selectionMetric == "peak":
        # Find n highest peak or total powers
        growth_rates_to_plot = sorted(growthRateData, key=lambda gamma: gamma.peakPower, reverse=True)[:numToPlot]
    elif selectionMetric == "total":
        growth_rates_to_plot = sorted(growthRateData, key=lambda gamma: gamma.totalPower, reverse=True)[:numToPlot]
    else:
        raise NotImplementedError("Only \'peak\' or \'total\' power selection criteria implemented.")
    
    rank = 0
    for g in growth_rates_to_plot:
        signal = spectrum.sel(wavenumber=g.wavenumber)
        timeVals = signal.coords['time'][g.timeStartIndex:g.timeEndIndex]
        plt.close("all")
        fig, ax = plt.subplots(figsize=(12, 8))
        logSignal = np.log10(signal)
        logSignal.plot(ax=ax, alpha = 0.5, color = "blue")
        if g.smoothingFunction is not None:
            ax.plot(logSignal.coords['time'], np.log(g.smoothingFunction(logSignal.coords['time'])), linestyle = "dashed", color="purple", label = "Smoothed signal")
        ax.plot(timeVals, g.gamma * timeVals + g.yIntercept, color = "orange", label = r"$\gamma = $" + f"{g.gamma:.3f}" + r"$\pm$" + f"{g.stdErr:.3f}" + r"$\Omega_{ci}$")
        ax.set_xlabel(r"Time [$\tau_{ci}$]")
        ax.set_ylabel(f"Log10 of {field} signal power")
        ax.grid()
        ax.legend()
        if not noTitle:
            plt.title(f'{runName}_{field}_{signString}_growth_k_{g.wavenumber:.3f}_{selectionMetric}_power_rank_{rank}')
        if save:
            plt.savefig(saveFolder / Path(f"{runName}_{field}_{signString}_growth_k_{g.wavenumber:.3f}_{selectionMetric}Power_rank_{rank}.png"))
        if display:
            plt.show()
        rank += 1
        plt.close('all')

def find_best_growth_rates(
        tkSpectrum : xr.DataArray,
        gammaWindowPctMin : int,
        gammaWindowPctMax : int,
        wavenumberToFrequencyTable : dict,
        useSmoothing : bool = True,
        debug : bool = False):

    spectrum = np.abs(tkSpectrum)
    # Change: This now only evaluates at integer values of percentage, i.e. if 5% - 15% range is given, this evaluates at 5%, 6%, 7% etc.
    gammaWindowIndicesMin = int(np.rint((gammaWindowPctMin / 100.0) * spectrum.coords['time'].size))
    gammaWindowIndicesMax = int(np.rint((gammaWindowPctMax / 100.0) * spectrum.coords['time'].size))
    onePercentIndices = int(np.rint(0.01 * spectrum.coords['time'].size))

    best_pos_growth_rates = []
    best_neg_growth_rates = []

    num_wavenumbers = spectrum.sizes["wavenumber"]

    for index in range(0, num_wavenumbers):

        # Must load data here for quick iteration over windows
        # Changed: this is pre-loaded in create_t_k_spectrum()
        rawSignal = spectrum.isel(wavenumber=index)

        signalK=float(spectrum.coords['wavenumber'][index])
        signalPeak=float(rawSignal.max())
        signalTotal=float(rawSignal.sum())

        if useSmoothing:
            # Smooth signal for finding growth rates
            smoothingFunction = make_smoothing_spline(rawSignal.coords["time"], np.nan_to_num(rawSignal), lam = 0.01)
            signal = smoothingFunction(rawSignal.coords["time"])
            # rawSignal.plot(alpha = 0.7)
            # plt.plot(rawSignal.coords["time"], signal, linestyle = "dashed", color = "purple")
            # plt.show()    
        else:
            smoothingFunction = None
            signal = rawSignal        

        windowWidths = range(gammaWindowIndicesMin, gammaWindowIndicesMax + 1, onePercentIndices)
        len_widths = len(windowWidths)

        best_pos_params = None
        best_neg_params = None
        best_pos_r_squared = float('-inf')
        best_neg_r_squared = float('-inf')

        width_count = 0
        for width in windowWidths: # For each possible window width
            
            windowStarts = range(0, len(signal) - (width + 1), onePercentIndices)

            if debug:
                width_count += 1
                len_windows = len(windowStarts)
                window_count = 0

            for window in windowStarts: # For each window
                
                if debug:
                    window_count += 1
                    print(f"Processing width {width} ({(width/spectrum.coords['time'].size)*100.0}% of signal) starting at {window} in k = {signalK}. Width {width_count}/{len_widths} window {window_count}/{len_windows}....")

                t_k_window = signal[window:(width + window)]

                result = stats.linregress(rawSignal.coords["time"][window:(width + window)], np.log10(t_k_window))
                r_squared = result.rvalue ** 2
                
                if not np.isnan(result.slope):
                    if result.slope > 0.0:
                        if r_squared > best_pos_r_squared:
                            best_pos_r_squared = r_squared
                            best_pos_params = (result.slope, result.intercept, width, window, r_squared, result.stderr, result.pvalue)
                    else:
                        if r_squared > best_neg_r_squared:
                            best_neg_r_squared = r_squared
                            best_neg_params = (result.slope, result.intercept, width, window, r_squared, result.stderr, result.pvalue)

        if best_pos_params is not None:

            gamma, y_int, window_width, window_start, r_sqrd, std_err, p_value = best_pos_params

            # Re-fit to raw data to get stats
            rawDataFit = stats.linregress(rawSignal.coords["time"][window:(width + window)], np.log10(rawSignal[window:(width + window)]))

            best_pos_growth_rates.append(
                LinearGrowthRate(timeStartIndex=window_start,
                                timeEndIndex=(window_start + window_width),
                                timeMidpointIndex=window_start+(int(window_width/2)),
                                gamma=gamma,
                                yIntercept=y_int,
                                rSquared=r_sqrd,
                                stdErr=std_err,
                                pValue=p_value,
                                rawRSquared=rawDataFit.rvalue ** 2,
                                rawStdErr=rawDataFit.stderr,
                                rawPValue=rawDataFit.pvalue,
                                wavenumber=signalK,
                                maxPowerFrequency=wavenumberToFrequencyTable[signalK],
                                timeMidpoint=float(spectrum.coords['time'][window_start+(int(window_width/2))]),
                                peakPower = signalPeak,
                                totalPower = signalTotal,
                                smoothingFunction=smoothingFunction))
            
        if best_neg_params is not None:
            gamma, y_int, window_width, window_start, r_sqrd, std_err, p_value = best_neg_params

            # Re-fit to raw data to get stats
            rawDataFit = stats.linregress(rawSignal.coords["time"][window:(width + window)], np.log10(rawSignal[window:(width + window)]))

            best_neg_growth_rates.append(
                LinearGrowthRate(timeStartIndex=window_start,
                                timeEndIndex=(window_start + window_width),
                                timeMidpointIndex=window_start+(int(window_width/2)),
                                gamma=gamma,
                                yIntercept=y_int,
                                rSquared=r_sqrd,
                                stdErr=std_err,
                                pValue=p_value,
                                rawRSquared=rawDataFit.rvalue ** 2,
                                rawStdErr=rawDataFit.stderr,
                                rawPValue=rawDataFit.pvalue,
                                wavenumber=signalK,
                                maxPowerFrequency=wavenumberToFrequencyTable[signalK],
                                timeMidpoint=float(spectrum.coords['time'][window_start+(int(window_width/2))]),
                                peakPower = signalPeak,
                                totalPower = signalTotal,
                                smoothingFunction=smoothingFunction))
            
        del(rawSignal)
        del(signal)
        
        if debug: 
            if best_pos_params is not None:
                print(f"Max positive gamma found: {best_pos_growth_rates[-1].to_string()}")
            if best_neg_params is not None:
                print(f"Max negative gamma found: {best_neg_growth_rates[-1].to_string()}")

    return best_pos_growth_rates, best_neg_growth_rates

def process_growth_rates(
        tkSpectrum : xr.DataArray,
        fieldRoot : nc.Dataset,
        plotFieldFolder : Path,
        simFolder : Path,
        field : str,
        gammaWindowPctMin : float,
        gammaWindowPctMax : float,
        saveGrowthRatePlots : bool,
        numGrowthRatesToPlot : int,
        wavenumberToFrequencyTable : dict,
        displayPlots : bool,
        noTitle : bool,
        debug : bool):

    print("Processing growth rates....")

    best_pos_gammas, best_neg_gammas = find_best_growth_rates(tkSpectrum, gammaWindowPctMin, gammaWindowPctMax, wavenumberToFrequencyTable, useSmoothing = True, debug = debug)
    maxNumGammas = np.max([len(best_pos_gammas), len(best_neg_gammas)])
    growthRateStatsRoot = create_netCDF_fieldGrowthRate_structure(fieldRoot, maxNumGammas)

    if saveGrowthRatePlots:
        gammaPlotFolder = plotFieldFolder / "growth_rates"
        if not os.path.exists(gammaPlotFolder):
            os.mkdir(gammaPlotFolder)
        plot_growth_rates(tkSpectrum, field, best_pos_gammas, numGrowthRatesToPlot, "peak", saveGrowthRatePlots, displayPlots, noTitle, gammaPlotFolder, simFolder.name, debug)
        plot_growth_rates(tkSpectrum, field, best_neg_gammas, numGrowthRatesToPlot, "total", saveGrowthRatePlots, displayPlots, noTitle, gammaPlotFolder, simFolder.name, debug)

    # Positive growth
    posGammaNc = growthRateStatsRoot.groups["positive"]
    for i in range(len(best_pos_gammas)):
        gamma = best_pos_gammas[i]
        posGammaNc.variables["wavenumber"][i] = gamma.wavenumber
        posGammaNc.variables["frequencyOfMaxPowerInK"][i] = gamma.maxPowerFrequency
        posGammaNc.variables["peakPower"][i] = gamma.peakPower
        posGammaNc.variables["totalPower"][i] = gamma.totalPower
        posGammaNc.variables["time"][i] = gamma.timeMidpoint
        posGammaNc.variables["growthRate"][i] = gamma.gamma
        posGammaNc.variables["rSquared"][i] = gamma.rSquared
        posGammaNc.variables["stdErr"][i] = gamma.stdErr
        posGammaNc.variables["pValue"][i] = gamma.pValue
        posGammaNc.variables["rawRSquared"][i] = gamma.rawRSquared
        posGammaNc.variables["rawStdErr"][i] = gamma.rawStdErr
        posGammaNc.variables["rawPValue"][i] = gamma.rawPValue
        posGammaNc.variables["yIntercept"][i] = gamma.yIntercept

    keyMetricsIndices = {
        "maxFoundInSimulation" : np.argmax([g.gamma for g in best_pos_gammas]),
        "bestInHighestPeakPowerK" : np.argmax([g.peakPower for g in best_pos_gammas]),
        "bestInHighestTotalPowerK" : np.argmax([g.totalPower for g in best_pos_gammas])
    }
    for metric, index in keyMetricsIndices.items():
        group = posGammaNc.createGroup(metric)
        gamma = best_pos_gammas[index]
        group.growthRate = float(gamma.gamma)
        group.time=float(gamma.timeMidpoint)
        group.yIntercept=float(gamma.yIntercept)
        group.rSquared=float(gamma.rSquared)
        group.stdErr = float(gamma.stdErr)
        group.pValue = float(gamma.pValue)
        group.rawRSquared = float(gamma.rawRSquared)
        group.rawStdErr = float(gamma.rawStdErr)
        group.rawPValue = float(gamma.rawPValue)
        group.wavenumber=float(gamma.wavenumber)
        group.frequencyOfMaxPowerInK=float(gamma.maxPowerFrequency)
        group.peakPower=float(gamma.peakPower)
        group.totalPower=float(gamma.totalPower)

    # Negative growth
    negGammaNc = growthRateStatsRoot.groups["negative"]
    for i in range(len(best_neg_gammas)):
        gamma = best_neg_gammas[i]
        negGammaNc.variables["wavenumber"][i] = gamma.wavenumber
        negGammaNc.variables["frequencyOfMaxPowerInK"][i] = gamma.maxPowerFrequency
        negGammaNc.variables["peakPower"][i] = gamma.peakPower
        negGammaNc.variables["totalPower"][i] = gamma.totalPower
        negGammaNc.variables["time"][i] = gamma.timeMidpoint
        negGammaNc.variables["growthRate"][i] = gamma.gamma
        negGammaNc.variables["rSquared"][i] = gamma.rSquared
        negGammaNc.variables["stdErr"][i] = gamma.stdErr
        negGammaNc.variables["pValue"][i] = gamma.pValue
        negGammaNc.variables["rawRSquared"][i] = gamma.rawRSquared
        negGammaNc.variables["rawStdErr"][i] = gamma.rawStdErr
        negGammaNc.variables["rawPValue"][i] = gamma.rawPValue
        negGammaNc.variables["yIntercept"][i] = gamma.yIntercept

    keyMetricsIndices["maxFoundInSimulation"] = np.argmin([g.gamma for g in best_neg_gammas])
    keyMetricsIndices["bestInHighestPeakPowerK"] = np.argmax([g.peakPower for g in best_neg_gammas])
    keyMetricsIndices["bestInHighestTotalPowerK"] = np.argmax([g.totalPower for g in best_neg_gammas])
    for metric, index in keyMetricsIndices.items():
        group = negGammaNc.createGroup(metric)
        gamma = best_neg_gammas[index]
        group.growthRate = float(gamma.gamma)
        group.time=float(gamma.timeMidpoint)
        group.yIntercept=float(gamma.yIntercept)
        group.rSquared=float(gamma.rSquared)
        group.stdErr = float(gamma.stdErr)
        group.pValue = float(gamma.pValue)
        group.rawRSquared = float(gamma.rawRSquared)
        group.rawStdErr = float(gamma.rawStdErr)
        group.rawPValue = float(gamma.rawPValue)
        group.wavenumber=float(gamma.wavenumber)
        group.frequencyOfMaxPowerInK=float(gamma.maxPowerFrequency)
        group.peakPower=float(gamma.peakPower)
        group.totalPower=float(gamma.totalPower)

def my_matrix_plot(
    data_series: list[list],
    series_labels: list[str] = [],
    parameter_labels: list[str] = None,
    show: bool = True,
    reference: Sequence[float] = None,
    filename: str = None,
    plot_style: str = "contour",
    colormap_list: list = ["Blues", "Greens"],
    show_ticks: bool = None,
    equalise_pdf_heights: bool = True,
    point_colors: Sequence[float] = None,
    hdi_fractions=(0.35, 0.65, 0.95),
    point_size: int = 1,
    label_size: int = 10
):
    """
    Construct a 'matrix plot' for a set of variables which shows all possible
    1D and 2D marginal distributions.

    :param data_series: \
        A list of data series', each of which is a list of array-like objects containing the samples for each variable.
    
    :param series_labels: \
        A list of strings to be used as labels for each data series being plotted, in the same order as those in data_series.

    :param parameter_labels: \
        A list of strings to be used as axis labels for each parameter being plotted, in the same order as those in data_series.

    :param bool show: \
        Sets whether the plot is displayed.

    :param reference: \
        A list of reference values for each parameter which will be over-plotted.

    :param str filename: \
        File path to which the matrix plot will be saved (if specified).

    :param str plot_style: \
        Specifies the type of plot used to display the 2D marginal distributions.
        Available styles are 'contour' for filled contour plotting, 'hdi' for
        highest-density interval contouring, 'histogram' for hexagonal-bin histogram,
        and 'scatter' for scatterplot.

    :param str colormap_list: \
        A list of colormaps to be used for plotting. Must be at least as long as len(data_series)
        and contain names of valid colormaps present in ``matplotlib.colormaps``.

    :param bool show_ticks: \
        By default, axis ticks are only shown when plotting less than 6 variables.
        This behaviour can be overridden for any number of parameters by setting
        show_ticks to either True or False.

    :param point_colors: \
        An array containing data which will be used to set the colors of the points
        if the plot_style argument is set to 'scatter'.

    :param point_size: \
        An array containing data which will be used to set the size of the points
        if the plot_style argument is set to 'scatter'.

    :param hdi_fractions: \
        The highest-density intervals used for contouring, specified in terms of
        the fraction of the total probability contained in each interval. Should
        be given as an iterable of floats, each in the range [0, 1].

    :param int label_size: \
        The font-size used for axis labels.
    """
    N_series = len(data_series)
    N_par = len(data_series[0])
    if parameter_labels is None:  # set default axis labels if none are given
        if N_par >= 10:
            parameter_labels = [f"p{i}" for i in range(N_par)]
        else:
            parameter_labels = [f"param {i}" for i in range(N_par)]
    else:
        if len(parameter_labels) != N_par:
            raise ValueError(
                """\n
                \r[ matrix_plot error ]
                \r>> The number of labels given does not match
                \r>> the number of plotted parameters.
                """
            )

    if reference is not None:
        if len(reference) != N_par:
            raise ValueError(
                """\n
                \r[ matrix_plot error ]
                \r>> The number of reference values given does not match
                \r>> the number of plotted parameters.
                """
            )
    # check that given plot style is valid, else default to a histogram
    if plot_style not in ["contour", "hdi", "histogram", "scatter"]:
        plot_style = "contour"
        warn(
            "'plot_style' must be set as either 'contour', 'hdi', 'histogram' or 'scatter'"
        )

    iterable = hasattr(hdi_fractions, "__iter__")
    if not iterable or not all(0 < f < 1 for f in hdi_fractions):
        raise ValueError(
            """\n
            \r[ matrix_plot error ]
            \r>> The 'hdi_fractions' argument must be given as an
            \r>> iterable of floats, each in the range [0, 1].
            """
        )

    # by default, we suppress axis ticks if there are 6 parameters or more to keep things tidy
    if show_ticks is None:
        show_ticks = N_par < 6

    L = 200
    cmaps = []
    marginal_colors = []
    cmap_count = 0
    for c in colormap_list:
        if c in colormaps:
            cmaps.append(colormaps[c])
        else:
            cmaps.append(colormaps[cmap_count])
            cmap_count += 1
            warn(f"'{c}' is not a valid colormap from matplotlib.colormaps. Using default.")
        # find the darker of the two ends of the colormap, and use it for the marginal plots
        marginal_colors.append(sorted([cmaps[-1](10), cmaps[-1](245)], key=lambda x: sum(x[:-1]))[0])

    # build axis arrays and determine limits for all variables
    axis_limits = []
    axis_arrays = []
    if type(data_series[0][0]) is list:
        parameters = copy.deepcopy(data_series[0])
    else:
        parameters = copy.deepcopy([s.tolist() for s in data_series[0]]) 
    for n_series in range(1, N_series):
        for n_sample in range(N_par):
            if type(data_series[n_series][n_sample]) is list:
                parameters[n_sample] += data_series[n_series][n_sample]
            else:
                parameters[n_sample] += data_series[n_series][n_sample].tolist()
    for sample in parameters:
        # get the 98% HDI to calculate plot limits
        lwr, upr = sample_hdi(sample, fraction=0.98)
        # store the limits and axis array
        axis_limits.append([lwr - (upr - lwr) * 0.3, upr + (upr - lwr) * 0.3])
        axis_arrays.append(
            np.linspace(lwr - (upr - lwr) * 0.35, upr + (upr - lwr) * 0.35, L)
        )

    fig = plt.figure(figsize=(14, 14))
    # build a lower-triangular indices list in diagonal-striped order
    inds_list = [(N_par - 1, 0)]  # start with bottom-left corner
    for k in range(1, N_par):
        inds_list.extend([(N_par - 1 - i, k - i) for i in range(k + 1)])

    # now create a dictionary of axis objects with correct sharing
    axes = {}
    for tup in inds_list:
        i, j = tup
        x_share = None
        y_share = None

        if i < N_par - 1:
            x_share = axes[(N_par - 1, j)]

        if (j > 0) and (i != j):  # diagonal doesnt share y-axis
            y_share = axes[(i, 0)]

        axes[tup] = plt.subplot2grid(
            (N_par, N_par), (i, j), sharex=x_share, sharey=y_share
        )

    # Pre-compute estimates for correct scaling of data_series
    all_estimates = [[] for _ in range(N_par)]
    for n_series in range(N_series):
        samples = data_series[n_series]
        for tup in inds_list:
            i, j = tup
            ax = axes[tup]
            # are we on the diagonal?
            if i == j:
                sample = samples[i]
                pdf = GaussianKDE(sample)
                estimate = np.array(pdf(axis_arrays[i], equalise_pdf_heights))
                all_estimates[i].append(estimate)

    all_estimate_maxes = [np.max(m) for m in all_estimates]
    
    initialiseAxes = True
    for n_series in range(N_series):
        
        samples = data_series[n_series]
        estimates = [par[n_series] for par in all_estimates]
        #estimate_maxes = [par[n_series] for par in parameter_maxes]
        marginal_color = marginal_colors[n_series]
        cmap = cmaps[n_series]

        # now loop over grid and plot
        for tup in inds_list:
            i, j = tup
            ax = axes[tup]
            # are we on the diagonal?
            if i == j:
                estimate = estimates[i]
                ax.plot(
                    axis_arrays[i],
                    0.9 * (estimate / estimate.max()) if equalise_pdf_heights else 0.9 * (estimate / all_estimate_maxes[i]),
                    lw=1,
                    color=marginal_color,
                    label = series_labels[n_series]
                )
                ax.fill_between(
                    axis_arrays[i],
                    0.9 * (estimate / estimate.max()) if equalise_pdf_heights else 0.9 * (estimate / all_estimate_maxes[i]),
                    color=marginal_color,
                    alpha=0.1,
                )
                if reference is not None:
                    ax.plot(
                        [reference[i], reference[i]],
                        [0, 1],
                        lw=1.5,
                        ls="dashed",
                        color="red",
                    )
                ax.set_ylim([0, 1])
            else:
                x = samples[j]
                y = samples[i]

                # plot the 2D marginals
                if plot_style == "hdi":
                    # Filled contour plotting using 2D gaussian KDE
                    pdf = KDE2D(x=x, y=y)
                    sample_probs = pdf(x, y)
                    pcts = [100 * (1 - f) for f in hdi_fractions]
                    levels = [l for l in np.percentile(sample_probs, pcts)]

                    x_ax = axis_arrays[j][::4]
                    y_ax = axis_arrays[i][::4]
                    X, Y = np.meshgrid(x_ax, y_ax)
                    prob = np.array(pdf(X.flatten(), Y.flatten())).reshape([L // 4, L // 4])
                    levels.append(prob.max())
                    levels = sorted(levels)
                    ax.contourf(X, Y, prob, levels=levels, cmap=cmap, alpha=0.7)
                    ax.contour(X, Y, prob, levels=levels, alpha=0.2)
                else:
                    # scatterplot
                    if point_colors is None:
                        ax.scatter(x, y, color=marginal_color, s=point_size)
                    else:
                        ax.scatter(x, y, c=point_colors, s=point_size, cmap=cmap)

                # plot any reference points if given
                if reference is not None:
                    ax.plot(
                        reference[j],
                        reference[i],
                        marker="o",
                        markersize=7,
                        markerfacecolor="none",
                        markeredgecolor="white",
                        markeredgewidth=3.5,
                    )
                    ax.plot(
                        reference[j],
                        reference[i],
                        marker="o",
                        markersize=7,
                        markerfacecolor="none",
                        markeredgecolor="red",
                        markeredgewidth=2,
                    )

            if initialiseAxes:
                # assign axis labels
                if i == N_par - 1:
                    ax.set_xlabel(parameter_labels[j], fontsize=label_size)
                if j == 0 and i != 0:
                    ax.set_ylabel(parameter_labels[i], fontsize=label_size)
                # impose x-limits on bottom row
                if i == N_par - 1:
                    ax.set_xlim(axis_limits[j])
                # impose y-limits on left column, except the top-left corner
                if j == 0 and i != 0:
                    ax.set_ylim(axis_limits[i])

                if show_ticks:  # set up ticks for the edge plots if they are to be shown
                    # hide x-tick labels for plots not on the bottom row
                    if i < N_par - 1:
                        plt.setp(ax.get_xticklabels(), visible=False)
                    # hide y-tick labels for plots not in the left column
                    if j > 0:
                        plt.setp(ax.get_yticklabels(), visible=False)
                    # remove all y-ticks for 1D marginal plots on the diagonal
                    if i == j:
                        ax.set_yticks([])
                else:  # else remove all ticks from all axes
                    ax.set_xticks([])
                    ax.set_yticks([])
        
        initialiseAxes = False

    # Set legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper right")

    # set the plot spacing
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.0, hspace=0.0)
    
    # save/show the figure if required
    if filename is not None:
        plt.savefig(filename)
    if show:
        plt.show()

    return fig

def keV_to_Kelvin(value_in_keV : float) -> float :
    value_in_keV = value_in_keV * u.keV
    value_in_J = value_in_keV.to(u.J)
    kb = constants.k * u.J / u.K
    value_in_Kelvin = value_in_J / kb
    assert value_in_Kelvin.unit == u.K
    return value_in_Kelvin

def MeV_to_Kelvin(value_in_MeV : float) -> float :
    value_in_MeV = value_in_MeV * u.MeV
    value_in_J = value_in_MeV.to(u.J)
    kb = constants.k * u.J / u.K
    value_in_Kelvin = value_in_J / kb
    assert value_in_Kelvin.unit == u.K
    return value_in_Kelvin

def add_field_from_input_deck_to_dataFile(inputDeckFolder : Path, dataFileFolder : Path, field : str):

    parsed_field = field.split("/")

    inputDeck_folders = [Path(f) for f in glob.glob(str(inputDeckFolder / "run_*"))]
    inputDecks = dict.fromkeys([f.name for f in inputDeck_folders])
    for path in inputDeck_folders:
        inputDecks[path.name] = path / "input.deck"

    statsFile_paths = glob.glob(str(dataFileFolder / "run_*_stats.nc"))
    statsFiles = dict.fromkeys(inputDecks.keys())
    for path in statsFile_paths:
        pattern = re.compile(r"run_\d+")
        sim_id = re.search(pattern, path).group()
        statsFiles[sim_id] = path

    assert len(inputDecks.items()) == len(statsFiles.items())
    assert inputDecks.keys() == statsFiles.keys()

    for sim_id, deckPath in inputDecks.items():
        # Read input deck
        inputDeck = {}
        with open(str(deckPath.absolute())) as id:

            # Get input deck
            inputDeck = epydeck.loads(id.read())

            # Get datafile
            dataPath = statsFiles[sim_id]
            data_nc = nc.Dataset(dataPath, "a", format="NETCDF4")

            # Confirm this is the right input deck
            assert inputDeck["constant"]["b0_strength"] == data_nc.B0strength
            assert inputDeck["constant"]["background_density"] == data_nc.backgroundDensity
            assert inputDeck["constant"]["frac_beam"] == data_nc.beamFraction

            # Get field value
            field_value = inputDeck
            for fieldComponent in parsed_field:
                field_value = field_value[fieldComponent]

            print(f"{sim_id}: Copying {field} = {field_value} from {deckPath} to {dataPath}...")
            
            # Set value. NOTE: Only works on root group attributes at present
            data_nc.setncattr(parsed_field[-1], field_value)
            data_nc.close()
