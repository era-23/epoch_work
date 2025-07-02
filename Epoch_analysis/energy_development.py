import glob
from sdf_xarray import SDFPreprocess
from pathlib import Path
from scipy import constants
from inference.plotting import matrix_plot
import epoch_utils as utils
import pandas as pd
import matplotlib.pyplot as plt
import epydeck
import numpy as np
import xarray as xr
import argparse

def calculate_energy(directory : Path, irb : bool, pct : bool):

    # Read dataset
    ds = xr.open_mfdataset(
        str(directory / "*.sdf"),
        data_vars='minimal', 
        coords='minimal', 
        compat='override', 
        preprocess=SDFPreprocess()
    )

    # Drop initial conditions because they may not represent a solution
    ds = ds.sel(time=ds.coords["time"]>ds.coords["time"][0])

    # Read input deck
    input = {}
    with open(str(directory / "input.deck")) as id:
        input = epydeck.loads(id.read())

    density = input['constant']['background_density']
    frac_beam = input['constant']['frac_beam']
        
    proton_density = density * (1.0 - frac_beam) # m^-3
    electron_density = density # m^-3

    proton_E_array : xr.DataArray = ds['Derived_Average_Particle_Energy_proton']
    electron_E_array : xr.DataArray = ds['Derived_Average_Particle_Energy_electron']
    Bx_array : xr.DataArray = ds['Magnetic_Field_Bx']
    By_array : xr.DataArray = ds['Magnetic_Field_By']
    Bz_array : xr.DataArray = ds['Magnetic_Field_Bz']
    Ex_array : xr.DataArray = ds['Electric_Field_Ex']
    Ey_array : xr.DataArray = ds['Electric_Field_Ey']
    Ez_array : xr.DataArray = ds['Electric_Field_Ez']

    proton_KE_load = proton_E_array.load() # J
    electron_KE_load = electron_E_array.load() # J
    Bx_load = Bx_array.load() # T
    By_load = By_array.load() # T
    Bz_load = Bz_array.load() # T
    Ex_load = Ex_array.load() # V/m
    Ey_load = Ey_array.load() # V/m
    Ez_load = Ez_array.load() # V/m

    plt.rcParams.update({'axes.labelsize': 16})
    plt.rcParams.update({'axes.titlesize': 18})
    plt.rcParams.update({'xtick.labelsize': 14})
    plt.rcParams.update({'ytick.labelsize': 14})

    # Time conversion
    TWO_PI = 2.0 * np.pi
    B0 = input['constant']['b0_strength']

    ion_bkgd_mass = input['constant']['ion_mass_e'] * constants.electron_mass
    ion_bkgd_charge = input['species']['proton']['charge'] * constants.elementary_charge
    ion_gyroperiod = (TWO_PI * ion_bkgd_mass) / (ion_bkgd_charge * B0)
    t_in_ci = proton_KE_load.coords["time"] / ion_gyroperiod
    
    # Mean over all cells for mean particle/field energy
    pro_ke_mean : xr.DataArray = proton_KE_load.mean(dim = "X_Grid_mid") # J
    e_ke_mean : xr.DataArray = electron_KE_load.mean(dim = "X_Grid_mid") # J

    E_field = np.sqrt(Ex_load**2 + Ey_load**2 + Ez_load**2)
    E_E_density : xr.DataArray = (constants.epsilon_0 * E_field**2) / 2.0 # J / m^3
    E_mean : xr.DataArray = E_E_density.mean(dim="X_Grid_mid") # J / m^3

    B_field = np.sqrt(Bx_load**2 + By_load**2 + Bz_load**2)
    B_E_density : xr.DataArray = (B_field**2 / (2.0 * constants.mu_0)) # J / m^3
    B_mean : xr.DataArray = B_E_density.mean(dim = "X_Grid_mid") # J / m^3

    # Calculate B energy and convert others to to J/m3
    B_dE_density = B_mean - B_mean[0] # J / m^3
    E_dE_density = E_mean - E_mean[0] # J / m^3
    proton_dKE = pro_ke_mean - pro_ke_mean[0] # J
    electron_dKE = e_ke_mean - e_ke_mean[0] # J
    
    proton_dKE_density = proton_dKE * proton_density # J / m^3
    electron_dKE_density = electron_dKE * electron_density # J / m^3

    #total_dKE_density = proton_dKE_density + B_dE_density + E_dE_density
    total_dKE_density = proton_dKE_density + electron_dKE_density + B_dE_density + E_dE_density

    if irb:
        irb_density = density * frac_beam # m^-3
        irb_E_array : xr.DataArray = ds['Derived_Average_Particle_Energy_ion_ring_beam']
        irb_KE_load = irb_E_array.load() # J
        irb_ke_mean : xr.DataArray = irb_KE_load.mean(dim = "X_Grid_mid") # J
        irb_dKE = irb_ke_mean - irb_ke_mean[0] # J
        irb_dKE_density = irb_dKE * irb_density # J / m^3
        total_dKE_density += irb_dKE_density
        plt.plot(t_in_ci, irb_dKE_density.data, label = r"ion ring beam KE")

    plt.plot(t_in_ci, proton_dKE_density.data, label = r"background proton KE")
    plt.plot(t_in_ci, electron_dKE_density.data, label = r"background electron KE")
    plt.plot(t_in_ci, B_dE_density.data, label = r"Magnetic field E")
    plt.plot(t_in_ci, E_dE_density.data, label = r"Electric field E")
    plt.plot(t_in_ci, total_dKE_density.data, label = r"Total E")
    plt.xlabel(r'Time [$\tau_{ci}$]')
    plt.ylabel(r"Change in energy density [$J/m^3$]")
    plt.title(f"{directory.name}: Evolution of energy in fast minority ions, background ions/electrons and EM fields")
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.show()

    if pct:
        # Calculate B energy and convert others to to J/m3
        B_dE_pct = 100.0 * (B_mean - B_mean[0])/B_mean[0] # %
        E_dE_pct = 100.0 * (E_mean - E_mean[0])/E_mean[0] # %
        proton_dKE_density = pro_ke_mean * proton_density # J / m^3
        electron_dKE_density = e_ke_mean * electron_density # J / m^3
        proton_dKE_pct = 100.0 * (proton_dKE_density - proton_dKE_density[0])/proton_dKE_density[0] # %
        electron_dKE_pct = 100.0 * (electron_dKE_density - electron_dKE_density[0])/electron_dKE_density[0] # %
        total_E_density = B_mean + E_mean + proton_dKE_density + electron_dKE_density

        if irb:
            irb_dKE_density = irb_ke_mean * irb_density # J / m^3
            irb_dKE_pct = 100.0 * (irb_dKE_density - irb_dKE_density[0])/irb_dKE_density[0] # J
            plt.plot(t_in_ci, irb_dKE_pct.data, label = "ion ring beam KE")
            total_E_density += irb_dKE_density
            #total_dE_pct  = np.mean([B_dE_pct, E_dE_pct, proton_dKE_pct, electron_dKE_pct, irb_dKE_pct], axis=0)
        #else:
            #total_dE_pct  = np.mean(B_dE_pct + E_dE_pct + proton_dKE_pct + electron_dKE_pct, axis=0)

        total_dE_density_pct = 100.0 * (total_E_density-total_E_density[0])/total_E_density[0]
        
        plt.plot(t_in_ci, proton_dKE_pct.data, label = "background proton KE")
        plt.plot(t_in_ci, electron_dKE_pct.data, label = "background electron KE")
        plt.plot(t_in_ci, B_dE_pct.data, label = "Magnetic field E")
        plt.plot(t_in_ci, E_dE_pct.data, label = "Electric field E")
        plt.plot(t_in_ci, total_dE_density_pct, label = "Total E")
        plt.yscale('symlog')
        plt.xlabel(r'Time [$\tau_{ci}$]')
        plt.ylabel("Percentage change in energy density [%]")
        plt.title(f"{directory.name}: Evolution of energy in fast minority ions, background ions/electrons and EM fields")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
            
    e_KE_start = float(e_ke_mean[0].data)
    e_KE_end = float(e_ke_mean[-1].data)
    print(f"Change in electron energy density: e- KE t=start: {e_KE_start:.4f}, e- KE t=end: {e_KE_end:.4f} (+{((e_KE_end - e_KE_start)/e_KE_start)*100.0:.4f}%)")
    E_start = total_E_density[0]
    E_end = total_E_density[-1]
    print(f"Change in overall energy density: E t=start: {E_start:.4f}, E t=end: {E_end:.4f} (+{((E_end - E_start)/E_start)*100.0:.4f}%)")
        
def analyse_electron_heating(analysisDirectory : Path):

    electronDeltaPct = []
    cellWidth_rLe = []
    cellWidth_dl = []
    totalConservationPct = []

    dataFiles = glob.glob(str(analysisDirectory / "*.nc"))
    
    for simulation in dataFiles:

        data = xr.open_datatree(
            simulation,
            engine="netcdf4"
        )

        cellWidth_dl.append(data.attrs["cellWidth_dL"])
        cellWidth_rLe.append(data.attrs["cellWidth_rLe"])

        energyStats = data["Energy"]
        #electronDeltaEnergy = ((energyStats.electronEnergyDensity_end - energyStats.electronEnergyDensity_start) / energyStats.electronEnergyDensity_start) * 100.0
        electronDeltaEnergy = 100.0 * (energyStats.electronEnergyDensity_delta/energyStats.electronEnergyDensity_start)
        # electronDeltaEnergy = energyStats.electronEnergyDensity_delta
        electronDeltaPct.append(electronDeltaEnergy)
        totalConservationPct.append(energyStats.totalEnergyDensityConservation_pct)

        print(f"{simulation.split('/')[-1]}: cell width: {data.attrs['cellWidth_dL']:.4f} Debye lengths/{data.attrs['cellWidth_rLe']:.4f} electron gyroradii (B0: {data.attrs['B0strength']:.4f}, background density: {data.attrs['backgroundDensity']:.4E}), electron delta energy: {electronDeltaEnergy:.4f}%, total energy change: {energyStats.totalEnergyDensityConservation_pct:.4f}")
    
    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax2 = ax1.twiny()
    ax1.scatter(cellWidth_rLe, electronDeltaPct, color="blue")
    # res = linregress(cellWidth_rLe, electronDeltaPct)
    ax2.scatter(cellWidth_dl, electronDeltaPct, color="blue")
    # x = np.linspace(0.0, np.max(cellWidth_rLe), 1000)
    # plt.plot(x, res.intercept + res.slope*x, "g--", label=f"R^2 = {res.rvalue**2:.5f}")
    # plt.yscale("log")
    # plt.legend()
    ax1.set_xlabel("Cell width / electron gyroradii")
    ax2.set_xlabel("Cell width / Debye lengths")
    plt.title("Electrons - run 33 (B0: 0.9549, background density: 7.7898E+19)")
    ax1.set_ylabel("Change in energy density by end of simulation / %")
    plt.tight_layout()
    plt.show()

    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax2 = ax1.twiny()
    # res = linregress(cellWidth_rLe, totalConservationPct)
    ax1.scatter(cellWidth_rLe, totalConservationPct, color="orange")
    ax2.scatter(cellWidth_dl, totalConservationPct, color="orange")
    # x = np.linspace(0.0, np.max(cellWidth_rLe), 1000)
    # plt.plot(x, res.intercept + res.slope*x, "r--", label=f"R^2 = {res.rvalue**2:.5f}")
    # plt.yscale("log")
    ax1.set_xlabel("Cell width / electron gyroradii")
    ax2.set_xlabel("Cell width / Debye lengths")
    ax1.set_ylabel("Change in energy density by end of simulation / %")
    plt.title("Total energy - (B0: 0.9549, background density: 7.7898E+19)")
    plt.tight_layout()
    plt.show()

def correlate_cellWidth_energy_transer(analysisDirectory : Path, logColourbar : bool = False):

    # Axes
    rLe_cellWidth_ratio = []
    beamFraction = []
    B0_str = []
    density = []

    # Colour data
    hasOverallFiLossBiGain = []
    hasFiTroughBiPeak = []
    bkgdIonChangeAtFastIonTrough_pct = []
    fiTroughEnergy_pct = []
    fiDelta_pct = []
    biDelta_pct = []
    netFiToBiDelta = []

    dataFiles = glob.glob(str(analysisDirectory / "*.nc"))
    
    for simulation in dataFiles:

        data = xr.open_datatree(
            simulation,
            engine="netcdf4"
        )

        # Independent vars
        rLe_cellWidth_ratio.append(float(data.cellWidth_rLe/data.cellWidth_dL))
        beamFraction.append(data.beamFraction)
        B0_str.append(data.B0strength)
        density.append(data.backgroundDensity)

        # Dependent vars
        energyStats = data["Energy"]
        hasOverallFiLossBiGain.append(True if (not energyStats.hasOverallFastIonGain) and energyStats.hasOverallBkgdIonGain else False)
        hasFiTroughBiPeak.append(True if energyStats["fastIonMeanEnergyDensity"].hasTroughs and energyStats["protonMeanEnergyDensity"].hasPeaks else False)
        try:
            bkgdIonChangeAtFastIonTrough_pct.append(energyStats.bkgdIonChangeAtFastIonTrough_pct)
        except AttributeError:
            bkgdIonChangeAtFastIonTrough_pct.append(0.0)
        try:
            fiTroughEnergy_pct.append(energyStats["fastIonMeanEnergyDensity"].troughValues_pct)
        except AttributeError:
            fiTroughEnergy_pct.append(0.0)
        fiDelta = 100.0 * (energyStats.fastIonEnergyDensity_delta / energyStats.fastIonEnergyDensity_start)
        fiDelta_pct.append(fiDelta)
        biDelta = 100.0 * (energyStats.backgroundIonEnergyDensity_delta / energyStats.fastIonEnergyDensity_start)
        biDelta_pct.append(biDelta)
        netFiToBiDelta.append(biDelta - fiDelta)

    permanentVars = {"e- Larmor radius : Debye L (cell width)" : rLe_cellWidth_ratio}
    independentVars = {
        "Beam fraction" : beamFraction, 
        "B0 [T]" : B0_str, 
        "Density [m^-3]" : density
    }
    categoricalDependentVars = {
        "Has overall FI loss and BI gain" : hasOverallFiLossBiGain,
        "Has FI trough and BI peak" : hasFiTroughBiPeak
    }
    continuousDependentVars = {
        "BI energy delta at FI trough [% of FI energy]" : bkgdIonChangeAtFastIonTrough_pct,
        "FI energy delta at trough [% of FI energy]" : fiTroughEnergy_pct,
        "FI overall energy delta [% of FI energy]" : fiDelta_pct,
        "BI overall energy delta [% of FI energy]" : biDelta_pct,
        "Net energy transfer from FI to BI [% of FI energy]" : netFiToBiDelta
    }

    # Categorical dependent variables
    for pointName, pointData in categoricalDependentVars.items():
        for yAxisName, yData in permanentVars.items():
            fig, axs = plt.subplots(1, 3, sharey=True, figsize=[12,8])
            fig.suptitle(pointName)
            fig.supylabel(yAxisName)
            axCount = 0
            for xAxisName, xData in independentVars.items():
                df = pd.DataFrame({
                    xAxisName : xData,
                    yAxisName : yData,
                    pointName : pointData
                })
                groups = df.groupby(pointName)
                for label, group in groups:
                    axs[axCount].scatter(group.T.loc[xAxisName], group.T.loc[yAxisName], marker='o', label=label)
                axs[axCount].set_xlabel(xAxisName)
                axs[axCount].legend()
                if "fraction" in xAxisName.lower() or "density" in xAxisName.lower():
                    axs[axCount].set_xscale('log')
                axCount += 1
            plt.show()
            plt.close('all')

    # Continuous dependent variables
    for pointName, pointData in continuousDependentVars.items():
        for yAxisName, yData in permanentVars.items():
            fig, axs = plt.subplots(1, 3, sharey=True, figsize=[12,8])
            fig.suptitle(pointName)
            fig.supylabel(yAxisName)
            axCount = 0
            for xAxisName, xData in independentVars.items():
                if logColourbar:
                    im = axs[axCount].scatter(xData, yData, marker='o', c=pointData, cmap = "Spectral_r", norm='symlog')
                else:
                    im = axs[axCount].scatter(xData, yData, marker='o', c=pointData, cmap = "Spectral_r")
                axs[axCount].set_xlabel(xAxisName)
                if "fraction" in xAxisName.lower() or "density" in xAxisName.lower():
                    axs[axCount].set_xscale('log')
                axCount += 1
            fig.colorbar(im, ax=axs.ravel().tolist())
            plt.show()
            plt.close('all')
    
    # fig.suptitle("Fast ion energy transfer")
    # im = ax1.scatter(beamFraction, rLe_cellWidth_ratio, c=hasOverallFiLossBiGain, cmap="Reds")
    # fig.colorbar(im, ax=ax1)
    # ax2.scatter(B0_str, rLe_cellWidth_ratio, c=hasFiTroughBiPeak, cmap="Greens")
    # ax3.scatter(density, rLe_cellWidth_ratio, c=bkgdIonChangeAtFastIonTrough_pct, cmap="Blues")

def analyse_peak_characteristics(analysisDirectory : Path):
    dataFiles = glob.glob(str(analysisDirectory / "*.nc"))

    B0strength = []
    B0angle = []
    densities = []
    beamFractions = []
    hasFastIonTrough = []
    hasBkgdIonPeak = []
    hasOverallFiLoss = []
    hasOverallBiGain = []
    fiDeltaEnergyPct = []
    biDeltaEnergyPct = []

    for simulation in dataFiles:

        data = xr.open_datatree(
            simulation,
            engine="netcdf4"
        )

        energyStats = data["Energy"]

        B0strength.append(data.attrs["B0strength"])
        B0angle.append(data.attrs["B0angle"])
        beamFractions.append(data.attrs["beamFraction"])
        densities.append(data.attrs["backgroundDensity"])

        hasFastIonTrough.append(bool(energyStats["fastIonMeanEnergyDensity"].attrs["hasTroughs"]))
        hasBkgdIonPeak.append(bool(energyStats["protonMeanEnergyDensity"].attrs["hasPeaks"]))
        hasOverallFiLoss.append(False if energyStats.hasOverallFastIonGain else True)
        hasOverallBiGain.append(bool(energyStats.hasOverallBkgdIonGain))
        fiDelta = (energyStats.fastIonEnergyDensity_delta / energyStats.fastIonEnergyDensity_start) * 100.0
        biDelta = (energyStats.backgroundIonEnergyDensity_delta / energyStats.fastIonEnergyDensity_start) * 100.0
        fiDeltaEnergyPct.append(fiDelta)
        biDeltaEnergyPct.append(biDelta)
    
    # simParameters = {"B0" : np.array(B0strength), "B0_angle" : np.array(B0angle), "Density" : np.log10(np.array(densities)), "Beam_fraction" : np.log10(np.array(beamFractions))}
    simParameters = {"B0" : np.array(B0strength), "B0_angle" : np.array(B0angle), "Density" : np.log10(np.array(densities)), "Beam_fraction" : np.log10(np.array(beamFractions))}
    variables = {
        "Has_FI_trough" : hasFastIonTrough, 
        "Has_BI_peak" : hasBkgdIonPeak, 
        "Has_overall_FI_loss" : hasOverallFiLoss, 
        "Has_overall_BI_gain" : hasOverallBiGain,
        "FI_delta_pct" : fiDeltaEnergyPct,
        "BI_delta_pct" : biDeltaEnergyPct
    }

    fiTroughIndices = np.where(variables["Has_FI_trough"])[0]
    biPeakIndices = np.where(variables["Has_BI_peak"])[0]
    overallFiLossIndices = np.where(variables["Has_overall_FI_loss"])[0]
    overallBiGainIndices = np.where(variables["Has_overall_BI_gain"])[0]
    mciIndices = np.intersect1d(fiTroughIndices, biPeakIndices)
    partialMciIndices = np.union1d(fiTroughIndices, biPeakIndices)
    noMciIndices = np.setdiff1d(np.arange(len(simParameters["B0"])), partialMciIndices)
    fiTroughBiPeakParams = {"B0 (T)" : simParameters["B0"][mciIndices], "B0 angle (deg)" : simParameters["B0_angle"][mciIndices], "Log_10 density" : simParameters["Density"][mciIndices], "Log_10 beam fraction" : simParameters["Beam_fraction"][mciIndices]}
    noFiTroughAndBiPeakParams = {"B0" : simParameters["B0"][noMciIndices], "B0_angle" : simParameters["B0_angle"][noMciIndices], "Density" : simParameters["Density"][noMciIndices], "Beam_fraction" : simParameters["Beam_fraction"][noMciIndices]}
    fiTroughOrBiPeakParams = {"B0" : simParameters["B0"][partialMciIndices], "B0_angle" : simParameters["B0_angle"][partialMciIndices], "Density" : simParameters["Density"][partialMciIndices], "Beam_fraction" : simParameters["Beam_fraction"][partialMciIndices]}

    labels = [k for k in fiTroughBiPeakParams.keys()]
    mciSamples = [v for v in fiTroughBiPeakParams.values()]
    partialMciSamples = [v for v in fiTroughOrBiPeakParams.values()]
    noMciSamples = [v for v in noFiTroughAndBiPeakParams.values()]
    # utils.my_matrix_plot(data_series=[noMciSamples, partialMciSamples, mciSamples], series_labels=["No E transfer", "Partial E transfer", "Clear E transfer"], parameter_labels=labels, plot_style="hdi", colormap_list=["Blues", "Greens", "Reds"], show=True)
    # utils.my_matrix_plot(data_series=[noMciSamples, mciSamples], series_labels=["No E transfer", "Clear E transfer"], parameter_labels=labels, plot_style="hdi", colormap_list=["Blues", "Reds"], show=True)

    fiLossParams = {
        "B0 (T)" : simParameters["B0"][overallFiLossIndices], 
        "B0 angle (deg)" : simParameters["B0_angle"][overallFiLossIndices], 
        "Log_10 density" : simParameters["Density"][overallFiLossIndices], 
        "Log_10 beam fraction" : simParameters["Beam_fraction"][overallFiLossIndices]
    }
    biGainParams = {
        "B0 (T)" : simParameters["B0"][overallBiGainIndices], 
        "B0 angle (deg)" : simParameters["B0_angle"][overallBiGainIndices], 
        "Log_10 density" : simParameters["Density"][overallBiGainIndices], 
        "Log_10 beam fraction" : simParameters["Beam_fraction"][overallBiGainIndices]
    }
    fiLossAndBiGainIndices = np.intersect1d(overallFiLossIndices, overallBiGainIndices)
    fiLossOrBiGainIndices = np.union1d(overallFiLossIndices, overallBiGainIndices)
    noFiToBiTransferIndices = np.setdiff1d(np.arange(len(simParameters["B0"])), fiLossOrBiGainIndices)
    fiLossAndBiGainParams = {
        "B0 (T)" : simParameters["B0"][fiLossAndBiGainIndices], 
        "B0 angle (deg)" : simParameters["B0_angle"][fiLossAndBiGainIndices], 
        "Log_10 density" : simParameters["Density"][fiLossAndBiGainIndices], 
        "Log_10 beam fraction" : simParameters["Beam_fraction"][fiLossAndBiGainIndices]
    }
    noFiToBiTransferParams = {
        "B0 (T)" : simParameters["B0"][noFiToBiTransferIndices], 
        "B0 angle (deg)" : simParameters["B0_angle"][noFiToBiTransferIndices], 
        "Log_10 density" : simParameters["Density"][noFiToBiTransferIndices], 
        "Log_10 beam fraction" : simParameters["Beam_fraction"][noFiToBiTransferIndices]
    }
    fiLossSamples = [v for v in fiLossParams.values()]
    biGainSamples = [v for v in biGainParams.values()]
    noTransferSamples = [v for v in noFiToBiTransferParams.values()]
    fiLossBiGainSamples = [v for v in fiLossAndBiGainParams.values()]

    print(f"FI loss samples: {len(fiLossSamples[0])}")
    print(f"BI gain samples: {len(biGainSamples[0])}")
    print(f"No transfer samples: {len(noTransferSamples[0])}")
    print(f"FI loss and BI gain samples: {len(fiLossBiGainSamples[0])}")

    utils.my_matrix_plot(
        data_series=[fiLossSamples, biGainSamples, noTransferSamples, fiLossBiGainSamples], 
        series_labels=["Overall FI Loss", "Overall BI Gain", "Neither FI loss or BI gain", "Both FI loss and BI gain"], 
        parameter_labels=labels, 
        plot_style="hdi", 
        colormap_list=["Greys", "Greens", "Blues", "Reds"], 
        show=True,
        equalise_pdf_heights=False,
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser("parser")
    parser.add_argument(
        "--dir",
        action="store",
        help="Directory.",
        required = True,
        type=Path
    )
    parser.add_argument(
        "--electronHeating",
        action="store_true",
        help="Analyse electron heating.",
        required = False
    )
    parser.add_argument(
        "--correlate",
        action="store_true",
        help="Correlate cell width with energy characteristics.",
        required = False
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Plot FFT with log of spectral power.",
        required = False
    )
    parser.add_argument(
        "--savePath",
        action="store",
        help="Directory to which figures should be saved.",
        required = False,
        type=Path
    )
    parser.add_argument(
        "--irb",
        action="store_true",
        help="Whether or not the ion ring beam should be accounted for or not.",
        required = False
    )
    parser.add_argument(
        "--pct",
        action="store_true",
        help="Calculate percentages.",
        required = False
    )

    args = parser.parse_args()

    analyse_peak_characteristics(Path("/home/era536/Documents/Epoch/Data/irb_may25_analysis/data/"))

    # if args.electronHeating:
    #     analyse_electron_heating(args.dir)
    # elif args.correlate:
    #     correlate_cellWidth_energy_transer(args.dir)
    # else:
    #     calculate_energy(args.dir, args.irb, args.pct)