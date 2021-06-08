import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Read in the data and set engine parameter to "openpyxl"
df = pd.read_excel(open('model data.xlsx', 'rb'), sheet_name='data', engine = 'openpyxl')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True) # set column to index
numberOfHours = len(df.index)

df_inputs = pd.read_excel(open('model data.xlsx', 'rb'), sheet_name='inputs', engine = 'openpyxl')

# Hourly electric grid demand
demandScaleFactor = df_inputs[df_inputs['Key inputs']=='Grid scale down factor'].Value.item()
gridLoad = demandScaleFactor * df['Demand (MW)'].to_numpy()
# Demand above which other sources step in 
B = df_inputs[df_inputs['Key inputs']=='Demand above which other sources step in (MW)'].Value.item()
b = df_inputs[df_inputs['Key inputs']=='Baseload power (MW)'].Value.item() # Baseload supply
# Electricity demand system seeking to fill above input baseload and below some specified ancillary services
sbbSystemDemand = np.minimum(np.maximum(gridLoad - b, 0), B - b)

# Day ahead pricing max realized price electricity
indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=24)
df['Max day ahead price ($/MWh)'] = df['day ahead price'].rolling(window=indexer, min_periods=1).max()
# Current miner realizable revenue net of variable costs
minerPowerUsage = 1e-3 * df_inputs[df_inputs['Key inputs']=='Power usage (kW)'].Value.item()
minerEfficiency = df_inputs[df_inputs['Key inputs']=='Hardware efficiency'].Value.item()
minerMaxHashrate = df_inputs[df_inputs['Key inputs']=='Max hashrate (TH/s)'].Value.item()
variableCost = 1e3 * df_inputs[df_inputs['Key inputs']=='Variable cost ($/kWh)'].Value.item()
hashratePerPower = minerMaxHashrate * minerEfficiency / minerPowerUsage # TH/s / kW
df['Miner Realizable Price ($/MWh)'] = hashratePerPower * df['Total Miner Revenue'] / df['Hashrate (TH/s)'] - variableCost

# Miner dispatch decision made on [input] trailing average hash/revenue information
offsetWindow = df_inputs[df_inputs['Key inputs']=='Miner lookback offset for dispatch decision'].Value.item()
df['Trailing Avg Miner Realizable Price ($/MWh)'] = df['Miner Realizable Price ($/MWh)'].rolling(offsetWindow).mean() # Trailing average miner realizable price
# Move towards this battery fill-level given current realizable revenue in crypto vs electricity sales
targetFrac = df_inputs[df_inputs['Key inputs']=='Target battery fill before shift to miner'].Value.item() # Breakeven battery fill percent
x = np.linspace(0, 1, num=101)
a = 12 * (x - targetFrac)
y = (2 / (1 + np.exp(-a)))**2
batteryDispatchY = (df['Max day ahead price ($/MWh)'] / df['Trailing Avg Miner Realizable Price ($/MWh)']).to_numpy()
miningAdjusterThreshold = 1.8
Y = np.where(batteryDispatchY < miningAdjusterThreshold, batteryDispatchY, miningAdjusterThreshold)
batteryDispatchX = targetFrac - np.log(2 / np.sqrt(Y) - 1) / 12
batteryDispatchX = np.minimum(batteryDispatchX, 1.0) # Breakeven battery fill percent

# netFillToBattery =
# + With battery and grid in supply state, surplus energy to battery after mining
# + With battery in demand state, remaining possible excess power devoted to battery
# + With battery in demand state, fill to battery
# - With battery and grid in supply state, additional battery energy to mining after surplus energy dispatch
# - With battery in supply state and grid in demand, energy used from battery by miner
# - Battery drawdown to meet electricity dispatch

hoursPerYear = 24 * 365
batterySizesStr = df_inputs[df_inputs['Key inputs']=='Battery system size (MWh)'].Value.item()
batterySizes = np.fromiter(map(int, batterySizesStr.split(';')), dtype=np.int)
batteryRelativePower = df_inputs[df_inputs['Key inputs']=='Battery power size ratio (1/h)'].Value.item()
batteryStartingFill = df_inputs[df_inputs['Key inputs']=='Battery starting fill'].Value.item()
miningSystemSizesStr = df_inputs[df_inputs['Key inputs']=='Mining system size (MW)'].Value.item()
miningSystemSizes = np.fromiter(map(int, miningSystemSizesStr.split(';')), dtype=np.int)
solarSystemSizesStr = df_inputs[df_inputs['Key inputs']=='Solar system size (MW)'].Value.item()
solarSystemSizes = np.fromiter(map(int, solarSystemSizesStr.split(';')), dtype=np.int)
solarLifespanHours = hoursPerYear * df_inputs[df_inputs['Key inputs']=='Lifespan (years)'].Value.item()
solarCapCost = 1e6 * df_inputs[df_inputs['Key inputs']=='Solar capital cost ($/W)'].Value.item()
batteryCapCost = 1e3 * df_inputs[df_inputs['Key inputs']=='Battery capital cost ($/kWh)'].Value.item()
minerCapCost = df_inputs[df_inputs['Key inputs']=='Rig cost ($)'].Value.item() / minerPowerUsage
minerLifespanHours = hoursPerYear * df_inputs[df_inputs['Key inputs']=='Miner lifespan (years)'].Value.item()
solarFixedOandMCost = 1e3 * df_inputs[df_inputs['Key inputs']=='Solar Fixed O&M cost ($/kW-yr)'].Value.item() / hoursPerYear
batteryFixedOandMCost = 1e3 * df_inputs[df_inputs['Key inputs']=='Battery O&M cost ($/kWh)'].Value.item() / hoursPerYear

# Battery energy in storage
electricitySuppliedDemandedRatioList = []
ratioHoursSystemMeetsGridDemandList = []
# Loop over various system sizes
for solarSystemSize, batterySize, miningSystemSize in zip(solarSystemSizes, batterySizes, miningSystemSizes):
    # Determine solar generation
    solarGeneration = 1e-6 * solarSystemSize * df['AC System Output (W)'].to_numpy()
    batteryPower = batteryRelativePower * batterySize
    currentBatteryEnergyInStorage = batteryStartingFill * batterySize
    netFillToBattery = 0
    batteryEnergyInStorage = np.zeros(numberOfHours) # Initialize array
    batteryDrawdown = np.zeros(numberOfHours) # Initialize array
    minerPowerDraw = np.zeros(numberOfHours) # Initialze array
    # Loop over hours
    for n in range(numberOfHours):
        # Battery energy in storage
        batteryEnergyInStorage[n] = currentBatteryEnergyInStorage
        # Amount of electricity generated by solar in excess of demand, surplus for battery/mining
        surplusForBatteryOrMining = max(solarGeneration[n] - sbbSystemDemand[n], 0)
        # Battery drawdown to meet electricity dispatch
        if sbbSystemDemand[n] > solarGeneration[n]:
            batteryDrawdown[n] = min(sbbSystemDemand[n] - solarGeneration[n], batteryPower, batteryEnergyInStorage[n])

        # Desired fill to/draw from battery
        desiredDrawFromBattery = batteryDispatchX[n] * batterySize - batteryEnergyInStorage[n] - batteryDrawdown[n]
        # With battery in demand state, fill to battery and excess power not dispatched to battery
        if desiredDrawFromBattery > 0:
            demandStateFillToBattery = min(desiredDrawFromBattery, batteryPower, surplusForBatteryOrMining)
            excessPowerNotDispatchedToBattery = surplusForBatteryOrMining - demandStateFillToBattery
        else:
            demandStateFillToBattery = 0
            excessPowerNotDispatchedToBattery = 0

        # With battery in demand state, devote excess power to miner
        excessPowerToMiner = min(miningSystemSize, excessPowerNotDispatchedToBattery)
        # With battery in demand state, incremental excess power available
        incrementalExcessPowerAvailable = excessPowerNotDispatchedToBattery - excessPowerToMiner
        # With battery in supply state and grid in demand, additional energy available for miner. Grid is in a demand state when batteryDrawdown larger than zero
        if batteryDrawdown[n] > 0 and desiredDrawFromBattery < 0:
            additionalAvailableMinerEnergy = min(-desiredDrawFromBattery, batteryPower - batteryDrawdown[n])
        else:
            additionalAvailableMinerEnergy = 0

        # With battery in supply state and grid in demand, energy used from battery by miner
        energyFromBatteryToMiner = min(additionalAvailableMinerEnergy, miningSystemSize)

        # With battery in demand state determine remaining possible excess power devoted to battery
        excessPowerToBattery = min(incrementalExcessPowerAvailable + demandStateFillToBattery, batteryPower, batterySize - batteryEnergyInStorage[n]) - demandStateFillToBattery

        # With battery in demand state, excess power unused
        excessPowerUnused = incrementalExcessPowerAvailable - excessPowerToBattery

        if batteryDrawdown[n] > 0 or desiredDrawFromBattery > 0:
            # With battery and grid in surplus state determine supply energy generated to miner
            supplyEnergyGeneratedToMiner = 0
            # With battery and grid in supply state determine additional surplus energy available for battery
            addSurplusEnergyAvailableForBattery = 0
            # With battery and grid in supply state, additional battery energy to mining after surplus energy dispatch
            addBatteryEnergyToMiningAfterSurplusEnergyDispatch = 0
        else:
            supplyEnergyGeneratedToMiner = min(surplusForBatteryOrMining, miningSystemSize)
            addSurplusEnergyAvailableForBattery = surplusForBatteryOrMining - supplyEnergyGeneratedToMiner
            addBatteryEnergyToMiningAfterSurplusEnergyDispatch = min(miningSystemSize - supplyEnergyGeneratedToMiner, batteryPower)

        # With battery and grid in supply state determine surplus energy to battery after mining
        surplusEnergyToBatteryAfterMining = min(addSurplusEnergyAvailableForBattery, batteryPower, batterySize - batteryEnergyInStorage[n])
        # Net fill to or draw from battery
        netFillToBattery = surplusEnergyToBatteryAfterMining + excessPowerToBattery + demandStateFillToBattery - addBatteryEnergyToMiningAfterSurplusEnergyDispatch - energyFromBatteryToMiner - batteryDrawdown[n]
        # Update battery energy storage
        currentBatteryEnergyInStorage += netFillToBattery
        # Miner power draw
        minerPowerDraw[n] = excessPowerToMiner + energyFromBatteryToMiner + supplyEnergyGeneratedToMiner + addBatteryEnergyToMiningAfterSurplusEnergyDispatch
        
    # Total electricity supplied and sold
    electricitySold = np.minimum(sbbSystemDemand, solarGeneration + batteryDrawdown)
    # The difference between electricity supplied and the amount demanded of solar+battery, generation gap
    supplyGap = electricitySold - sbbSystemDemand
    # Percent of hours the system meets grid demand
    r = 1. - float(np.count_nonzero(supplyGap < 0))/float(numberOfHours)
    electricitySuppliedDemandedRatio = np.sum(solarGeneration)/np.sum(sbbSystemDemand)
    electricitySuppliedDemandedRatioList.append(electricitySuppliedDemandedRatio)
    ratioHoursSystemMeetsGridDemandList.append(r)
    # Financial stuff
    revenueElectricPower = electricitySold * df['real time price'].to_numpy()
    solarDepreciation = solarSystemSize * solarCapCost / solarLifespanHours
    batteryDepreciation = batterySize * batteryCapCost / solarLifespanHours
    solarOandM = -solarFixedOandMCost * solarSystemSize
    batteryOandM = -batteryFixedOandMCost * batterySize
    # Where does the battery system price levelize with day ahead pricing
    solarBatteryProfit = revenueElectricPower - solarDepreciation - batteryDepreciation + solarOandM + batteryOandM
    deployedHashrate = minerPowerDraw * hashratePerPower
    shareOfGlobalHashrate = deployedHashrate / df['Hashrate (TH/s)'].to_numpy()
    minerRevenue = shareOfGlobalHashrate * df['Total Miner Revenue'].to_numpy()
    miningVariableCost = np.where(deployedHashrate > 0, miningSystemSize * variableCost, 0)
    minerDepreciation = miningSystemSize * minerCapCost / minerLifespanHours
    minerProfit = minerRevenue - miningVariableCost - minerDepreciation
    totalProfit = solarBatteryProfit + minerProfit
    print(f'For scenario Solar {solarSystemSize} MW, Battery {batterySize} MWh and Mining {miningSystemSize} MW:')
    print(f'Electricity supplied/electricty demanded: {100*electricitySuppliedDemandedRatio:.1f}%')
    print(f'Percentage of hours system meets grid demand: {100*r:.2f}%')
    print(f'Total profit/loss: ${np.sum(totalProfit):,.0f}')
    print('--')

# Plot results, set values to determine marker and plot size
markerStartingSize = 3
markerSizeFactor = 5
markerAreas = markerSizeFactor * (miningSystemSizes + markerStartingSize)
fig, ax = plt.subplots(figsize=(1.6 * markerSizeFactor, markerSizeFactor))
ax.grid(True)
ax.scatter(batterySizes, solarSystemSizes, s=markerAreas) # Scatter plot
for posX, posY, labelRaw in zip(batterySizes, solarSystemSizes, ratioHoursSystemMeetsGridDemandList):
    label = str(int(round(100*labelRaw)))+'%'
    ax.annotate(label, xy=(posX, posY))

ax.set(xlabel='Battery Storage Capacity (MWh)', ylabel='Solar System Size (MW)')
ax.margins(0.2)
ax.set_xlim(xmin=0)
ax.set_ylim(ymin=0)
ax.grid(True)
ax.set_axisbelow(True)
plt.show()
