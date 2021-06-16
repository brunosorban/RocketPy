from datetime import datetime
from time import process_time, perf_counter, time
import glob

from rocketpy import Environment, SolidMotor, Rocket, Flight, Function

import numpy as np
from numpy.random import normal, uniform, choice
from IPython.display import display

import SisRec

analysis_parameters = {
    # Mass Details
    "rocketMass": (25.879, 0.1*25.879), # Rocket's dry mass (kg) and its uncertainty (standard deviation)
    
    # Propulsion Details - run help(SolidMotor) for more information
    "impulse": (9010.466, 0.1*9010.466),                         # Motor total impulse (N*s)
    "burnOut": (11.8, 0.1*11.8),                              # Motor burn out time (s)
    "nozzleRadius": (31.7/1000, 0.01*31.7/1000),            # Motor's nozzle radius (m)
    "throatRadius": (13/1000, 0.01*13/1000) ,                # Motor's nozzle throat radius (m)
    "grainSeparation": (0, 0),                          # Motor's grain separation (axial distance between two grains) (m)
    "grainDensity": (4034, 0.05*4034),                         # Motor's grain density (kg/m^3)
    "grainOuterRadius": (48.5/1000, 0.05*48.5/1000),        # Motor's grain outer radius (m)
    "grainInitialInnerRadius": (25/1000, 0.05*25/1000), # Motor's grain inner radius (m) 
    "grainInitialHeight": (230/1000, 0.05*230/1000),           # Motor's grain height (m)

    # Aerodynamic Details - run help(Rocket) for more information
    "inertiaI": (20.035, 0.1*20.035),                       # Rocket's inertia moment perpendicular to its axis (kg*m^2) 
    "inertiaZ": (0.064, 0.1*0.064),                       # Rocket's inertia moment relative to its axis (kg*m^2)
    "radius": (127/2000, 0.002),                      # Rocket's radius (kg*m^2)
    "distanceRocketNozzle": (-1.365084, 0.1*1.36419),             # Distance between rocket's center of dry mass and nozzle exit plane (m) (negative)
    "distanceRocketPropellant": (-0.6034939999999995, 0.1*0.6026),         # Distance between rocket's center of dry mass and and center of propellant mass (m) (negative)
    "powerOffDrag": (0.4624/0.366, 0.1*0.4624/0.366),               # Multiplier for rocket's drag curve. Usually has a mean value of 1 and a uncertainty of 5% to 10%
    "powerOnDrag": (0.4624/0.366, 0.1*0.4624/0.366),                # Multiplier for rocket's drag curve. Usually has a mean value of 1 and a uncertainty of 5% to 10%
    "noseLength": (558.29/1000, 0.01),                       # Rocket's nose cone length (m)
    "noseDistanceToCM": (1.562416, 0.05),                 # Axial distance between rocket's center of dry mass and nearest point in its nose cone (m)
    "finSpan": (0.130, 0.001),                         # Fin span (m)
    "finRootChord": (0.120, 0.001),                    # Fin root chord (m)
    "finTipChord": (0.060, 0.001),                     # Fin tip chord (m)
    "finDistanceToCM": ( -1.14539, 0.05),                 # Axial distance between rocket's center of dry mass and nearest point in its fin (m) 
    "tailDistanceToCM": (-1.35739, 0.05),
    
    # Launch and Environment Details - run help(Environment) and help(Flight) for more information
    "inclination": (80, 1),                           # Launch rail inclination angle relative to the horizontal plane (degrees)
    "heading": (-45, 2),                                 # Launch rail heading relative to north (degrees)
    "railLength": (5.2 - 1.2870366 , 0.0005),                       # Launch rail length (m)
    "ensembleMember": list(range(10)),                  # Members of the ensemble forecast to be used
    
    # Launch and Environment Details
    "windX":(1, 0.03),
    "windY":(1, 0.03),
    "inclination": (85, 1),
    "heading": (60, 3), 
    "railLength": (5.2 - 1.2870366, 0.01*(5.2 - 1.2870366)),
    # "time":[0,6,12,18],
    # "day":[15,16,17,18],

    # Parachute Details - run help(Rocket) for more information
    "CdSDrogue": (1.077 , 0.1*1.077),                     # Drag coefficient times reference area for the drogue chute (m^2)
    "CdSMain": (6.184, 0.1*6.184),
    "lag_rec": (1 , 0.5),                               # Time delay between parachute ejection signal is detected and parachute is inflated (s)
    
    # Electronic Systems Details - run help(Rocket) for more information
    "lag_se": (0, 0)                              # Time delay between sensor signal is received and ejection signal is fired (s)
}

def flight_settings(analysis_parameters, total_number):
    i = 0
    while i < total_number:
        # Generate a flight setting
        flight_setting = {}
        for parameter_key, parameter_value in analysis_parameters.items():
            if type(parameter_value) is tuple:
                flight_setting[parameter_key] =  normal(*parameter_value)
            else:
                flight_setting[parameter_key] =  choice(parameter_value)

        # Skip if certain values are negative, which happens due to the normal curve but isnt realistic
        if flight_setting['lag_rec'] < 0 or flight_setting['lag_se'] < 0: continue
        
        # Update counter
        i += 1
        # Yield a flight setting
        yield flight_setting

def export_flight_data(flight_setting, flight_data, exec_time):
    # Generate flight results
    flight_result = {"outOfRailTime": flight_data.outOfRailTime,
                 "outOfRailVelocity": flight_data.outOfRailVelocity,
                        "apogeeTime": flight_data.apogeeTime,
                    "apogeeAltitude": flight_data.apogee - Env.elevation,
                           "apogeeX": flight_data.apogeeX,
                           "apogeeY": flight_data.apogeeY,
                        "impactTime": flight_data.tFinal,
                           "impactX": flight_data.xImpact,
                           "impactY": flight_data.yImpact,
                    "impactVelocity": flight_data.impactVelocity,
               "initialStaticMargin": flight_data.rocket.staticMargin(0),
             "outOfRailStaticMargin": flight_data.rocket.staticMargin(TestFlight.outOfRailTime),
                 "finalStaticMargin": flight_data.rocket.staticMargin(TestFlight.rocket.motor.burnOutTime),
                    "numberOfEvents": len(flight_data.parachuteEvents),
                     "executionTime": exec_time}
    
    # Calculate maximum reached velocity
    sol = np.array(flight_data.solution)
    flight_data.vx = Function(sol[:, [0, 4]], 'Time (s)', 'Vx (m/s)', 'linear', extrapolation="natural")
    flight_data.vy = Function(sol[:, [0, 5]], 'Time (s)', 'Vy (m/s)', 'linear', extrapolation="natural")
    flight_data.vz = Function(sol[:, [0, 6]], 'Time (s)', 'Vz (m/s)', 'linear', extrapolation="natural")
    flight_data.v = (flight_data.vx**2 + flight_data.vy**2 + flight_data.vz**2)**0.5
    flight_data.maxVel = np.amax(flight_data.v.source[:, 1])
    flight_result['maxVelocity'] = flight_data.maxVel
    
    # Take care of parachute results
    if len(flight_data.parachuteEvents) > 0:
        flight_result['drogueTriggerTime'] = flight_data.parachuteEvents[0][0]
        flight_result['drogueInflatedTime'] = flight_data.parachuteEvents[0][0] + flight_data.parachuteEvents[0][1].lag
        flight_result['drogueInflatedVelocity'] = flight_data.v(flight_data.parachuteEvents[0][0] + flight_data.parachuteEvents[0][1].lag)
    else:
        flight_result['drogueTriggerTime'] = 0
        flight_result['drogueInflatedTime'] = 0
        flight_result['drogueInflatedVelocity'] = 0
    
    # Write flight setting and results to file
    dispersion_input_file.write(str(flight_setting) + '\n')
    dispersion_output_file.write(str(flight_result) + '\n')

def export_flight_error(flight_setting):
    dispersion_error_file.write(str(flight_setting) + '\n')

    # Basic analysis info
filename = 'Dispersao/dispersionOutput2/Europa_dispersion_full_recovery3'
number_of_simulations = 5000

# Create data files for inputs, outputs and error logging
dispersion_error_file = open(str(filename)+'.disp_errors.txt', 'w')
dispersion_input_file = open(str(filename)+'.disp_inputs.txt', 'w')
dispersion_output_file = open(str(filename)+'.disp_outputs.txt', 'w')

# Initialize counter and timer
i = 0

initial_wall_time = time()
initial_cpu_time = process_time()

Env = Environment(
        railLength= 5.2 - 1.2870366,
        date=(2021, 5, 15, 12),
        latitude=32.94258333,   # Celular do Faruk
        longitude=-106.91488889,  # Celular do Faruk
        elevation=1189
    )

Env.maxExpectedHeight = 7000

Env.setAtmosphericModel(type="Ensemble", file = "GEFS")

motorThrust = r'Dispersao\Data\EmpSimHibNHNE final LASC.csv'
NACA0012 = r'Dispersao\Data\NACA0012_completa_RE_160000.csv'
PowerOffDrag = r'Dispersao\Data\Europa_IREC_Power_off_drag.csv'
PowerOnDrag = r'Dispersao\Data\Europa_IREC_Power_on_drag.csv'

# Set up parachutes
sisRecDrogue = SisRec.SisRecSt(0.8102412376980768, 0.2)
sisRecMain = SisRec.SisRecSt(0.8102412376980768, 0.2)

def drogueTrigger(p, y):
    return True if sisRecDrogue.update(p/100000) == 2 else False
def mainTrigger(p, y):
    return True if sisRecMain.update(p/100000) == 3 else False

# Iterate over flight settings
#out = display('Starting', display_id=True)
#print(analysis_parameters)
residual = 0
media = 0
for setting in flight_settings(analysis_parameters, number_of_simulations):
    start_time = process_time()
    i += 1
          
    # Update environment object

    # Define basic Environment object

    Env.railLength = setting['railLength']
    #Env.setDate((2021, 5, setting['day'], setting['time']))
    Env.selectEnsembleMember(setting['ensembleMember'])
    Env.windVelocityX = Env.windVelocityX*setting['windX']
    Env.windVelocityY = Env.windVelocityY*setting['windY']

    # Create motor
    Marimbondo =  SolidMotor(
        thrustSource=motorThrust,
        burnOut=11.8,
        reshapeThrustCurve=(setting['burnOut'], setting['impulse']),
        nozzleRadius=setting['nozzleRadius'],
        throatRadius=setting['throatRadius'],
        grainNumber=1,
        grainSeparation=setting['grainSeparation'],
        grainDensity=setting['grainDensity'],
        grainOuterRadius=setting['grainOuterRadius'],
        grainInitialInnerRadius=setting['grainInitialInnerRadius'],
        grainInitialHeight=setting['grainInitialHeight'],
        interpolationMethod='linear'
    )
    

    Posição_CMgraos = 2724.2 /1000       # Posição do centro de massa de todos os grãos e espaçadores, em metros 
    Posição_aletas  = (3273.1 - 7) /1000          # Posição das Aletas no CAD, considerar o ponto de encontro da aleta com o corpo do foguete mais próximo da origem
    Posição_cauda   = 3478.1 /1000       # Posição da transição entre o corpo cilíndrico do foguete e a cauda, que em geral é um tronco de cone ou não existe mesmo
    Posição_nozzle  = 3485.79 /1000       # Posição da saída do Nozzle (bocal). É também a posição da seção de maior diâmetro do nozzle
    Posição_CM_descarregado = 2120.706 /1000     # Posição do Centro de Massa do foguete sem os grãos de propelente ou espaçadores entre grãos

    # Create rocket
    EUROPA = Rocket(
        motor=Marimbondo,
        radius=setting['radius'],
        mass=setting['rocketMass'],
        inertiaI=setting['inertiaI'],
        inertiaZ=setting['inertiaZ'],
        distanceRocketNozzle=setting['distanceRocketNozzle'],
        distanceRocketPropellant=setting['distanceRocketPropellant'],
        powerOffDrag=PowerOffDrag,
        powerOnDrag=PowerOnDrag
    )
    EUROPA.setRailButtons([Posição_CM_descarregado - 2.06429, Posição_CM_descarregado - 3.460], 60)
    # Edit rocket drag
    EUROPA.powerOffDrag *= setting["powerOffDrag"]
    EUROPA.powerOnDrag *= setting["powerOnDrag"]
    # Add rocket nose, fins and tail
    NoseCone = EUROPA.addNose(
        length=setting['noseLength'],
        kind='vonKarman',
        distanceToCM=setting['noseDistanceToCM']
    )
    FinSet = EUROPA.addFins(
        n=3,
        rootChord=setting['finRootChord'],
        tipChord=setting['finTipChord'],
        span=setting['finSpan'],
        distanceToCM=setting['finDistanceToCM'],
        airfoil = NACA0012
        
    )
    Tail = EUROPA.addTail(topRadius=127/2000, bottomRadius=102/2000, length=50/1000, distanceToCM= Posição_CM_descarregado - Posição_cauda)
    # Add parachute
    Drogue = EUROPA.addParachute(
        'Drogue',
        CdS=setting['CdSDrogue'],
        trigger=drogueTrigger, 
        samplingRate=105,
        lag=setting['lag_rec'] + setting['lag_se'],
        noise=(0, 8.3, 0.5)
    )

    Main = EUROPA.addParachute(
        'Main',
        CdS=setting['CdSMain'],
        trigger=mainTrigger, 
        samplingRate=105,
        noise=(0, 8.3, 0.5),
        lag=setting['lag_rec'] + setting['lag_se']
        )
#     EUROPA.allInfo()
    # Run trajectory simulation
    try:
        sisRecDrogue.reset()
        sisRecDrogue.enable()
        sisRecMain.reset()
        sisRecMain.enable()
        TestFlight = Flight(
            rocket=EUROPA,
            environment=Env,
            inclination=setting['inclination'],
            heading=setting['heading'],
            maxTime=600
        )
        export_flight_data(setting, TestFlight, process_time() - start_time)
    except Exception as E:
        print(E)
        export_flight_error(setting)

    apogee = TestFlight.apogee - TestFlight.env.elevation
    media = (media * i + apogee) / (i+1)
    residual = abs(media - apogee)
    if residual <= 1e-3:
        break
    # Register time
    print("Iteration = ", i)
    print("Residual = ", residual)
    #out.update(f"Curent iteration: {i:06d} | Average Time per Iteration: {(process_time() - initial_cpu_time)/i:2.6f} s")

# Done

## Print and save total time
final_string = f"Completed {i} iterations successfully. Total CPU time: {process_time() - initial_cpu_time} s. Total wall time: {time() - initial_wall_time} s"
out.update(final_string)
dispersion_input_file.write(final_string + '\n')
dispersion_output_file.write(final_string + '\n')
dispersion_error_file.write(final_string + '\n')

## Close files
dispersion_input_file.close()
dispersion_output_file.close()
dispersion_error_file.close()