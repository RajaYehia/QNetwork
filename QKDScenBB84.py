from lossmodel2 import FixedSatelliteLossModel
from scipy.special import i0, i1
from numpy.random import weibull
import model as mdl
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
from math import radians
from org.orekit.frames import FramesFactory
from orekit.pyhelpers import datetime_to_absolutedate, absolutedate_to_datetime
from org.orekit.orbits import PositionAngleType
from org.orekit.utils import IERSConventions, Constants
#For QKD 
from QBEuropeFunctions2 import *  

'''This script estimates the secret key rates for each link in the BB84 QKD protoccol described
 for different discrete points in the satelite orbit of the Micius satellite'''

#Parameters for loss models of QuantumCity
#Satellite to ground channel parameters (QuanutmCity)
txDiv = 5e-6
sigmaPoint = 0.5e-6
rx_aperture_sat = 1
Cn2_sat = 0 #not used

#Free space channel parameters(QuantumCity)
W0 = 1550*1e-9/(txDiv*np.pi)
rx_aperture_drone = 0.4#radius of the aperture of the satellite
rx_aperture_ground = 1#radius of the aperture of the ground station. Same for the balloon
Cn2_drone_to_ground = 10e-16#1e-15
Cn2_drone_to_drone = 10e-18
wavelength = 1550*1e-9
c = 299792.458 #speed of light in km/s
Tatm = 1

#Parameters for for the balloon to ground link
W0b= 0.1 #initial beam waist from the balloon(b)
#rx_aperture_balloon= 0.3#radius of the aperture of the balloon (reason for 30cm: gives the highest channel efficiency for a balloon at 20km based on the paper "Free-space model for a balloon-based quantum network")
sigmaPointb = 1e-6 #Pointing error due to balloon(b) retransmition
pcouplb= 0.8 #coupling efficiency when the photon is captured by the balloon(b). Can also be a distribution(with Gaussian beam) not just a value


#For the QKD part
Qonnector_meas_succ = 0.85 #Detector efficiency at the receiver
drone_meas_succ = 0.25 #Detector efficiency in the balloon
tracking_efficiency = 0.8 #Tracking efficiency
h_balloonP = 20 #Altitude of the balloonP in km
h_balloonA = 20 #Altitude of the balloonA in km
dist_cities = 2118 #distance between the cities(paris and athens distance accounting for earth's shape) in km
distAlice = 16 #distance between Alice and her Qonnector(in Paris) in km
distBob = 21 #distance between Bob and his Qonnector(in Athens) in km
zenith_angle = 0 #because each balloon is above it's corresponding ground station

#Parameters and function to calculate the secret key rate
ratesources = 80e6
sourceeff=0.01
QBER = 0.01#WAS 0.04 BEFORE I CHANGED IT
def h(p):#binary entropy function
    return -p*np.log2(p)-(1-p)*np.log2(1-p)

simtime = 100000 #in ns

#Creating time objects
hi = '15 May 2025 01:15:00.330'
format = '%d %b %Y %H:%M:%S.%f'
starttime = datetime.strptime(hi, format)
start = pd.Timestamp(starttime)
end = pd.Timestamp(start+timedelta(days= 2))
t = np.linspace(start.value, end.value, 40)#lat term=number of spacings(=number of elements)
datetime_list = pd.to_datetime(t)
gcrf = FramesFactory.getGCRF()

epoch = datetime_to_absolutedate(starttime)
MU = Constants.WGS84_EARTH_MU

#Micius satellite TLE 
tle_line_1 = "1 41731U 16051A   25265.33409832  .00080539  00000-0  89175-3 0  9999"
tle_line_2 = "2 41731  97.2945 187.6583 0004596  12.3887 347.7483 15.62724089508176"
tle = (tle_line_1, tle_line_2)

wl = 810e-9  #wavelength (this wavelength is not used)
parisparams = [48.85, 2, 35, "Paris"]#latitude, longitute, altitude in meters, label
balloonPparams= [48.85, 2, 20000, "BalloonP" ]#exactly above the paris ground station, orekit works with distances in meters
athensparams = [37.98, 23.7, 70, "Athens"]
balloonAparams= [37.98, 23.7, 20000, "BalloonA"]#exactly above athens ground station


 
micius = mdl.Satellite(tle, simType="tle")
paris = mdl.GroundStation(*parisparams)
balloonP= mdl.GroundStation(*balloonPparams)
athens = mdl.GroundStation(*athensparams)
balloonA= mdl.GroundStation(*balloonAparams)

TESTCHANNEL_balloonP = mdl.SimpleDownlinkChannel(micius, balloonP)
results_balloonP = TESTCHANNEL_balloonP.calculateChannelParameters(datetime_list)

TESTCHANNEL_balloonA = mdl.SimpleDownlinkChannel(micius, balloonA)
results_balloonA = TESTCHANNEL_balloonA.calculateChannelParameters(datetime_list)

(altitudes1,elevations1,times1) = results_balloonP
for i in range(len(altitudes1)):
    if elevations1[i]<0.0: #For balloon we consider it 0 degrees due to its 20km vertical position in the atmosphere
        altitudes1[i]=0.0

(altitudes2,elevations2,times2) = results_balloonA
for i in range(len(altitudes2)):
    if elevations2[i]<0.0: #For balloon we consider it 0 degrees due to its 20km vertical position in the atmosphere
        altitudes2[i]=0.0



#Now we perform the BB84 QKD protoccol in our setup

SKR1=[]#lists for secret key rates for each link in each discrete point in the orbit
SKR2=[]
SKR3=[]
SKR4=[]
SKR5=[]
SKR6=[]
for i in range(len(times1)):
    visP = elevations1[i] > 0.0   #BalloonP must be visible to satellite
    visA = elevations2[i] > 0.0   #BalloonA must be visible to satellite 
    
    if not (visP and visA):#to run protoccol we assume that both balloons have to be visible by the satellite(>availability(time) same for all links) or you will have problems with the altitudes=0 with one of the two loss models
        # append zeros (or compute only the city<->balloon parts if desired)
        SKR1.append(0.0); SKR2.append(0.0); SKR3.append(0.0)
        SKR4.append(0.0); SKR5.append(0.0); SKR6.append(0.0)
        continue
    # Initialize network for this specific point
    net = QEurope("Europe")#Or maybe call it "World" in this project

    # Create two quantum Cities
    net.Add_Qonnector("QonnectorParis")
    net.Add_Qlient("QlientAlice",distAlice,"QonnectorParis")

    net.Add_Qonnector("QonnectorAthens")
    net.Add_Qlient("QlientBob",distBob,"QonnectorAthens")
    
    # Create drones
    net.Add_Qonnector("QonnectorBalloonP")
    net.Add_Qonnector("QonnectorBalloonA")
    net.Add_Qonnector("QonnectorMicius")
    
    # Create channels (in a way they are the loss models/channels)
    lossesSB= FixedSatelliteLossModel(txDiv=txDiv, sigmaPoint=sigmaPoint, rx_aperture=rx_aperture_ground, Cn2=Cn2_drone_to_drone, wavelength=wavelength)#we pass the aperture of the reciever and also use this Cn2
    lossesBG= FixedSatelliteLossModel(txDiv=txDiv, sigmaPoint=sigmaPointb, rx_aperture=rx_aperture_ground, Cn2=Cn2_drone_to_ground, wavelength=wavelength)#we pass the aperture of the reciever and also use this Cn2
    
    # Connect nodes in the network (c..._q...)
    net.connect_qonnectors("QonnectorParis", "QonnectorBalloonP", distance = 20, loss_model = lossesBG )#distances in km here (distance from balloon to ground=20km)
    net.connect_qonnectors("QonnectorAthens", "QonnectorBalloonA", distance = 20, loss_model = lossesBG)
    net.connect_qonnectors("QonnectorBalloonP", "QonnectorMicius", distance = altitudes1[i]/1e3, loss_model = lossesSB)#distances in km here, but orekit outputted meters 
    net.connect_qonnectors("QonnectorBalloonA", "QonnectorMicius", distance = altitudes2[i]/1e3, loss_model = lossesSB)

    # Get node instances
    city1 = net.network.get_node("QonnectorParis")
    city2 = net.network.get_node("QonnectorAthens")
    balloon1 = net.network.get_node("QonnectorBalloonP")
    balloon2 = net.network.get_node("QonnectorBalloonA")
    satel= net.network.get_node("QonnectorMicius")
    Alice = net.network.get_node("QlientAlice")
    Bob= net.network.get_node("QlientBob")
    
    #Performing the QKD BB84 protoccol in the links (key generation)
    send1 = SendBB84(city1, 1, 0, Alice)
    send1.start()
    receive1 = ReceiveProtocol(Alice, Qonnector_meas_succ, Qonnector_meas_flip, True, city1)#Qonnector_meas_flip is in the QBEuropeFunctions2.py ,and the last argument in the recieve protoccol is the reciever
    receive1.start()
    
    send2 = SendBB84(city1, 1, 0, balloon1)
    send2.start()
    receive2 = ReceiveProtocol(balloon1, Qonnector_meas_succ, Qonnector_meas_flip, True, city1)#city1 is the reciever here as explained above
    receive2.start()
    
    send3 = SendBB84(balloon1, 1, 0, satel)
    send3.start()
    receive3 = ReceiveProtocol(satel, Qonnector_meas_succ, Qonnector_meas_flip, True, balloon1)
    receive3.start()
    
    send4 = SendBB84(balloon2, 1, 0, satel)
    send4.start()
    receive4 = ReceiveProtocol(satel, Qonnector_meas_succ, Qonnector_meas_flip, True, balloon2)
    receive4.start()

    send5 = SendBB84(city2, 1, 0, balloon2)
    send5.start()
    receive5 = ReceiveProtocol(balloon2, Qonnector_meas_succ, Qonnector_meas_flip, True, city2)
    receive5.start()
    
    send6 = SendBB84(city2, 1, 0, Bob)
    send6.start()
    receive6 = ReceiveProtocol(Bob, Qonnector_meas_succ, Qonnector_meas_flip, True, city2)
    receive6.start()

    #Run simulation
    stat = ns.sim_run(duration = simtime)

    #Calculate the rates
    sentAlice = len(Alice.keylist)
    recfromAlice = len(city1.QlientKeys[Alice.name])#recieved from balloon1
    eff1 = recfromAlice/sentAlice
    rate1 = ratesources*sourceeff*eff1
    skr1 = rate1*(1-2*h(QBER))/1000 #all skr's in kbits/sec

    sentballoon1 = len(balloon1.QlientKeys[city1.name])
    recfromballoon1 = len(city1.QlientKeys[balloon1.name])
    eff2 = recfromballoon1/sentballoon1
    rate2 = ratesources*sourceeff*eff2
    skr2 = rate2*(1-2*h(QBER))/1000

    sentsatelB1 = len(satel.QlientKeys[balloon1.name])#satellite(satel) sending photons(qubits) to balloon1(in paris)
    recfromsatelB1 = len(balloon1.QlientKeys[satel.name])
    eff3 = recfromsatelB1/sentsatelB1
    rate3 = ratesources*sourceeff*eff3
    skr3= rate3*(1-2*h(QBER))/1000

    sentsatelB2 = len(satel.QlientKeys[balloon2.name])#satellite(satel) sending photons(qubits) to balloon2(in athens)
    recfromsatelB2 = len(balloon2.QlientKeys[satel.name])
    eff4 = recfromsatelB2/sentsatelB2
    rate4 = ratesources*sourceeff*eff4
    skr4 = rate4*(1-2*h(QBER))/1000


    sentballoon2 = len(balloon2.QlientKeys[city2.name])
    recfromballoon2 = len(city2.QlientKeys[balloon2.name])
    eff5 = recfromballoon2/sentballoon2
    rate5 = ratesources*sourceeff*eff5
    skr5 = rate5*(1-2*h(QBER))/1000


    sentBob = len(Bob.keylist)
    recfromBob = len(city2.QlientKeys[Bob.name])
    eff6 = recfromBob/sentBob
    rate6 = ratesources*sourceeff*eff6
    skr6 = rate6*(1-2*h(QBER))/1000
    

    SKR1.append(float(skr1))
    SKR2.append(float(skr2))    
    SKR3.append(float(skr3))
    SKR4.append(float(skr4))
    SKR5.append(float(skr5))
    SKR6.append(float(skr6))



print(SKR1)
print(SKR3)
print(SKR2)


skr_lists = [SKR1, SKR2, SKR3, SKR4, SKR5, SKR6]
names = [
    "Alice - Paris",
    "BalloonP - Paris",
    "Satellite - BalloonP",
    "Satellite - BalloonA",
    "BalloonA - Athens",
    "Bob - Athens"
]

#Plotting skr with respect to time for each link
for i, skr in enumerate(skr_lists, start=1):
    n = min(len(times1), len(skr))
    x = times1[:n]
    y = skr[:n]

    plt.figure()
    plt.plot(x, y, color='blue', label=names[i-1])
    plt.xlabel('Time')
    plt.ylabel('SKR (kbits/s)') 
    plt.title(names[i-1])
    plt.legend()
    plt.grid(True)
    plt.show()

#ADD A SAVEFIG OR 6 SAVEFIGS
###DO FOR VERY LITTLE TIME FOR  1 FULL PASSAGE IN A SPECIFIC DAY

#To compute mean skr value of each link  
names = [
    "Alice -> Paris Qonnector",
    "BalloonP -> Paris Qonnector",
    "Satellite -> BalloonP",
    "Satellite -> BalloonA",
    "BalloonA -> Athens Qonnector",
    "Bob -> Athens Qonnector"
]

skr_lists = [SKR1, SKR2, SKR3, SKR4, SKR5, SKR6]
print()
for name, skr in zip(names, skr_lists):
    # keep only non-zero values
    vals = [v for v in skr if v != 0.0]
    mean_val = sum(vals) / len(vals) if vals else 0.0
    print(f"{name:<35} {mean_val:8.2f}")
print()




































































