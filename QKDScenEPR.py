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

'''This script estimates the secret key rate in the
untrusted node scenario described  for different discrete points in the satelite orbit
 of the Micius satellite. The satellite sends photonic Bell pairs to the balloons that transmit
 them to the ground stations and then the ground stations transmit them to the Qlients'''

#Parameters for loss models of QuantumCity
#Satellite to ground channel parameters (QuanutmCity)
txDiv = 5e-6
sigmaPoint = 0.5e-6
rx_aperture_sat = 1
Cn2_sat = 0 #not used

#Free space channel parameters(QuantumCity)
W0 = 1550*1e-9/(txDiv*np.pi)
rx_aperture_drone = 0.4#radius of the aperture of the satellite
rx_aperture_ground = 1#radius of the aperture of the ground station. Same used for the balloon 
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
sourceeff=0.01#was 0.01 before I changed it 
QBER = 0.01#was 0.04 before I changed it

EPR_succ = 1
p_transmit=1

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



#Now we perform the Entaglement-based QKD protoccol in our setup

SKRlist=[]#list for the secret key rate for each point in the satellite orbit in the untrusted node scenario
#BellPairsList=[]#list for the number   is the eff,rate,   is sent/ 60 to become in seconds??
for i in range(len(times1)):#For each discrete point in the satellite orbit 

    visP = elevations1[i] > 0.0   #BalloonP must be visible to satellite
    visA = elevations2[i] > 0.0   #BalloonA must be visible to satellite 
    if not (visP and visA):#to run protoccol we assume that both balloons have to be visible by the satellite(>availability(time) same for all links) or you will have problems with the altitudes=0 with one of the two loss models
        SKRlist.append(0.0)
        continue
    # Initialize network
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
    net.connect_qonnectors("QonnectorBalloonP", "QonnectorMicius", distance = altitudes1[i]/1e3, loss_model = lossesSB)
    net.connect_qonnectors("QonnectorBalloonA", "QonnectorMicius", distance = altitudes2[i]/1e3, loss_model = lossesSB)

    # Get node instances
    city1 = net.network.get_node("QonnectorParis")
    city2 = net.network.get_node("QonnectorAthens")
    balloon1 = net.network.get_node("QonnectorBalloonP")
    balloon2 = net.network.get_node("QonnectorBalloonA")
    satel= net.network.get_node("QonnectorMicius")
    Alice = net.network.get_node("QlientAlice")
    Bob= net.network.get_node("QlientBob")
    
    #Now we send the photonic Bell pair(EPR pair)
    send = SendEPR(balloon1, balloon2, EPR_succ, satel) #satellite sends photonic bell pairs to each balloon. Then they are retrasmitted.
    send.start()

    transmit1 = TransmitProtocol(satel, city1, p_transmit, balloon1)
    transmit1.start()

    transmit2 = TransmitProtocol(satel, city2, p_transmit, balloon2)
    transmit2.start()

    transmit3 = TransmitProtocol(balloon1, Alice, p_transmit, city1)
    transmit3.start()

    transmit4 = TransmitProtocol(balloon2, Bob, p_transmit, city2)
    transmit4.start()


    receive1 = ReceiveProtocol(city1, Qonnector_meas_succ, Qonnector_meas_flip, True, Alice)
    receive1.start()

    receive2 = ReceiveProtocol(city2, Qonnector_meas_succ, Qonnector_meas_flip, True, Bob)
    receive2.start()


    # Run simulation
    stat = ns.sim_run(duration = simtime)

    # Calculate the rates
    L1 = Sifting(Alice.keylist, Bob.keylist) #sifting for the entanglement process
    sent = len(satel.QlientKeys[balloon1.name])
    rec = len(L1)
    eff = rec/sent
    rate = ratesources*sourceeff*eff
    skr = rate*(1-2*h(QBER)) #here the skr is in bits/sec. In the BB84 QKD protoccol it is in kbits/sec

    SKRlist.append(float(skr))


print(SKRlist)

#Plotting the skr for the process with respect to time for a certain time period
plt.figure()
plt.plot(times1, SKRlist, color='blue')
plt.xlabel('Time')
plt.ylabel('SKR (bits/s)') 
plt.title("Secret key rate with respect to time")
plt.grid(True)
plt.show()

#Calculation of the mean skr for 1 whole passage (we accept only values where the satellite is visible)
print()
vals = [v for v in SKRlist if v != 0.0]
mean_val = sum(vals)/len(vals)  if vals else 0.0
print("Mean secret key rate = ", mean_val)#mean skr for the entanglement-based protocol between the Qlients Alice and Bob
print()




