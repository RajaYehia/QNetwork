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

'''This script plots the channel efficiency with respect to time for the channels 
satellite-to-balloon-to-ground and satellite-to-ground'''

#Parameters for loss models of QuantumCity
#Satellite to ground channel parameters (QuanutmCity)
txDiv = 5e-6
sigmaPoint = 0.5e-6
rx_aperture_sat = 1
Cn2_sat = 0 #not used currently

#Free space channel parameters(QuantumCity)
W0 = (1550*1e-9)/(txDiv*np.pi)
rx_aperture_drone = 0.4#radius of the aperture of the satellite
rx_aperture_ground = 1#radius of the aperture of the ground station. Same for the balloon 
Cn2_drone_to_ground = 10e-16#1e-15
Cn2_drone_to_drone = 10e-18
wavelength = 1550*1e-9
c = 299792.458 #speed of light in km/s
Tatm = 1

#Parameters for for the balloon to ground link
W0b= 0.1 #initial beam waist from the balloon(b)
sigmaPointb = 1e-6 #Pointing error due to balloon(b) retransmition
pcouplb= 0.8 #coupling efficiency when the photon is captured by the balloon(b). Can also be a distribution(with Gaussian beam) not just a value

#Creating the time objects
h = '15 May 2025 01:15:00.330'
format = '%d %b %Y %H:%M:%S.%f'
starttime = datetime.strptime(h, format)
start = pd.Timestamp(starttime)
end = pd.Timestamp(start+timedelta(days= 30))
t = np.linspace(start.value, end.value, 20000)
datetime_list = pd.to_datetime(t)
gcrf = FramesFactory.getGCRF()


epoch = datetime_to_absolutedate(starttime)

MU = Constants.WGS84_EARTH_MU

#Micius satellite TLE 
tle_line_1 = "1 41731U 16051A   25265.33409832  .00080539  00000-0  89175-3 0  9999"
tle_line_2 = "2 41731  97.2945 187.6583 0004596  12.3887 347.7483 15.62724089508176"
tle = (tle_line_1, tle_line_2)


wl = 810e-9
#delftparams = [53.8008, -1.5491, 63, "Delft"]
parisparams = [48.85, 2, 35, "Paris"]  
balloonparams= [48.85, 2, 20000, "Balloon" ]


micius = mdl.Satellite(tle, simType="tle")
paris = mdl.GroundStation(*parisparams)
balloon= mdl.GroundStation(*balloonparams)

TESTCHANNEL_paris = mdl.SimpleDownlinkChannel(micius, paris)
results_paris = TESTCHANNEL_paris.calculateChannelParameters(datetime_list)

TESTCHANNEL_balloon = mdl.SimpleDownlinkChannel(micius, balloon)
results_balloon = TESTCHANNEL_balloon.calculateChannelParameters(datetime_list)

(altitudes1,elevations1,times1) = results_paris
for i in range(len(altitudes1)):
    if elevations1[i]<10: #For the ground station we consider that for a passage we must exceed the 10 degree value due to the existance of mountains on the horizon
        altitudes1[i]=0.0

(altitudes2,elevations2,times2) = results_balloon
for i in range(len(altitudes2)):
    if elevations2[i]<0:  #For balloon we consider it 0 degrees due to its 20km vertical position in the atmosphere
        altitudes2[i]=0.0



#Here we use the FixedSatelliteLossModel class for the setup that we have
lossesSG= FixedSatelliteLossModel(txDiv=txDiv, sigmaPoint=sigmaPoint, rx_aperture=rx_aperture_ground, Cn2=Cn2_drone_to_ground, wavelength=wavelength)#we pass the aperture of the reciever and also use this Cn2
lossesSB= FixedSatelliteLossModel(txDiv=txDiv, sigmaPoint=sigmaPoint, rx_aperture=rx_aperture_ground, Cn2=Cn2_drone_to_drone, wavelength=wavelength)#we pass the aperture of the reciever and also use this Cn2
lossesBG= FixedSatelliteLossModel(txDiv=txDiv, sigmaPoint=sigmaPointb, rx_aperture=rx_aperture_ground, Cn2=Cn2_drone_to_ground, wavelength=wavelength)#we pass the aperture of the reciever and also use this Cn2



#Define channel efficiency function for satellite to balloon downlink channel, vertical link from balloon to ground, and downlink from satellite to ground
def cheffSB(alti):#alti=altitude  SB=Satellite to Balloon
    result = lossesSB._compute_weibull_loss_model_parameters(alti)
    (a, scaleL, T0) = result
    x = weibull(a, 1)
    scaleX = scaleL * x
    T = T0*np.exp(-scaleX/2)
    chaneff1 = lossesSB.Tatm * T**2  
    return chaneff1

def cheffBG(alti):#alti=altitude  BG=Balloon to Ground
    result = lossesBG._compute_weibull_loss_model_parameters(alti)
    (a, scaleL, T0) = result
    x = weibull(a, 1)
    scaleX = scaleL * x
    T = T0*np.exp(-scaleX/2)
    chaneff2 = lossesBG.Tatm * T**2 
    return chaneff2

def cheffSG(alti):#alti=altitude  SG=Satellite to Ground
    result = lossesSG._compute_weibull_loss_model_parameters(alti)
    (a, scaleL, T0) = result
    x = weibull(a, 1)
    scaleX = scaleL * x
    T = T0*np.exp(-scaleX/2)
    chaneff3 = lossesSG.Tatm * T**2 
    return chaneff3

#Lists with efficiencies
efficiencies1= [cheffSG((h/1e3)) for h in altitudes1]
efficiencies2= [cheffBG(20)*pcouplb*cheffSB(h/1e3) for h in altitudes2] #Channel eff of balloon to ground link is considered always constant and have to also consider coupling efficiency at balloon  
efficiencies3= [cheffSB(h/1e3) for h in altitudes2]


#Creating text file displaying various data in vertical columns (also creating efficient method of storing a lot of data for each passage)
with open("Datafile.txt", "w") as f:
    f.write("Datetimes\t\t\t\t\tChannel_length_km\tElevation_deg\n")
    for i in range(len(times1)):
        if elevations1[i]>20:#we do this only for passages of more than 20 degrees
            tti=times1[i] #changed a lot of variable names to not be the same as the other date creation in the main code
            #h = '15 May 2021 01:15:00.330'
            tt = pd.to_datetime(str(tti))
            ms = tt.nanosecond // 1_000_000
            ti = tt.strftime('%d %b %Y %H:%M:%S.') + f'{ms:03d}'
            formati = '%d %b %Y %H:%M:%S.%f'
            starttimei = datetime.strptime(str(ti), formati)
            starti = pd.Timestamp(starttimei-timedelta(minutes=10))
            endi = pd.Timestamp(starti+timedelta(minutes= 20))
            tii = np.linspace(starti.value, endi.value, 200)
            datetime_listi = pd.to_datetime(tii)
            print(pd.to_datetime(tti))
            print(elevations1[i])
            gcrfi = FramesFactory.getGCRF()

            epochi = datetime_to_absolutedate(starttimei)
            MUi = Constants.WGS84_EARTH_MU

            tle_line_1 = "1 41731U 16051A   25265.33409832  .00080539  00000-0  89175-3 0  9999"
            tle_line_2 = "2 41731  97.2945 187.6583 0004596  12.3887 347.7483 15.62724089508176"
            tle = (tle_line_1, tle_line_2)
            
            parisparams = [48.85, 2, 35, "Paris"]  
            balloonparams= [48.85, 2, 20000, "Balloon" ]

            miciusi = mdl.Satellite(tle, simType="tle")
            parisi = mdl.GroundStation(*parisparams)
            ballooni= mdl.GroundStation(*balloonparams)

            TESTCHANNEL_parisi = mdl.SimpleDownlinkChannel(miciusi, parisi)
            results_parisi = TESTCHANNEL_parisi.calculateChannelParameters(datetime_listi)

            TESTCHANNEL_ballooni = mdl.SimpleDownlinkChannel(miciusi, ballooni)
            results_ballooni = TESTCHANNEL_ballooni.calculateChannelParameters(datetime_listi)

            (altitudes1i,elevations1i,times1i) = results_parisi
            for k in range(len(altitudes1i)):
                if elevations1i[k]<10: #For the ground station we consider that for a passage we must exceed the 10 degree value due to the existance of mountains on the horizon
                    altitudes1i[k]=0.0

            (altitudes2i,elevations2i,times2i) = results_ballooni
            for j in range(len(altitudes2i)):
                if elevations2i[j]<0:  #For balloon we consider it 0 degrees due to its 20km vertical position in the atmosphere
                    altitudes2i[j]=0.0

            for p in range(len(times1i)):
                if elevations1i[p]>10: #we tell the code to consider only the elevations >10 degrees
                    datr=str(times1i[p])
                    datrr = pd.to_datetime(datr)
                    datrrr= datrr.strftime('%d %b %Y %H:%M:%S.%f')[:-3] 
                    chlength=altitudes1i[p]/1e3 #to convert to km
                    elev=elevations1i[p]#in degrees
                    f.write(f"{datrrr}\t{chlength:.6f}\t\t\t{elev:.3f}\n")

f.close()



plt.plot(times1, efficiencies1, marker='.' , color='red', label="Satellite-to-Ground link")#For SG link
plt.plot(times2, efficiencies2, marker='.' , color='blue', label="Satellite-to-Balloon-to-Ground link")#For SBG link
#plt.plot(times2, efficiencies3, marker='.' ,color='green' )#For SB
plt.xlabel("Time")#in seconds
plt.ylabel("Channel efficiency")
plt.title("Satellite-to-ground efficiency vs time")
plt.legend()
plt.savefig("ChannelEfficTheor.pdf")
plt.grid(True)
plt.show()





