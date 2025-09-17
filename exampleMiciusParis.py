import model as ns
from model import inertialFrame
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
from math import radians
from org.orekit.frames import FramesFactory
from orekit.pyhelpers import datetime_to_absolutedate, absolutedate_to_datetime
from org.orekit.orbits import PositionAngleType


start = pd.Timestamp(datetime.now())
end = pd.Timestamp(datetime.now()+timedelta(hours= 200))
t = np.linspace(start.value, end.value, 100)
datetime_list = pd.to_datetime(t)
gcrf = FramesFactory.getGCRF()


a = 6872181.5020032395
e = 0.0013200
i = radians(97.3699)
omega = radians(178.5836)
Omega = radians(267.4500)
v = radians(246.0824)
from org.orekit.utils import IERSConventions, Constants

h = '18 Dec 2016 21:38:11.330'
format = '%d %b %Y %H:%M:%S.%f'
date = datetime.strptime(h, format)

epoch = datetime_to_absolutedate(date)

MU = Constants.WGS84_EARTH_MU

tle_line_1 = "1 41731U 16051A   24016.15735159  .00011450  00000-0  34540-3 0  9998"
tle_line_2 = "2 41731  97.3167 289.0989 0012522  59.2544 300.9930 15.34373256413200"
tle = (tle_line_1, tle_line_2)

wl = 810e-9
#delftparams = [53.8008, -1.5491, 63, "Delft"]
parisparams = [48.85, 2, 35, "Paris"]
kepler=[a, e, i, omega, Omega, v, PositionAngleType.TRUE, gcrf, epoch, MU]

micius = ns.Satellite(tle, simType="tle")
#delft = ns.GroundStation(*delftparams)
paris = ns.GroundStation(*parisparams)

TESTCHANNEL_paris = ns.SimpleDownlinkChannel(micius, paris)
#TESTCHANNEL_delft = ns.SimpleDownlinkChannel(micius, delft, wl)

results_paris = TESTCHANNEL_paris.calculateChannelParameters(datetime_list)
#results_delft= TESTCHANNEL_delft.calculateChannelParameters(datetime_list)

#             - length [m]
#             - elevation [degrees]

plt.figure(figsize=(21,9))
plt.plot(results_paris[2], results_paris[0])
#plt.plot(datetime_list, results_delft[1])
plt.xticks(rotation=45)
plt.show()
