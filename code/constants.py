import os

'''
The exact formulation of the problem will be as follows: 

given data covering the previous five days and sampled once per hour, 
can we predict the temperature in 24 hours?

Jena Climate dataset is made up of 14 different quantities 
(such air temperature, atmospheric pressure, humidity, wind direction, 
and so on) were recorded every 10 minutes, over several years. 
This dataset covers data from January 1st 2009 to December 31st 2016.
'''

DATA = "data/"
CSV = "jena_climate_2009_2016.csv"
FILE = os.path.join(DATA, CSV)