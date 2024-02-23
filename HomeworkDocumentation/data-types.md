# Data Types Used in This Project

In this project, several different data types will be used. Here we will demonstrate some different data types used using python annotations. However, since the project will be largely written in python, most occasions will not call for the explicit casting of datatypes

# String

The first data type we'll look at is string. This is crucial in communicating with the function generator used to control the input voltage to the LCVR's, since it communicates via SCPI (Standard Commands for Programmable Instruments) queries. An example of an SCPI query is as follows:
```Python
import pyvisa
rm = pyvisa.ResourceManager()

# String initialization for query that returns wave data
getInfo = str("C1:BSWV?")

# Now sending the query with pyvisa
waveInfo = sdg.query(getInfo)
```

It's crucial that the data sent with `query()` is a string, since otherwise the function generator will not be able to interpret it.
