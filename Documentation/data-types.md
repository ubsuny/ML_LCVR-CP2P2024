# Data Types Used in This Project

In this project, several different data types will be used. Here we will demonstrate some different data types used using python annotations. However, since the project will be largely written in python, most occasions will not call for the explicit casting of datatypes

# String

## Example 1
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

## Example 2 (String and Float)

An interesting example comes when we look at the output of `query()`. It returns a string in the form of
```Python
waveInfo = "<channel>:BSWV<type>,FRQ,<frequency>,AMP,<amplitude>,OFST,<offset>,DUTY,<duty>"
```
With more info depending on the query.

So how do we programatically check a given value? For example, the LCVR's should not have a driving voltage over 20 volts, so we need to check AMP and ensure that it is not greater than 20. Here, python annotations are extremely useful. We can pick out the point in the response that gives the amplitude, then cast that string to a float and check to make sure it's no greater than 20.
```Python
def volt_check():
  waveInfo = "<channel>:BSWV<type>,FRQ,<frequency>,AMP,<amplitude>,OFST,<offset>,DUTY,<duty>"
  voltIndexStart = waveInfo.find("AMP")
  voltIndexEnd = waveInfo.find("V,AMPVRMS")
  if float(waveInfo[voltIndexStart+4:voltIndexEnd]) > 20.0:
      sdg.write("C1:AMP 1")
      print("WARNING: VOLTAGE TOO HIGH. VOLTAGE SHOULD BE NO GREATER THAN 20 V")
      raise SystemExit
```
Where here we made sure to cast our string to a float so that our if statement can accurately compare this to our voltage threshold. We can run this function periodically as a safety measure to ensure that at no point is the voltage pushed above the threshold.

# Int

A space-saving measure that may prove useful is to store certain training data values as integers, rather than allowing python to assign a type. The model is trained off of 3 parameters, as is shown in the example svm_regression_test.ipynb. Two of these parameters are polarization angle and wavlength of input light. Both of them will probably only be measurable to integer precision, so it may be beneficial to ensure they are stored/sent as such, since this will save much needed memory.
