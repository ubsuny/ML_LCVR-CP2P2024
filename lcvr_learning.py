from ADCDifferentialPi import ADCDifferentialPi as adc
from time import sleep
import numpy as np
import pyvisa
rm = pyvisa.ResourceManager('@py')
sdg = rm.open_resource('USB0::62700::4354::SDG2XCAD5R3372::0::INSTR')

class lcvr_learning:
    
    def __init__(self,i2c1,i2c2,input_channel = 1, sample_rate = 18, funcgen = sdg):
        """Initializes the object

        Args:
            i2c1: First i2c address of the ADC.
            i2c2: Second i2c address of the ADC. (Maybe scan for these later?)
            input_channel: Input channel for photodiode data. Using ABElectronics ADC hat. Should be an int (and typically 1)
            sample_rate: Sampling bitrate. Highest is 18
            funcgen: Function generator resource with pyvisa. Here using siglent 2042
        """
        self.i2c1 = i2c1
        self.i2c2 = i2c2
        self.input_channel = input_channel
        self.signal = adc(i2c1,i2c2,sample_rate)
        self.funcgen = funcgen
        self.funcgen.write("C1:BSWV WVTP, SQUARE")
        self.funcgen.write("C1:BSWV FRQ, 2000") # Wave must be 2000 Hz Square wave, no exceptions

    def get_voltage(self):
        """
        Returns voltage from input channel
        """

        return self.signal.read_voltage(self.input_channel)
    
    def set_ch1_volts(self,voltage):

        if float(voltage) < 20.0:
            self.funcgen.write("C1:BSWV AMP,  "+ str(voltage))
        else:
            raise("VOLTAGE CANNOT EXCEED 20 V LEST YOU HARM THE LCVR'S")
    
    def set_ch2_volts(self,voltage):

        if float(voltage) < 20.0:
            self.funcgen.write("C2:BSWV AMP,  "+ str(voltage))
        else:
            raise("VOLTAGE CANNOT EXCEED 20 V LEST YOU HARM THE LCVR'S")
        
    def check_params(self):
        """
        This can definitely be way nicer, but right now this is just a quick check that makes sure both channels are running
        2 kHz, < 20 V square waves. Should do nothing if all is good, but the idea is just to have a function to call that
        will be able to periodically check that the operating conditions are good.
        """
        waveInfo1 = self.funcgen.query("C1:BSWV?")
        waveInfo2 = self.funcgen.query("C2:BSWV?")

        while 1 < 2:
            if waveInfo1[0] != "C":
                waveInfo1 = self.funcgen.query("C1:BSWV?")
            else:
                break

        while 1 < 2:
            if waveInfo2[0] != "C":
                waveInfo2 = self.funcgen.query("C2:BSWV?")
            else:
                break

        freqIndex1 = waveInfo1.find("FRQ")
        voltIndexStart1 = waveInfo1.find("AMP")
        voltIndexEnd1 = waveInfo1.find("V,AMPVRMS")
        freqIndex2 = waveInfo2.find("FRQ")
        voltIndexStart2 = waveInfo2.find("AMP")
        voltIndexEnd2 = waveInfo2.find("V,AMPVRMS")

        if waveInfo1[freqIndex1 + 4:freqIndex1 + 8] != "2000":
            self.funcgen.write("C1:BSWV FRQ, 2000")
            print("WARNING: INCORRECT FREQUENCY, MUST BE 2 kHz")
            raise SystemExit
        if float(waveInfo1[voltIndexStart1+4:voltIndexEnd1]) > 20.0:
            self.funcgen.write("C1:BSWV AMP, 1")
            print("WARNING: VOLTAGE TOO HIGH. VOLTAGE SHOULD BE NO GREATER THAN 20 V")
            raise SystemExit
        
        if waveInfo2[freqIndex2 + 4:freqIndex2 + 8] != "2000":
            self.funcgen.write("C2:BSWV FRQ, 2000")
            print("WARNING: INCORRECT FREQUENCY, MUST BE 2 kHz")
            raise SystemExit
        if float(waveInfo2[voltIndexStart2+4:voltIndexEnd2]) > 20.0:
            self.funcgen.write("C2:BSWV AMP, 1")
            print("WARNING: VOLTAGE TOO HIGH. VOLTAGE SHOULD BE NO GREATER THAN 20 V")
            raise SystemExit
        
    def outputs_on(self):
        self.check_params()
        self.funcgen.write("C1:OUTP ON")
        self.funcgen.write("C2:OUTP ON")
    
    def get_training_data(self, num_iterations):

        realnum = int(num_iterations/4)

        print("Starting training data scan. Don't touch anything please")

        # First check to make sure the parameters are in a safe range, then set voltage to a low value on both
        self.check_params()
        self.set_ch1_volts(1)
        self.set_ch2_volts(1)
        self.funcgen.write()
        
        volt_range = np.linspace(0,20,realnum)

        # Now iterate across a wide range of voltage configs for channel 1 and 2 to record training data
        # Note response time of LCVR is ~30 ms max, so that limit is hard coded in right now. For speed
        # later there may be ways to lower that (i.e. time how long each step takes  and subtract it so we can save a few
        # ms in case we need to do a lot of iterations?)
        trainingdata = []
        
        #First keep ch2 constant and iterate over ch1
        for i in range(realnum):
            time.sleep(.03)
            ch1_volts = volt_range[i]
            ch2_volts = 1 #I know this is bad. Probably should just implement a function to read it straight will be quick
            self.set_ch1_volts(ch1_volts)
            new_row = {'V1': ch1_volts, 'V2': ch2_volts, 'Out': self.get_voltage}
            trainingdata.append(new_row)

        self.set_ch1_volts(1)

        #Now ch1 constant iterate over ch2
        for i in range(realnum):
            time.sleep(.03)
            ch1_volts = 1 #I know this is bad. Probably should just implement a function to read it straight will be quick
            ch2_volts = volt_range[i]
            self.set_ch2_volts(ch2_volts)
            new_row = {'V1': ch1_volts, 'V2': ch2_volts, 'Out': self.get_voltage}
            trainingdata.append(new_row)
        
        #Now both increasing together
        for i in range(realnum):
            time.sleep(.03)
            ch1_volts = volt_range[i]
            ch2_volts = volt_range[i]
            self.set_ch2_volts(ch2_volts)
            new_row = {'V1': ch1_volts, 'V2': ch2_volts, 'Out': self.get_voltage}
            trainingdata.append(new_row)

        #Now opposite directions
        for i in range(realnum):
            time.sleep(.03)
            ch1_volts = volt_range[i]
            ch2_volts = volt_range[len(volt_range) - i - 1]
            self.set_ch2_volts(ch2_volts)
            new_row = {'V1': ch1_volts, 'V2': ch2_volts, 'Out': self.get_voltage}
            trainingdata.append(new_row)

        trainingdataframe = pd.DataFrame(trainingdata)

        return trainingdataframe
