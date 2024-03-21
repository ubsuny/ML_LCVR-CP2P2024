from ADCDifferentialPi import ADCDifferentialPi as adc
import time
import numpy as np
import pyvisa
import pandas as pd
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
        self.funcgen.write("C1:BSWV AMP, 1")
        self.funcgen.write("C2:BSWV WVTP, SQUARE")
        self.funcgen.write("C2:BSWV FRQ, 2000")
        self.funcgen.write("C2:BSWV AMP, 1")

    def get_voltage(self):
        """
        Returns voltage from input channel
        """

        return self.signal.read_voltage(self.input_channel)
    
    def get_wave_info(self, channel: int):
        """
        Gets output wave info in a way that's slightly less annoying than using SCPI

        Args:
            channel: Channel number, should be 1 or 2 for Siglent 2042x
        
        Returns:
            freq: Frequency (in Hz)
            amp: Amplitude (in V)
        """
    
        waveInfo = self.funcgen.query("C" + str(channel) + ":BSWV?")

        # This is weird, but sometimes the function generator returns a null statement
        # The first char of the proper return is "C", but not for the null
        while 1 < 2:
            if waveInfo[0] != "C":
                waveInfo = self.funcgen.query("C" + str(channel) + ":BSWV?")
            else:
                break
        
        voltIndexStart = waveInfo.find("AMP")
        voltIndexEnd = waveInfo.find("V,AMPVRMS")
        freqIndex = waveInfo.find("FRQ")
    
        freq = float(waveInfo[freqIndex + 4:freqIndex + 8])
        volt = float(waveInfo[voltIndexStart+4:voltIndexEnd])
    
        return freq, volt
    
    def set_ch1_volts(self,voltage):

        if float(voltage) <= 20.0:
            self.funcgen.write("C1:BSWV AMP,  "+ str(voltage))
        else:
            raise("VOLTAGE CANNOT EXCEED 20 V LEST YOU HARM THE LCVR'S")
    
    def set_ch2_volts(self,voltage):

        if float(voltage) <= 20.0:
            self.funcgen.write("C2:BSWV AMP,  "+ str(voltage))
        else:
            raise("VOLTAGE CANNOT EXCEED 20 V LEST YOU HARM THE LCVR'S")
    
        
    def check_params(self):
        """
        This can definitely be way nicer, but right now this is just a quick check that makes sure both channels are running
        2 kHz, < 20 V square waves. Should do nothing if all is good, but the idea is just to have a function to call that
        will be able to periodically check that the operating conditions are good.
        """
        freq1, volt1 = self.get_wave_info(1)
        freq2, volt2 = self.get_wave_info(2)

        if freq1 != 2000.0:
            self.funcgen.write("C1:BSWV FRQ, 2000")
            print("WARNING: INCORRECT FREQUENCY, MUST BE 2 kHz")
            raise SystemExit
        if volt1 > 20.0:
            self.funcgen.write("C1:BSWV AMP, 1")
            print("WARNING: VOLTAGE TOO HIGH. VOLTAGE SHOULD BE NO GREATER THAN 20 V")
            raise SystemExit
        
        if freq2 != 2000.0:
            self.funcgen.write("C2:BSWV FRQ, 2000")
            print("WARNING: INCORRECT FREQUENCY, MUST BE 2 kHz")
            raise SystemExit
        if volt2 > 20.0:
            self.funcgen.write("C2:BSWV AMP, 1")
            print("WARNING: VOLTAGE TOO HIGH. VOLTAGE SHOULD BE NO GREATER THAN 20 V")
            raise SystemExit
        
    def outputs_on(self):
        self.check_params()
        self.funcgen.write("C1:OUTP ON")
        self.funcgen.write("C2:OUTP ON")

    def outputs_off(self):
        self.funcgen.write("C1:OUTP OFF")
        self.funcgen.write("C2:OUTP OFF")
    
    def get_training_data(self, num_iterations):

        realnum = int(num_iterations/4)

        print("Starting training data scan. Don't touch anything please")

        # First check to make sure the parameters are in a safe range, then set voltage to a low value on both
        self.set_ch1_volts(1)
        self.set_ch2_volts(1)
        self.outputs_on()
        
        volt_range = np.linspace(0,20,realnum) #MAX VOLTAGE IS 20 V!
        delay = 1 #Based on response time of LCVR and *SLEW RATE OF FUNCTION GENERATOR*! Check this in data sheet


        # Now iterate across a wide range of voltage configs for channel 1 and 2 to record training data
        # Note response time of LCVR is ~30 ms max, so that limit is hard coded in right now. For speed
        # later there may be ways to lower that (i.e. time how long each step takes  and subtract it so we can save a few
        # ms in case we need to do a lot of iterations?)
        trainingdata = []

        #This could be its own function maybe? Then call it 4 times
        #First keep ch2 constant and iterate over ch1
        for i in range(realnum):
            self.check_params()
            ch1_volts = volt_range[i]
            ch2_volts = self.get_wave_info(2)[1]
            self.set_ch1_volts(ch1_volts)
            self.set_ch2_volts(ch2_volts)
            time.sleep(delay)
            new_row = {'V1': ch1_volts, 'V2': ch2_volts, 'Out': self.get_voltage}
            trainingdata.append(new_row)

        self.set_ch1_volts(1)

        #Now ch1 constant iterate over ch2
        for i in range(realnum):
            self.check_params()
            ch1_volts = self.get_wave_info(1)[1]
            ch2_volts = volt_range[i]
            self.set_ch1_volts(ch1_volts)
            self.set_ch2_volts(ch2_volts)
            time.sleep(delay)
            new_row = {'V1': ch1_volts, 'V2': ch2_volts, 'Out': self.get_voltage}
            trainingdata.append(new_row)
        
        #Now both increasing together
        for i in range(realnum):
            self.check_params()
            ch1_volts = volt_range[i]
            ch2_volts = volt_range[i]
            self.set_ch1_volts(ch1_volts)
            self.set_ch2_volts(ch2_volts)
            time.sleep(delay)
            new_row = {'V1': ch1_volts, 'V2': ch2_volts, 'Out': self.get_voltage}
            trainingdata.append(new_row)

        #Now opposite directions
        for i in range(realnum):
            self.check_params()
            ch1_volts = volt_range[i]
            ch2_volts = volt_range[len(volt_range) - i - 1]
            self.set_ch1_volts(ch1_volts)
            self.set_ch2_volts(ch2_volts)
            time.sleep(delay)
            new_row = {'V1': ch1_volts, 'V2': ch2_volts, 'Out': self.get_voltage}
            trainingdata.append(new_row)

        self.outputs_off()
        self.set_ch1_volts(1)
        self.set_ch2_volts(1)


        trainingdataframe = pd.DataFrame(trainingdata)

        return trainingdataframe
