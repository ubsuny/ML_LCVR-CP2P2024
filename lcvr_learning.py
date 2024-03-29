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
        if volt1 > 10.0:
            self.funcgen.write("C1:BSWV AMP, 1")
            print("WARNING: VOLTAGE TOO HIGH. VOLTAGE SHOULD BE NO GREATER THAN 10 V")
            raise SystemExit
        
        if freq2 != 2000.0:
            self.funcgen.write("C2:BSWV FRQ, 2000")
            print("WARNING: INCORRECT FREQUENCY, MUST BE 2 kHz")
            raise SystemExit
        if volt2 > 10.0:
            self.funcgen.write("C2:BSWV AMP, 1")
            print("WARNING: VOLTAGE TOO HIGH. VOLTAGE SHOULD BE NO GREATER THAN 10 V")
            raise SystemExit
        
    def outputs_on(self):
        self.check_params()
        self.funcgen.write("C1:OUTP ON")
        self.funcgen.write("C2:OUTP ON")

    def outputs_off(self):
        self.funcgen.write("C1:OUTP OFF")
        self.funcgen.write("C2:OUTP OFF")

    def set_input_volts(self,target_volts,channel:int):
        current_volts = self.get_wave_info(channel)[1]
        change_range = np.linspace(current_volts,target_volts,3)
        for i in range(len(change_range)):
            self.funcgen.write("C"+str(channel)+":BSWV AMP, " + str(change_range[i]))
        self.outputs_on()

    def get_training_data(self, num_iterations: int, wavelength,gain = 1):
        """
            Generates training data by scanning a range of input voltages for each lcvr and measuring the
            differential output signal from the photodetectors

            Args:
                num_iterations: Number of times you wish to iterate. More = better quality data
                wavelength: Input wavelength for given set. Necessary for 3D fitting
                gain = gain factor on the ADC
        """

        realnum = int(num_iterations/4)

        print("Starting training data scan. Don't touch anything please")

        min_volt = .05
        volt_range = np.linspace(min_volt,4,realnum) #Retardance greatly diminished by ~5 V


        delay = .05 #Based on response time of LCVR, which is around 30 ms, Right now 50 for safety/accuracy

        # First check to make sure the parameters are in a safe range, then set voltage to a low value on both
        self.set_input_volts(min_volt,1)
        self.set_input_volts(min_volt,2)
        self.outputs_on()
        time.sleep(delay)


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
            self.set_input_volts(ch1_volts,1)
            self.set_input_volts(ch2_volts,2)
            time.sleep(delay)
            new_row = {'Wavelength': wavelength, 'V1': ch1_volts, 'V2': ch2_volts, 'Gain': gain, 'Out': self.get_voltage()}
            trainingdata.append(new_row)

        self.set_input_volts(min_volt,1)
        time.sleep(delay)

        #Now ch1 constant iterate over ch2
        for i in range(realnum):
            self.check_params()
            ch1_volts = self.get_wave_info(1)[1]
            ch2_volts = volt_range[i]
            self.set_input_volts(ch1_volts,1)
            self.set_input_volts(ch2_volts,2)
            time.sleep(delay)
            new_row = {'Wavelength': wavelength, 'V1': ch1_volts, 'V2': ch2_volts, 'Gain': gain, 'Out': self.get_voltage()}
            trainingdata.append(new_row)
        
        #Now both increasing together
        for i in range(realnum):
            self.check_params()
            ch1_volts = volt_range[i]
            ch2_volts = volt_range[i]
            self.set_input_volts(ch1_volts,1)
            self.set_input_volts(ch2_volts,2)
            time.sleep(delay)
            new_row = {'Wavelength': wavelength, 'V1': ch1_volts, 'V2': ch2_volts, 'Gain': gain, 'Out': self.get_voltage()}
            trainingdata.append(new_row)

        #Now opposite directions
        for i in range(realnum):
            self.check_params()
            ch1_volts = volt_range[i]
            ch2_volts = volt_range[len(volt_range) - i - 1]
            self.set_input_volts(ch1_volts,1)
            self.set_input_volts(ch2_volts,2)
            time.sleep(delay)
            new_row = {'Wavelength': wavelength, 'V1': ch1_volts, 'V2': ch2_volts, 'Gain': gain, 'Out': self.get_voltage()}
            trainingdata.append(new_row)

        self.outputs_off()
        self.set_input_volts(min_volt,1)
        self.set_input_volts(min_volt,2)


        trainingdataframe = pd.DataFrame(trainingdata)

        return trainingdataframe
