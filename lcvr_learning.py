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

    def get_voltage(self, mode = "single"):
        """
        Returns voltage from input channel

        Args:
            mode: single or avg. Single returns one reading, avg takes an average to try and account for noisy signals
        """
        if mode == "avg":
            num = 20
            reads = []
            for i in range(num):
                reads.append(self.signal.read_voltage(self.input_channel))
            return np.average(reads)
        else:
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
            raise SystemExit("WARNING: INCORRECT FREQUENCY, MUST BE 2 kHz")
        if volt1 > 10.0:
            self.funcgen.write("C1:BSWV AMP, 1")
            raise SystemExit("WARNING: VOLTAGE TOO HIGH. VOLTAGE SHOULD BE NO GREATER THAN 10 V")
        
        if freq2 != 2000.0:
            self.funcgen.write("C2:BSWV FRQ, 2000")
            raise SystemExit("WARNING: INCORRECT FREQUENCY, MUST BE 2 kHz")
        if volt2 > 10.0:
            self.funcgen.write("C2:BSWV AMP, 1")
            raise SystemExit("WARNING: VOLTAGE TOO HIGH. VOLTAGE SHOULD BE NO GREATER THAN 10 V")
        
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

    def get_training_data(self, num_iterations: int, wavelength,gain = 1,mode = "all",v1 = 0):
        """
            Generates training data by scanning a range of input voltages for each lcvr and measuring the
            differential output signal from the photodetectors

            Args:
                num_iterations: Number of times you wish to iterate. More = better quality data
                wavelength: Input wavelength for given set. Necessary for 3D fitting
                gain = gain factor on the ADC
                mode: String, can be "all" or "fixed_v1"
                v1: only needed for fixed_v1 mode, sets constant V1 during scan
            Returns:
                trainingdataframe: A pandas dataframe containing the training data
        """
        self.signal.set_pga(gain)
        readmode = "single"

        print("Starting training data scan. Don't touch anything please")

        min_volt = .6
        delay = .05 #Based on response time of LCVR, which is around 30 ms, Right now 50 for safety/accuracy

        # First check to make sure the parameters are in a safe range, then set voltage to a low value on both
        self.set_input_volts(min_volt,1)
        self.set_input_volts(min_volt,2)
        self.outputs_on()
        time.sleep(delay)

        if mode =="all":
            # Now iterate across a wide range of voltage configs for channel 1 and 2 to record training data
            # Note response time of LCVR is ~30 ms max, so that limit is hard coded in right now. For speed
            # later there may be ways to lower that (i.e. time how long each step takes  and subtract it so we can save a few
            # ms in case we need to do a lot of iterations?)
            realnum = int(num_iterations/5)
            volt_range = np.linspace(min_volt,10,realnum)
            trainingdata = []

            for i in range(realnum):
                self.check_params()
                ch1_volts = volt_range[i]
                ch2_volts = self.get_wave_info(2)[1]
                self.set_input_volts(ch1_volts,1)
                self.set_input_volts(ch2_volts,2)
                time.sleep(delay)
                new_row = {'Wavelength': wavelength, 'V1': ch1_volts, 'V2': ch2_volts, 'Gain': gain, 'Out': self.get_voltage(mode = readmode)}
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
                new_row = {'Wavelength': wavelength, 'V1': ch1_volts, 'V2': ch2_volts, 'Gain': gain, 'Out': self.get_voltage(mode = readmode)}
                trainingdata.append(new_row)
            
            #Now both increasing together
            for i in range(realnum):
                self.check_params()
                ch1_volts = volt_range[i]
                ch2_volts = volt_range[i]
                self.set_input_volts(ch1_volts,1)
                self.set_input_volts(ch2_volts,2)
                time.sleep(delay)
                new_row = {'Wavelength': wavelength, 'V1': ch1_volts, 'V2': ch2_volts, 'Gain': gain, 'Out': self.get_voltage(mode = readmode)}
                trainingdata.append(new_row)

            #Now opposite directions
            for i in range(realnum):
                self.check_params()
                ch1_volts = volt_range[i]
                ch2_volts = volt_range[len(volt_range) - i - 1]
                self.set_input_volts(ch1_volts,1)
                self.set_input_volts(ch2_volts,2)
                time.sleep(delay)
                new_row = {'Wavelength': wavelength, 'V1': ch1_volts, 'V2': ch2_volts, 'Gain': gain, 'Out': self.get_voltage(mode = readmode)}
                trainingdata.append(new_row)
            
            #Now other way (order matters so this may be wise)
            for i in range(realnum):
                self.check_params()
                ch1_volts = volt_range[len(volt_range) - i - 1]
                ch2_volts = volt_range[i]
                self.set_input_volts(ch1_volts,1)
                self.set_input_volts(ch2_volts,2)
                time.sleep(delay)
                new_row = {'Wavelength': wavelength, 'V1': ch1_volts, 'V2': ch2_volts, 'Gain': gain, 'Out': self.get_voltage(mode = readmode)}
                trainingdata.append(new_row)

        # Allows us to scan over an axis with fixed V1. Very useful for single wavelength fitting
        elif mode == "fixed_v1":

            realnum = int(num_iterations)
            volt_range = np.linspace(min_volt,10,realnum)
            trainingdata = []
            ch1_volts = v1
            self.set_input_volts(ch1_volts,1)

            for i in range(realnum):
                self.check_params()
                ch2_volts = volt_range[i]
                self.set_input_volts(ch2_volts,2)
                time.sleep(delay)
                new_row = {'Wavelength': wavelength, 'V1': ch1_volts, 'V2': ch2_volts, 'Gain': gain, 'Out': self.get_voltage(mode = readmode)}
                trainingdata.append(new_row)

        else:
            raise ValueError("Invalid Scan Mode")

        self.set_input_volts(min_volt,1)
        self.set_input_volts(min_volt,2)
        self.outputs_off()


        trainingdataframe = pd.DataFrame(trainingdata)

        return trainingdataframe
    
    def add_angle(self,training_data):
        """
        Adds a column to the output data that represents the output as a polarization angle
        Maybe put this in get_training_data?

        Args:
            training_data: Training data from get_training_data (pandas dataframe)
        
        Returns:
            training_data: Training data with Angle column attached
        
        """

        range = training_data['Out'].max() - training_data['Out'].min()
        scale = 90/range
        offset = abs(training_data['Out'].min())
        training_data['Angle'] = (training_data['Out'] + offset)*scale

        return training_data
