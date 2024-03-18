from ADCDifferentialPi import ADCDifferentialPi as adc
from time import sleep
import pyvisa
rm = pyvisa.ResourceManager('@py')
sdg = rm.open_resource('USB0::62700::4354::SDG2XCAD5R3372::0::INSTR')

class lcvr_learning:

    
    def __init__(i2c1,i2c2,input_channel = 1, sample_rate = 18, funcgen = sdg):
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

        if waveInfo1[freqIndex + 4:freqIndex + 8] != "2000":
            self.funcgen.write("C1:FRQ, 2000")
            print("WARNING: INCORRECT FREQUENCY, MUST BE 2 kHz")
            raise SystemExit
        if float(waveInfo1[voltIndexStart+4:voltIndexEnd]) > 20.0:
            self.funcgen.write("C1:AMP 1")
            print("WARNING: VOLTAGE TOO HIGH. VOLTAGE SHOULD BE NO GREATER THAN 20 V")
            raise SystemExit
        
        if waveInfo2[freqIndex + 4:freqIndex + 8] != "2000":
            self.funcgen.write("C2:FRQ, 2000")
            print("WARNING: INCORRECT FREQUENCY, MUST BE 2 kHz")
            raise SystemExit
        if float(waveInfo2[voltIndexStart+4:voltIndexEnd]) > 20.0:
            self.funcgen.write("C2:AMP 1")
            print("WARNING: VOLTAGE TOO HIGH. VOLTAGE SHOULD BE NO GREATER THAN 20 V")
            raise SystemExit
        

    
    def get_training_data(self):

        print("Starting training data scan. Don't touch anything please")

        # First check to make sure the parameters are in a safe range, then set voltage to a low value on both
        self.check_params()
