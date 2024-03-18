from ADCDifferentialPi import ADCDifferentialPi as adc


class lcvr_learning:

    
    def __init__(i2c1,i2c2,input_channel = 1, sample_rate = 18):
        """Initializes the object

        Args:
            i2c1: First i2c address of the ADC.
            i2c2: Second i2c address of the ADC. (Maybe scan for these later?)
            input_channel: Input channel for photodiode data. Using ABElectronics ADC hat. Should be an int (and typically 1)
            sample_rate: Sampling bitrate. Highest is 18
        """
        self.i2c1 = i2c1
        self.i2c2 = i2c2
        self.input_channel = input_channel
        self.signal = adc(i2c1,i2c2,sample_rate)

    def get_voltage(self):
        """
        Returns voltage from input channel
        """


        return self.signal.read_voltage(self.input_channel)