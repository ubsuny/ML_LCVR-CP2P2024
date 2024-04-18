import time
from tqdm import tqdm
import math
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from scipy.optimize import minimize, Bounds
from skopt import BayesSearchCV
from skopt.space import Real
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.acquisition import UpperConfidenceBound 
from botorch.sampling import SobolQMCNormalSampler
import random


class lcvr_learning:

    def __init__(self,i2c1 = 0x6a,i2c2 = 0x6b,input_channel = 1, sample_rate = 18, funcgen = "null",max_attempts = 20,attempt_delay = 1):
        """Initializes the object

        Args:
            i2c1: First i2c address of the ADC.
            i2c2: Second i2c address of the ADC. (Maybe scan for these later?)
            input_channel: Input channel for photodiode data. Using ABElectronics ADC hat. Should be an int (and typically 1)
            sample_rate: Sampling bitrate. Highest is 18
            funcgen: Function generator resource with pyvisa. Here using siglent 2042
            max_attempts: Max attempts at reading for exception handling. Should only take 2 or 3 max
        """

        from ADCDifferentialPi import ADCDifferentialPi as adc
        import pyvisa
        rm = pyvisa.ResourceManager('@py')
        sdg = rm.open_resource('USB0::62700::4354::SDG2XCAD5R3372::0::INSTR')
        funcgen = sdg

        self.i2c1 = i2c1
        self.i2c2 = i2c2
        self.input_channel = input_channel
        self.signal = adc(i2c1,i2c2,sample_rate)
        self.funcgen = funcgen
        self.max_attempts = max_attempts
        self.attempt_delay = attempt_delay
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
        #Rarely the function generator gives a pipe (buffer?) error which can throw off the whole measurement set,
        #So I want to just except and retry that should it happen. It should only take 1 or 2 attempts,
        #but I wanted to add *some* maximum in case for some reason there's some real issues.
        #This is repeated for any of the other SCPI read/write functions
        max_attempts = self.max_attempts
        delay = self.attempt_delay

        for attempt in range(max_attempts):
            try:
                waveInfo = self.funcgen.query("C" + str(channel) + ":BSWV?")
                break
            except:
                print("Read error, retrying")
                time.sleep(delay)

        # This is weird, but sometimes the function generator returns a null statement
        # The first char of the proper return is "C", but not for the null
        while 1 < 2:
            if waveInfo[0] != "C":
                for attempt in range(max_attempts):
                    try:
                        waveInfo = self.funcgen.query("C" + str(channel) + ":BSWV?")
                        break
                    except:
                        print("Read error, retrying")
                        time.sleep(delay)
            else:
                break
        
        voltIndexStart = waveInfo.find("AMP")
        voltIndexEnd = waveInfo.find("V,AMPVRMS")
        freqIndex = waveInfo.find("FRQ")
    
        freq = float(waveInfo[freqIndex + 4:freqIndex + 8])
        volt = float(waveInfo[voltIndexStart+4:voltIndexEnd])
    
        return freq, volt
    
    def outputs_off(self):
        self.funcgen.write("C1:OUTP OFF")
        self.funcgen.write("C2:OUTP OFF")
        
    def check_params(self):
        """
        This can definitely be way nicer, but right now this is just a quick check that makes sure both channels are running
        2 kHz, < 20 V square waves. Should do nothing if all is good, but the idea is just to have a function to call that
        will be able to periodically check that the operating conditions are good.
        """
        freq1, volt1 = self.get_wave_info(1)
        freq2, volt2 = self.get_wave_info(2)

        if freq1 != 2000.0:
            self.outputs_off()
            self.funcgen.write("C1:BSWV FRQ, 2000")
            raise SystemExit("WARNING: INCORRECT FREQUENCY, MUST BE 2 kHz")
        if volt1 > 10.0:
            self.outputs_off()
            self.funcgen.write("C1:BSWV AMP, 1")
            raise SystemExit("WARNING: VOLTAGE TOO HIGH. VOLTAGE SHOULD BE NO GREATER THAN 10 V")
        
        if freq2 != 2000.0:
            self.outputs_off()
            self.funcgen.write("C2:BSWV FRQ, 2000")
            raise SystemExit("WARNING: INCORRECT FREQUENCY, MUST BE 2 kHz")
        if volt2 > 10.0:
            self.outputs_off()
            self.funcgen.write("C2:BSWV AMP, 1")
            raise SystemExit("WARNING: VOLTAGE TOO HIGH. VOLTAGE SHOULD BE NO GREATER THAN 10 V")
        
    def outputs_on(self):
        self.check_params()
        max_attempts = self.max_attempts
        delay = self.attempt_delay
        for attempt in range(max_attempts):
                try:
                    self.funcgen.write("C1:OUTP ON")
                    break
                except:
                    print("Read error, retrying")
                    time.sleep(delay)
        
        for attempt in range(max_attempts):
                try:
                    self.funcgen.write("C2:OUTP ON")
                    break
                except:
                    print("Read error, retrying")
                    time.sleep(delay)
        

    def set_input_volts(self,target_volts,channel:int):
        max_attempts = self.max_attempts
        delay = self.attempt_delay
        current_volts = self.get_wave_info(channel)[1]
        change_range = np.linspace(current_volts,target_volts,3)
        for i in range(len(change_range)):
            for attempt in range(max_attempts):
                try:
                    self.funcgen.write("C"+str(channel)+":BSWV AMP, " + str(change_range[i]))
                    break
                except:
                    print("Read error, retrying")
                    time.sleep(delay)
        self.outputs_on()
    
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

    def get_training_data(self, num_iterations: int, wavelength,gain = 4,mode = "all",v1 = 0):
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
        delay = 1 #Response time of lcvr is ~30 ms, but can take longer to actually relax? So for training data purposes
                    # a large response time is *very* useful

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

            for i in tqdm(range(realnum)):
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

        trainingdataframe = self.add_angle(trainingdataframe)

        return trainingdataframe

    def find_3d_max(self,data_3d):
        """
        Uses Nelder-Mead to find true maximum of 3d data given some randomly sampled data
        """

        max_index = data_3d['Out'].idxmax()
        step = data_3d['V1'][1] - data_3d['V1'][0] #volt_step from the initial reading

        max_point = data_3d.iloc[max_index]
        x_max = max_point.V1
        y_max = max_point.V2
        z_max = max_point.Out
        bounds = Bounds([0.6, 0.6], [10, 10])  

        vertex_1 = np.array([x_max, y_max])
        vertex_2 = vertex_1 + np.array([2*step, 0])
        vertex_3 = vertex_1 + np.array([0, 2*step]) 
        initial_simplex = np.array([vertex_1, vertex_2, vertex_3])

        # Guided Sampling Function
        def sample_near_high_values(data, current_max_value, radius=0.5):
            sample_center = data[0]
            x_min = 0.6
            x_max = 10.0
            y_min = x_min
            y_max = x_max

            while True:  # Loop until we generate a valid sample
                new_point = sample_center + radius * (np.random.rand(2) - 0.5) 
                new_x, new_y = new_point  # Extract x and y
                x_min = 0.6
                x_max = 10.0
                y_min = x_min
                y_max = x_max

                if x_min <= new_x <= x_max and y_min <= new_y <= y_max: 
                    break  # Valid sample found 

            return new_point 

        def optimization_wrapper(point):
            x = np.round(point[0],decimals = 1)
            y = np.round(point[1],decimals = 1)
            delay = 1
            z = self.read_output(x, y,delay)
            return -z
        


        result = minimize(optimization_wrapper, initial_simplex[0], method='Nelder-Mead', 
                        options={'initial_simplex': initial_simplex},bounds = bounds)

        # Sampling iterations for refinement                         
        for _ in tqdm(range(5)):  # Run for a few iterations
            new_sample = sample_near_high_values(result.x.reshape(1,-1), result.fun)
            result = minimize(optimization_wrapper, new_sample, method='Nelder-Mead', bounds = bounds) 

        point_max = result.x
        out_max = result.fun

        return point_max,out_max

    def get_training_data_rand(self, num_iterations: int, wavelength,gain = 4,mode = "all",v1 = 0):
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
            max_volt = 10
            delay = 1 #Response time of lcvr is ~30 ms, but can take longer to actually relax? So for training data purposes
                        # a large response time is *very* useful

            # First check to make sure the parameters are in a safe range, then set voltage to a low value on both
            self.set_input_volts(min_volt,1)
            self.set_input_volts(min_volt,2)
            self.outputs_on()
            time.sleep(delay)

            v1_vals = np.random.rand(num_iterations)  * (max_volt - min_volt) + min_volt
            v2_vals = np.random.rand(num_iterations)  * (max_volt-min_volt) + min_volt

            if mode =="all":
                trainingdata = []

                for i in tqdm(range(num_iterations)):
                    self.check_params()
                    ch1_volts = v1_vals[i]
                    ch2_volts = v2_vals[i]
                    self.set_input_volts(ch1_volts,1)
                    self.set_input_volts(ch2_volts,2)
                    time.sleep(delay)
                    new_row = {'Wavelength': wavelength, 'V1': ch1_volts, 'V2': ch2_volts, 'Gain': gain, 'Out': self.get_voltage(mode = readmode)}
                    trainingdata.append(new_row)

                self.set_input_volts(min_volt,1)
                time.sleep(delay)

            else:
                raise ValueError("Invalid Scan Mode")

            self.set_input_volts(min_volt,1)
            self.set_input_volts(min_volt,2)
            self.outputs_off()


            trainingdataframe = pd.DataFrame(trainingdata)

            trainingdataframe = self.add_angle(trainingdataframe)

            return trainingdataframe

    def read_output(self,ch1_volts,ch2_volts,delay):

        self.set_input_volts(ch1_volts,1)
        self.set_input_volts(ch2_volts,2)
        time.sleep(delay)
        out = self.get_voltage()

        return out


    def get_2d_data(self, training_data, num_steps: int,optimize = False):
        """
        Gets a 2D fit for a *single wavelength* that can generate an arbitrary polarization with
        fixed V1. It takes the data and checks for a fixed V1 axis where max polarization
        range is achievable, then rescans over there and models this range

        Args:
            training_data: The 3D scan data obtained from get_training_data()
            num_steps: Number of steps for 2d data collection
            optimize: If True, will use find_3d_max to get the absolute best global max. (Adds about 10 minutes, give or take, and improvement is not *masssive*)

        Returns:
            data_2d: Data used for the 2D fit
        """

        #Finds V1 with max range
        print("Finding optimal V1")
        maxind3 = training_data['Out'].idxmax()
        best_v1 = training_data['V1'][maxind3]

        if optimize:
            max_point, out_max = self.find_3d_max(training_data)
            best_v1 = max_point[0]
        
    
        #Gets more thorough data along fixed V1 axis
        print("Rescanning along new axis")
        wavelength = training_data['Wavelength'][2]
        data_2d = self.get_training_data(num_steps, wavelength, mode = "fixed_v1", v1 = best_v1)

        return data_2d

    def close_connection(self):
        self.funcgen.close()

    def open_connection(self):
        self.funcgen.open()

class optimize_model:

    def __init__(self, data_2d):

        self.data_2d = data_2d

    def get_scale(self,data_3d):
        """
        Needed to preserve scaling between training and validation
        """

        top = data_3d['Out'].max()
        bot = data_3d['Out'].min()
        offset = abs(bot)
        range = top - bot
        scale = 90/range

        return scale,range,offset


    def optimize_model_2d(self):
        """
        Optimizes SVM regressor with given data

        Args:
            data_2d: 2d data for regressor to optimize

        Returns:
            best_c: Optimized c for svm regressor
            best_gamma: Optimized gamma for svm regressor
        """

        x = np.array(self.data_2d['V2'])
        X = x.reshape(-1, 1) 
        y = np.array(self.data_2d['Angle'])
        precision = 0.5
        best_c = 300 #A priori I know that these are approximate values for c and gamma via testing, speeds up convergence
        best_gamma = 20
        c_step = 50
        gamma_step = 1
        min_step_size = 0.05
        improvement_threshold = 0.01
        prev_score = 0
        param_grid = {
                'C': np.linspace(max(0.1, best_c - 2*c_step), best_c + 2*c_step, 10),
                'gamma': np.linspace(max(0.001, best_gamma - 2*gamma_step), best_gamma + 2*gamma_step, 10)
                }

        #Searches 10x10 parameter grids for the best fitting parameters
        #Finds the best parameter then re-searches in a progressively narrower range
        #If both parameters are unchanged
        #Should add some more explanation
        while True:
            print("Loop: " + str(best_c) + " " + str(best_gamma))
            grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5)
            grid_search.fit(X,y)

            current_c = grid_search.best_params_['C']
            current_gamma = grid_search.best_params_['gamma']
            current_score = grid_search.best_score_
            improvement = current_score - prev_score
            prev_score = current_score
            
            if not math.isclose(current_c,best_c,abs_tol = precision) or not math.isclose(current_gamma, best_gamma,abs_tol = precision):
                if improvement > improvement_threshold and c_step > min_step_size:
                    c_step /= 2
                if improvement > improvement_threshold and gamma_step > min_step_size: 
                    gamma_step /= 2
                best_c = current_c
                best_gamma = current_gamma
                param_grid = {
                'C': np.linspace(max(0.1, best_c - 2*c_step), best_c + 2*c_step, 10),
                'gamma': np.linspace(max(0.001, best_gamma - 2*gamma_step), best_gamma + 2*gamma_step, 10)
                }
            else:
                break
        
        return best_c, best_gamma
    
    def fit_2d(self,input_c,input_gamma):
        
        x = np.array(self.data_2d['V2'])
        X = x.reshape(-1,1)
        y = np.array(self.data_2d['Angle'])

        model = SVR(kernel='rbf', C=input_c, gamma=input_gamma)
        model.fit(X,y)

        return model
    
    def optimize_model_bayes(self):
        """
        Using scikit optimize for Bayesnian parameter tuning, should be faster/more accurate than old method
        """

        X = self.data_2d['V2']
        y = self.data_2d['Angle']

        search_space = {
        'C': Real(1e-2, 1e3, prior='log-uniform'), 
        'gamma': Real(1e-3, 1e2, prior='log-uniform')}

        opt = BayesSearchCV(SVR(kernel='rbf'),search_space,n_iter=30,cv=5,random_state=13)

        opt.fit(X,y)

        best_c = opt.best_params_['C']
        best_gamma = opt.best_params_['gamma']

        return best_c,best_gamma



    
    def calc_rmse(self,model,measurements,scale,range,offset):
        """
        Finds the RMS error between the model's predictions and actual measurements using random sampling (in real time)
        of the polarization
        NOTE: needs scale,range,offset from original data
        """

        measured_raw = []
        predicted = []
        v1 = self.data_2d['V1'][1]
        v2_low = self.data_2d['V2'].min()
        v2_high = self.data_2d['V2'].max()

        #Need to initialize the lcvrs for measurement. Note that input channels may change depending on your configuration
        lcvrs = lcvr_learning(0x6a,0x6b)
        lcvrs.set_input_volts(v1,1)

        #Generates random V2 values to test the fit against
        v2_inputs = np.random.rand(measurements) * (v2_high - v2_low) + v2_low

        v2_inputs = np.sort(v2_inputs) # If the increments are small the response time is better/better for the function generator

        for input in v2_inputs:
            v2 = np.array(input).reshape(-1,1) #Needed shape for model prediction
            lcvrs.set_input_volts(input,2)
            time.sleep(1)
            predicted.append(model.predict(v2)[0])
            measured_raw.append(lcvrs.get_voltage())

        
        lcvrs.outputs_off()
        # Need to change measured to an angle
        measured_raw = np.array(measured_raw)
        measured_angle = (measured_raw + offset)*scale
        rmse_measurements = [v2_inputs,measured_angle,predicted]

        mse = np.mean((measured_angle - predicted) ** 2)
        rmse = np.sqrt(mse)
        lcvrs.close_connection()
        
        return rmse, rmse_measurements

class complete_fit_2d:
    """
    Class to automate the entire process of fitting from start to finish. Just lumps the above together and 
    performs the process automatically. Should make it easy to go from aligned lcvr's w/ photodiodes
    to arbitrary polarization control hands free.
    """

    def __init__(self,wavelength, num_measurements = 500,val_meas = 200, num_models = 5):
        """
        Initializes the object

        Args:
            wavelength: Input wavelength in nm. This works for arb. wl so it is only important if you wish to record it
            num_measurements: Number of measurements for training data. More measurements ~ more accuracy
            num_models: Number of 2d models to compare against each other
            val_meas: Number of validation measurements
        """

        self.wavelength = wavelength
        self.num_measurements = num_measurements
        self.num_models = num_models
        self.val_meas = val_meas

        return None


    def compare_models(self,models,data_2d):
        """
        Takes an arbitrary list of models and compares their RMSE against random measurements

        ARgs:
            models: List of models fitted with optimize_model.fit_2d()

        Returns:
            best_index: Index of model with lowest error
            model_errors: List of RMSE errors of models
            data_2d: List of 2d data sets used for modelling
        """

        error = []
        for i in range(len(models)):
            optimize = optimize_model(data_2d[i])
            scale,range,offset = optimize.get_scale()
            v1 = data_2d['V1'][2]
            
            rmse, meas = optimize.calc_rmse(models[i],self.val_meas,v1,scale,range,offset)

            error.append(rmse)
        
        return np.argmin(error)

    def get_2d_model(self):
        """
        This runs the *whole* process outlined in data_collection.ipynb from start to finish. Takes ~30 minutes depending on
        your measurement parameters.

        Returns:
            model: Optimized, best selected 2d model to get an arbitrary polarization angle

        """

        lcvrs = lcvr_learning()
        print("Collecting initial 3D training data")
        data = lcvrs.get_training_data(self.num_measurements,self.wavelength)

        print("Collecting 2D scan(s)")
        data_2d = lcvrs.get_2d_data(data,self.val_meas,num_axes = int(self.num_models))

        lcvrs.close_connection()
        errors = []
        models = []
        voltages = []
        for i in range(len(data_2d)):
            print("Optimizing 2D Model #" + str(i))
            optimizer = optimize_model(data_2d[i])
            v1 = data_2d[i]['V1'][2]
            voltages.append(v1)
            scale,rang,offset = optimizer.get_scale(data) #This is sloppy implementation overall, would like to rework
            best_c, best_gamma = optimizer.optimize_model_2d()
            model = optimizer.fit_2d(best_c,best_gamma)
            models.append(model)
            rmse, error_meas = optimizer.calc_rmse(model,self.val_meas,scale,rang,offset)
            errors.append(rmse)
        
        min_index = np.argmin(errors)
        best_model = models[min_index]
        best_v1 = voltages[min_index]

        self.data_3d = data
        self.data_2d = data_2d[min_index]
        self.best_v1 = best_v1

        print("Returning model with RMSE " + str(errors[min_index]) + " degrees at " + str(best_v1) + " Volts")

        return best_model

    def set_polarization_angle(self,model,angle):
        """
        Sets the output polarization to the chosen angle using the model found before
        """

        x_model = np.linspace(0.6, 10, 2000).reshape(-1, 1)
        model_arr = model.predict(x_model)
        abs_diffs = np.abs(model_arr - angle)
        closest_index = np.argmin(abs_diffs)
        closest_v2 = x_model[closest_index][0]

        return closest_v2
            
            
        



