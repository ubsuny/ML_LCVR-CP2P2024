#BOOT W/ gunicorn --worker-class eventlet -w 1 -b 0.0.0.0:5000 app:app --timeout 200
#IF DOESN'T OCCUR AUTOMATICALLY


from threading import Lock
from flask import Flask, render_template, session, request, \
    copy_current_request_context
from flask_socketio import SocketIO, emit, join_room, leave_room, \
    close_room, rooms, disconnect
import sys
import numpy as np
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import lcvr_learning as lcl
sys.path.remove(parent_dir) 
import pandas as pd
import pickle
import eventlet
eventlet.monkey_patch()
from gevent.pywsgi import WSGIServer
from geventwebsocket.handler import WebSocketHandler
# Set this variable to "threading", "eventlet" or "gevent" to test the
# different async modes, or leave it set to None for the application to choose
# the best option based on installed packages.
async_mode = 'eventlet'
from engineio.payload import Payload

Payload.max_decode_packets = 1500


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode=async_mode)
thread = 'eventlet'
thread_lock = Lock()

lcvrs = lcl.lcvr_learning()
lcvrs.close_connection()
wavelength = 0
x_model = np.linspace(0.6, 10, 2000).reshape(-1, 1) # Inputs for setting V2


def background_thread():
    """Example of how to send server generated events to clients.not using it"""
    count = 0
    while count<20:
        socketio.sleep(10)
        count += 1
        #socketio.emit('my_response',
         #             {'data': 'Server Up', 'count': count}) 



'''how to get the html template and initiate the server with route'''
@app.route('/')
def index():
    return render_template('index.html', async_mode=socketio.async_mode)
   

'''backend for freq and amplitude frequency is not connected to anything just returns values'''
@socketio.event
def freq_1(message):
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response',
         {'data': message['data'], 'count': session['receive_count']})
    inst = message['data']
    print(inst)
'''Frequency is hard coded to 2KHz because LCVR only wants that'''
@socketio.event
def ampl_1(message):
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response',
         {'data': message['data'], 'count': session['receive_count']})
    inst = message['data']
    lcvrs.open_connection()
    lcvrs.set_input_volts(inst,1)
    v1, f1 = lcvrs.get_wave_info(1)
    lcvrs.close_connection()
    stat1 = str("CH1 Voltage:  CH1 Frequency: ")
    emit('status',
                {'status1': stat1})
    

@socketio.event
def freq_2(message):
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response',
         {'data': message['data'], 'count': session['receive_count']})
    inst = message['data']
    print(inst)    

@socketio.event
def ampl_2(message):
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response',
         {'data': message['data'], 'count': session['receive_count']})
    inst = message['data']
    lcvrs.open_connection()
    lcvrs.set_input_volts(inst,2)
    v2, f2 = lcvrs.get_wave_info(2)
    lcvrs.close_connection()
    stat1 = str("CH2 Voltage: " + str(v2) + " CH2 Frequency: " + str(f2))
    emit('status',
                {'status1': stat1})
    
    
@socketio.event
def wvlngth(message):
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response',
         {'data': message['data'], 'count': session['receive_count']})
    
    
    angle = message['data2']
    wavelength=message['data']    
    funcGen.sq_wave2(2000,inerp.ch1volt(wavelength))
    funcGen.sq_wave(2000,interp.voltage(angle,wavelength))
    funcGen.ch1_on()
    funcGen.ch2_on()
    
    print(angle,wavelength)    

@socketio.on('getdata')
def getdata(message):
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response',
         {'data': message['data'], 'count': session['receive_count']})
    
    lcvrs.close_connection()
    print("Input Received")
    wavelength=message['data']
    print("Starting data taking, please be patient")
    modeller = lcl.complete_fit_2d(wavelength,num_measurements=100,val_meas=100,num_models=1)
    model = modeller.get_2d_model()
    with open(str(wavelength) + '.pkl', 'wb') as f:
        pickle.dump(model, f)

    data_2d = pd.DataFrame(modeller.data_2d)
    data_2d.to_csv(str(wavelength) + '_2d.csv')
    data_3d = modeller.data_3d
    data_3d.to_csv(str(wavelength) + '_3d.csv')

    print("Modelling Complete, Please use the interface at the collected wavelength")   

@socketio.on('set_model')
def set_model(message):
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response',
         {'data': message['data'], 'count': session['receive_count']})
    
    global wavelength
    wavelength=int(message['data'])

    if os.path.exists("models/" + str(wavelength) + ".pkl"): 
        with open('models/'+ str(wavelength)+'.pkl', 'rb') as f:
            model = pickle.load(f)
        data_2d = pd.read_csv("data/"+str(wavelength)+"_2d.csv")
        data_3d = pd.read_csv("data/"+str(wavelength)+"_3d.csv")
        v1 = data_2d['V1'][2]
        lcvrs.open_connection()
        lcvrs.set_input_volts(v1,1)
        lcvrs.close_connection()
        global scaler,scale,rang,offset,model_arr
        scaler = lcl.optimize_model(data_3d)
        scale,rang,offset = scaler.get_scale(data_3d)
        model_arr = np.array(model.predict(x_model))
        emit('model_success', 'Model and V1 set successfully, angle can now be set') 

    else:
        wavelength = 0
        msg = message.get('message', 'No model with this wavelength, please calibrate above')
        emit('model_fail', msg)

@socketio.on('set_angle')
def set_angle(message):
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response',
         {'data': message['data'], 'count': session['receive_count']})

    if wavelength == 0:
        msg = "Error: Need to set wavelength before setting polarization angle!"
        emit('ang_set', msg)
    else:
        des_angle = float(message['data'])
        abs_diffs = np.abs(model_arr - des_angle)
        closest_index = np.argmin(abs_diffs)
        closest_v2 = x_model[closest_index][0]

        lcvrs.open_connection()
        lcvrs.set_input_volts(closest_v2,2)
        lcvrs.close_connection()

        msg = "Angle set to " + str(des_angle)

        emit('ang_set', msg)

'''

@socketio.event
def angle(message):
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response',
         {'data': message['data'], 'count': session['receive_count']})
    freq = message['data']
    print(freq)    
'''








@socketio.event
def disconnect_request():
    
    lcvrs.open_connection()
    lcvrs.outputs_off()
    lcvrs.close_connection()


    session['receive_count'] = session.get('receive_count', 0) + 1
    # for this emit we use a callback function
    # when the callback function is invoked we know that the message has been
    # received and it is safe to disconnect
    msg = "Successfully turned off"

    emit('off_msg', msg)
    


@socketio.event
def my_ping():
    emit('my_pong')



@socketio.event
def connect():
    global thread
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(background_thread)
    emit('my_response', {'data': 'Connected', 'count': 0})


@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected', request.sid)

@socketio.on('off_1')
def off_1():
    funcGen.ch1_off()
    emit('status',
            {'status1': 'ch1 Off'})

@socketio.on('off_2')
def off_2():
    funcGen.ch1_off()
    emit('status',
            {'status1': 'ch2 Off'})



if __name__ == '__main__':
    socketio.run(app)
