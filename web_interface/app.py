from threading import Lock
from flask import Flask, render_template, session, request, \
    copy_current_request_context
from flask_socketio import SocketIO, emit, join_room, leave_room, \
    close_room, rooms, disconnect
import lcvr_learning as lcl
import pandas as pd
import pickle
# Set this variable to "threading", "eventlet" or "gevent" to test the
# different async modes, or leave it set to None for the application to choose
# the best option based on installed packages.
async_mode = None
from engineio.payload import Payload

Payload.max_decode_packets = 1500


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode=async_mode)
thread = None
thread_lock = Lock()

lcvrs = lcl.lcvr_learning()

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
    lcvrs.set_input_volts(inst,1)
    v1, f1 = lcvrs.get_wave_info(1)
    stat1 = str("CH1 Voltage: " + str(v1) + " CH1 Frequency: " + str(f1))
    emit('status',
                {'status1': stat1()})
    

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
    lcvrs.set_input_volts(inst,2)
    v2, f2 = lcvrs.get_wave_info(2)
    stat1 = str("CH2 Voltage: " + str(v1) + " CH2 Frequency: " + str(f1))
    emit('status',
                {'status1': stat1()})
    
    
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

@socketio.event
def getdata(message):
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response',
         {'data': message['data'], 'count': session['receive_count']})
    
    
    wavelength=message['data']
    lcvrs.close_connection()
    print("Starting data taking, please be patient")
    modeller = lcl.complete_fit_2d(wavlength,num_measurements=200,val_meas=300,num_models=1)
    model = modeller.get_2d_model()
    with open(str(wavelength) + '.pkl', 'wb') as f:
        pickle.dump(model, f)

    data_2d = pd.DataFrame(modeller.data_2d)
    data_2d.to_csv(str(wavelength) + '2d.csv')
    data_3d = modeller.data_3d
    data_3d.to_csv(str(wavelength) + '3d.csv')
    
    print(angle,wavelength)   


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
    @copy_current_request_context
    def can_disconnect():
        disconnect()


    session['receive_count'] = session.get('receive_count', 0) + 1
    # for this emit we use a callback function
    # when the callback function is invoked we know that the message has been
    # received and it is safe to disconnect
    emit('my_response',
         {'data': 'Disconnected!', 'count': session['receive_count']},
         callback=can_disconnect)
    


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
