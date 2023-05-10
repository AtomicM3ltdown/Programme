import serial
import dash
from dash import html
from dash import dcc


# create the Dash app
app = dash.Dash(__name__)

# configure the serial port
ser = serial.Serial(port='/dev/ttyUSB0', baudrate=9600, timeout=0.5)

# create the layout of the app
app.layout = html.Div(children=[
    html.H1('Temperature Monitor'),
    html.Div(id='temperature-value'),
    dcc.Interval(id='update-interval', interval=1000, n_intervals=0)
])

# create a callback function to update the temperature value
@app.callback(Output('temperature-value', 'children'),
              events=[Event('update-interval', 'interval')])
def update_temperature():
    # send the command to read the temperature value
    ser.write(b'#01RD\r\n')

    # read the response and extract the temperature value
    response = ser.readline()
    temperature = float(response[4:9])

    # convert the temperature to degrees Celsius
    temperature_celsius = (temperature - 1000) / 10.0

    # return the updated temperature value
    return f'Temperature: {temperature_celsius:.2f} degrees Celsius'

# start the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
    
# close the serial port when the app is closed
app.callback(Output('dummy', 'children'), [Input('dummy', 'value')])(lambda x: ser.close())
