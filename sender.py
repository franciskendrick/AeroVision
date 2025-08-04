import time
from pyfirmata import Arduino, util

# Replace this with your actual serial port (COMx on Windows, /dev/ttyUSBx on Linux/Mac)
board = Arduino('/dev/ttyUSB0')

# Define pins (corresponding to Arduino pin numbers, not NodeMCU)
# For example: pin 5 is digital pin 5 on Arduino Uno
BUTTON_PINS = [5, 4, 0, 2, 14, 12]  # Adjust if you're using Uno (pins above 13 are invalid)

# Start iterator thread to avoid buffer overflow
it = util.Iterator(board)
it.start()

# Set pins as input with pull-up (PyFirmata doesn't support internal pull-up, so use external resistor)
buttons = [board.get_pin(f'd:{pin}:i') for pin in BUTTON_PINS]

# Store last command
last_command = ' '

def get_command():
    current_command = 'S'  # Default: Stop

    # Read button states
    states = [btn.read() for btn in buttons]

    if states[0] == False:
        current_command = 'F'  # Forward
    elif states[1] == False:
        current_command = 'B'  # Backward
    elif states[2] == False:
        current_command = 'L'  # Left
    elif states[3] == False:
        current_command = 'R'  # Right

    if states[4] == False:
        current_command = 'N'  # eNgine ON
    elif states[5] == False:
        current_command = 'O'  # engine Off

    return current_command

while True:
    command = get_command()

    if command != last_command:
        print(f"[SENT] {command}")
        # Here you can send it over socket, MQTT, UDP, etc.
        last_command = command

    time.sleep(0.05)
