# This script listens for keyboard input and sends motor control commands to your ESP8266.
# The motor will stop when the movement key is released.
# Make sure to install the required libraries first by running:
# pip install pynput requests

import requests
from pynput import keyboard

# IMPORTANT: This IP address must match the ESP8266's Access Point IP.
ESP8266_IP = "192.168.4.1" 

# --- State Variables ---
# Keep track of the engine state to toggle it
engine_is_on = False
# Keep track of currently pressed movement keys to handle multiple key presses
active_movement_keys = set()

# --- URL Endpoints ---
# Construct the base URL for the ESP8266 server
base_url = f"http://{ESP8266_IP}"

print("Python Motor Controller is running (Hold Mode).")
print("-----------------------------------")
print("W, A, S, D: Hold to Move")
print("E:          Toggle Engine ON/OFF")
print("Esc:        Exit Script")
print("-----------------------------------")


def send_command(command):
    """Sends a specific command to the ESP8266 web server."""
    url = f"{base_url}/{command}"
    try:
        # The timeout prevents the script from hanging if the ESP8266 is not reachable.
        requests.get(url, timeout=1.0)
        print(f"Sent command: {command}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending request to {url}: {e}")
        print("Check that you are connected to the ESP8266's Wi-Fi network.")


def on_press(key):
    """Callback function for when a key is pressed."""
    global engine_is_on
    
    try:
        char = key.char
        # --- Movement Controls (only send if it's a new key) ---
        if char in ['w', 'a', 's', 'd'] and char not in active_movement_keys:
            active_movement_keys.add(char)
            if char == 'w':
                send_command('forward')
            elif char == 's':
                send_command('backward')
            elif char == 'a':
                send_command('left')
            elif char == 'd':
                send_command('right')
        
        # --- Engine Toggle ---
        elif char == 'e':
            engine_is_on = not engine_is_on # Toggle the state
            if engine_is_on:
                send_command('engine_on')
            else:
                send_command('engine_off')

    except AttributeError:
        # Stop the listener if the 'Esc' key is pressed
        if key == keyboard.Key.esc:
            send_command('stop') # Ensure motors are stopped on exit
            print("Exiting script.")
            return False


def on_release(key):
    """Callback function for when a key is released."""
    try:
        char = key.char
        if char in ['w', 'a', 's', 'd']:
            active_movement_keys.discard(char)
            # Only stop if no other movement keys are being held down
            if not active_movement_keys:
                send_command('stop')
    except AttributeError:
        pass


# Set up the listener to monitor keyboard events
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()