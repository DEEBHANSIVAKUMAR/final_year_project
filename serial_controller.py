"""
serial_controller.py - Serial Communication Manager between Python and ESP32
"""
import time
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

class ESP32SerialController:
    def __init__(self, port='COM13', baudrate=115200):
        """
        Initializes Serial connection to ESP32.
        On Linux/Raspberry Pi, port is typically '/dev/ttyUSB0' or '/dev/ttyACM0'.
        On Windows, port is typically 'COM3', 'COM4', 'COM13', etc.
        """
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.last_command = None
        self._last_err_time = 0

        if not SERIAL_AVAILABLE:
            print("[SerialController] Warning: 'pyserial' not installed. Install using: pip install pyserial")
            return

        self.connect()

    def connect(self):
        if self.ser and self.ser.is_open:
            return True
        if not SERIAL_AVAILABLE:
            return False
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(1)  # Wait for ESP32 serial reset
            print(f"============================================================")
            print(f"[SerialController] SUCCESS! Connected to ESP32 on {self.port} at {self.baudrate} baud.")
            print(f"============================================================")
            return True
        except Exception as e:
            now = time.time()
            if now - self._last_err_time > 3.0:  # Throttle warning to once every 3 sec
                self._last_err_time = now
                if "Access is denied" in str(e) or "PermissionError" in str(e):
                    print(f"------------------------------------------------------------")
                    print(f"[Serial ERROR] Cannot open {self.port}: ACCESS DENIED!")
                    print(f"  -> Reason: Arduino IDE (or Serial Monitor) is holding {self.port} open!")
                    print(f"  -> Fix: Please CLOSE the Serial Monitor in Arduino IDE.")
                    print(f"------------------------------------------------------------")
                else:
                    print(f"[Serial ERROR] Could not open port {self.port}: {e}")
            self.ser = None
            return False

    def send_command(self, command: str):
        """
        Sends direction command to ESP32 only if command has changed.
        Valid commands: 'FORWARD' (or 'F'), 'BACKWARD' (or 'B'), 'LEFT' (or 'L'), 'RIGHT' (or 'R'), 'STOP' (or 'S')
        """
        if not self.ser or not self.ser.is_open:
            # Try to auto-reconnect if port was locked previously
            if not self.connect():
                return

        # Map long direction names to single character commands (or full string)
        cmd_map = {
            "FORWARD": "F",
            "BACKWARD": "B",
            "BRAKE": "S",
            "LEFT": "L",
            "RIGHT": "R",
            "STOP": "S"
        }

        
        cmd_to_send = cmd_map.get(command.upper(), command.upper())

        # Avoid redundant commands
        if cmd_to_send != self.last_command:
            try:
                # Send command followed by newline character \n
                msg = f"{cmd_to_send}\n"
                self.ser.write(msg.encode('utf-8'))
                self.last_command = cmd_to_send
                print(f"[SerialController] Predicted Direction: {command} -> Sent '{cmd_to_send}' to ESP32 ({self.port} @ {self.baudrate} baud)")
            except Exception as e:
                print(f"[SerialController] Error sending command to ESP32: {e}")
                self.ser = None  # mark closed so it auto-reconnects


    def read_response(self):
        """
        Reads and prints any feedback lines received from ESP32 over serial.
        """
        if self.ser and self.ser.is_open and self.ser.in_waiting:
            try:
                lines = self.ser.read_all().decode('utf-8', errors='ignore')
                for line in lines.splitlines():
                    if line.strip():
                        print(f"[ESP32 Feedback] {line.strip()}")
            except Exception as e:
                pass

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("[SerialController] Closed Serial connection.")

