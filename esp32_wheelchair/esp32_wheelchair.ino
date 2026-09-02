// ESP32 Serial Motor Controller for Smart Wheelchair
// Pin Configuration for BTS7960 / H-Bridge Motor Drivers

// Motor Pins configuration
#define L_RPWM 22
#define L_LPWM 23
#define R_RPWM 18
#define R_LPWM 19

// PWM Speed Control (0 to 255)
const int freq = 5000;
const int resolution = 8;
const int speedVal = 80; // Speed set to 80 (Slow & Safe for Wheelchair)

void stopMotors() {
  ledcWrite(L_RPWM, 0); ledcWrite(L_LPWM, 0);
  ledcWrite(R_RPWM, 0); ledcWrite(R_LPWM, 0);
}

void moveForward() {
  // Both motors forward
  ledcWrite(L_LPWM, 0); ledcWrite(L_RPWM, speedVal);
  ledcWrite(R_LPWM, 0); ledcWrite(R_RPWM, speedVal);
}

void moveBackward() {
  // Both motors backward
  ledcWrite(L_RPWM, 0); ledcWrite(L_LPWM, speedVal);
  ledcWrite(R_RPWM, 0); ledcWrite(R_LPWM, speedVal);
}

void turnLeft() {
  // Left motor reverse, Right motor forward
  ledcWrite(L_RPWM, 0); ledcWrite(L_LPWM, speedVal);
  ledcWrite(R_LPWM, 0); ledcWrite(R_RPWM, speedVal);
}

void turnRight() {
  // Left motor forward, Right motor reverse
  ledcWrite(L_LPWM, 0); ledcWrite(L_RPWM, speedVal);
  ledcWrite(R_RPWM, 0); ledcWrite(R_LPWM, speedVal);
}

void setup() {
  // Initialize Hardware Serial at 115200 Baud Rate
  Serial.begin(115200);

  // Setup ESP32 PWM Output Channels
  ledcAttach(L_RPWM, freq, resolution);
  ledcAttach(L_LPWM, freq, resolution);
  ledcAttach(R_RPWM, freq, resolution);
  ledcAttach(R_LPWM, freq, resolution);

  stopMotors();

  Serial.println("Serial Controller Ready!");
}

void loop() {
  // Listen for commands via Hardware Serial (USB connection)
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    cmd.toUpperCase();

    if (cmd.length() > 0) {
      Serial.println("Command Received: " + cmd);

      if (cmd == "F" || cmd == "FORWARD") {
        moveForward();
        Serial.println("Status: Moving FORWARD");
      } 
      else if (cmd == "B" || cmd == "BACKWARD") {
        moveBackward();
        Serial.println("Status: Moving BACKWARD");
      } 
      else if (cmd == "L" || cmd == "LEFT") {
        turnLeft();
        Serial.println("Status: Turning LEFT");
      } 
      else if (cmd == "R" || cmd == "RIGHT") {
        turnRight();
        Serial.println("Status: Turning RIGHT");
      } 
      else if (cmd == "S" || cmd == "STOP") {
        stopMotors();
        Serial.println("Status: STOPPED");
      }
    }
  }
}
