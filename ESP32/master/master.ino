#include <painlessMesh.h>

#define SSID "ba_con_bon_banh_omni"
#define PASSWORD "24681012"
#define PORT 5555

painlessMesh mesh;

void sendCommand(String input) {
  int firstSpace = input.indexOf(' ');
  int secondSpace = input.indexOf(' ', firstSpace + 1);
  int thirdSpace = input.indexOf(' ', secondSpace + 1);

  int slaveID = input.substring(0, firstSpace).toInt();
  float vx = input.substring(firstSpace + 1, secondSpace).toFloat();
  float vy = input.substring(secondSpace + 1, thirdSpace).toFloat();
  float omega = input.substring(thirdSpace + 1).toFloat();

  char buffer[50];
  snprintf(buffer, sizeof(buffer), "%u %.1f %.1f %.1f", slaveID, vx, vy, omega);
  
  mesh.sendBroadcast(String(buffer));
  Serial.printf("Sent: %s\n", buffer);
}

void setup() {
  Serial.begin(38400);
  // mesh.setDebugMsgTypes(ERROR | STARTUP | CONNECTION);
  mesh.init(SSID, PASSWORD, PORT);
}

void loop() {
  mesh.update();

  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    
    sendCommand(input);
  }
}