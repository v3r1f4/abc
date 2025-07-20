#include <painlessMesh.h>

#define SSID "ba_con_bon_banh_omni"
#define PASSWORD "24681012"
#define PORT 5555
#define RX2 16
#define TX2 17

uint32_t slaveID = 1;

painlessMesh mesh;

void receivedCallback(uint32_t from, String &msg) {
  int sep = msg.indexOf(' ');
  String targetID = msg.substring(0, sep);
  String command = msg.substring(sep + 1);

  if (targetID.toInt() == slaveID) {
    command.trim();
  
    Serial.printf("Received from %u: %s\n", from, command.c_str());

    Serial2.println(command);
  }
}

void setup() {
  Serial.begin(38400);
  Serial2.begin(38400, RX2, TX2);
  
  // mesh.setDebugMsgTypes(ERROR | STARTUP | CONNECTION);
  mesh.init(SSID, PASSWORD, PORT);
  mesh.onReceive(&receivedCallback);
}

void loop() {
  mesh.update();
}