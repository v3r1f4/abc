#include <painlessMesh.h>

#define MESH_PREFIX     "robot_mesh"
#define MESH_PASSWORD   "12345678"
#define MESH_PORT       5555
#define RXD2 16
#define TXD2 17

Scheduler userScheduler;
painlessMesh mesh;

void receivedCallback(uint32_t from, String &msg) {
  msg.trim();
  
  char buffer[50];
  snprintf(buffer, sizeof(buffer), "%s\n", msg.c_str());
  Serial2.print(buffer);
}

void setup() {
  Serial.begin(115200);
  Serial2.begin(115200, SERIAL_8N1, RXD2, TXD2);
  mesh.setDebugMsgTypes(ERROR | STARTUP | CONNECTION);
  mesh.init(MESH_PREFIX, MESH_PASSWORD, &userScheduler, MESH_PORT);
  mesh.onReceive(&receivedCallback);
}

void loop() {
  mesh.update();
}