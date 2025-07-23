#include <painlessMesh.h>

#define MESH_PREFIX     "robot_mesh"
#define MESH_PASSWORD   "12345678"
#define MESH_PORT       5555

#define RXD2 16
#define TXD2 17
#define NODE_ID 1  

Scheduler userScheduler;
painlessMesh mesh;

void receivedCallback(uint32_t from, String &msg) {
  msg.trim();

  // Tách phần đầu là ID
  int spaceIndex = msg.indexOf(' ');
  if (spaceIndex == -1) return;

  String targetID = msg.substring(0, spaceIndex);
  String payload = msg.substring(spaceIndex + 1);

  // Kiểm tra ID khớp
  if (targetID != "all" && targetID.toInt() != NODE_ID) {
    return;  
  }

  // Gửi payload (vx vy omega) ra Serial2
  Serial2.print(payload + "\n");
  Serial.printf("NODE %d nhận: %s\n", NODE_ID, payload.c_str());
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
