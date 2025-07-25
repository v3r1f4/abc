#include <painlessMesh.h>

#define MESH_PREFIX     "robot_mesh"
#define MESH_PASSWORD   "12345678"
#define MESH_PORT       5555

Scheduler userScheduler;
painlessMesh mesh;

// Hàm gửi lệnh
void sendCommand(String targetId, float vx, float vy, float omega) {
  char buffer[64];
  snprintf(buffer, sizeof(buffer), "%s %.2f %.2f %.2f", targetId.c_str(), vx, vy, omega);
  mesh.sendBroadcast(String(buffer));
}

// Hàm phân tích lệnh
bool parseCommand(String input, String &id, float &vx, float &vy, float &omega) {
  input.trim();
  int firstSpace = input.indexOf(' ');
  int secondSpace = input.indexOf(' ', firstSpace + 1);
  int thirdSpace = input.indexOf(' ', secondSpace + 1);

  if (firstSpace == -1 || secondSpace == -1 || thirdSpace == -1) return false;

  id = input.substring(0, firstSpace);
  String vxStr = input.substring(firstSpace + 1, secondSpace);
  String vyStr = input.substring(secondSpace + 1, thirdSpace);
  String omegaStr = input.substring(thirdSpace + 1);

  vx = vxStr.toFloat();
  vy = vyStr.toFloat();
  omega = omegaStr.toFloat();

  return true;
}

void setup() {
  Serial.begin(115200);
  mesh.setDebugMsgTypes(ERROR | STARTUP | CONNECTION);
  mesh.init(MESH_PREFIX, MESH_PASSWORD, &userScheduler, MESH_PORT);
}

void loop() {
  mesh.update();

  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    String id;
    float vx, vy, omega;
    if (parseCommand(input, id, vx, vy, omega)) {
      sendCommand(id, vx, vy, omega);
    } else {
      Serial.println("Lệnh sai định dạng! Định dạng đúng: ID vx vy omega");
    }
  }
}
