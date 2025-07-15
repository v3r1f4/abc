#include <painlessMesh.h>

#define MESH_PREFIX     "robot_mesh"
#define MESH_PASSWORD   "12345678"
#define MESH_PORT       5555

Scheduler userScheduler;
painlessMesh mesh;

// Hàm phân tích lệnh từ Serial Monitor
bool parseCommand(String input, float &a, float &b, float &c) {
  input.trim();
  int firstSpace = input.indexOf(' ');
  int secondSpace = input.indexOf(' ', firstSpace + 1);
  
  if (firstSpace == -1 || secondSpace == -1) {
    return false;
  }
  
  String aStr = input.substring(0, firstSpace);
  String bStr = input.substring(firstSpace + 1, secondSpace);
  String cStr = input.substring(secondSpace + 1);
  
  a = aStr.toFloat();
  b = bStr.toFloat();
  c = cStr.toFloat();
  
  if (aStr.length() == 0 || bStr.length() == 0 || cStr.length() == 0) {
    return false;
  }
  
  return true;
}

void sendCommand(float vx, float vy, float omega) {
  char buffer[50];
  snprintf(buffer, sizeof(buffer), "%.1f %.1f %.1f", vx, vy, omega);
  mesh.sendBroadcast(String(buffer));
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
    input.trim();
    float vx, vy, omega;
    if (parseCommand(input, vx, vy, omega)) {
      sendCommand(vx, vy, omega);
    } else {
      Serial.println("Lệnh nhập không đúng định dạng!");
    }
  }
}