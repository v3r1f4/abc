#include <painlessMesh.h>
#include <ArduinoJson.h>

#define MESH_PREFIX "ESP32MESH"
#define MESH_PASSWORD "12345678"
#define MESH_PORT 5555

painlessMesh mesh;

const int myID = 3;  // ⚠️ Đổi thành 1, 2, hoặc 3 cho từng con ESP32

void receivedCallback(uint32_t from, String &msg) {
  DynamicJsonDocument doc(1024);
  DeserializationError error = deserializeJson(doc, msg);

  if (!error) {
    int target = doc["target"];
    const char* message = doc["msg"];

    if (target == myID) {
      Serial.printf("👉 Message for me (ID %d): %s\n", myID, message);
    } else {
      Serial.printf("⛔ Not for me (My ID: %d, Target: %d)\n", myID, target);
    }
  } else {
    Serial.println("❌ JSON Parse Failed");
  }
}

void setup() {
  Serial.begin(115200);
  mesh.setDebugMsgTypes(ERROR | STARTUP | CONNECTION);
  mesh.init(MESH_PREFIX, MESH_PASSWORD, MESH_PORT);
  mesh.onReceive(&receivedCallback);
}

void loop() {
  mesh.update();
}
