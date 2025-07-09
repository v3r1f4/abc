#include <painlessMesh.h>
#include <ArduinoJson.h>

#define MESH_PREFIX "ESP32MESH"
#define MESH_PASSWORD "12345678"
#define MESH_PORT 5555

painlessMesh mesh;

void setup() {
  Serial.begin(115200);
  mesh.setDebugMsgTypes(ERROR | STARTUP | CONNECTION);
  mesh.init(MESH_PREFIX, MESH_PASSWORD, MESH_PORT);
}

void loop() {
  mesh.update();

  // Gửi cho ID 1
  {
    DynamicJsonDocument doc(1024);
    doc["target"] = 1;
    doc["msg"] = "Hello Node 1!";
    String msg;
    serializeJson(doc, msg);
    mesh.sendBroadcast(msg);
    Serial.println("✅ Sent to ID1: " + msg);
    delay(1000);
  }

  // Gửi cho ID 2
  {
    DynamicJsonDocument doc(1024);
    doc["target"] = 2;
    doc["msg"] = "Hello Node 2!";
    String msg;
    serializeJson(doc, msg);
    mesh.sendBroadcast(msg);
    Serial.println("✅ Sent to ID2: " + msg);
    delay(1000);
  }

  // Gửi cho ID 3
  {
    DynamicJsonDocument doc(1024);
    doc["target"] = 3;
    doc["msg"] = "Hello Node 3!";
    String msg;
    serializeJson(doc, msg);
    mesh.sendBroadcast(msg);
    Serial.println("✅ Sent to ID3: " + msg);
    delay(1000);
  }
}
