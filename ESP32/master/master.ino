#include <esp_now.h>
#include <WiFi.h>

char c;
char str[255];
uint8_t idx = 0;

uint8_t mac1[] = {0x14, 0x33, 0x5C, 0x04, 0x61, 0x18};  // MAC của slave 1
uint8_t mac2[] = {0x1C, 0x69, 0x20, 0xA4, 0xD0, 0x58};  // MAC của slave 2
uint8_t mac3[] = {0x94, 0xB9, 0x7E, 0xFB, 0x23, 0xF0};  // MAC của slave 3

void sendTo(uint8_t *mac, const String &msg) {
  esp_now_send(mac, (const uint8_t *)msg.c_str(), msg.length());
}

void setup() {
  Serial.begin(115200);
  WiFi.mode(WIFI_STA);

  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW init thất bại");
    while (1);
  }

  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, mac1, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;
  esp_now_add_peer(&peerInfo);

  Serial.println("ESP32 Master sẵn sàng...");
}

void loop() {
  if (Serial.available()) {
    String msg = Serial.readStringUntil('\n');
    msg.trim();
    
    if (msg.length() > 0) {
      sendTo(mac1, msg);
      Serial.println("Đã gửi: " + msg);
    }
  }
}
