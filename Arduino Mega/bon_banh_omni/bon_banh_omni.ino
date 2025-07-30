void setup() {
  Serial.begin(38400);
  Serial1.begin(38400);
}

void loop() {
  if (Serial1.available()) {
    char c = Serial1.read();
    Serial.print(c); // In ra monitor
  } else {
    Serial.println("Serial1's not available");
  }
}