const int IN1 = 5;
const int IN2 = 4;
const int IN3 = 6;
const int IN4 = 7;

void dk_dc(int pwmL, int pwmR) {
  analogWrite(IN1, pwmL > 0 ? pwmL : 0);
  analogWrite(IN2, pwmL < 0 ? -pwmL : 0);
  analogWrite(IN3, pwmR > 0 ? pwmR : 0);
  analogWrite(IN4, pwmR < 0 ? -pwmR : 0);
}

void setup() {
  Serial.begin(115200);
  Serial1.begin(115200);  // RX1 = pin 19
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
}

void loop() {
  if (Serial1.available()) {
    String msg = Serial1.readStringUntil('\n');
    msg.trim();
    Serial.print("Mega nhận: "); Serial.println(msg);
    int id;
    int vx, vy, omega;
    int n = sscanf(msg.c_str(), "%d %d %d %d", &id, &vx, &vy, &omega);
    if (n == 4 && id == 1) {
      dk_dc(vx, vy);  // với robot 2 bánh thì vx=trái, vy=phải
    }
  }
}
