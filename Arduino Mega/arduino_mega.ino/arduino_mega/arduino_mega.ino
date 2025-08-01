const int IN1 = 5;
const int IN2 = 4;
const int IN3 = 6;
const int IN4 = 7;

float target[100][2];
int i = 0;

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
    float x, y, omega;
    float target_x, target_y;
    int n = sscanf(msg.c_str(), "%f %f %f", &x, &y, &omega);
    Serial.print("Mega nhận: ");
    Serial.println(msg);
    Serial.print("n = ");
    Serial.println(n);
    if (n == 3) { // gửi vị trí
      Serial.println("n == 3");
    } else if (n == 2) { // thêm vị trí đích
      target[i][0] = x;
      target[i][1] = y;
      Serial.println("Mảng vị trí: ");
      Serial.print("target[");
      Serial.print(i);
      Serial.print("][0] = ");
      Serial.print(target[i][0]);
      Serial.print(", target[");
      Serial.print(i);
      Serial.print("][1] = ");
      Serial.println(target[i][1]);
      i += 1;
    }
    delay(1000);
  }
}
