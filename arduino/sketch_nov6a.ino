
#define RPWM 5
#define LPWM 6
#define REN  7
#define LEN  8

int motorSpeed = 150;

void setup() {
  Serial.begin(9600);

  pinMode(RPWM, OUTPUT);
  pinMode(LPWM, OUTPUT);
  pinMode(REN, OUTPUT);
  pinMode(LEN, OUTPUT);

  digitalWrite(REN, HIGH);
  digitalWrite(LEN, HIGH);

  stopMotor();

  Serial.println("BTS7960 Motor Control Ready");
  Serial.println("W = Forward");
  Serial.println("S = Reverse");
  Serial.println("X = Stop");
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();

    switch (command) {
      case 'S':
      case 's':
        forward();
        break;

      case 'W':
      case 'w':
        backward();
        break;

      case 'X':
      case 'x':
        stopMotor();
        break;
    }
  }
}

void forward() {
  analogWrite(LPWM, 0);
  analogWrite(RPWM, motorSpeed);
}

void backward() {
  analogWrite(RPWM, 0);
  analogWrite(LPWM, motorSpeed);
}

void stopMotor() {
  analogWrite(RPWM, 0);
  analogWrite(LPWM, 0);
}
