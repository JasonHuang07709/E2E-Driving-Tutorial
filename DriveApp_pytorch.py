import numpy as np
import cv2
import torch
import torch.nn as nn

class AutopilotNet(nn.Module):
    def __init__(self):
        super(AutopilotNet, self).__init__()
        self.normalize = LambdaLayer()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.normalize(x)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 5 * 5)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LambdaLayer(nn.Module):
    def __init__(self, lambd=lambda x: x / 127.5 - 1.):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

# 加载模型
model = AutopilotNet()
model.load_state_dict(torch.load('Autopilot_test.pth'))
model.eval()

def pytorch_predict(model, image):
    processed = pytorch_process_image(image)
    with torch.no_grad():
        steering_angle = float(model(processed).item())
    steering_angle = steering_angle * 100
    return steering_angle

def pytorch_process_image(img):
    image_x = 40
    image_y = 40
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (1, 1, image_x, image_y))
    img = torch.tensor(img)
    return img

steer = cv2.imread('steering_wheel_image.jpg', 0)
rows, cols = steer.shape
smoothed_angle = 0

cap = cv2.VideoCapture('run.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    #print(frame.shape) # (160, 320, 3) 在图像处理中，图像的形状通常以 (height, width, channels) 表示
    if not ret:
        break
    gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)[:, :, 1], (40, 40))
    print(gray.shape) # (40, 40)
    print(gray)
    steering_angle = pytorch_predict(model, gray)
    print(steering_angle)
    cv2.imshow('frame', cv2.resize(frame, (500, 300), interpolation=cv2.INTER_AREA))
    smoothed_angle += 0.2 * pow(abs(steering_angle - smoothed_angle), 2.0 / 3.0) * (
        steering_angle - smoothed_angle) / abs(steering_angle - smoothed_angle)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
    dst = cv2.warpAffine(steer, M, (cols, rows))
    cv2.imshow("steering wheel", dst)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
