from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QPixmap
from ui import Ui_MainWindow

import sys
import os
import numpy
import cv2
import matplotlib.pyplot as plt

import torch
import torchsummary
import torchvision.models

from torchvision.transforms import transforms

from PIL import Image

from VGG19 import VGG19

class main(QMainWindow):
    def __init__(self): # 初始化UI視窗
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # click button event
        self.ui.loadButton1.clicked.connect(self.load_image)
        self.ui.Button1_1.clicked.connect(self.color_separation)
        self.ui.Button1_2.clicked.connect(self.color_transform)
        self.ui.Button1_3.clicked.connect(self.color_extraction)
        self.ui.Button1_4.clicked.connect(self.gaussian_blur)
        self.ui.Button1_5.clicked.connect(self.bilateral_filter)
        self.ui.Button1_6.clicked.connect(self.median_filter)
        self.ui.Button1_7.clicked.connect(self.sobelX)
        self.ui.Button1_8.clicked.connect(self.sobelY)
        self.ui.Button1_9.clicked.connect(self.combAndThreshold)
        self.ui.Button1_10.clicked.connect(self.gradient_angle)
        self.ui.Button1_11.clicked.connect(self.transforms)
        self.ui.Button1_12.clicked.connect(self.load_image_inference)
        self.ui.Button1_13.clicked.connect(self.show_argumented_image)
        self.ui.Button1_14.clicked.connect(self.show_structure)
        self.ui.Button1_15.clicked.connect(self.show_AccLoss) 
        self.ui.Button1_16.clicked.connect(self.inference)

    def load_image(self): # 載入圖片
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        global file_name
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image File', '', 'Images (*.png *.xpm *.jpg *.bmp);;All Files (*)', options=options) # 回傳為圖片路徑
        if file_name:
            self.ui.loadLabel1.setText(os.path.basename(file_name)) # argv為 file_name 的文件名

    def color_separation(self):
        global blue_image, green_image, red_image
        image = cv2.imread(file_name)

        b, g, r = cv2.split(image) # 分離顏色channel
        zeros = numpy.zeros_like(b) # zeros為全零channel(形狀像r, g, b)
        blue_image = cv2.merge([b, zeros, zeros])
        green_image = cv2.merge([zeros, g, zeros])
        red_image = cv2.merge([zeros, zeros, r])
        
        cv2.imshow('Blue', blue_image) # 標題為Blue
        cv2.imshow('Green', green_image) # 標題為Green
        cv2.imshow('Red', red_image) # 標題為Red
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def color_transform(self):
        image = cv2.imread(file_name)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # RGB to Grayscale
        b, g, r = cv2.split(image) # 分離顏色channel
        average_gray_image = (r + b + g) / 3 # 取平均值
        average_gray_image = average_gray_image.astype(numpy.uint8) # unsigned 8 bit integer, 確保在0-255之間, 無小數

        cv2.imshow('Gray', gray_image)
        cv2.imshow('Average Gray', average_gray_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def color_extraction(self):
        image = cv2.imread(file_name)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # BGR to HSV

        lower_bound = numpy.array([18, 25, 25]) # 設定[色相, 飽和度, 亮度] / [Hue, Saturation, Value] 的下限
        upper_bound = numpy.array([85, 255, 255])

        mask_i1 = cv2.inRange(hsv_image, lower_bound, upper_bound) # extraction yellow and green(HSV)
        i1_bgr = cv2.cvtColor(mask_i1, cv2.COLOR_GRAY2BGR) # HSV to BGR
        i2_bgr = cv2.bitwise_not(i1_bgr, image, mask_i1) # (I1 in BGR, image, I1 in HSV)

        cv2.imshow('I1', i1_bgr)
        cv2.imshow('I2', i2_bgr)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def gaussian_blur(self): # 平滑圖像去除噪聲和細節
        global filter_type # filter_type (gaussian_blur or bliateral_filter or median_filter)
        filter_type = 0

        image = cv2.imread(file_name)
        cv2.namedWindow('Gaussian Blur', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Gaussian Blur', 800, 600)
        cv2.imshow('Gaussian Blur', image)
        cv2.createTrackbar('magnitude', 'Gaussian Blur', 0, 5, self.update_magnitude) # 設定trackbar由0到5

    def bilateral_filter(self): # 平滑圖像同時保留紋理和邊緣
        global filter_type
        filter_type = 1

        image = cv2.imread(file_name)
        cv2.namedWindow('Bilateral filter', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Bilateral filter', 800, 600)
        cv2.imshow('Bilateral filter', image)
        cv2.createTrackbar('magnitude', 'Bilateral filter', 0, 5, self.update_magnitude)

    def median_filter(self): # 利用平均值去除強烈噪聲
        global filter_type
        filter_type = 2

        image = cv2.imread(file_name)
        cv2.namedWindow('Median filter', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Median filter', 800, 600)
        cv2.imshow('Median filter', image)
        cv2.createTrackbar('magnitude', 'Median filter', 0, 5, self.update_magnitude)

    def update_magnitude(self, value): # 動態調整trackbar的value
        m = value
        image = cv2.imread(file_name)
        kernel_size = (2 * m + 1, 2 * m + 1)
        kernel_int = (2 * m + 1) * (2 * m + 1)

        # check filter type
        if filter_type == 0:
            blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
            cv2.imshow('Gaussian Blur', blurred_image)
        elif filter_type == 1:
            bliateral_image = cv2.bilateralFilter(image, kernel_int, sigmaColor = 90, sigmaSpace = 90)
            cv2.imshow('Bilateral filter', bliateral_image)
        else:
            median_image = cv2.medianBlur(image, kernel_int)
            cv2.imshow('Median filter', median_image)

    def sobelX(self): # 偵測垂直邊緣
        image = cv2.imread(file_name)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # RGB to Grayscale
        smoothed_image = cv2.GaussianBlur(gray_image, (3, 3), 0) # 使用 Gaussian filter 使Grayscale平滑
        sobel_x = numpy.array([[-1, 0, 1], # 自定義 sobel x filter
                               [-2, 0, 2],
                               [-1, 0, 1]])
        
        rows, cols = smoothed_image.shape # 獲取行數和列數
        global sobelX_image, sobelX_for_gradient
        sobelX_image = numpy.zeros((rows, cols), dtype=numpy.uint8) # 創建大小相同的空白圖像
        sobelX_for_gradient = numpy.zeros((rows, cols), dtype=numpy.int16) # 創建大小相同的空白圖像

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                gx = numpy.sum(smoothed_image[i - 1:i + 2, j - 1:j + 2] * sobel_x) # convolution, 相乘後求和
                sobelX_for_gradient[i, j] = gx # 保留原本負值去做gradient
                sobelX_image[i, j] = numpy.abs(gx) # 將負值轉成正值

        cv2.imshow('Sobel X', sobelX_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def sobelY(self): # 偵測水平邊緣
        image = cv2.imread(file_name)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # RGB to Grayscale
        smoothed_image = cv2.GaussianBlur(gray_image, (3, 3), 0) # 使用 Gaussian filter 使Grayscale平滑
        sobel_y = numpy.array([[-1, -2, -1], # 自定義 sobel x filter
                              [0, 0, 0],
                              [1, 2, 1]])
        
        rows, cols = smoothed_image.shape # 獲取行數和列數
        global sobelY_image, sobelY_for_gradient
        sobelY_image = numpy.zeros((rows, cols), dtype=numpy.uint8) # 創建大小相同的空白圖像
        sobelY_for_gradient = numpy.zeros((rows, cols), dtype=numpy.int16) # 創建大小相同的空白圖像

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                gx = numpy.sum(smoothed_image[i - 1:i + 2, j - 1:j + 2] * sobel_y) # convolution, 相乘後求和
                sobelY_for_gradient[i, j] = gx # 保留原本負值去做gradient
                sobelY_image[i, j] = numpy.abs(gx) # 將負值轉成正值

        cv2.imshow('Sobel Y', sobelY_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def combAndThreshold(self):
        comb_image = numpy.sqrt(sobelX_image.astype(numpy.uint16) ** 2 + sobelY_image.astype(numpy.uint16) ** 2) # 平和相加後開根號
        global normalized_image
        normalized_image = cv2.normalize(comb_image, None, 0, 255, cv2.NORM_MINMAX) # 確保值在0-255
        threshold = 128
        threshold_image = numpy.where(normalized_image < threshold, 0, 255).astype(numpy.uint8) # 小於threshold就改為0, 反之改為255

        cv2.imshow('Combination', normalized_image.astype(numpy.uint8))
        cv2.imshow('Threshold', threshold_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def gradient_angle(self):
        gradient_angle = numpy.arctan2(sobelY_for_gradient, sobelX_for_gradient) * (180 / numpy.pi) # tan值換算成角度
        adjust_angle = numpy.mod(gradient_angle + 360, 360) # ex. 將 -150 變成 210
        mask1 = ((adjust_angle >= 120) & (adjust_angle <= 180)).astype(numpy.uint8) * 255 # 範圍內的值設定為255, 否則設為0
        mask2 = ((adjust_angle >= 210) & (adjust_angle <= 330)).astype(numpy.uint8) * 255

        result1 = cv2.bitwise_and(normalized_image.astype(numpy.uint8), mask1)
        result2 = cv2.bitwise_and(normalized_image.astype(numpy.uint8), mask2)

        cv2.imshow('Result1', result1)
        cv2.imshow('Result2', result2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def transforms(self):
        image = cv2.imread(file_name)

        if self.ui.lineEdit.text() == "":
            angle = 0 # 預設值為 0
        else:
            angle = float(self.ui.lineEdit.text()) # 讀入 linetext 上的文字
        if self.ui.lineEdit_2.text() == "":
            scale = 1
        else:
            scale = float(self.ui.lineEdit_2.text())
        if self.ui.lineEdit_3.text() == "":
            tx = 0
        else:
            tx = float(self.ui.lineEdit_3.text())   
        if self.ui.lineEdit_4.text() == "":
            ty = 0
        else:
            ty = float(self.ui.lineEdit_4.text())

        rotation_matrix = cv2.getRotationMatrix2D((240, 200), angle, scale) # 中心為 (240, 200)
        translation_matrix = numpy.float32([[1, 0, tx],
                                            [0, 1, ty]])
        
        transformed_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0])) # 旋轉和縮放
        transformed_image = cv2.warpAffine(transformed_image, translation_matrix, (image.shape[1], image.shape[0])) # 平移

        cv2.imshow('Transforms', transformed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_argumented_image(self):
        # data augmentation
        data_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=30)
        ])
        
        folder_path = '.\\Dataset_OpenCvDl_Hw1\\Dataset_OpenCvDl_Hw1\\Q5_image\\Q5_1'
        image_files = [file for file in os.listdir(folder_path) if file.endswith(('.jpg', '.png'))] # 找到folder_path內部結尾有.jpg或.png的檔案

        plt.figure(figsize=(12, 8)) # 設定視窗大小
        plt.ion() # 即時更新 (交互式模式)
        plt.clf() # 清除圖形內容

        for i, image_file in enumerate(image_files[:9]):
            image_path = os.path.join(folder_path, image_file)
            image = Image.open(image_path) # 使用PIL function路徑上的圖片
            augmented_image = data_transforms(image) # 將圖片做augmentation

            plt.subplot(3, 3, i + 1)
            plt.title(image_file.replace(".png", ""))
            plt.imshow(augmented_image)

        plt.show()
        plt.ioff() # 關閉交互式模式

    def show_structure(self):
        model = torchvision.models.vgg19_bn(num_classes=10)
        torchsummary.summary(model, (3, 32, 32))

    def show_AccLoss(self):
        image_acc_path = '.\\VGG19 Acc.png'
        image_loss_path = '.\\VGG19 Loss.png'

        image_acc = cv2.imread(image_acc_path)
        image_loss = cv2.imread(image_loss_path)

        cv2.imshow('Acc', image_acc)
        cv2.imshow('Loss', image_loss)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def load_image_inference(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        global inference_file_name
        inference_file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image File', '', 'Images (*.png *.xpm *.jpg *.bmp);;All Files (*)', options=options) # 回傳為圖片路徑

        pixmap = QPixmap(inference_file_name).scaled(128, 128)
        self.ui.image_label.setPixmap(pixmap)

    def inference(self):
        model = VGG19(num_classes = 10)
        model.load_state_dict(torch.load("Model Weights.pth", map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
        model.eval()

        # Load and preprocess the image
        image = Image.open(inference_file_name).convert("RGB")
        transform = transforms.Compose([transforms.Resize((32, 32)),
                                        transforms.ToTensor()])
        
        image = transform(image).unsqueeze(0)

        # Make a prediction using the model
        with torch.no_grad():
            outputs, _ = model(image)
            predictions = torch.softmax(outputs, dim = 1).numpy()[0]

        class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.ui.label5_1.setText('Predict = ' + class_labels[numpy.argmax(predictions)])

        # Create a histogram for the predictions
        plt.figure()
        plt.bar(class_labels, predictions)
        plt.xticks(range(len(predictions)))
        plt.xlabel('Class')
        plt.ylabel("Probability")
        plt.title("Prediction Probability Distribution")
        plt.show()
            

app = QApplication([])
window = main()
window.show()
sys.exit(app.exec_())
