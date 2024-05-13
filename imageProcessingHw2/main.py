from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QPixmap
from ui import Ui_MainWindow

import sys
import os
import random

import numpy
import cv2
import matplotlib.pyplot as plt

import torch
import torchsummary
import torchvision.models

from torchvision.transforms import transforms

from PIL import Image

from VGG19 import VGG19
from ResNet50 import ResNet50

class main(QMainWindow):
    def __init__(self): # 初始化UI視窗
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # click button event
        self.ui.loadButton1.clicked.connect(self.load_image)
        self.ui.Button1_1.clicked.connect(self.draw_contour)
        self.ui.Button1_2.clicked.connect(self.count_coins)
        self.ui.Button1_3.clicked.connect(self.histogram_equalization)
        self.ui.Button1_4.clicked.connect(self.morphology_closing)
        self.ui.Button1_5.clicked.connect(self.morphology_opening)
        self.ui.Button1_6.clicked.connect(self.structure_VGG19)
        self.ui.Button1_7.clicked.connect(self.showAccLoss)
        self.ui.Button1_8.clicked.connect(self.predict)
        self.ui.Button1_9.clicked.connect(self.reset)
        self.ui.Button1_10.clicked.connect(self.load_CatDog)
        self.ui.Button1_11.clicked.connect(self.show_image)
        self.ui.Button1_12.clicked.connect(self.structure_ResNet50)
        self.ui.Button1_13.clicked.connect(self.show_Comparison)
        self.ui.Button1_14.clicked.connect(self.inference)

    def load_image(self): # 載入圖片
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        global file_name
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image File', '', 'Images (*.png *.xpm *.jpg *.bmp);;All Files (*)', options=options) # 回傳為圖片路徑
        if file_name:
            self.ui.loadLabel1.setText(os.path.basename(file_name)) # argv為 file_name 的文件名
    
    def draw_contour(self):
        image = cv2.imread(file_name)

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply GaussianBlur to remove noise
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        _, thresh = cv2.threshold(blurred_image, 90, 255, cv2.THRESH_BINARY)

        # Detect circles using HoughCircles
        circles = cv2.HoughCircles(
            blurred_image,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=50
        )

        if circles is not None:
            circles = numpy.uint16(numpy.around(circles))

            processed_image = image.copy()
            centers_image = numpy.zeros_like(gray_image)

            for i in circles[0, :]:
                cv2.circle(processed_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(centers_image, (i[0], i[1]), 2, 255, 3)

            # Display the original image, processed image, and circle center image
            plt.figure(figsize = (12, 4))

            plt.subplot(131)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title('Original Image')

            plt.subplot(132)
            plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
            plt.title('Processed Image')

            plt.subplot(133)
            plt.imshow(centers_image, cmap = 'gray')
            plt.title('Circle Centers')

            global circleCount
            circleCount = len(circles[0])

            plt.show()
        else:
            print("No circles found in the image.")

    def count_coins(self):
        self.ui.count_coin.setText("There are " + str(circleCount) + " coins in the image")

    def histogram_equalization(self):
        image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

        equalized_image = cv2.equalizeHist(image)

        plt.figure(figsize = (15, 10))

        plt.subplot(2, 3, 1)
        plt.imshow(image, cmap = 'gray')
        plt.title('Original Image')

        plt.subplot(2, 3, 2)
        plt.imshow(equalized_image, cmap = 'gray')
        plt.title('Equalized with OpenCV')

        plt.subplot(2, 3, 4)
        plt.hist(image.flatten(), 300, [0, 256], color = 'b')
        plt.title('Histogram of Original Image')

        plt.subplot(2, 3, 5)
        plt.hist(equalized_image.flatten(), 300, [0, 256], color = 'b')
        plt.title('Histogram of Equalized Image (OpenCV)')

        plt.show()

        hist, bins = numpy.histogram(image.flatten(), 256, [0, 256])

        pdf = hist / sum(hist)

        cdf = numpy.cumsum(pdf)

        lookup_table = numpy.interp(image.flatten(), bins[:-1], numpy.round(cdf * 255))

        equalized_image_manual = lookup_table.reshape(image.shape).astype('uint8')

        plt.subplot(2, 3, 3)
        plt.imshow(equalized_image_manual, cmap ='gray')
        plt.title('Equalized Manually')
        
        plt.subplot(2, 3, 6)
        plt.hist(equalized_image_manual.flatten(), 300, [0, 256], color = 'b')
        plt.title('Histogram of Equalized (Manual)')

        plt.show()

    def morphology_closing(self):
        image = cv2.imread(file_name)

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        threshold = 127
        _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)

        kernel_size = 3
        padded_image = numpy.pad(binary_image, pad_width = kernel_size // 2, mode = 'constant', constant_values = 0)

        structuring_element = numpy.ones((kernel_size, kernel_size), dtype = numpy.uint8)
        
        # Dilation
        dilated_image = numpy.zeros_like(binary_image)
        for i in range(binary_image.shape[0]):
            for j in range(binary_image.shape[1]):
                if numpy.any(padded_image[i:i + kernel_size, j:j + kernel_size] & structuring_element):
                    dilated_image[i, j] = 255

        # Erosion
        eroded_image = numpy.copy(dilated_image)
        padded_image = numpy.pad(dilated_image, pad_width = kernel_size // 2, mode = 'constant', constant_values = 0)
        for i in range(binary_image.shape[0]):
            for j in range(binary_image.shape[1]):
                if numpy.all(padded_image[i:i + kernel_size, j:j + kernel_size] & structuring_element == structuring_element):
                    eroded_image[i, j] = 255

        plt.figure(figsize = (12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(binary_image, cmap = 'gray')
        plt.title('Binary Image')

        plt.subplot(1, 2, 2)
        plt.imshow(eroded_image, cmap = 'gray')
        plt.title('Closing Image')

        plt.show()

    def morphology_opening(self):
        image = cv2.imread(file_name)

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        threshold = 127
        _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)

        kernel_size = 3
        padded_image = numpy.pad(binary_image, pad_width = kernel_size // 2, mode = 'constant', constant_values = 0)

        structuring_element = numpy.ones((kernel_size, kernel_size), dtype = numpy.uint8)
        
        # Erosion
        eroded_image = numpy.zeros_like(binary_image)
        for i in range(binary_image.shape[0]):
            for j in range(binary_image.shape[1]):
                if numpy.all(padded_image[i:i + kernel_size, j:j + kernel_size] & structuring_element == structuring_element):
                    eroded_image[i, j] = 255
        
        # Dilation
        dilated_image = numpy.copy(eroded_image)
        padded_image = numpy.pad(eroded_image, pad_width = kernel_size // 2, mode = 'constant', constant_values = 0)
        for i in range(binary_image.shape[0]):
            for j in range(binary_image.shape[1]):
                if numpy.any(padded_image[i:i + kernel_size, j:j + kernel_size] & structuring_element):
                    dilated_image[i, j] = 255

        plt.figure(figsize = (12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(binary_image, cmap = 'gray')
        plt.title('Binary Image')

        plt.subplot(1, 2, 2)
        plt.imshow(dilated_image, cmap = 'gray')
        plt.title('Opening Image')

        plt.show()

    def structure_VGG19(self):
        model = torchvision.models.vgg19_bn(num_classes = 10)
        torchsummary.summary(model.cuda(), (3, 32, 32))

    def showAccLoss(self):
        image_acc_path = '.\\VGG19 Acc.png'
        image_loss_path = '.\\VGG19 Loss.png'

        image_acc = cv2.imread(image_acc_path)
        image_loss = cv2.imread(image_loss_path)

        cv2.imshow('Acc', image_acc)
        cv2.imshow('Loss', image_loss)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def predict(self):
        pixmap = self.ui.drawingWidget.grab()
        image = pixmap.toImage()

        image = image.convertToFormat(4)

        width = image.width()
        height = image.height()

        ptr = image.bits()
        ptr.setsize(image.byteCount())

        arr = numpy.array(ptr).reshape(height, width, 4)

        model = VGG19(num_classes = 10)
        model.load_state_dict(torch.load("VGG19_Model Weights.pth", map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
        model.eval()

        RGB_image = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        predict_image = Image.fromarray(RGB_image.astype('uint8'))

        transform = transforms.Compose([transforms.Resize((32, 32)),
                                        transforms.ToTensor()])
        
        predict_image = transform(predict_image).unsqueeze(0)

        # Make a prediction using the model
        with torch.no_grad():
            outputs, _ = model(predict_image)
            predictions = torch.softmax(outputs, dim = 1).numpy()[0]
            
        class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        # Find the index of the maximum prediction
        max_index = numpy.argmax(predictions)

        # Create a one-hot encoded array
        one_hot_prediction = numpy.zeros_like(predictions)
        one_hot_prediction[max_index] = 1

        # Create a histogram for the predictions
        plt.figure()
        plt.bar(class_labels, one_hot_prediction)
        plt.xticks(range(len(predictions)))
        plt.xlabel('Class')
        plt.ylabel("Probability")
        plt.title("Prediction Probability Distribution")
        plt.show()
        
    def reset(self):
        self.ui.drawingWidget.clearDrawing()

    def load_CatDog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        global inference_file_name
        inference_file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image File', '', 'Images (*.png *.xpm *.jpg *.bmp);;All Files (*)', options=options) # 回傳為圖片路徑

        pixmap = QPixmap(inference_file_name).scaled(300, 300)
        self.ui.image_label.setPixmap(pixmap)

    def show_image(self):
        image_dog_path = './inference_dataset/Dog/'

        all_dog_images = os.listdir(image_dog_path)
        dog_file = [file for file in all_dog_images if file.lower().endswith(('.jpg', '.jpeg', '.png'))]

        random_dog_file = random.choice(dog_file)
        random_dog_path = os.path.join(image_dog_path, random_dog_file)
        dog_image = cv2.imread(random_dog_path)
        dog_image = cv2.cvtColor(dog_image, cv2.COLOR_BGR2RGB)
        dog_image = cv2.resize(dog_image, (224, 224))

        image_cat_path = './inference_dataset/Cat/'

        all_cat_images = os.listdir(image_cat_path)
        cat_file = [file for file in all_cat_images if file.lower().endswith(('.jpg', '.jpeg', '.png'))]

        random_cat_file = random.choice(cat_file)
        random_cat_path = os.path.join(image_cat_path, random_cat_file)
        cat_image = cv2.imread(random_cat_path)
        cat_image = cv2.cvtColor(cat_image, cv2.COLOR_BGR2RGB)
        cat_image = cv2.resize(cat_image, (224, 224))

        plt.figure(figsize = (10, 8))

        plt.subplot(1, 2, 1)
        plt.imshow(dog_image)
        plt.title("Dog")

        plt.subplot(1, 2, 2)
        plt.imshow(cat_image)
        plt.title("Cat")

        plt.show()

    def structure_ResNet50(self):
        model = ResNet50()
        torchsummary.summary(model.cuda(), (3, 224, 224))     

    def show_Comparison(self):
        image_path = '.\\Show Comparison.png'

        image = cv2.imread(image_path)

        cv2.imshow('Comparison', image)

    def inference(self):
        model = ResNet50()
        model.load_state_dict(torch.load("ResNet50_Model Weights.pth", map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
        model.eval()

        image = Image.open(inference_file_name).convert("RGB")
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels = 3),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0, 0, 0),(1, 1, 1)),
        ])

        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image)
            prediction = outputs.item()

        thresh = 0.5

        if(prediction >= thresh):
            self.ui.label5_1.setText("Predict = Cat")
        else:
            self.ui.label5_1.setText("Predict = Dog")


app = QApplication([])
window = main()
window.show()
sys.exit(app.exec_())
