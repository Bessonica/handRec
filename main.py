# import
import os
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow import keras

# создаем свой датасет

# адрес фото
dir_train = "D:/programming/deeplearning/dataset/hands/train"
dir_test = "D:/programming/deeplearning/dataset/hands/test"
# можно лиD:/programming/deeplearning/dataset/hands/train/*.png???7!!!
# что такоке glob?
labels = ["0L", "1L", "2L", "3L", "4L", "5L", "0R", "1R", "2R", "3R", "4R", "5R", ]

# создаем массивы изображений и ярлыков,которые обозначают руку и кол пальцев


arr_train = []
arr_test = []

# listdir дает имена файлов  img_train = str_train
# не забывай что у имени в конце есть еще .png. -1 = идем справо налево

for str_train in os.listdir(dir_train):
    str_label = str_train[-5: -7: -1][::-1]
    label = labels.index(str_label)
    img = cv2.imread(os.path.join(dir_train, str_train), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    arr_train.append([img, label])
    print(len(arr_train))

# arr_test
for str_test in os.listdir(dir_test):
    str_label = str_test[-5: -7: -1][::-1]
    label = labels.index(str_label)
    img = cv2.imread(os.path.join(dir_test, str_test), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    arr_test.append([img, label])
    print(len(arr_test))

# reshape и сделать черно белім изображение

# отделяем массивы фото от меток

arr_train_img = []
arr_train_label = []

arr_test_img = []
arr_test_label = []

for img_train, label_train in arr_train:
    arr_train_img.append(img_train)
    arr_train_label.append(label_train)

for img_test, label_test in arr_test:
    arr_test_img.append(img_test)
    arr_test_label.append(label_test)

arr_train_img = np.array(arr_train_img).reshape(-1, 128, 128, 1)

# arr_test_img это лист (list) из массивов,в каждом массиве 128 масивов с 128 элементов????


arr_test_img = np.array(arr_test_img).reshape(-1, 128, 128, 1)
# print(arr_test_img[0].shape)#128,128 до ришейпа.   после (128, 128, 1)
# осле ришейпа 1 изображение  это массив из масивов в которых числа(пиксели)в отдельных
# массивах  [ [[15]...(128 элеиентов)[15] ](строка 1) ... [[0]...(128 элеиентов)[0]](строка 128)  ]

arr_test_img = arr_test_img / 255.0
arr_train_img = arr_train_img / 255.0

arr_train_label = np.array(arr_train_label)
arr_test_label = np.array(arr_test_label)

# проверка
# print("test image shape", arr_test_img.shape) #3600, 128, 128, 1
# print("train image shape", arr_train_img.shape)#18000, 128, 128, 1

#загружаем модель
model = keras.models.load_model("saved_model_CNN")


test_loss, test_acc = model.evaluate(arr_test_img, arr_test_label)
print("test acc", test_acc)  # 0.99

# 21.12.2020 выводы
# CNN готова но я не доволен ее скоростью,почитать способы оптимизации
# все еще нет поддержки веб камеры,попытайся не использовать object detection api от тенсерфлоу,а просто сделать все используя только cv2
# ОТДЕЛИ МОДЕЛЬ ОТ ОСНОВНОЙ ЧАСТИ КОДА,что бы не создавать и учить модель каждый раз когда запускаешь код
# ПОСТАВЬ ГИТ В pycharm

