#import
import os
import cv2
import numpy
#создаем свой датасет

#адрес фото
dir_train = "D:/programming/deeplearning/dataset/hands/train"
dir_test = "D:/programming/deeplearning/dataset/hands/test"

labels = ["0L", "1L", "2L", "3L", "4L", "5L", "0R", "1R", "2R", "3R", "4R", "5R", ]

#создаем массивы изображений и ярлыков,которые обозначают руку и кол пальцев


arr_train = []
arr_test = []

#listdir дает имена файлов  img_train = str_train
#не забывай что у имени в конце есть еще .png. -1 = идем справо налево
for str_train in os.listdir(dir_train):
    str_label = str_train[-5: -7: -1][::-1]
    label = labels.index(str_label)
    img = cv2.imread(os.path.join(dir_train, str_train), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    arr_train.append([img, label])
    #print(len(arr_train))

#arr_test
for str_test in os.listdir(dir_test):
    str_label = str_test[-5: -7: -1][::-1]
    label = labels.index(str_label)
    img = cv2.imread(os.path.join(dir_test, str_test), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    arr_test.append([img, label])
    #print(len(arr_test))









