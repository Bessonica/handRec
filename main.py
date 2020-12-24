#import
import os
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
#создаем свой датасет

#адрес фото
dir_train = "D:/programming/deeplearning/dataset/hands/train"
dir_test = "D:/programming/deeplearning/dataset/hands/test"
#можно лиD:/programming/deeplearning/dataset/hands/train/*.png???7!!!

labels = ["0L", "1L", "2L", "3L", "4L", "5L", "0R", "1R", "2R", "3R", "4R", "5R", ]

#создаем массивы изображений и ярлыков,которые обозначают руку и кол пальцев


arr_train = []
arr_test = []

#listdir дает имена файлов  img_train = str_train
#не забывай что у имени в конце есть еще .png. -1 = идем справо налево

#arr_train
for str_train in os.listdir(dir_train):
    str_label = str_train[-5: -7: -1][::-1]
    label = labels.index(str_label)
    img = cv2.imread(os.path.join(dir_train, str_train), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    arr_train.append([img, label])
    print(len(arr_train))


#arr_test
for str_test in os.listdir(dir_test):
    str_label = str_test[-5: -7: -1][::-1]
    label = labels.index(str_label)
    img = cv2.imread(os.path.join(dir_test, str_test), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    arr_test.append([img, label])
    print(len(arr_test))




#отделяем фото от меток

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



#print(arr_test_img.shape)#'list' object has no attribute 'shape'.решение- использовать np

#arr_test_img    это лист наполненный массивами.превращаем этот лист в один массив и делаем изображение черно белым
arr_test_img = np.array(arr_test_img)
arr_test_img = arr_test_img / 255.0

#print(arr_test_img) #.shape = 3600, 128, 128 заметь что команда почти точно такаяже как и ришейп ниже
#можно ли теперь делить?



#print(arr_test_img)
#найс,вроде все норм поделилось,!!!!но протестируй NN на тестовом массиве

#arr_train_img
arr_train_img = np.array(arr_train_img)
arr_train_img = arr_train_img / 255.0
print(arr_train_img.shape)#.shape 18000, 128, 128


arr_train_label = np.array(arr_train_label)
arr_test_label = np.array(arr_test_label)

#проверка
#print("test image shape", arr_test_img.shape) #3600, 128, 128, 1
#print("train image shape", arr_train_img.shape)#18000, 128, 128, 1



#создаем модель

#сколько нейронов выбрать?прочитал что оптимальный выбор числа нейронов между
#инпутом(128*128 = 16 384) и аутпутом(12) многовато,нужно тестить
#начну с 400+может быть сделать 2 внутренних слоя
#точность 0.99 так что трех слоев достаточно.!!!не забудь протестировать на новой инфе(массив test и
# еще есть папка fingers)


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(128, 128)),
    tf.keras.layers.Dense(400, activation="relu"),
    tf.keras.layers.Dense(12, activation="softmax")
])
#разобраться в разнице между разными optimizer что делает RMSprop
model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#batc size по дефолту они и так 32,но у нас датасет и нужно обязательно указать.по крайней мере так прочитал
#в документации
#вспомни в чем разница между .fit и .evaluate
model.fit(arr_train_img, arr_train_label,batch_size=32, epochs=5)

#тестируем
test_loss, test_acc = model.evaluate(arr_test_img, arr_test_label)

print("Окончательная точность", test_acc)# 0.995

#выводы 19.12.20
#медленный,можно оптимизировать?
#сделать CNN и добавить поддержку вебкамеры






