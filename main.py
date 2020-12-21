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
# что такоке glob?
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
    print(len(arr_train))


#arr_test
for str_test in os.listdir(dir_test):
    str_label = str_test[-5: -7: -1][::-1]
    label = labels.index(str_label)
    img = cv2.imread(os.path.join(dir_test, str_test), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    arr_test.append([img, label])
    print(len(arr_test))






#reshape и сделать черно белім изображение

#проверяем датасет
#print(arr_test[0][0])
#print(labels[arr_test[0][1]])
#print(len(arr_train))#18000
#print(arr_train[0][0].shape)#128,128
#print(arr_train[0].shape)#'list' object has no attribute 'shape'
#print(arr_train.shape) 18000, 2





#отделяем массивы фото от меток

arr_train_img = []
arr_train_label = []

arr_test_img = []
arr_test_label = []

for img_train, label_train in arr_train:
    arr_train_img.append(img_train)
    arr_train_label.append(label_train)


# #проверка
# print("img : ", len(arr_train_img)) #18000
# print("label : ", len(arr_train_label))  #18000
# plt.imshow(arr_train_img[0], cmap = 'gray')
# plt.show()
# print(labels[arr_train_label[0]])

for img_test, label_test in arr_test:
    arr_test_img.append(img_test)
    arr_test_label.append(label_test)

# #проверка
# print("img : ", len(arr_test_img)) #3600
# print("label : ", len(arr_test_label))  #3600
# plt.imshow(arr_test_img[0], cmap = 'gray')
# plt.show()
# print(labels[arr_test_label[0]])


arr_train_img = np.array(arr_train_img).reshape(-1, 128, 128, 1)

#arr_test_img это лист (list) из массивов,в каждом массиве 128 масивов с 128 элементов????

#!!что если сделать цикл,где ті проходишь arr_test_img[i]( и делишь их на 255.0!!



arr_test_img = np.array(arr_test_img).reshape(-1, 128, 128, 1)
#print(arr_test_img[0].shape)#128,128 до ришейпа.   после (128, 128, 1)
#осле ришейпа 1 изображение  это массив из масивов в которых числа(пиксели)в отдельных
#массивах  [ [[15]...[15](128 элеиентов) ](строка 1) ... [[0]...(128 элеиентов)[0]](строка 128)  ]

arr_test_img = arr_test_img / 255.0
arr_train_img = arr_train_img / 255.0


arr_train_label = np.array(arr_train_label)
arr_test_label = np.array(arr_test_label)

#проверка
#print("test image shape", arr_test_img.shape) #3600, 128, 128, 1
#print("train image shape", arr_train_img.shape)#18000, 128, 128, 1
# print("test img ", len(arr_test_img))
# print("train img ", len(arr_train_img))
# print("train label", len(arr_train_label))
# print("test label", len(arr_test_label))


#создаем модель
#в ANN можно было скормить массивы без ришейпа
#но с CNN необходимо изменить лист массивов,что бы скормить моделе.
#иначе будет ошибка  input 0 of layer sequential is incompatible with the layer expected min_ndim=4 found ndim=3
#
#print("shape img train", arr_train_img.shape[1:]) #128, 128, 1


#!!!!!!!!!!!!!!!!!!!    ЭКСПЕРЕМЕНТИРУЮ С МОДЕЛЬЮ      !!!!!!!!!!!!
#модель неверотно массивная(параметров больше ляма)  conv2d(20) еще 40 размер=3, 3

#уменьшил кол функций conv2d(20,(10, 10)) и увеличил их размеры(10,10) параметров меньше 184тыс,но невероятно медленна

#уменьшил кол функций conv2d(20) и уменьшил их размеры(3,3)
#модель работает лучше,но все еще невероятно медленно,но модель быстрее учится.ей достаточно 2 epochs что бы быть +-100% точным

#попробую увеличить пулинг,было 2, 2.сделаю 5, 5
#модель учится намного быстрее, но меня все еще раздражает что моя CNN работает медленее ANN,хотя в теории должно быть
#наоборот(параметров 9тыс)

#сделаю пул 10, 10.Появилась ошибка на строке где последний conv2d - Value error negative dimension size caused by subtracting 3 from 1 for ...

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(20, (3, 3), activation="relu", input_shape=(128, 128, 1)))
model.add(tf.keras.layers.MaxPool2D((5, 5)))
model.add(tf.keras.layers.Conv2D(20, (3, 3), activation="relu"))
model.add(tf.keras.layers.MaxPool2D((5, 5)))
model.add(tf.keras.layers.Conv2D(20, (3, 3), activation="relu"))
model.summary()

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(20, activation="relu"))
model.add(tf.keras.layers.Dense(12))
model.summary()

model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.fit(arr_train_img, arr_train_label, batch_size=32, epochs=2)


test_loss, test_acc = model.evaluate(arr_test_img, arr_test_label)
print("test acc", test_acc)  #0.99

#21.12.2020 выводы
#CNN готова но я не доволен ее скоростью,почитать способы оптимизации
#все еще нет поддержки веб камеры,попытайся не использовать object detection api от тенсерфлоу,а просто сделать все используя только cv2
#ОТДЕЛИ МОДЕЛЬ ОТ ОСНОВНОЙ ЧАСТИ КОДА,что бы не создавать и учить модель каждый раз когда запускаешь код
#ПОСТАВЬ ГИТ В pycharm


