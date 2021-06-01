# from PIL import Image as pimage
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
# from tensorflow.keras.preprocessing import image
# from matplotlib import pyplot as plt
# import os
import os.path
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout


class_list = []
# Список всех букв и их перевод на русский
classnames = ['a', 'b', 'v', 'g', 'd', 'je', 'jo', 'zh', 'z', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't',
              'u', 'f', 'h', 'c', 'ch', 'sh', 'sch', 'tvz', 'y', 'mhz', 'e', 'ju', 'ja']
ru_classnames = ['а', 'б', 'в', 'г', 'д', 'e', 'ё', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т',
                 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я']
# Создание листа директорий с буквами
for class_name in classnames:
    dir_list = os.listdir(os.path.join('energizers03', class_name))
    class_list.extend([classnames] * len(dir_list))
# Создание
datagen = ImageDataGenerator(rescale=1.0 / 255)
# Пути до папопок с тренировойчными, тестовый набооры и набор для окончательной проверки.
train_dir = os.getcwd() + "\\energizers03"
test_dir = os.getcwd() + "\\energizers03test"
val_dir = os.getcwd() + "\\energizers03val"
# Ширина и высота изображения
img_width = 28
img_height = 28
# Количество эпох
epochs = 50
# ? Количество изображений в
batch_size = 30
# Количество фотографий в наборах
nb_train_samples = 26466
nb_test_samples = 10560
nb_validation_samples = 15906

# Обучающая модель Sequential
model = Sequential()
# Создаём свёрточную нейросеть
# Conv2D - convolutional 2 dimension (свёрточная нейросеть в двух измерениях)
# stride - шаг смещения фильтра
# padding -
# input_shape -
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1)))  # (28, 28, 1)
# Добавляем функцию активации
model.add(Activation('relu'))
# Изменение масштаба получаемого изображения, выбирая максимальные значения
# MaxPooling2D - слой дискретизации по выборке pool_size
# padding -
model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

model.add(Dropout(0.1))
#
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
#

# Превращает результат работы нейросети в (одномерный) вектор
# Flatten - смягчающий слой
model.add(Flatten())
# Добавляем слой из 64 нейронов
model.add(Dropout(0.5))
model.add(Dense(64))
# Добавляем этому слою функцию активации
model.add(Activation('relu'))
# Исключение некоторых нейронов для борьбы с переобучением
# Добавляем слой из 33 нейронов
model.add(Dense(33))
# Добавляем этому слою функцию активации
model.add(Activation('softmax'))
# Вывод структуры нейронной сети
model.summary()
#
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
# Создание тренировочного набора
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical",
    color_mode="grayscale")

# Создание данных для обучения
test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    color_mode="grayscale")

# Создание данных для самопроверки нейросетью
val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    color_mode="grayscale")

# Обучение нейросети
model.fit(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    verbose=1,
    shuffle=False,
    validation_data=None,
    batch_size=None,
    callbacks=None
)

model.save("first_model_with_big_dataset.h5")

# не нужно, пока храню как память о былом
# validation_data = val_generator
# validation_steps = nb_validation_samples // batch_size

# Устаревший evaluate_generator из примера Созыкина
# scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
# print("Аккуратность на тестовых данных: %.2f%%" % (scores[1]*100))

# современный evaluate
scores1 = model.evaluate(test_generator, steps=1, verbose=1)
print("Аккуратность на тестовых данных: %.2f%%" % (scores1[1] * 100))
prediction = model.predict(val_generator, steps=20)

# цикл вывода на экран результатов распознавания validation set
f = open('result.txt', 'w', encoding='UTF-8')
for i in range(len(prediction)):
    np.argmax(prediction[i])
    print(ru_classnames[classnames.index(val_generator.filenames[i].split('''\\''')[0])], "---->", ru_classnames[np.argmax(prediction[i])], file=f)
