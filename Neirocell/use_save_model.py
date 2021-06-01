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
from tensorflow.keras.models import load_model


img_width = 28
img_height = 28
# Количество эпох
epochs = 50
# ? Количество изображений в
batch_size = 30
model = load_model('first_model_with_big_dataset.h5')
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
datagen = ImageDataGenerator(rescale=1.0 / 255)
# Пути до папопок с тренировойчными, тестовый набооры и набор для окончательной проверки.
train_dir = os.getcwd() + "\\energizers03"
test_dir = os.getcwd() + "\\energizers03test"
val_dir = os.getcwd() + "\\energizers03val"

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

scores1 = model.evaluate(test_generator, steps=1, verbose=1)
print("Аккуратность на тестовых данных: %.2f%%" % (scores1[1] * 100))
prediction = model.predict(val_generator, steps=20)

# цикл вывода на экран результатов распознавания validation set
f = open('result.txt', 'w', encoding='UTF-8')
print(len(prediction))
for i in range(len(prediction)):
    np.argmax(prediction[i])
    print(ru_classnames[classnames.index(val_generator.filenames[i].split('''\\''')[0])], "---->", ru_classnames[np.argmax(prediction[i])], file=f)
f.close()