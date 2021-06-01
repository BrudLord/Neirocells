import pygame
import os
import shutil
from random import randint


# Создание папок для обучения
def build_dirs(from_dir, to_dir):
    for i in os.walk(to_dir):
        print(i)
        for j in i[-1]:
            try:
                os.remove(i[0] + '''\\''' + j)
            except Exception:
                print(9)
    dl = kol_pic(from_dir)[0][0]
    f03 = dl // 2
    f03test = dl // 10 * 2
    f03val = min([dl - f03 - f03test, 18])
    print(f03, f03val, f03test)
    for i in os.walk(from_dir):
        print(i[0].split('\\')[-1])
        for j in range(len(i[-1])):
            try:
                if j < f03:
                    shutil.copy(i[0] + '''\\''' + i[-1][j],
                                to_dir + '''\\''' + 'energizers03' + '''\\''' + i[0].split('\\')[-1])
                elif j < f03 + f03test:
                    shutil.copy(i[0] + '''\\''' + i[-1][j],
                                to_dir + '''\\''' + 'energizers03test' + '''\\''' + i[0].split('\\')[-1])
                elif j < f03 + f03test + f03val:
                    shutil.copy(i[0] + '''\\''' + i[-1][j],
                                to_dir + '''\\''' + 'energizers03val' + '''\\''' + i[0].split('\\')[-1])
                else:
                    break
            except Exception:
                print(9)


# Перенос картинок в одну папку
def travel_files(from_dir, to_dir):
    for i in os.walk(from_dir):
        print(i)
        for j in i[-1]:
            try:
                shutil.copy(i[0] + '''\\''' + j, to_dir + '''\\''' + i[0].split('\\')[-1])
            except Exception:
                print(9)


# Количество изображений в общей папке
def kol_pic(direc):
    buk = []
    for i in os.walk(direc):
        if len(i[-1]) != 0:
            buk.append([len(i[-1]), i[0].split('\\')[-1]])
    return sorted(buk)


def load_image(name, colorkey=None):
    # Загрузка изображения
    fullname = os.path.join(name)
    # если файл не существует, то выходим
    if not os.path.isfile(fullname):
        return '0'
    image = pygame.image.load(fullname)
    return image


def transform(name):
    im = load_image(name)
    im = pygame.transform.scale(im, (randint(20, 36), randint(20, 36)))
    im = pygame.transform.rotate(im, randint(-10, 10))
    im = pygame.transform.scale(im, (28, 28))
    pygame.draw.rect(im, (255, 255, 255), (0, 0, 28, 28), 2)
    pygame.image.save(im, name.split('.')[0] + ''.join([str(randint(0, 9)),
                                                        str(randint(0, 9)),
                                                        str(randint(0, 9)),
                                                        str(randint(0, 9)),
                                                        str(randint(0, 9)),
                                                        str(randint(0, 9)),
                                                        str(randint(0, 9)),
                                                        str(randint(0, 9)),
                                                        str(randint(0, 9)),
                                                        str(randint(0, 9)),
                                                        str(randint(0, 9)),
                                                        str(randint(0, 9))]) + '.' + name.split('.')[-1])


def transform_all_pic(direct):
    for i in os.walk(direct):
        print(i)
        if len(i[-1]) == 0:
            continue
        k = 0
        while k < 1000:
            for j in i[-1]:
                try:
                    transform(i[0] + '''\\''' + j)
                    k += 1
                except Exception:
                    print(9)


def del_useless_files(direct):
    for i in os.walk(direct):
        print(i)
        for j in i[-1]:
            try:
                os.remove(i[0] + '''\\''' + j)
            except Exception:
                print(9)


# copy_from_dir = 'to_join'
# copy_to_dir = 'after_join'
# travel_files(copy_from_dir, copy_to_dir)
# print(kol_pic(copy_to_dir))

# build_dirs_from = 'after_join'
# build_dirs_to = 'build_dirs'
# build_dirs(build_dirs_from, build_dirs_to)
# print(kol_pic('build_dirs\\energizers03'))
# print(kol_pic('build_dirs\\energizers03test'))
# print(kol_pic('build_dirs\\energizers03val'))

# build_dirs_from = 'after_join'
# build_dirs_to = 'build_dirs'
# build_dirs(build_dirs_from, build_dirs_to)
# print(kol_pic('build_dirs\\energizers03'))
# print(kol_pic('build_dirs\\energizers03test'))
# print(kol_pic('build_dirs\\energizers03val'))

# del_useless_files('after_join')

# transform_all_pic('after_join')
# print(kol_pic('after_join'))
