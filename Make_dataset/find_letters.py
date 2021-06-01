from typing import Any, List
import random as rd
import numpy as np
import cv2


image_file = "images/123.jpg"
# Считываем изображение, создавая лист листов BGR
img = cv2.imread(image_file)
# Переводим изображение из различных оттенков серого
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Переводим в чёрно-белый формат
# threshold(0) - серая картинка
# threshold(1) - если 0, то threshold(3)
# threshold(2) - иначе
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
# Зачем iterations?
img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=15)

# Get contours
contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
output = img.copy()

for idx, contour in enumerate(contours):
    (x, y, w, h) = cv2.boundingRect(contour)
    # print("R", idx, x, y, w, h, cv2.contourArea(contour), hierarchy[0][idx])
    # иерархия[i][0] — индекс следующего контура на текущем слое;
    # иерархия[i][1] — индекс предыдущего контура на текущем слое:
    # иерархия[i][2] — индекс первого контура на вложенном слое;
    # иерархия[i][3] — индекс родительского контура.
    if hierarchy[0][idx][3] == 0:
        cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)


cv2.imshow("Input", img)
cv2.imshow("Enlarged", img_erode)
cv2.imshow("Output", output)
cv2.waitKey(0)

def letters_extract(image_file: str, out_size=28):
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=5)

    # Get contours
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    output = img.copy()

    letters = []
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        # print("R", idx, x, y, w, h, cv2.contourArea(contour), hierarchy[0][idx])
        # иерархия[i][0] — индекс следующего контура на текущем слое;
        # иерархия[i][1] — индекс предыдущего контура на текущем слое:
        # иерархия[i][2] — индекс первого контура на вложенном слое;
        # иерархия[i][3] — индекс родительского контура.
        if hierarchy[0][idx][3] == 0:
            cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
            letter_crop = gray[y:y + h, x:x + w]
            # print(letter_crop.shape)

            # Resize letter canvas to square
            size_max = max(w, h)
            letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
            if w > h:
                # Enlarge image top-bottom
                # ------
                # ======
                # ------
                y_pos = size_max//2 - h//2
                letter_square[y_pos:y_pos + h, 0:w] = letter_crop
            elif w < h:
                # Enlarge image left-right
                # --||--
                x_pos = size_max//2 - w//2
                letter_square[0:h, x_pos:x_pos + w] = letter_crop
            else:
                letter_square = letter_crop

            # Resize letter to 28x28 and add letter and its X-coordinate
            letters.append((x, w, cv2.resize(letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA)))

    # Sort array in place by X-coordinate
    letters.sort(key=lambda x: x[0], reverse=False)

    return letters

letters = letters_extract(image_file)
flag = True
s = ''
for i in range(rd.randint(2, 7)):
    s += chr(rd.randint(ord('a'), ord('z')))
for i in range(len(letters) - 1):
    filestring = 'vf1' + s + str(i) + '.png'
    cv2.imwrite(filestring,letters[i][2])
    cv2.imshow("0", letters[i][2])
#cv2.imshow("0", letters[0][2])
#cv2.imshow("1", letters[1][2])
#cv2.imshow("2", letters[2][2])
#cv2.imshow("3", letters[3][2])
#cv2.imshow("4", letters[4][2])
cv2.waitKey(0)