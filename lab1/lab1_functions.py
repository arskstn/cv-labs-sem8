import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import random
from IPython.display import Image
import math
import os

def median_1d(arr_initial, aperture):
    if aperture % 2 != 0:
        if aperture != 1:
            arr = [arr_initial[0]] + arr_initial + [arr_initial[-1]]
        else:
            arr = arr_initial
        curr_start = 0
        result = []
        for i in range(len(arr) - (aperture - 1)):
            curr_slice = arr[i : (aperture - 1 + i)+1]
            curr_slice = sorted(curr_slice)
            print(curr_slice)
            result.append(curr_slice[aperture // 2])
        return result

    else:
        print("Используйте окно нечетного размера")

def median_2d(image, minor_size = 3):
    if minor_size % 2 == 0:
        print("Выбрана четная размерность апертуры")
        return image
    else:
        image_median = image.copy()
        for rows in range (image.shape[0] - minor_size + 1):
            for columns in range(image.shape[1] - minor_size + 1):
                current_minor = image[rows:rows+minor_size, columns:columns+minor_size]
                #print(current_minor)

                current_minor = np.reshape(current_minor, (1,current_minor.shape[0]**2))
                current_minor = np.sort(current_minor)
                #print(current_minor)

                median_value = int(np.median(current_minor))
                #print(median_value)

                image_median[rows + minor_size//2][columns + minor_size//2] = median_value

        return image_median

def intensity(image):
    height, width, _ = image.shape
    image_intensity = np.zeros((height, width), dtype='uint8')

    for rows in range(image.shape[0]):
        for columns in range(image.shape[1]):
            image_intensity[rows][columns] = sum(image[rows][columns]) // 3

    return image_intensity

def median_2d_RGB(image, minor_size = 3):
    if minor_size % 2 == 0:
        print("Выбрана четная размерность апертуры")
        return image
    else:
        if len(image.shape) == 2:
            return median_2d(image, minor_size)

        elif len(image.shape) == 3 and image.shape[2] == 3:
            height, width, _ = image.shape
            image_R, image_G, image_B = np.split(image, 3, axis=2)

            image_R_median = median_2d(image3_R, 13)
            image_G_median = median_2d(image3_G, 13)
            image_B_median = median_2d(image3_B, 13)

            image_RGB_median = np.dstack((image_R_median, image_G_median, image_B_median))
            return image_RGB_median
        else:
            print("Изображение не RGB, должно быть 3 цветовых канала")

def Gauss(x, sigma):
    return (1/math.sqrt(2*math.pi*(sigma**2)))*(math.e**(-(x**2)/(2*(sigma**2))))

def gauss_kernel_visualize(sigma, minor_size=3):
    #АХТУНГ! По сути тот выйдет на выходе кернел 200 на 200 - почему? смотри коммент ниже
    if minor_size % 2 == 0:
        print("Выбрана четная размерность апертуры")
        return None

    c = minor_size // 2
    x_values = np.linspace(-c, c, 200)
    '''
    Зачем здесь такая громоздкая хреновина и что делает? Мы задаем x_values на 200 точек от -с до +с значениями.
    Почему? потому что при маленьком количестве точек, получается, что функция для визуализации contourf.
    Она работает строя изолинии по интерполяциий между узлами сетки. А у нас она выходит с шагом в 1 по хорошему.
    (так как минор небольшой, там не будет 200 элементов по одной оси... Обычно 9 или 21 берут. То бишь раз в 10 меньше)
    Как быть? в целях визуализации моя апертура будет 200 на 200 элементов. Все красиво, работает.
    А если хочу визуализировать в натуральном размере? Не вопрос, делай, но ты выведешь ромб, а не круг...
    При маленьких расстояниях интерполяция начинает себя вести не как евклидово расстояние, а манхэттенское - лесенкой.
    Собственно ты и получаешь не круг, а круг из майнкрафта - ромб.
    Если горит визуализировать как круг - надо искать альтернативный способ. На момент написания мне лень это делать,
    это не является целью реализации метода. Но если мне станет не лень, напишу ниже.
    Так что же такое X,Y которые я швыряю в Z. Изображение к которому применят функцию гаусса? нет! Это координатная сетка
    для финальной визуализации, вот и все!
    '''
    y_values = np.linspace(-c, c, 200)
    X, Y = np.meshgrid(x_values, y_values)

    Z = Gauss(X, sigma) * Gauss(Y, sigma)
    Z /= Z.sum()  # нормализация
    return X, Y, Z

def gaussian_2d(image, sigma, minor_size = 9):
    if minor_size % 2 == 0:
        print("Выбрана четная размерность апертуры")
        return image
    else:
        image_gaussian = image.astype(float).copy()
        pad = minor_size // 2
        image_padded = np.pad(image, pad, mode='edge') #исправим ошибку с потерей краев изображений при применении фильтров добавив паддинг, см. визуализацию медианного фильтра, если непонятно.

        x_values = np.arange(-pad, pad+1, 1)
        y_values = np.arange(-pad, pad+1, 1)
        X, Y = np.meshgrid(x_values, y_values)

        Z = np.ones_like(X, dtype=float)
        print('limits for Z kernel', np.min(Z), np.max(Z))
        for i in range(len(x_values)):
            for j in range(len(y_values)):
                Z[j, i] = Gauss(X[j, i], sigma) * Gauss(Y[j, i], sigma)
        Z /= Z.sum() #нормализуем. зачем? потому что в сумме все должно единичку дать. не будет этого - будут приколы с избыточной/недостаточной интенсивностью. можно отключить и проверить результат...
        print('limits for Z kernel after normalization', np.min(Z), np.max(Z))
        for rows in range (image_padded.shape[0] - minor_size + 1):
            for columns in range(image_padded.shape[1] - minor_size + 1):

                current_minor = image_padded[rows:rows+minor_size, columns:columns+minor_size]
                accumulator = 0

                accumulator = np.sum(current_minor * Z)

                image_gaussian[rows][columns] = accumulator

        print('limits for gaussian output', np.min(image_gaussian), np.max(image_gaussian))
        return image_gaussian #почему не обрезал паддинг который добавляли? потому что он у меня при свертке только для чтения был. перезапись пикселей сделана только в копии исходного изображения, собственно, размеры и не поменялись...


def gaussian_2d_RGB(image, sigma, minor_size = 9):
    if minor_size % 2 == 0:
        print("Выбрана четная размерность апертуры")
        return image
    else:
        if len(image.shape) == 2:
            return gaussian_2d(image, sigma, minor_size)

        elif len(image.shape) == 3 and image.shape[2] == 3:
            height, width, _ = image.shape
            image_R, image_G, image_B = np.split(image, 3, axis=2)
            image_R = image_R.squeeze()
            image_G = image_G.squeeze()
            image_B = image_B.squeeze()

            image_R_gaussian = gaussian_2d(image_R, sigma, minor_size)
            print('limits for R channel', np.min(image_R_gaussian), np.max(image_R_gaussian))
            image_G_gaussian = gaussian_2d(image_G, sigma, minor_size)
            print('limits for G channel', np.min(image_G_gaussian), np.max(image_G_gaussian))
            image_B_gaussian = gaussian_2d(image_B, sigma, minor_size)
            print('limits for B channel', np.min(image_B_gaussian), np.max(image_B_gaussian))

            image_RGB_gaussian = np.dstack((image_R_gaussian, image_G_gaussian, image_B_gaussian))
            image_RGB_gaussian = np.clip(image_RGB_gaussian, 0, 255).astype(np.uint8)
            return image_RGB_gaussian
        else:
            print("Изображение не RGB, должно быть 3 цветовых канала")


def intensity_grayscale(image):
    height, width, _ = image.shape
    image_intensity = np.zeros((height, width), dtype='uint8')

    for rows in range(image.shape[0]):
        for columns in range(image.shape[1]):
            #отталкиваемся от того, что исходник RGB и порядок каналов сохранен
            image_intensity[rows][columns] = 0.299*image[rows][columns][0] + 0.587*image[rows][columns][1] + 0.114*image[rows][columns][2]

    return image_intensity


def eroded_threshold(image, threshold, minor_size = 9):
    if minor_size % 2 == 0:
        print("Выбрана четная размерность апертуры")
        return image
    else:
        image_eroded = image.astype(float).copy()
        pad = minor_size // 2
        image_padded = np.pad(image, pad, mode='edge') #исправим ошибку с потерей краев изображений при применении фильтров добавив паддинг, см. визуализацию медианного фильтра, если непонятно.

        # x_values = np.arange(-pad, pad+1, 1)
        # y_values = np.arange(-pad, pad+1, 1)
        # X, Y = np.meshgrid(x_values, y_values)

        # Z = np.ones_like(X, dtype=float)
        # print('limits for Z kernel', np.min(Z), np.max(Z))

        for rows in range (image_padded.shape[0] - minor_size + 1):
            for columns in range(image_padded.shape[1] - minor_size + 1):
                current_minor = image_padded[rows:rows+minor_size, columns:columns+minor_size]
                for i in current_minor:
                    for j in i:
                        if j <= threshold:
                            image_eroded[rows][columns] = threshold #а если соседей подходящих на условие несколько, мы будем несколько раз присваивать значение таргету? Да. Не хочется - можно ввести переменную для флага и тогда все равно его переопределять несколько раз. Смысл?

        print('limits for eroded output', np.min(image_eroded), np.max(image_eroded))
        return image_eroded #почему не обрезал паддинг который добавляли? потому что он у меня при свертке только для чтения был. перезапись пикселей сделана только в копии исходного изображения, собственно, размеры и не поменялись...


def dilated_threshold(image, threshold, minor_size = 9):
    if minor_size % 2 == 0:
        print("Выбрана четная размерность апертуры")
        return image
    else:
        image_dilated = image.astype(float).copy()
        pad = minor_size // 2
        image_padded = np.pad(image, pad, mode='edge') #исправим ошибку с потерей краев изображений при применении фильтров добавив паддинг, см. визуализацию медианного фильтра, если непонятно.

        for rows in range (image_padded.shape[0] - minor_size + 1):
            for columns in range(image_padded.shape[1] - minor_size + 1):
                current_minor = image_padded[rows:rows+minor_size, columns:columns+minor_size]
                for i in current_minor:
                    for j in i:
                        if j >= threshold:
                            image_dilated[rows][columns] = threshold #а если соседей подходящих на условие несколько, мы будем несколько раз присваивать значение таргету? Да. Не хочется - можно ввести переменную для флага и тогда все равно его переопределять несколько раз. Смысл?

        print('limits for dilated output', np.min(image_dilated), np.max(image_dilated))
        return image_dilated #почему не обрезал паддинг который добавляли? потому что он у меня при свертке только для чтения был. перезапись пикселей сделана только в копии исходного изображения, собственно, размеры и не поменялись...



def binarized(image, threshold):
    image1 = image.copy()

    for rows in range (image1.shape[0]):
            for columns in range(image1.shape[1]):
                if image1[rows][columns] <= threshold:
                    image1[rows][columns] = 0
                else:
                    image1[rows][columns] = 1

    return image1


def eroded_bin(image, minor_size = 3):
    if minor_size % 2 == 0:
        print("Выбрана четная размерность апертуры")
        return image
    else:
        image_eroded = image.astype(float).copy()
        pad = minor_size // 2
        image_padded = np.pad(image, pad, mode='edge') #исправим ошибку с потерей краев изображений при применении фильтров добавив паддинг, см. визуализацию медианного фильтра, если непонятно.

        for rows in range (image_padded.shape[0] - minor_size + 1):
            for columns in range(image_padded.shape[1] - minor_size + 1):
                current_minor = image_padded[rows:rows+minor_size, columns:columns+minor_size]
                for i in current_minor:
                    for j in i:
                        if j == 1:
                            image_eroded[rows][columns] = 1 #а если соседей подходящих на условие несколько, мы будем несколько раз присваивать значение таргету? Да. Не хочется - можно ввести переменную для флага и тогда все равно его переопределять несколько раз. Смысл?

        print('limits for eroded output', np.min(image_eroded), np.max(image_eroded))
        return image_eroded #почему не обрезал паддинг который добавляли? потому что он у меня при свертке только для чтения был. перезапись пикселей сделана только в копии исходного изображения, собственно, размеры и не поменялись...


def dilated_bin(image, minor_size = 3):
    if minor_size % 2 == 0:
        print("Выбрана четная размерность апертуры")
        return image
    else:
        image_dilated = image.astype(float).copy()
        pad = minor_size // 2
        image_padded = np.pad(image, pad, mode='edge') #исправим ошибку с потерей краев изображений при применении фильтров добавив паддинг, см. визуализацию медианного фильтра, если непонятно.

        for rows in range (image_padded.shape[0] - minor_size + 1):
            for columns in range(image_padded.shape[1] - minor_size + 1):
                current_minor = image_padded[rows:rows+minor_size, columns:columns+minor_size]
                for i in current_minor:
                    for j in i:
                        if j == 0:
                            image_dilated[rows][columns] = 0 #а если соседей подходящих на условие несколько, мы будем несколько раз присваивать значение таргету? Да. Не хочется - можно ввести переменную для флага и тогда все равно его переопределять несколько раз. Смысл?

        print('limits for eroded output', np.min(image_dilated), np.max(image_dilated))
        return image_dilated #почему не обрезал паддинг который добавляли? потому что он у меня при свертке только для чтения был. перезапись пикселей сделана только в копии исходного изображения, собственно, размеры и не поменялись...


def binarized_RGB(image, thresholdR, thresholdG, thresholdB):
    image1 = image.copy()
    if len(image1.shape) == 2:
        print("Используй функцию binarized, эта функция нужна для RGB, а здесь больше 1 канала")
    else:
        image_R, image_G, image_B = np.split(image1, 3, axis=2)
        image_R = image_R.squeeze()
        image_G = image_G.squeeze()
        image_B = image_B.squeeze()
        image_R_binarized = binarized(image_R, thresholdR)
        image_G_binarized = binarized(image_G, thresholdG)
        image_B_binarized = binarized(image_B, thresholdB)
        image_RGB_binarized = np.dstack((image_R_binarized, image_G_binarized, image_B_binarized))
        #image_RGB_gaussian = np.clip(image_RGB_gaussian, 0, 255).astype(np.uint8)
        return image_RGB_binarized


def image_hist(image):
    counts = dict.fromkeys(range(256), 0)

    for rows in range (image.shape[0]):
            for columns in range(image.shape[1]):
                counts[image[rows][columns]] += 1
    return counts

def hist_equalize(image):
    counts = dict.fromkeys(range(256), 0)

    for rows in range (image.shape[0]):
            for columns in range(image.shape[1]):
                counts[image[rows][columns]] += 1

    total_pixels = image.shape[0]*image.shape[1]

    probabilities = list()
    names = counts.keys()

    for i in names:
        temp = counts[i]/total_pixels
        probabilities.append(counts[i]/total_pixels)


    cdf = [0] * 256
    cumulative = 0.0

    for i in range(256):
        cumulative += probabilities[i]
        cdf[i] = cumulative

    image_equalized = np.zeros_like(image)

    for rows in range(image.shape[0]):
        for columns in range(image.shape[1]):
            r = image[rows][columns]
            s = int(round(cdf[r] * 255)) #зачем округление? у нас интенсивности - целые числа, а не дробные. А кумулятивы дробные.
            image_equalized[rows][columns] = s

    return image_equalized


def hist_equalize_RGB(image):
    image1 = image.copy()
    image_R, image_G, image_B = np.split(image1, 3, axis=2)
    image_R = image_R.squeeze()
    image_G = image_G.squeeze()
    image_B = image_B.squeeze()
    image_R_equalized = hist_equalize(image_R)
    image_G_equalized = hist_equalize(image_G)
    image_B_equalized = hist_equalize(image_B)
    image_RGB_equalized = np.dstack((image_R_equalized, image_G_equalized, image_B_equalized))
    image_RGB_equalized = np.clip(image_RGB_equalized, 0, 255).astype(np.uint8)
    return image_RGB_equalized


def rotate_90_cw(image, amount):
    image_90_cw = image.copy()

    for i in range(amount):
        image_90_cw = np.transpose(image_90_cw)

        for rows in range(len(image_90_cw)):
            curr_row = image_90_cw[rows]
            rev_row = curr_row[::-1]
            image_90_cw[rows] = rev_row

    return image_90_cw

def rotate_90_cw_rgb(image, amount):
    image1 = image.copy()
    image_R, image_G, image_B = np.split(image1, 3, axis=2)
    image_R = image_R.squeeze()
    image_G = image_G.squeeze()
    image_B = image_B.squeeze()
    image_R_rotated = rotate_90_cw(image_R, amount)
    image_G_rotated = rotate_90_cw(image_G, amount)
    image_B_rotated = rotate_90_cw(image_B, amount)
    image_RGB_rotated = np.dstack((image_R_rotated, image_G_rotated, image_B_rotated))
    image_RGB_rotated = np.clip(image_RGB_rotated, 0, 255).astype(np.uint8)
    return image_RGB_rotated

