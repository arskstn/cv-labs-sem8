#!/usr/bin/env python
# coding: utf-8

# Костин Арсений, 8Е21, вариант 3.

# Для 1 лаб работы по CV необходимо реализовать базовый минимум операций над изображениями
# Входное изображение в формате (RGB, не чёрно-белое)
# 1. Фильтры
# <br>1.1 Медианный фильтр
# <br>1.2 Фильтр гаусса
# 2. Морфологические операции
# <br>2.1 Эрозия
# <br>2.2 Дилатация
# 3. Прочие операции
# <br>3.1 пороговая бинаризация (для rgb и grayscale изображения)
# <br>3.2 выравнивание гистограммы
# <br>3.3 поворот изображений на угол кратный 90 градусов
# 
# 
# Использовать методы OpenCV для реализации операций нельзя. Допустимы только методы cv2.imread() и cv2.imshow(). Все методы должны быть реализованы вручную.

# In[302]:


import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import random
#from PIL import Image
from IPython.display import Image
get_ipython().run_line_magic('matplotlib', 'inline')
import math

import os
print(os.getcwd())


# In[303]:


image1=cv2.imread('sample_image.jpg')
image2=cv2.imread('sample_image2.png')
image3=cv2.imread('sample_image3.png')


# In[304]:


plt.imshow(image1)


# In[305]:


image1_RGB = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)


# In[306]:


plt.imshow(image1_RGB)


# In[307]:


print(image1_RGB)


# # 1. ФИЛЬТРЫ

# ### 1.1 Медианный фильтр

# Медианный фильтр - один из методов борьбы с "шумами". Суть заключается в том, что создается "окно" для проверки. Внутри окна элементы упорядочиваются по возрастанию/убыванию. Как медианное значение берется число в середине этого окна. Если таких чисел несколько, то берется среднее значение двух чисел посередине окна. Если рассмотреть одномерный массив как объект, к котрому будет применен фильтр:
# (Используя пример из википедии: https://ru.wikipedia.org/wiki/%D0%9C%D0%B5%D0%B4%D0%B8%D0%B0%D0%BD%D0%BD%D1%8B%D0%B9_%D1%84%D0%B8%D0%BB%D1%8C%D1%82%D1%80)
# <p>
# Пусть есть одномерный массив x = [2 80 6 3]
# Пусть окно проверки будет размером 3, обозначено круглыми скобками
# 
# 1 итерация: (2 80 6)
# упорядочить
# (2 6 80) = медианное значение 6 = выход итерации = 6
# 
# 2 итерация: (80 6 3)
# упорядочить
# (3 6 80) = медианное значение 6 = выход итерации 6
# 
# Алогритм выполнен, выход фильтра [6 6], потеряны 2 элемента. Тренд сохраняется и при других размерах "окна". Таким образом: (длина окна - 1)/2 = количество потерянных элементов с одного края. То есть, в нашем случае были потеряны первый и последний элементы исходного массива. Продублируем элементы. Получаем:
# [2 2 80 6 3 3]
# Применим к исправленному исходному массиву медианный фильтр
# 
# 1 итерация: (2 2 80)
# упорядочить
# (2 2 80) = медианное значение 2 = выход итерации = 2
# 
# 2 итерация: (2 80 6)
# упорядочить
# (2 6 80) = медианное значение 6 = выход итерации = 6
# 
# 3 итерация: (80 6 3)
# упорядочить
# (3 6 80) = медианное значение 6 = выход итерации = 6
# 
# 4 итерация: (6 3 3)
# упорядочить
# (3 3 6) = медианное значение 3 = выход итерации = 3
# 
# Выход функции [2 6 6 3]. Значения были существенно сглажены.
# 
# Стоит упомянуть, что размер окна так же может быть четным. Но даже в одномерных массивах возникают определенные трудности по его применению. Например, можно брать левое медианное значение в окне, можно брать правое, существует путь с применением среднего арифметического обоих чисел и округленное для целого числа. Для проверки алгоритмических способностей фильтра в текущей задаче это будет избыточно. </p>
# 

# In[308]:


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

print(median_1d([1,2,3,4,5], 3))

#random_arr = [6, 2, 4, 1, 2, 6, 9, 3, 1, 7] - 8 times for 3el window; 10 elements
#random_arr_ext = [6, 6, 2, 4, 1, 2, 6, 9, 3, 1, 7, 7] - 10 times for 3el window; 12 elements


# aperture 3 = ind 1
# aperture 5 = ind 2
# aperture 7 = ind 3


# Проверим на примере из википедии

# In[309]:


print(median_1d([2,80,6,3], 3))


# Результат совпал. Проверим при окне размером 1.

# In[310]:


print(median_1d([2,80,6,3], 1))


# <p>
# Смысла делать с окном ноль нет, не берется медианы.
# <p>
# Рассмотрим двумерный массив.
# <p>
# По сути алгоритм тот же, но в двумерном пространстве.
# <p>
# Берем окно квадратного размера, по сути матрицу меньшего порядка, чем изначальную. При этом рекомендация брать нечетную размерность аргументируется схожим образом как для одномерных массивов. Проходим этим окном по изображению. На каждой итерации разворачиваем текущий минор в ряд и применяем медианный фильтр. После этого заменяем центральный элемент минора на медианное значение от всех элеемнтов внутри этого минора - то есть, стоящее по середине отсортированного ряда.
# <p>
# Зададим матрицу:

# In[311]:


sample_matrix = []
for i in range(100):
    temp_row = []
    for j in range(100):
        temp_row.append(random.randint(1,100))
    sample_matrix.append(temp_row)
for i in sample_matrix:
    print(i)


# Первый минор, размер 3:

# In[312]:


sample_matrix = np.array(sample_matrix)
minor_size = 3
sample_matrix_minor = sample_matrix[0:minor_size, 0:minor_size]
for i in sample_matrix_minor:
    print(i)


# Развернем и возьмем медианное значение

# In[313]:


unfolded_minor = np.reshape(sample_matrix_minor, (1,minor_size**2))
print(unfolded_minor)
print("Сортируем")
print(np.sort(unfolded_minor))
median_value = int(np.median(unfolded_minor))
print("Медианное значение =", median_value)



# Присвоим центральному элементу минора его же медианное значение

# In[314]:


sample_matrix_minor[(minor_size//2)][(minor_size//2)] = median_value
print(minor_size//2)
for i in sample_matrix_minor:
    print(i)


# Отлично, все работает. Теперь попробуем применить фильтр ко всей этой матрице:

# In[315]:


sample_matrix_median = sample_matrix.copy()
sample_matrix_median = np.pad(sample_matrix_median, (minor_size-1)//2)
for rows in range (sample_matrix.shape[1] - minor_size + 1):
    for columns in range(sample_matrix.shape[0] - minor_size + 1):

        current_minor = sample_matrix[rows:rows+minor_size, columns:columns+minor_size]
        print(current_minor)

        current_minor = np.reshape(current_minor, (1,current_minor.shape[0]**2))
        current_minor = np.sort(current_minor)
        print(current_minor)

        median_value = int(np.median(current_minor))
        print(median_value)

        sample_matrix_median[rows + minor_size//2][columns + minor_size//2] = median_value
for i in sample_matrix_median:
    print(i)


# In[316]:


f, axarr = plt.subplots(1,2)
axarr[0].imshow(sample_matrix, cmap='gray')
axarr[0].set_title('Исходная матрица')

axarr[1].imshow(sample_matrix_median, cmap='gray')
axarr[1].set_title('После медианного фильтра')


# In[317]:


f, axarr = plt.subplots(1,2)
axarr[0].imshow(sample_matrix)
axarr[0].set_title('Исходная матрица')

axarr[1].imshow(sample_matrix_median)
axarr[1].set_title('После медианного фильтра')


# Медианный фильтр применен. Попробуем на изображении. Поскольку входное изображение имеет цвета, его можно представить как матрицу, где кол-во строк = высота изображения, кол-во столбцов = ширина, и каждый пиксель является одномерным массивом из трех элементов = интенсивность red, green, blue соотвественно. Получается 3-х ранговый тензор. Поскольку мы применяем фильтр сейчас к двумерному массиву, нам нужно преобразовать изображение в карту интенсивностей. То есть, мы потеряем цвет, но получим карту интенсивностей изображения в градациях серого. Самый простой способ - взять сумму всех интенсивностей по каналам и разделить на количество каналов.
# <p>
# Попробуем:

# Изначальное изображение

# In[318]:


plt.imshow(image1_RGB)


# In[319]:


height, width, _ = image1_RGB.shape
image1_RGB_intensity = np.zeros((height, width), dtype='uint8')
print(image1_RGB.shape)
for rows in range(image1_RGB.shape[0]):
    for columns in range(image1_RGB.shape[1]):
        image1_RGB_intensity[rows][columns] = sum(image1_RGB[rows][columns]) // 3
plt.imshow(image1_RGB_intensity, cmap='gray')
print(image1_RGB_intensity)


# Теперь соберем это все в медианный фильтр для двумерного массива:

# In[320]:


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

#print(image1_RGB_intensity)
image = image1_RGB_intensity

#plt.imshow(image)
#image_median = median_2d(image)

# plt.imshow(image)
# plt.imshow(image_median)

#plt.figure(figsize=(40,40))
f, axarr = plt.subplots(1,2, figsize = (12,6))
axarr[0].imshow(image, cmap='gray')
axarr[0].set_title('Исходная матрица')

axarr[1].imshow(median_2d(image, 13), cmap='gray')
axarr[1].set_title('После медианного фильтра')



# Последнее - завернем перевод в карту интенсивностей в функцию

# In[321]:


def intensity(image):
    height, width, _ = image.shape
    image_intensity = np.zeros((height, width), dtype='uint8')

    for rows in range(image.shape[0]):
        for columns in range(image.shape[1]):
            image_intensity[rows][columns] = sum(image[rows][columns]) // 3

    return image_intensity

image = image2

f, axarr = plt.subplots(1,2, figsize = (12,6))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
axarr[0].imshow(image)
axarr[0].set_title('Исходная матрица')

axarr[1].imshow(intensity(image), cmap = 'gray')
axarr[1].set_title('Карта инстенсивностей')


# Посмотрим на результаты применения медианного фильтра к изображению 2

# In[322]:


image = intensity(image2)

f, axarr = plt.subplots(1,2, figsize = (12,6))
axarr[0].imshow(image, cmap='gray')
axarr[0].set_title('Исходная матрица')

axarr[1].imshow(median_2d(image, 13), cmap='gray')
axarr[1].set_title('После медианного фильтра')


# In[323]:


f, axarr = plt.subplots(1,2, figsize = (12,6))
axarr[0].imshow(image, cmap='gray')
axarr[0].set_title('Исходная матрица')

axarr[1].imshow(median_2d(image, 13), cmap='gray')
axarr[1].set_title('После медианного фильтра')


# А что насчет цветных изображений, как поступить там? наша функция работает же лишь для 2х мерных массивов. Решить эту проблему можно применим медианный фильтр для каждого из каналов отдельно. Рассмотрим изображение 3:

# In[324]:


image3_RGB = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB) #изначально opencv видит как BGR, переведем в RGB
plt.imshow(image3_RGB)


# In[325]:


height, width, _ = image3_RGB.shape
image3_R, image3_G, image3_B = np.split(image3_RGB, 3, axis=2)

image3_R_median = median_2d(image3_R, 13)
image3_G_median = median_2d(image3_G, 13)
image3_B_median = median_2d(image3_B, 13)
image3_R_median.shape

image3_RGB_median = np.dstack((image3_R_median, image3_G_median, image3_B_median))
plt.imshow(image3_RGB_median)


# Получилось! Сравним:

# In[326]:


image = image3_RGB
image1 = image3_RGB_median
f, axarr = plt.subplots(1,2, figsize = (12,6))
axarr[0].imshow(image)
axarr[0].set_title('Исходная матрица')

axarr[1].imshow(image1)
axarr[1].set_title('После медианного фильтра')


# Фоформиим как функцию:

# In[327]:


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

image = image3_RGB
image1 = image3_RGB_median
plt.imshow(median_2d_RGB(image3_RGB, 13))


# In[328]:


image = image3_RGB
image1 = image3_RGB_median
image2 = median_2d_RGB(image1, 39)


# In[329]:


f, axarr = plt.subplots(1,3, figsize = (15,9))
axarr[0].imshow(image)
axarr[0].set_title('Исходная матрица')

axarr[1].imshow(image1)
axarr[1].set_title('После медианного фильтра')

axarr[2].imshow(image2)
axarr[2].set_title('Результат фильтра после медианного фильтра')


# Поскольку значения на изображении остались теми же, фильтровать повторно там нечего. Соответственно, результаты 2 и 3 абсолютно идентичны.

# ### 1.2 Фильтр гаусса

# По определению: Размытие по Гауссу в цифровой обработке изображений — способ размытия изображения с помощью функции Гаусса, названной в честь немецкого математика Карла Фридриха Гаусса.
# 
# Этот эффект широко используется в графических редакторах для уменьшения шума изображения и снижения детализации. Визуальный эффект этого способа размытия напоминает эффект просмотра изображения через полупрозрачный экран, и отчётливо отличается от эффекта боке, создаваемого расфокусированным объективом или тенью объекта при обычном освещении. 

# Математика: Поскольку преобразование Фурье функции Гаусса само является функцией Гаусса, применение размытия по Гауссу приводит к уменьшению высокочастотных компонентов изображения. Таким образом, размытие по Гауссу является фильтром нижних частот. 
# В этом способе размытия функция Гаусса (которая также используется для описания нормального распределения в теории вероятностей) используется для вычисления преобразования, применяемого к каждому пикселю изображения. Формула функции Гаусса в одном измерении: 

# In[330]:


img = cv2.imread('./gaussfunc.png')
plt.imshow(img)


# Сигма в этой функции это среднеквадратическое отклонение нормального распределения. Визуализируем в desmos.

# In[331]:


img = Image('./output.gif')
img


# В двухмерном пространстве по определению это произведение двух функций Гаусса, для каждого измерения. Зададим функцию Гаусса как функцию в коде:

# In[332]:


def Gauss(x, sigma):
    return (1/math.sqrt(2*math.pi*(sigma**2)))*(math.e**(-(x**2)/(2*(sigma**2))))

print(Gauss(0, 0.7))


# Визуализируем

# In[333]:


x_values = np.arange(-3, 3, 0.01)

sigma = 0.7
y_values = [Gauss(x, sigma) for x in x_values]

plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, linewidth=1)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('G(x)')
plt.title('Функция Гаусса')
plt.show()


# Попробуем задать второе измерение и визуализировать

# In[334]:


x_values = np.arange(-3, 3, 0.01)
y_values = np.arange(-3, 3, 0.01)
X, Y = np.meshgrid(x_values, y_values)
sigma = 0.7

Z = np.zeros_like(X)
print(Z.shape)
for i in range(len(x_values)):
    for j in range(len(y_values)):
        Z[j, i] = Gauss(X[j, i], sigma) * Gauss(Y[j, i], sigma)

plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
contour = plt.contourf(X, Y, Z, levels=50, cmap='gray')
plt.colorbar(contour)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Двумерная функция Гаусса (произведение)')
plt.axis('equal')

ax = plt.subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='gray')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('G(x,y)')

plt.tight_layout()
plt.show()


# Как видим, образуется своебразный колокол. Интенсивность в центре выше. В этом строится основная идея применения фильтра. Если мы представим изображение, по которому будем проходиться апертурой, то станет ясно что то, что окажется в ее центре имеет больший вес. А то что к краям - там значение меньше, важность меньше. Соответственно, поэтому фильтр Гаусса является фильтром низких частот. Практически это значит, что если есть изображения с маленькими шумами, то фильтр их должен убрать, попробуем:

# In[335]:


import numpy as np
import matplotlib.pyplot as plt

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

minor_size = 3
X, Y, Z = gauss_kernel_visualize(0.7, minor_size)

plt.figure(figsize=(10, 6))
contour = plt.contourf(X, Y, Z, levels=50, cmap='gray')
plt.colorbar(contour)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Двумерная функция Гаусса (произведение)')
plt.axis('equal')
plt.show()
print(Z.shape)


# In[336]:


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


#plt.imshow(gaussian_2d(image1_RGB_intensity, 0.7, 3), cmap='gray')


# In[337]:


f, axarr = plt.subplots(1,2, figsize = (12,6))
axarr[0].imshow(image1_RGB_intensity, cmap='gray')
axarr[0].set_title('Исходная матрица')

image1_RGB_intensity_gaussian = gaussian_2d(image1_RGB_intensity, 3.7, 121)
axarr[1].imshow(image1_RGB_intensity_gaussian, cmap='gray')
axarr[1].set_title('После фильтра Гаусса')

print(image1_RGB_intensity.shape, image1_RGB_intensity_gaussian.shape)


# Супер, все работает. А что делать с RGB? То же что и с медианным. Сделаем все ровно то же самое, просто по каналам.

# In[338]:


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


# Отлично, можно визуализировать:

# In[339]:


image = image3_RGB
image1 = gaussian_2d_RGB(image3_RGB, 3.7, 121)
f, axarr = plt.subplots(1,2, figsize = (12,6))
axarr[0].imshow(image)
axarr[0].set_title('Исходная матрица')

axarr[1].imshow(image1)
axarr[1].set_title('После фильтра Гаусса')


# # 2. Морфологические операции

# По опредению: Морфология является широким набором операций обработки изображений, которые процесс отображает на основе форм. Морфологические операции применяют элемент структурирования к входному изображению, создавая выходное изображение, одного размера. В морфологической операции значение каждого пикселя в выходном изображении основано на сравнении соответствующего пикселя во входном изображении с его соседями. Источник: https://docs.exponenta.ru/images/morphological-dilation-and-erosion.html
# 

# ### 2.1 Эрозия
# 
# Значение выходного пикселя является минимальным значением всех пикселей в окружении. В бинарном изображении пиксель установлен в 0 если какой-либо из соседних пикселей имеет значение 0.
# 
# Морфологическая эрозия удаляет острова и маленькие объекты так, чтобы только независимые объекты остались.

# То есть, мы смотрим на соседей в окрестности апертуры относительно таргета - целевого пикселя. Если хоть один выполняет условие - таргет приобретает значение из условия. Импортируем новое изображение для этой главы:

# In[340]:


image4 = cv2.imread('sample_image4.jpg')
image4 = cv2.cvtColor(image4, cv2.COLOR_BGR2RGB)
plt.imshow(image4)


# Так же, как и раньше для простоты переведем в чернобелое изображений исходник. В этот раз я захотел использовать не карту инстенсивности, а реальный перевод в черно-белый формат с полутонами. По сути, та же карта интенсивности. Только в этот раз мы не просто будем усреднять значения по каналам, а использовать корректную фотограмметрическую формулу для такого перевода: Result = 0.299 R + 0.587 G + 0.114 B

# In[341]:


def intensity_grayscale(image):
    height, width, _ = image.shape
    image_intensity = np.zeros((height, width), dtype='uint8')

    for rows in range(image.shape[0]):
        for columns in range(image.shape[1]):
            #отталкиваемся от того, что исходник RGB и порядок каналов сохранен
            image_intensity[rows][columns] = 0.299*image[rows][columns][0] + 0.587*image[rows][columns][1] + 0.114*image[rows][columns][2]

    return image_intensity


# In[342]:


image = image4
image4_gray = intensity_grayscale(image)


# In[343]:


f, axarr = plt.subplots(1,2, figsize = (12,6))
axarr[0].imshow(image)
axarr[0].set_title('Исходная матрица')

axarr[1].imshow(image4_gray, cmap='gray')
axarr[1].set_title('После перевода')


# In[344]:


print(image4_gray.dtype)


# Теперь можем приступать к тому же алгоритму! У нас будет кернел - апертура, в окрестности которой мы будем сравнивать пиксели. Как проходить мы знаем, как добавлять паддинг для избежания потери краев изображения при свертке знаем. Принципиально один вопрос - как сравнивать. Наше изображение, как показано выше, закодировано в формате uint8. То бишь, глубина цвета 8 бит. Unsigned = нет знака. Соответственно максимум у нас каждый цвет кодируется 8 битами, а не 7. Значит значение интенсивности цвета лежит в промежутке от 0 до 2^8. 0...255. Где 256 - его нет, так как отчет мы ведем с нуля.

# Отлично, мы будем сравнивать значения пикселей под апертурой в диапазоне от 0 до 255. Соответственно, нужно то, с чем мы будем сравнивать, условие. Для этих целей вводим понятие - порог, threshold. Пускай этот порог будет иметь некоторое значение. Например, 122. Значит то, что будет в окрестности апертуры ниже или равно получит логически ноль. Если такие пиксели в окрестности есть - такое значение получит и таргет. Все что больше - на результат не повлияет.

# Стоит отметить, что та же логика будет применена к главе 2.2. Но в обратную сторону. То есть если больше или равно условию/порогу - повлияет на таргет. Это будет рассмотрено в следующей главе. 

# Приступим к реализации:

# In[345]:


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


#plt.imshow(gaussian_2d(image1_RGB_intensity, 0.7, 3), cmap='gray')


# In[346]:


image = image4_gray
image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4), 
                         interpolation=cv2.INTER_AREA) #зачем? исходное было очень большое, программа долго выполнялась. Хотите без потери времени работать с большими - делайте многопоток и кидайте PR, вообще капитальными красавчиками будете!

image4_gray_eroded = eroded_threshold(image, 42, 3)


# In[347]:


f, axarr = plt.subplots(1,2, figsize = (12,6))
axarr[0].imshow(image, cmap='gray')
axarr[0].set_title('Исходная матрица')

axarr[1].imshow(image4_gray_eroded, cmap='gray')
axarr[1].set_title('После перевода')


# Отлично! Как мы видим, при текущих параметрах, тонкие паутинки стали пропадать. При этом сам паук остался. Попробуем вообще избавиться от паутинок:

# In[348]:


image = image4_gray
image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4), 
                         interpolation=cv2.INTER_AREA) #зачем? исходное было очень большое, программа долго выполнялась. Хотите без потери времени работать с большими - делайте многопоток и кидайте PR, вообще капитальными красавчиками будете!

image4_gray_eroded = eroded_threshold(image, 68, 3)

f, axarr = plt.subplots(1,2, figsize = (12,6))
axarr[0].imshow(image, cmap='gray')
axarr[0].set_title('Исходная матрица')

axarr[1].imshow(image4_gray_eroded, cmap='gray')
axarr[1].set_title('После перевода')


# Получилось! Теперьв центре мы в состоянии выденить лишь самого паука, без паутины. То есть логика простая. Маленькие штуки - фильтруются. Чем тоньше цель - тем легче ее "съесть", проверяя то насколько она тонкая засчет соседей. Соотвественно размер апертуры - насколько хирургически мы действуем. А порог - таргет с которым мы сравниваем. Попробуем обратную операцию, расширение/диляция/дилатация.

# P.S. А как обстоят дела с RGB? Так же как с прошлыми. Но там мы применяем эту операцию отдельно к каждому из каналов. 
# Кстати, эрозия так же активно применяется к бинаризованным изображениям. О них упомянуто в главе 3.1. Посколько там порог задается и применяется на этапе бинаризации, функция эрозии или диляции уже не потребует задания какого либо порога. Там либо 0 у соседей ищем, либо 1 соответственно.

# ### 2.2 Дилатация a.k.a Диляция, Расширение
# 
# 
# Значение выходного пикселя является максимальным значением всех пикселей в окружении. В бинарном изображении пиксель установлен в 1 если какой-либо из соседних пикселей имеет значение 1.
# 
# Морфологическое расширение делает объекты более видимыми и заполняет маленькие отверстия в объектах.

# Используем прошлый код функции и перевернем логику. Гипотеза - паутинки должны стать толще. Пробуем:

# In[349]:


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



# In[350]:


image = image4_gray
image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4), 
                         interpolation=cv2.INTER_AREA) #зачем? исходное было очень большое, программа долго выполнялась. Хотите без потери времени работать с большими - делайте многопоток и кидайте PR, вообще капитальными красавчиками будете!

image4_gray_dilated = dilated_threshold(image, 122, 3)

f, axarr = plt.subplots(1,2, figsize = (12,6))
axarr[0].imshow(image, cmap='gray')
axarr[0].set_title('Исходная матрица')

axarr[1].imshow(image4_gray_dilated, cmap='gray')
axarr[1].set_title('После перевода')


# Выводы те же самые с проправкой на то, что операция обратная. Гипотеза верна, ч.т.д.

# Ради эксперимента в главе 3.1. применим диляцию к бинаризованному изображению для наглядности.

# # 3. Прочие операции

# ### 3.1 Пороговая бинаризация (для rgb и grayscale изображения)

# По определению:
# Процесс бинаризации – это перевод цветного (или в градациях серого) изображения в двухцветное черно-белое. Главным параметром такого преобразования является порог t – значение, с которым сравнивается яркость каждого пикселя. По результатам сравнения, пикселю присваивается значение 0 или 1. Существуют различные методы бинаризации, которые можно условно разделить на две группы – глобальные и локальные. В первом случае величина порога остается неизменной в течение всего процесса бинаризации. Во втором изображение разбивается на области, в каждой из которых вычисляется локальный порог.
# Источник: https://habr.com/ru/articles/278435/

# То есть, есть некоторый порог с которым мы сравниваем каждый пиксель изображения. Наш случай - первый, глобальный, простой. Меньше порога - присваиваем пикселю ноль, больше порога - единицу. Стоит отметить сразу, что для RGB логика та же, просто отдельно по каналам мы проверяем попиксельно интенсивности. 

# Начнем импортируя новое изображение и выделяя интересующую нас область:

# In[351]:


image5 = cv2.imread('sample_image5.jpg')
image5 = cv2.cvtColor(image5, cv2.COLOR_BGR2RGB)
plt.imshow(image5)


# In[352]:


image5_gray = intensity_grayscale(image5)
image5_gray_cropped = image5_gray[300:600, 200:600]

plt.imshow(image5_gray_cropped, cmap='gray')
plt.show()


# Супер! Теперь попробуем пройтись по изображению с порогом в 125. Реализуем функцию:

# In[353]:


def binarized(image, threshold):
    image1 = image.copy()

    for rows in range (image1.shape[0]):
            for columns in range(image1.shape[1]):
                if image1[rows][columns] <= threshold:
                    image1[rows][columns] = 0
                else:
                    image1[rows][columns] = 1

    return image1


# In[354]:


image = image5_gray_cropped
image_binarized = binarized(image, 125)

f, axarr = plt.subplots(1,2, figsize = (12,6))
axarr[0].imshow(image, cmap='gray')
axarr[0].set_title('Исходная матрица')

axarr[1].imshow(image_binarized, cmap='gray')
axarr[1].set_title('После перевода')


# Как видим, то что было светлее 125 стало единицей - белым - неважным. То, что было равно или темнее 125 стало выразительным. Отдельные элементы лица, с которыми нам, возможно, пришлось бы работать стали отдельными и их отделить от остального изображения станет легче.

# Как обещал, попробуем избавиться от неточностей при помощи морфологических операций. Внимание на нос - я хочу оставить ноздрю ии попробовать избавиться от остальных ненужных частей. Применим эрозию:

# Для этого поменяем функцию эрозии для бинаризованных изображений:

# In[355]:


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


# In[356]:


image_binarized_eroded = eroded_bin(image_binarized, 5)

f, axarr = plt.subplots(1,2, figsize = (12,6))
axarr[0].imshow(image_binarized, cmap='gray')
axarr[0].set_title('Исходная матрица')

axarr[1].imshow(image_binarized_eroded, cmap='gray')
axarr[1].set_title('После перевода')


# Получилось! Но мы слишком сильно избавились от частей носа. Наш таргет - ноздря тоже пострадала. Попробуем применить диляцию:

# In[357]:


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


# In[358]:


image_binarized_eroded_dilated = dilated_bin(image_binarized_eroded, 3)

f, axarr = plt.subplots(1,2, figsize = (12,6))
axarr[0].imshow(image_binarized_eroded, cmap='gray')
axarr[0].set_title('Исходная матрица')

axarr[1].imshow(image_binarized_eroded_dilated, cmap='gray')
axarr[1].set_title('После перевода')


# Отлично. Ноздря на месте и мы значительно избавились от лишних деталей. Вернемся к RGB:

# Попробуем сделать так же, но для RGB. Пускай для каждого канала мы зададим отдельно свой порог

# Определим для этого функцию:

# In[359]:


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


# In[360]:


image5_binarized_RGB = binarized_RGB(image5, 125, 125, 125)
print(image5_binarized_RGB.shape)

image_R, image_G, image_B = np.split(image5_binarized_RGB, 3, axis=2)
image_R = image_R.squeeze()
image_G = image_G.squeeze()
image_B = image_B.squeeze()

f, axarr = plt.subplots(1,2, figsize = (12,6))
axarr[0].imshow(image5)
axarr[0].set_title('Исходная матрица')

image5_binarized_RGB_display = (image5_binarized_RGB * 255).astype(np.uint8) # нормализуем в диапазон 0...255 для RGB отображения
axarr[1].imshow(image5_binarized_RGB_display)
# axarr[1].imshow(image_R, cmap='gray')
axarr[1].set_title('После перевода')


# ### 3.2 Выравнивание гистограммы

# По определению:
# Операция выравнивания гистограмм (увеличение контраста) часто используется для увеличения качества изображения.
# Гистограмма представляет из себя функцию h(x), которая возвращает суммарное количество пикселей, яркость которых равна x.
# 
# Гистограмма h полутонового изображения I задается выражением:

# In[361]:


img = cv2.imread('./histfunc.png')
plt.imshow(img)


# , где m соответствует интервалам значений яркости

# Визуально гистограмма представляет из себя прямоугольник, ширина которого равна максимально возможному значению яркости точки на исходном изображении. Для полутоновых изображений мы будем работать с диапазоном яркостей точек от 0 до 255, а значит и ширина гистограммы будет равна 256. Высота гистограммы может быть любой, но для наглядности мы будем работать с прямоугольными гистограммами.
# 
# С точки зрения программиста, гистограмма — это одномерный массив размерностью 256 (в нашем случае), где каждый элемент массива хранит в себе суммарное количество точек соответствующей яркостью. 

# Надо визуализировать. Сначала попробуем вывести гистограмму изображения:

# In[362]:


def image_hist(image):
    counts = dict.fromkeys(range(256), 0)

    for rows in range (image.shape[0]):
            for columns in range(image.shape[1]):
                counts[image[rows][columns]] += 1
    return counts

image = image4_gray
print(np.min(image), np.max(image))
counts = image_hist(image)

names = list(counts.keys())
values = list(counts.values())

plt.bar(names, values)
plt.show()


# Видно, что функция распределения распределена не равномерно.
# 
# В начале стабильные значения. Примерно вторую четверть графика она близка в нулю (относительно вертикальной координаты), потом идет небольшой рост и в последней четверти графика все спокойно за исключением конца, там опять резкий рост. Процедура выравнивания гистограммы заключается в том, чтобы сделать функцию распределения более равномерной, чтобы она возрастала примерно одинаково во всем своем диапазоне.

# Посчитаем PDF(вероятности) для каждой интенсивности. Напоминаю, что у нас 256 интенсивностей, а промежуток значений от 0 до 255.

# In[363]:


image = image4_gray
counts = image_hist(image)
total_pixels = image.shape[0]*image.shape[1]

names = list(counts.keys())
values = list(counts.values())
probabilities = list()

for i in names:
    temp = counts[i]/total_pixels
    probabilities.append(counts[i]/total_pixels)

names = list(counts.keys())
values = probabilities

plt.bar(names, values)
plt.show()



# Теперь считаем кумулятиву (CDF)

# In[364]:


cdf = [0] * 256
cumulative = 0.0

for i in range(256):
    cumulative += probabilities[i]
    cdf[i] = cumulative

names = list(counts.keys())
values = cdf

plt.bar(names, values)
plt.show()


# Все по теории, все отлично, финальная кумулятива = 1. Теперь попробуем выравнять. То есть пиксель с интенсивностью умножаем на соответствующую этой интенсивности кумулятиву.

# In[365]:


image_equalized = np.zeros_like(image)

for rows in range(image.shape[0]):
    for columns in range(image.shape[1]):
        r = image[rows][columns]
        s = int(round(cdf[r] * 255)) #зачем округление? у нас интенсивности - целые числа, а не дробные. А кумулятивы дробные.
        image_equalized[rows][columns] = s

f, axarr = plt.subplots(1,2, figsize = (12,6))
axarr[0].imshow(image, cmap='gray')
axarr[0].set_title('Исходная матрица')

image_hist_equalized = image_equalized
#image_hist_equalized = (image_hist_equalized * 255).astype(np.uint8) # нормализуем в диапазон 0...255 для RGB отображения
axarr[1].imshow(image_hist_equalized, cmap='gray')
# axarr[1].imshow(image_R, cmap='gray')
axarr[1].set_title('После перевода')


# как видим, изображение было чересчур темным, а стало достаточно сбланасированно ярким. Посмотрим на гистограммы:

# In[366]:


counts = image_hist(image)
total_pixels = image.shape[0]*image.shape[1]

names = list(counts.keys())
values = list(counts.values())

names = list(counts.keys())
values = list(counts.values())

plt.bar(names, values)
plt.show()


# In[367]:


image = image_hist_equalized
counts = image_hist(image)
total_pixels = image.shape[0]*image.shape[1]

names = list(counts.keys())
values = list(counts.values())

names = list(counts.keys())
values = list(counts.values())

plt.bar(names, values)
plt.show()


# Как видно, цель выполнена, гистограмма стала значительно равнее. Попробуем на RGB:

# In[368]:


def hist_equalize(image):
    counts = dict.fromkeys(range(256), 0)

    for rows in range (image.shape[0]):
            for columns in range(image.shape[1]):
                counts[image[rows][columns]] += 1

    total_pixels = image.shape[0]*image.shape[1]

    probabilities = list()

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


# In[369]:


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


# In[370]:


image = image4
image_hist_equalized = hist_equalize_RGB(image)
f, axarr = plt.subplots(1,2, figsize = (12,6))
axarr[0].imshow(image)
axarr[0].set_title('Исходная матрица')

axarr[1].imshow(image_hist_equalized)
axarr[1].set_title('После перевода')


# ### 3.3 поворот изображений на угол кратный 90 градусов

# Что такое поворот? По своей сути это смена рядов и столбцов исходной матрицы местами. То есть, транспонирование в каком то смысле. Давайте попробуем...

# In[371]:


plt.imshow(image4_gray, cmap='gray')


# In[372]:


image4_gray_transposed = np.transpose(image4_gray)
plt.imshow(image4_gray_transposed, cmap='gray')


# Получилось! Мы повернули картинку на -90 градусов. Попробуем повращать дальше:

# In[373]:


image4_gray_transposed = np.transpose(image4_gray_transposed)
plt.imshow(image4_gray_transposed, cmap='gray')


# Не вышло. Функция транспонирования реверсивна. Если ее применить повторно, то ряды и столбцы просто поменяются местами обратно. Как быть?

# Вообще, подсказка кроется в первом результате транспонирования. Если бы паучок смотрел в противоположную сторону, то это был бы поворот на +90 градусов, то есть, по часовой. Попробуем еще раз:

# In[374]:


image4_gray_transposed = np.transpose(image4_gray)
for rows in range(len(image4_gray_transposed)):
    curr_row = image4_gray_transposed[rows]
    rev_row = curr_row[::-1]
    image4_gray_transposed[rows] = rev_row

plt.imshow(image4_gray_transposed, cmap='gray')


# Что-то пошло не так... изображение отзеркалено по горизонтали. Паучок должен был смотреть наверх. Получается, мы перепутали порядок операций. Надо было разворачивать ряды, а мы развернули "столбцы" изначального изображения, потому что работали над транспонированным. попробуем изменить порядок действий и провести проверку:

# In[375]:


image = intensity_grayscale(image4)

image_90_ccw = image.copy()
image_90_cw = image.copy()

image_90_ccw = np.transpose(image_90_ccw)
#plt.imshow(image_90_ccw, cmap='gray')

for rows in range(len(image_90_cw)):
    curr_row = image_90_cw[rows]
    rev_row = curr_row[::-1]
    image_90_cw[rows] = rev_row

image_90_cw = np.transpose(image4_gray)

#plt.imshow(image_90_cw, cmap='gray')


# In[376]:


f, axarr = plt.subplots(1,3, figsize = (12,6))

axarr[0].imshow(image, cmap='gray')
axarr[0].set_title('изначально')

axarr[1].imshow(image_90_cw, cmap='gray')
axarr[1].set_title('по часовой')

axarr[2].imshow(image_90_ccw, cmap='gray')
axarr[2].set_title('против часовой')


# Супер, получилось! Получается, применив реверс рядов и транспонируя результат мы вращаем результат на 90 градусов по часовой стрелке. Сколько раз это будет сделано можно задать целым числом - как параметр для этой функции. Оформим как функцию для grayscale. Для RGB так же, как и везде делается ровно то же самое, просто по каждому из каналов...

# In[377]:


def rotate_90_cw(image, amount):
    image_90_cw = image.copy()

    for i in range(amount):
        image_90_cw = np.transpose(image_90_cw)

        for rows in range(len(image_90_cw)):
            curr_row = image_90_cw[rows]
            rev_row = curr_row[::-1]
            image_90_cw[rows] = rev_row

    return image_90_cw

image = image5_gray
plt.imshow(image, cmap='gray')



# In[378]:


plt.imshow(rotate_90_cw(image, 4), cmap='gray')


# In[379]:


plt.imshow(rotate_90_cw(image, 3), cmap='gray')


# In[380]:


plt.imshow(rotate_90_cw(image, 2), cmap='gray')


# In[381]:


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

image = image5
f, axarr = plt.subplots(1,3, figsize = (12,6))

axarr[0].imshow(image)
axarr[0].set_title('изначально')

axarr[1].imshow(rotate_90_cw_rgb(image, 1))
axarr[1].set_title('по часовой 1 раз')

axarr[2].imshow(rotate_90_cw_rgb(image, 2))
axarr[2].set_title('вверх ногами')


# # Вывод

# Все получилось. Кто сдох - тот лох. Кто дочитал - красавчик. Передохни и иди читай вторую лабу.
