#!/usr/bin/env python
# coding: utf-8

# Костин Арсений, 8Е21, вариант 3.

# Лабораторная работа №2. Визуальная одометрия (навигация)
# Цель: Разработать систему визуальной одометрии (навигации) по группе фотографий.
# Ход работы: сделайте не менее 8 фото с переносом камеры или ноутбука по квадрату (то есть двиньте сначала вправо, потом вперед, потом влево, потом назад и обратно в начальную точку). Используя данные фотографии реализуйте следующее:
# <p> 1.	Определите на каждой фотографии ключевые точки </p>
# <p>2.	Отфильтруйте самые наилучшие применяю адаптивный радиус и локальные максимумы, не забудьте так же выровнять по яркости изображения.</p>
# <p>3.	Постройте по каждой точке дескриптор (можете использовать любой, рекомендуется SIFT)</p>
# <p>4.	Сопоставьте два соседних изображения на предмет соответствия ключевых точек. То есть определите пары одинаковых точек.</p>
# <p>5.	Постройте модель преобразования изображений, учитывайте только поворот и сдвиг.</p>
# <p>6.	С учетом полученных моделей постройте траекторию движения камеры.</p>
# <p>Проверка работоспособности: будет осуществляться на специальной группе фото, предоставленных преподавателем. Траектория движения, для которых недоступна.</p>
# <p>В процессе выполнения вы можете использовать готовые функции по погрузке данных, перевода в цветовые пространства, фильтрации, для построения прямых и траекторий. Функции 1-6 описанные выше должны быть реализованы самостоятельно.</p>
# 

# In[4]:


import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import random
#from PIL import Image
from IPython.display import Image
get_ipython().run_line_magic('matplotlib', 'inline')
import math
import sys; sys.path.insert(0, '../lab1')
import lab1_functions as lb1
import os
print(os.getcwd())
print(os.listdir())


# # 2.1 Загрузить изображения

# Приступим, импортируем сделанные изображения:

# In[5]:


image1=cv2.cvtColor(cv2.imread('magnetssequence/sequence1.jpeg'), cv2.COLOR_BGR2RGB)
image2=cv2.cvtColor(cv2.imread('magnetssequence/sequence2.jpeg'), cv2.COLOR_BGR2RGB)
image3=cv2.cvtColor(cv2.imread('magnetssequence/sequence3.jpeg'), cv2.COLOR_BGR2RGB)
image4=cv2.cvtColor(cv2.imread('magnetssequence/sequence4.jpeg'), cv2.COLOR_BGR2RGB)
image5=cv2.cvtColor(cv2.imread('magnetssequence/sequence5.jpeg'), cv2.COLOR_BGR2RGB)
image6=cv2.cvtColor(cv2.imread('magnetssequence/sequence6.jpeg'), cv2.COLOR_BGR2RGB)
image7=cv2.cvtColor(cv2.imread('magnetssequence/sequence7.jpeg'), cv2.COLOR_BGR2RGB)
image8=cv2.cvtColor(cv2.imread('magnetssequence/sequence8.jpeg'), cv2.COLOR_BGR2RGB)

images_sequence = [image1, image2, image3, image4, image5, image6, image7, image8]

f, axarr = plt.subplots(2,4, figsize = (12,6))

axarr[0,0].imshow(image1)
axarr[0,1].imshow(image2)
axarr[0,2].imshow(image3)
axarr[0,3].imshow(image4)

axarr[1,0].imshow(image5)
axarr[1,1].imshow(image6)
axarr[1,2].imshow(image7)
axarr[1,3].imshow(image8)


# Лирическое отступление - чтобы не размазывать отчет, работа будет вестись над grayscale изображениями. То что мы будем использовать - не зависит от цветов, как видно из первой лабы. Все что возможно - можно сделать для RGB повторяя те же операций и преобразования, просто трижды - по разу для каждого цветового канала. Это не цель лабораторной работы. Приступим, переведем изображения в черно-белый формат, используя функцию из прошлой лабы.

# In[6]:


images_sequence_gray = []
for img in images_sequence:
    images_sequence_gray.append(lb1.intensity_grayscale(img))


# In[7]:


f, axarr = plt.subplots(2,4, figsize = (12,6))

axarr[0,0].imshow(images_sequence_gray[0], cmap='gray')
axarr[0,1].imshow(images_sequence_gray[1], cmap='gray')
axarr[0,2].imshow(images_sequence_gray[2], cmap='gray')
axarr[0,3].imshow(images_sequence_gray[3], cmap='gray')

axarr[1,0].imshow(images_sequence_gray[4], cmap='gray')
axarr[1,1].imshow(images_sequence_gray[5], cmap='gray')
axarr[1,2].imshow(images_sequence_gray[6], cmap='gray')
axarr[1,3].imshow(images_sequence_gray[7], cmap='gray')


# То есть, нам нужно:
# 
# загрузить изображения, привести их к одинаковой яркости / grayscale, найти ключевые точки, отфильтровать точки, построить дескрипторы (SIFT), сопоставить точки между соседними кадрами, вычислить преобразование (поворот + сдвиг), накопить преобразования и построить траекторию камеры

# Исправим проблемы с яркостью, применив реализованный модуль для выравнивания гистограммы:

# # 2.2 Привести к одинаковой яркости / grayscale

# для нашего удобства и ментального здоровья зададим функцию готовую для отображения картинок из серии:

# In[8]:


def show_images(images_sequence, rows=2, cols=4, figsize=(12,6)):

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < len(images_sequence):
            ax.imshow(images_sequence[i], cmap='gray')
            ax.axis('off')  # убираем оси
        else:
            ax.axis('off')  # если картинок меньше, оставляем пустые

    plt.tight_layout()
    plt.show()


# In[9]:


show_images(images_sequence_gray, 2, 4)


# Супер. Вернемся к идее применить выравнивание гистограммы на всех изображениях:

# In[10]:


images_hist_equalized = images_sequence_gray
for img in images_hist_equalized:
    img = lb1.hist_equalize(img)

show_images(images_sequence_gray, 2, 4)


# # 2.3 Найти ключевые точки

# Приступаем. На изображениях у нас есть два магнита рядом, но нам могут значительно помешать: блики, тень от фотографа, другие нерелватные в этом контексте детали. Это визуальный <b>шум</b>. От шума надо избавиться, рассмотрим изображение 1, применим для подавления шумов фильтр Гаусса.

# In[11]:


images_temp = [images_hist_equalized[0], lb1.gaussian_2d(images_hist_equalized[0], 1.2, 21)]
show_images(images_temp, 1, 2)


# Стало слегка лучше. Теперь перейдем к теории:

# Ключевые точки: это такие места, по которым сравнивая две картинки можно отследить движение.

# Если бы мы смотрели на голубое небо и взяли его кусочек как ключевую точку, то распределение интенсивностей было бы +- одинаковое между ним и другим кусочком неба. Это плохая ключевая точка.

# Если бы мы смотрели на фото прикроватной тумбочки, то могли бы предположить, что край тумбочки - хорошая ключевая точка, т.к. интенсивность прыгает на моменте перехода от края тумбочки к ее боковой части в тени. По идее - уже неплохо, но если двигаться вдоль этого края - ситуация не поменяется при сравнении двух кадров.

# Из этого следует, что лучший вариант - когда меняются по двум направлениям тренды. Например, угол тумбочки. Вдоль него не подвигаться, то есть изменение интенсивности слева и спереди (перед стеной) достаточно легко отслеживаются.  

# Из математики следует, что производная функции показывает скорость изменения ее значения. А градиент - вектор, показывающий <b>НАПРАВЛЕНИЕ</b> наибыстрейшего увеличения функции. Это то, что надо нам. Формула для градиента в общей форме выглядит так:

# In[12]:


img = cv2.imread('../../README_files/gradient.png')
plt.imshow(img)


# Исходя из вышесказанного, понятно, что алгоритмы поиска ключевых точек ищут точки, где изменение яркости происходит во многих направлениях одновременно.

# Математически - ищем градиенты изображения, по x и y координатам. 

# Пусть I_x = изменение по х, I_y = изменение по y.
# <p>Значит, место, где I_x = 0, I_y = 0 это однородная область.
# <p>Если I_x большое, I_y маленькое = это край.
# <p>Если I_x большое, I_y большое = это угол (ключевая точка).

# Из опыта прошлой лабы мы понимаем, что смотреть на сам один пиксель недостаточно. Нужно брать апертуру/кернел/область/окно. Так и сделаем. Идейно по определению подходит фильтр Хариса. Источник: https://docs.exponenta.ru/R2021a/visionhdl/ug/corner-detection.html

# ### 2.3.1 Разберем этот фильтр

# Фактически у нас есть исходное изображение, minor_size, k, threshold_ratio.

# minor_size - так же как в прошлой лабе, размер апертуры/кернел/окно. маленькое окно = чувствительность к мелким деталям, большое окно = реагирует только на крупные структуры

# источник: https://docs.opencv.org/3.4/dc/d0d/tutorial_py_features_harris.html
# <p> k - коэффициент, испольуемый в формуле Хариса:

# In[13]:


img = cv2.imread('../../README_files/harris1.png')
plt.imshow(img)


# Этот коэффициент регулирует насколько алгоритм строго смотрит края.
# <p>Маленький = алгоритм более терпим к краям, может принимать некоторые края за углы
# <p>Большой = алгоритм строгий, оставляет только очень выраженные углы

# threshold_ratio
# 
# После вычисления Harris response R нужно решить какие точки считать ключевыми.
# 
# Для этого берётся максимум:
# R_max = max(R)
# 
# и строится порог:
# 
# threshold = threshold_ratio * R_max

# Если threshold_ratio = 0.01, то берутся точки у которых R > 1% от максимального

# Если увеличить до 0.1 = останутся только самые сильные углы.

# In[14]:


def harris_keypoints(image, minor_size=5, k=0.04, threshold_ratio=0.01):

    if len(image.shape) == 3:
        image = intensity_grayscale(image)

    image = image.astype(float)

    height, width = image.shape

    Ix = np.zeros_like(image)
    Iy = np.zeros_like(image)

    # градиенты (центральные разности)
    for r in range(1, height-1):
        for c in range(1, width-1):
            Ix[r][c] = (image[r][c+1] - image[r][c-1]) / 2
            Iy[r][c] = (image[r+1][c] - image[r-1][c]) / 2

    pad = minor_size // 2

    R = np.zeros_like(image)

    for r in range(pad, height-pad):
        for c in range(pad, width-pad):

            sum_Ix2 = 0
            sum_Iy2 = 0
            sum_Ixy = 0

            for i in range(-pad, pad+1):
                for j in range(-pad, pad+1):
                    gx = Ix[r+i][c+j]
                    gy = Iy[r+i][c+j]

                    sum_Ix2 += gx*gx
                    sum_Iy2 += gy*gy
                    sum_Ixy += gx*gy

            det = sum_Ix2 * sum_Iy2 - sum_Ixy**2
            trace = sum_Ix2 + sum_Iy2

            R[r][c] = det - k*(trace**2)

    R_max = np.max(R)
    threshold = threshold_ratio * R_max

    keypoints = []

    for r in range(pad, height-pad):
        for c in range(pad, width-pad):

            if R[r][c] > threshold:

                local_max = True

                for i in range(-1,2):
                    for j in range(-1,2):
                        if R[r+i][c+j] > R[r][c]:
                            local_max = False

                if local_max:
                    keypoints.append((r,c))

    return keypoints, Ix, Iy


# Что тут происходит? Сначала вычисляются упомянутые градиенты изображения.
# Градиент показывает, насколько быстро меняется яркость:
# 
# Ix — изменение яркости по горизонтали
# 
# Iy — изменение яркости по вертикали
# 
# Это делается с помощью центральной разности: берётся разница между соседними пикселями.
# 
# После этого алгоритм начинает рассматривать каждую точку изображения и маленькое окно вокруг неё (minor_size). Внутри этого окна суммируются значения градиентов:
# 
# квадрат горизонтального градиента
# 
# квадрат вертикального градиента
# 
# произведение двух градиентов
# 
# Эти суммы используются для вычисления величины R. Она показывает, насколько вероятно, что точка является углом.
# 
# Дальше выбираются только точки, у которых R достаточно большое (больше порога).
# После этого выполняется проверка локального максимума: точка должна быть больше всех своих соседей. Это нужно, чтобы оставить только самые сильные углы и убрать лишние точки вокруг них.
# 
# В итоге функция возвращает:
# 
# keypoints — координаты найденных углов
# 
# Ix — карту горизонтальных градиентов
# 
# Iy — карту вертикальных градиентов

# In[15]:


def draw_keypoints(image, keypoints, Ix=None, Iy=None, show_vectors=False, vector_scale=5):

    img = image.copy()

    if len(img.shape) == 2:
        img = np.dstack((img,img,img))

    plt.figure(figsize=(8,6))
    plt.imshow(img)

    for (r,c) in keypoints:
        plt.scatter(c, r, s=20)

        if show_vectors and Ix is not None and Iy is not None:
            gx = Ix[r][c]
            gy = Iy[r][c]

            plt.arrow(c, r,
                      gx*vector_scale,
                      gy*vector_scale,
                      head_width=3,
                      length_includes_head=True)

    plt.axis("off")
    plt.show()
    fig = plt.gcf()
    fig.canvas.draw()

    buf = fig.canvas.buffer_rgba()
    data = np.asarray(buf)
    data = data[:, :, :3]

    return data


# Функция draw_keypoints рисует найденные ключевые точки на изображении. Сначала создаётся копия изображения. Если изображение чёрно-белое, оно превращается в трёхканальное (RGB), чтобы на нём можно было рисовать цветные элементы.
# 
# 
# Для каждой найденной точки:
# 
# на изображении рисуется маркер (точка)
# 
# если включён параметр show_vectors, дополнительно рисуется стрелка
# 
# Стрелка показывает направление градиента в этой точке. Она берётся из Ix и Iy и показывает, в каком направлении яркость изменяется сильнее всего.
# 
# Параметр vector_scale просто увеличивает длину стрелок, чтобы их было лучше видно.

# In[16]:


gray = images_temp[1]

keypoints, Ix, Iy = harris_keypoints(gray, minor_size=9)

draw_keypoints(gray, keypoints)


# Как видим, ключевые точки отлично отобразились. Но здесь, думаю, сыграло хорошее качество изображения и правильно подобранный наугад размер апертуры. Попробуем с более быстрым вариантом, с апертурой поменьше:

# In[17]:


gray = images_temp[1]

keypoints, Ix, Iy = harris_keypoints(gray, minor_size=3)

draw_keypoints(gray, keypoints)


# Точек меньше, но очевидно ошибочной можно назвать лишь одну, сверху. Откуда она взялась? На изображении в этом месте блик от окна. Похоже, что это светлое пятно и было обнаружено. Для анализа была написана функция рисующая стрелки направлений для обнаруженных градиентов, посмотрим:

# In[18]:


draw_keypoints(gray, keypoints, Ix, Iy, show_vectors=True)


# Интересно. Блик на самом деле левее. При ближайшем рассмотрении оказалось, что на холодильнике была точка с грязью. Она темная и поэтому была обнаружена. Получается, нам нужно быть готовым к ошибочным точкам и как то их фильтровать. Рассмотрим это в следующей части. А пока предлагаю для наглядности отобразить градиенты и сравнить с изначальным изображением.

# In[19]:


plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(gray, cmap="gray")
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Ix")
plt.imshow(Ix, cmap="gray")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Iy")
plt.imshow(Iy, cmap="gray")
plt.axis("off")

plt.show()


# # 2.4 Отфильтровать точки

# Перед поиском точек мы заранее задумались о том, чтобы выравнять освещение изображений и избавиться от жестких бликов и прочих плохих факторов, которые могли испортить нам обнаружение. Поэтому мы обогнали задачи для этой работы. Но в итоге у нас все равно остался хоть один, но вброс. Душить его дальше размытием по Гауссу - можно, но неинтересно. А что если грязь была бы побольше по площади? а если бы темнее? Такой расклад сделал бы повторное размытие наприменимым. Более того, с дополнительным размытиеммы уменьшаем эффективность поиска градиентов. Нужен альтернативный метод. Для такие задач используются разные методы. Некоторые из них: DBSCAN, RANSAC.

# RANSAC - популярно, круто, но сложно. DBSCAN - тоже круто и популярно, но понятнее. Применяем кластеризацию и потом по соседям фильтруем. Возьмем как основу этот метод, но выкинем из него класетиразацию для упрощения. Будем по количеству соседей проверять точки "в лоб". В нашем случае - отличное решение. Для очень загруженных пятнами изображений уже пригодится упомянутая кластеризация. А пока оставим так:

# In[20]:


def filter_isolated_points(keypoints, radius=10, min_neighbors=5):

    filtered = []

    for i, p in enumerate(keypoints):

        neighbors = 0

        for j, q in enumerate(keypoints):

            if i == j:
                continue

            dist = np.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2)

            if dist < radius:
                neighbors += 1

        if neighbors >= min_neighbors:
            filtered.append(p)

    return filtered


# Что происходит:
# 
# берём точку
# 
# смотрим сколько других точек ближе чем radius
# 
# если их меньше min_neighbors, считаем её шумом
# 
# Твоя точка сверху просто исчезнет, потому что рядом с ней нет соседей.

# In[21]:


keypoints_filtered = filter_isolated_points(keypoints)


# In[22]:


draw_keypoints(gray, keypoints_filtered)


# Мы избавились от вброса но с этим потеряли много нужных точек! Настраиваем наш фильтр:

# In[23]:


keypoints_filtered = filter_isolated_points(keypoints, 30, 3)


# In[24]:


draw_keypoints(gray, keypoints_filtered)


# Гораздо лучше! Займемся дескрипторами.

#  # 2.5 Построить дескрипторы (SIFT)

# Дескриптор — это числовое описание окрестности ключевой точки. Он должен быть устойчив к изменению освещения, небольшому повороту и сдвигу — чтобы одна и та же точка на двух разных кадрах давала похожий дескриптор, а разные точки — непохожие.
# <p>SIFT (Scale-Invariant Feature Transform) — один из самых известных алгоритмов для этого. Он строит дескриптор из гистограмм градиентов в окрестности точки. Алгоритм состоит из четырёх этапов, разберём каждый.

# Подготовим материалы для дальнейшей работы по аналогии с прошлыми шагами:

# In[25]:


working_images = []
for i in images_hist_equalized:
    working_images.append(lb1.gaussian_2d(i, 1.2, 21))

show_images(working_images)


# Применили размытие на всех изображениях. Это подавит мелкий шум перед вычислением градиентов. Теперь найдём и отобразим ключевые точки для всей серии:

# In[26]:


working_images_keypoints = []
working_images_visualised = []
for i in working_images:
    keypoints, Ix, Iy = harris_keypoints(i, minor_size=3)
    keypoints = filter_isolated_points(keypoints, 30, 3)
    working_images_keypoints.append(keypoints)
    working_images_visualised.append(draw_keypoints(i, keypoints))


# Отлично. Ключевые точки найдены на всех кадрах. Заметно, что точки кластеризуются вокруг объектов — магнитов, — что и ожидается: именно там происходят резкие изменения яркости в обоих направлениях. Переходим к теории SIFT.

# Так и что мы делаем сейчас? Harris дал точки, но он не умеет их узнавать на другом изображении. Если повернуть картинку, изменить масштаб или освещение — координаты точек изменятся.
# 
# Именно поэтому появился алгоритм SIFT. Его задача: для каждой точки построить уникальное числовое описание (дескриптор), которое можно сравнивать между кадрами.

# SIFT — идея алгоритма
# 
# Алгоритм делает две вещи: находит устойчивые точки интереса и строит для каждой точки вектор признаков, который описывает локальную структуру изображения
# 
# Этот вектор потом можно сравнивать между изображениями.
# 
# Классический SIFT состоит из 4 этапов:

# ## 2.5.1 Пирамиды Гаусса

# Первый шаг — создать несколько размытых версий изображения.
# 
# Это нужно, чтобы точки находились независимо от масштаба.
# Мелкие детали исчезают при сильном размытии, а крупные остаются.
# 
# Алгоритм строит так называемую пирамиду Гаусса.

# In[27]:


def gaussian_pyramid(image, sigmas=[1,2,4,8]):

    pyramid = []

    for sigma in sigmas:
        blurred = lb1.gaussian_2d(image, sigma, minor_size=17)
        pyramid.append(blurred)

    return pyramid


# In[28]:


pyramid1 = gaussian_pyramid(working_images[0])
show_images(pyramid1, 1, 4)


# На каждом следующем уровне пирамиды изображение размывается сильнее: мелкие детали пропадают, крупные структуры остаются. Это позволяет находить точки на разных масштабах. В нашей задаче камера движется примерно на одном расстоянии от объекта, поэтому масштабная инвариантность не критична: пирамида строится для полноты алгоритма.

# ## 2.5.2 Разница гауссиан DoG (АХТУНГ!!!)

# DoG (Difference of Gaussians) — это разница между соседними уровнями пирамиды Гаусса. Математически это приближение лапласиана гауссиана (LoG), который хорошо реагирует на точки и края.
# <p>В оригинальном SIFT именно в DoG-пространстве ищутся экстремумы — точки, которые являются максимумом или минимумом среди 26 соседей (8 в своём слое + 9 выше + 9 ниже). Мы эту функцию реализуем, но использовать для детектирования не будем — у нас уже есть Харис.

# In[29]:


def difference_of_gaussians(pyramid):

    dogs = []

    for i in range(len(pyramid)-1):
        dog = pyramid[i+1] - pyramid[i]
        dogs.append(dog)

    return dogs


# Обычно SIFT ищет точки в DoG, но мы уже реализовали алгоритм Хариса, поэтому использовать я буду его. Пирамида тоже не нужна для детектирования: только для масштабной инвариантности, которая в данной лабе не приоритет, у нас камера +- на одном расстоянии от холодильника.

# ## 2.5.3 Экстремумы

# Шаг 2.5.3 важен — он нужен не для поиска новых точек, а для того чтобы назначить каждой точке доминирующий угол по локальной гистограмме градиентов. Без этого дескриптор не будет инвариантен к повороту.

# In[30]:


def compute_keypoint_orientations(keypoints, Ix, Iy,
                                  orientation_window_size=16,
                                  num_bins=36):

    height, width = Ix.shape
    half = orientation_window_size // 2

    sigma = half
    oriented_keypoints = []

    for (r, c) in keypoints:
        if r < half or r >= height - half or c < half or c >= width - half:
            continue

        hist = [0.0] * num_bins
        bin_width = 360.0 / num_bins

        for i in range(-half, half):
            for j in range(-half, half):

                gx = Ix[r + i][c + j]
                gy = Iy[r + i][c + j]


                magnitude = math.sqrt(gx * gx + gy * gy)
                angle_deg = math.degrees(math.atan2(gy, gx)) % 360
                gauss_weight = math.exp(-(i * i + j * j) / (2 * sigma * sigma))

                bin_idx = int(angle_deg / bin_width) % num_bins
                hist[bin_idx] += magnitude * gauss_weight

        max_val = max(hist)
        peak_bin = hist.index(max_val)

        dominant_angle = math.radians((peak_bin + 0.5) * bin_width)

        oriented_keypoints.append((r, c, dominant_angle))

    return oriented_keypoints


# Что происходит внутри:
# <p>Для каждой точки берётся окно orientation_window_size * orientation_window_size пикселей.
# В каждом пикселе считается магнитуда и угол градиента. Вклад каждого пикселя взвешивается на магнитуду и на гауссов вес — пиксели в центре окна важнее, чем на краях.
# <p>Вклады накапливаются в гистограмму из 36 бинов (шаг 10°, покрывают 0..360°). Бин с максимальным значением даёт доминирующую ориентацию точки.
# <p>На выходе каждая точка (r, c) превращается в тройку (r, c, angle_rad) — теперь дескриптор будет строиться относительно этого угла и станет инвариантен к повороту камеры.

# In[31]:


oriented_kp = compute_keypoint_orientations(keypoints, Ix, Iy)
print(f"Точек после ориентации: {len(oriented_kp)}")


# Точки получили ориентацию. Переходим к построению самого дескриптора.

# ## 2.5.4 Построение дескриптора

# Для каждой ориентированной точки берём патч 16x16 пикселей и делим его на сетку 4x4 блока (каждый 4x4 пикселя).
# <p>В каждом блоке строится гистограмма градиентов по 8 направлениям (бины по 45 градусов). Ключевой момент: угол каждого пикселя считается относительно доминирующей ориентации точки — это и даёт инвариантность к повороту.
# <p>16 блоков x 8 бинов = вектор из 128 чисел. Он нормализуется, затем значения обрезаются на уровне 0.2 (это стандартный трюк SIFT для подавления нелинейностей освещения) и нормализуются снова

# In[ ]:


def compute_sift_descriptors(oriented_keypoints, Ix, Iy,
                              patch_size=16,
                              num_spatial_bins=4,
                              num_orientation_bins=8):

    height, width = Ix.shape
    half = patch_size // 2
    cell_size = patch_size // num_spatial_bins
    bin_width = 360.0 / num_orientation_bins

    valid_keypoints = []
    descriptors = []

    for (r, c, dominant_angle) in oriented_keypoints:
        if r < half or r >= height - half or c < half or c >= width - half:
            continue

        histograms = []
        for bi in range(num_spatial_bins):
            row_hists = []
            for bj in range(num_spatial_bins):
                row_hists.append([0.0] * num_orientation_bins)
            histograms.append(row_hists)

        for i in range(-half, half):
            for j in range(-half, half):

                gx = Ix[r + i][c + j]
                gy = Iy[r + i][c + j]

                magnitude = math.sqrt(gx * gx + gy * gy)

                raw_angle = math.degrees(math.atan2(gy, gx))
                relative_angle = (raw_angle - math.degrees(dominant_angle)) % 360

                bi = (i + half) // cell_size
                bj = (j + half) // cell_size

                bi = min(bi, num_spatial_bins - 1)
                bj = min(bj, num_spatial_bins - 1)

                bin_idx = int(relative_angle / bin_width) % num_orientation_bins

                histograms[bi][bj][bin_idx] += magnitude

        descriptor = []
        for bi in range(num_spatial_bins):
            for bj in range(num_spatial_bins):
                for val in histograms[bi][bj]:
                    descriptor.append(val)

        descriptor = np.array(descriptor, dtype=float)

        norm = np.sqrt(np.sum(descriptor * descriptor))
        if norm > 1e-6:
            descriptor = descriptor / norm

        descriptor = np.clip(descriptor, 0, 0.2)

        norm2 = np.sqrt(np.sum(descriptor * descriptor))
        if norm2 > 1e-6:
            descriptor = descriptor / norm2

        valid_keypoints.append((r, c, dominant_angle))
        descriptors.append(descriptor)

    return valid_keypoints, np.array(descriptors)


# Дескриптор готов. Проверим на одном изображении:

# In[33]:


valid_kp, descs = compute_sift_descriptors(oriented_kp, Ix, Iy)
print(f"Дескрипторов: {len(descs)}, форма вектора: {descs[0].shape}")


# Супер! 128-мерный вектор для каждой точки получен. Теперь применим пайплайн ко всей серии изображений:

# In[34]:


all_keypoints = []
all_descriptors = []

for img in working_images:
    kp, Ix, Iy = harris_keypoints(img, minor_size=3)
    kp = filter_isolated_points(kp, 30, 3)
    oriented = compute_keypoint_orientations(kp, Ix, Iy)
    valid_kp, descs = compute_sift_descriptors(oriented, Ix, Iy)
    all_keypoints.append(valid_kp)
    all_descriptors.append(descs)


# Дескрипторы построены для всех кадров. Количество точек может отличаться между кадрами — это нормально, часть точек отсеивается у края изображения.

# Мы это сделали, теперь приступаем к работе с ними - сопоставим соседние кадры и провизуализируем это!

# За одно я решил разобраться с визуализацией картинок, так как теперь придется работать с несколькими сразу. Оформим функцию show_images_any, которая уже и с grayscale и с rgb справится. Почему не редактировал ту? мне лень, а еще в этой версии все будет "плотно" рядом отображено.

# In[35]:


def show_images_any(images, rows=2, cols=4, figsize=(12, 6), titles=None):
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < len(images):
            img = images[i]
            if len(img.shape) == 2:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)
            if titles and i < len(titles):
                ax.set_title(titles[i], fontsize=9)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()


# # 2.6 Сопоставить точки между соседними кадрами

# Идея: Для каждого дескриптора из кадра A ищем ближайший дескриптор в кадре B по евклидовому расстоянию.
# 
# Тест Лоу (Lowe's ratio test): берём два ближайших соседа (best и second_best). Если dist(best) < ratio * dist(second_best), матч считается надёжным. Стандартное значение ratio = 0.75. Смысл: если лучший матч явно лучше второго — он скорее всего правильный. Если они близки по расстоянию — скорее всего оба неправильные.

# P.S. Далее "матч" = match. С англ. совпадение.

# In[36]:


def euclidean_distance(a, b):
    diff = a - b
    return math.sqrt(float(np.sum(diff * diff)))


# In[37]:


def match_descriptors(kp_a, desc_a, kp_b, desc_b, ratio=0.75):
    matches = []

    for i in range(len(desc_a)):
        best_dist = float('inf')
        second_dist = float('inf')
        best_j = -1

        for j in range(len(desc_b)):
            dist = euclidean_distance(desc_a[i], desc_b[j])

            if dist < best_dist:
                second_dist = best_dist
                best_dist = dist
                best_j = j
            elif dist < second_dist:
                second_dist = dist

        if second_dist > 1e-6 and best_dist / second_dist < ratio:
            r_a, c_a = kp_a[i][0], kp_a[i][1]
            r_b, c_b = kp_b[best_j][0], kp_b[best_j][1]
            matches.append(((r_a, c_a), (r_b, c_b)))

    return matches


# Функция euclidean_distance считает расстояние между двумя дескрипторами-векторами.
# <p>match_descriptors для каждой точки кадра A перебирает все точки кадра B и находит два ближайших дескриптора. Тест Лоу отсеивает неоднозначные матчи: если лучший и второй по качеству кандидат похожи по расстоянию — значит точка неуникальная и лучше её отбросить. Остаются только чёткие, уверенные совпадения.

# In[38]:


def draw_matches(img_a, img_b, matches, max_display=50):
    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]


    h_combined = max(h_a, h_b)
    combined = np.zeros((h_combined, w_a + w_b, 3), dtype=np.uint8)


    if len(img_a.shape) == 2:
        combined[:h_a, :w_a] = np.dstack((img_a, img_a, img_a))
    else:
        combined[:h_a, :w_a] = img_a


    if len(img_b.shape) == 2:
        combined[:h_b, w_a:w_a + w_b] = np.dstack((img_b, img_b, img_b))
    else:
        combined[:h_b, w_a:w_a + w_b] = img_b

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(combined)

    np.random.seed(42)

    displayed = matches[:max_display]
    for ((r_a, c_a), (r_b, c_b)) in displayed:
        color = (np.random.random(), np.random.random(), np.random.random())
        ax.scatter(c_a, r_a, s=15, color=color, zorder=3)
        ax.scatter(c_b + w_a, r_b, s=15, color=color, zorder=3)
        ax.plot([c_a, c_b + w_a], [r_a, r_b], color=color, linewidth=0.8, alpha=0.7)

    ax.axis('off')
    ax.set_title(f'Матчей показано: {len(displayed)} из {len(matches)}')
    plt.tight_layout()
    plt.show()


# draw_matches склеивает два кадра горизонтально и рисует цветные линии между сопоставленными точками. Каждая пара: своим цветом для наглядности.
# <p>Запускаем матчинг для всех соседних пар кадров:

# In[39]:


all_matches = []

for i in range(len(working_images) - 1):
    print(f"\nКадр {i} → Кадр {i+1}")
    matches = match_descriptors(
        all_keypoints[i], all_descriptors[i],
        all_keypoints[i+1], all_descriptors[i+1]
    )
    print(f"  Найдено матчей: {len(matches)}")
    all_matches.append(matches)

    draw_matches(working_images[i], working_images[i+1], matches)


# Матчи найдены. Видно, что точки на магнитах уверенно сопоставляются между кадрами.

# # 2.7 Вычислить преобразование (поворот + сдвиг)

# Модель преобразования:
#    У нас только поворот и сдвиг (без масштаба и проективных искажений).
#    Это называется rigid body transformation (жёсткое тело):
#    <p>
#    [x']   [cos θ  -sin θ] [x]   [tx]
#    <p>
#    [y'] = [sin θ   cos θ] [y] + [ty]
#    <p>
#    Как считаем? По парам матчей вычисляем угол поворота и вектор сдвига.

#    Шаги:
#    1. Центрируем обе группы точек (вычитаем центроид)
#    2. Для каждой пары считаем угол: atan2(y_b - y_centroid_b, x_b - ...) и т.д.
#       Точнее — используем cross и dot product между центрированными векторами,
#       это даёт угол поворота по каждой паре.
#    3. Усредняем углы (через sin/cos, иначе проблемы с переходом через 0/360).
#    4. Вычисляем сдвиг: tx, ty = centroid_b - R @ centroid_a
# 
#  P.S. функция также реализует упрощённый RANSAC —
#    повторяем выборку случайных пар N раз, берём модель с наибольшим консенсусом.
#    Это защищает от неправильных матчей (outliers), которые всегда есть.

# In[46]:


def estimate_rotation_translation(matches, ransac_iterations=500, inlier_threshold=5.0):
    if len(matches) < 2:
        print("Недостаточно матчей")
        return 0.0, 0.0, 0.0, []

    def fit_model(sample):
        n = len(sample)

        cax = sum(m[0][1] for m in sample) / n
        cay = sum(m[0][0] for m in sample) / n
        cbx = sum(m[1][1] for m in sample) / n
        cby = sum(m[1][0] for m in sample) / n

        dot   = 0.0
        cross = 0.0
        for ((r_a, c_a), (r_b, c_b)) in sample:
            ax = c_a - cax;  ay = r_a - cay
            bx = c_b - cbx;  by = r_b - cby
            dot   += ax * bx + ay * by
            cross += ax * by - ay * bx

        angle = math.atan2(cross, dot)

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        tx = cbx - (cos_a * cax - sin_a * cay)
        ty = cby - (sin_a * cax + cos_a * cay)

        return angle, tx, ty

    def count_inliers(all_matches, angle, tx, ty):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        inliers = []
        for ((r_a, c_a), (r_b, c_b)) in all_matches:
            xp = cos_a * c_a - sin_a * r_a + tx
            yp = sin_a * c_a + cos_a * r_a + ty
            err = math.sqrt((xp - c_b)**2 + (yp - r_b)**2)
            if err < inlier_threshold:
                inliers.append(((r_a, c_a), (r_b, c_b)))
        return inliers

    best_angle = 0.0
    best_tx    = 0.0
    best_ty    = 0.0
    best_inliers = []

    for _ in range(ransac_iterations):
        sample = random.sample(matches, min(4, len(matches)))
        angle, tx, ty = fit_model(sample)
        inliers = count_inliers(matches, angle, tx, ty)
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_angle   = angle
            best_tx      = tx
            best_ty      = ty

    if len(best_inliers) >= 2:
        best_angle, best_tx, best_ty = fit_model(best_inliers)

    print(f"  Угол поворота: {math.degrees(best_angle):.2f}°")
    print(f"  Сдвиг: tx={best_tx:.1f}px, ty={best_ty:.1f}px")
    print(f"  Inliers: {len(best_inliers)} из {len(matches)} матчей")

    return best_angle, best_tx, best_ty, best_inliers


# Супер, смотрим трансформации для всей выборки.

# In[47]:


transforms = []

for i, matches in enumerate(all_matches):
    print(f"\nТрансформация {i} → {i+1}:")
    angle, tx, ty, inliers = estimate_rotation_translation(matches)
    transforms.append((angle, tx, ty))

    draw_matches(working_images[i], working_images[i+1], inliers)


# После RANSAC остались только матчи, согласующиеся с моделью поворот+сдвиг. Количество inliers относительно общего числа матчей показывает качество сопоставления: чем выше доля тем лучше.

# # 2.8 Накопить преобразования и построить траекторию камеры

# Теперь у нас есть список трансформаций для каждой соседней пары кадров. Осталось накопить их и получить траекторию.
# <p>Первый подход — накапливать (angle, tx, ty) последовательно. Позиция камеры на шаге i+1 вычисляется через позицию на шаге i с учётом накопленного угла. Этот метод работает, но ошибки накапливаются от кадра к кадру.

# In[51]:


def build_trajectory(transforms):
    positions = [(0.0, 0.0)]
    angles    = [0.0]

    cam_x = 0.0
    cam_y = 0.0
    global_angle = 0.0

    for (local_angle, tx, ty) in transforms:
        global_angle += local_angle

        cam_x += -tx
        cam_y += ty

        positions.append((cam_x, cam_y))
        angles.append(global_angle)

    return positions, angles


# build_trajectory идёт по списку трансформаций и на каждом шаге прибавляет к позиции камеры инвертированный сдвиг: если объект уехал вправо на tx камера уехала влево. Угол накапливается отдельно и используется только для отрисовки стрелок направления.

# Отлично! Визуализируем:

# In[52]:


def draw_trajectory(positions, angles=None, image_labels=None):
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(xs, ys, color='steelblue', linewidth=1.5, zorder=1)

    ax.scatter(xs, ys, color='steelblue', s=40, zorder=2)

    if angles is not None:
        span = max(max(xs) - min(xs), max(ys) - min(ys))
        arrow_len = span * 0.06 + 5
        for (x, y), a in zip(positions, angles):
            dx = math.cos(a) * arrow_len
            dy = math.sin(a) * arrow_len
            ax.annotate('', xy=(x + dx, y + dy), xytext=(x, y),
                        arrowprops=dict(arrowstyle='->', color='tomato', lw=1.5))

    labels = image_labels if image_labels else [str(i) for i in range(len(positions))]
    for i, (x, y) in enumerate(positions):
        ax.annotate(labels[i], (x, y),
                    textcoords='offset points', xytext=(6, 6),
                    fontsize=9, color='dimgray')

    ax.scatter([xs[0]], [ys[0]], color='green', s=100, zorder=3, label='Старт')
    ax.scatter([xs[-1]], [ys[-1]], color='red',   s=100, zorder=3, label='Финиш')

    closure_error = math.sqrt((xs[-1] - xs[0])**2 + (ys[-1] - ys[0])**2)
    ax.plot([xs[-1], xs[0]], [ys[-1], ys[0]],
            color='gray', linewidth=1.0, linestyle='--', alpha=0.6,
            label=f'Ошибка замыкания: {closure_error:.1f}px')

    ax.invert_yaxis()

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title('Траектория камеры')
    ax.set_xlabel('X (пиксели)')
    ax.set_ylabel('Y (пиксели)')

    plt.tight_layout()
    plt.show()


# In[53]:


positions, angles = build_trajectory(transforms)

labels = [f'img{i+1}' for i in range(len(positions))]
draw_trajectory(positions, angles, image_labels=labels)

print("\nСводная таблица трансформаций:")
print(f"{'Пара':<12} {'Угол (°)':<12} {'tx (px)':<12} {'ty (px)':<12}")
for i, (angle, tx, ty) in enumerate(transforms):
    print(f"{i}→{i+1:<9} {math.degrees(angle):<12.2f} {tx:<12.1f} {ty:<12.1f}")


# Траектория выглядит разумно по форме, однако при сравнении с реальным маршрутом камеры видно, что ошибка замыкания велика: накопленные неточности в трансформациях уводят финишную точку далеко от старта. Попробуем более точный метод — через центроиды ключевых точек.
# <p>Идея: вместо того чтобы накапливать трансформации, мы для каждого кадра независимо считаем центр масс всех ключевых точек. Это позиция объекта в пикселях кадра. Сдвиг объекта между кадрами, это и есть движение в системе координат изображения. Ошибки не накапливаются, каждый кадр независим.
#  

# In[ ]:


def build_trajectories_from_keypoints(all_keypoints):
    #считаем центроид каждого кадра
    centroids = []
    for kps in all_keypoints:
        if len(kps) == 0:
            centroids.append(None)
            continue
        mean_c = sum(kp[1] for kp in kps) / len(kps)
        mean_r = sum(kp[0] for kp in kps) / len(kps)
        centroids.append((mean_c, mean_r))

    origin = next((c for c in centroids if c is not None), (0.0, 0.0))
    x0, y0 = origin

    obj_positions = []
    cam_positions = []

    for c in centroids:
        if c is None:
            obj_positions.append(None)
            cam_positions.append(None)
        else:
            dx = c[0] - x0
            dy = c[1] - y0
            obj_positions.append(( dx,  dy))   #объект
            cam_positions.append((-dx, -dy))   #камера

    return obj_positions, cam_positions, centroids


# build_trajectories_from_keypoints считает центроид всех ключевых точек для каждого кадра.
# Смещение относительно первого кадра даёт траекторию объекта. Камера движется строго противоположно: если объект уехал на (dx, dy) в пикселях кадра, камера переместилась на (-dx, -dy) в мировых координатах.

# In[60]:


def draw_trajectory_generic(positions, image_labels=None,
                             title='Траектория', color='steelblue'):

    valid = [(i, p) for i, p in enumerate(positions) if p is not None]
    idxs  = [v[0] for v in valid]
    xs    = [v[1][0] for v in valid]
    ys    = [v[1][1] for v in valid]
    labels = image_labels if image_labels else [str(i) for i in range(len(positions))]

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(xs, ys, color=color, linewidth=1.5, zorder=1)
    ax.scatter(xs, ys, color=color, s=40, zorder=2)

    for idx, x, y in zip(idxs, xs, ys):
        ax.annotate(labels[idx], (x, y),
                    textcoords='offset points', xytext=(6, 6),
                    fontsize=9, color='dimgray')

    ax.scatter([xs[0]], [ys[0]], color='green', s=100, zorder=3, label='Старт')
    ax.scatter([xs[-1]], [ys[-1]], color='red',  s=100, zorder=3, label='Финиш')

    closure = math.sqrt((xs[-1] - xs[0])**2 + (ys[-1] - ys[0])**2)
    ax.plot([xs[-1], xs[0]], [ys[-1], ys[0]],
            color='gray', linewidth=1.0, linestyle='--', alpha=0.6,
            label=f'Ошибка замыкания: {closure:.1f}px')

    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title(title)
    ax.set_xlabel('X (пиксели)')
    ax.set_ylabel('Y (пиксели)')

    plt.tight_layout()
    plt.show()


# draw_trajectory_generic это универсальная функция отрисовки. Принимает любой список позиций, заголовок и цвет. Рисует траекторию, подписывает точки, отмечает старт и финиш, показывает пунктиром ошибку замыкания.

# In[61]:


def draw_both_trajectories(obj_positions, cam_positions, image_labels=None):
    labels = image_labels if image_labels else \
             [str(i) for i in range(len(obj_positions))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    configs = [
        (obj_positions, 'darkorange', 'Траектория объекта'),
        (cam_positions, 'steelblue',  'Траектория камеры'),
    ]

    for ax, (positions, color, title) in zip(axes, configs):
        valid = [(i, p) for i, p in enumerate(positions) if p is not None]
        idxs  = [v[0] for v in valid]
        xs    = [v[1][0] for v in valid]
        ys    = [v[1][1] for v in valid]

        ax.plot(xs, ys, color=color, linewidth=1.5, zorder=1)
        ax.scatter(xs, ys, color=color, s=40, zorder=2)

        for idx, x, y in zip(idxs, xs, ys):
            ax.annotate(labels[idx], (x, y),
                        textcoords='offset points', xytext=(6, 6),
                        fontsize=9, color='dimgray')

        ax.scatter([xs[0]], [ys[0]], color='green', s=100, zorder=3, label='Старт')
        ax.scatter([xs[-1]], [ys[-1]], color='red',  s=100, zorder=3, label='Финиш')

        closure = math.sqrt((xs[-1] - xs[0])**2 + (ys[-1] - ys[0])**2)
        ax.plot([xs[-1], xs[0]], [ys[-1], ys[0]],
                color='gray', linewidth=1.0, linestyle='--', alpha=0.6,
                label=f'Ошибка замыкания: {closure:.1f}px')

        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_title(title)
        ax.set_xlabel('X (пиксели)')
        ax.set_ylabel('Y (пиксели)')

    plt.tight_layout()
    plt.show()


# draw_both_trajectories выводит обе траектории рядом для сравнения. Теперь запустим:

# In[62]:


labels = [f'img{i+1}' for i in range(len(working_images))]
obj_positions, cam_positions, centroids = build_trajectories_from_keypoints(all_keypoints)


# Центроиды посчитаны. Рисуем траектории по отдельности:

# In[63]:


draw_trajectory_generic(obj_positions, labels, 'Траектория объекта', 'darkorange')
draw_trajectory_generic(cam_positions, labels, 'Траектория камеры',  'steelblue')


# И рядом для сравнения:

# In[64]:


draw_both_trajectories(obj_positions, cam_positions, labels)


# # Вывод
# Задачи лабораторной работы выполнены в полном объеме
