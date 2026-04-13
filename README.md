Костин Арсений, 8Е21, вариант 3.

Для 1 лаб работы по CV необходимо реализовать базовый минимум операций над изображениями
Входное изображение в формате (RGB, не чёрно-белое)
1. Фильтры
<br>1.1 Медианный фильтр
<br>1.2 Фильтр гаусса
2. Морфологические операции
<br>2.1 Эрозия
<br>2.2 Дилатация
3. Прочие операции
<br>3.1 пороговая бинаризация (для rgb и grayscale изображения)
<br>3.2 выравнивание гистограммы
<br>3.3 поворот изображений на угол кратный 90 градусов


Использовать методы OpenCV для реализации операций нельзя. Допустимы только методы cv2.imread() и cv2.imshow(). Все методы должны быть реализованы вручную.


```python
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import random
#from PIL import Image
from IPython.display import Image
%matplotlib inline
import math

import os
print(os.getcwd())
```

    /home/ars/cv-labs-sem8/lab1



```python
image1=cv2.imread('sample_image.jpg')
image2=cv2.imread('sample_image2.png')
image3=cv2.imread('sample_image3.png')
```


```python
plt.imshow(image1)
```




    <matplotlib.image.AxesImage at 0x758be9a048c0>




    
![png](README_files/README_4_1.png)
    



```python
image1_RGB = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
```


```python
plt.imshow(image1_RGB)
```




    <matplotlib.image.AxesImage at 0x758c3c6d89b0>




    
![png](README_files/README_6_1.png)
    



```python
print(image1_RGB)
```

# 1. ФИЛЬТРЫ

### 1.1 Медианный фильтр

Медианный фильтр - один из методов борьбы с "шумами". Суть заключается в том, что создается "окно" для проверки. Внутри окна элементы упорядочиваются по возрастанию/убыванию. Как медианное значение берется число в середине этого окна. Если таких чисел несколько, то берется среднее значение двух чисел посередине окна. Если рассмотреть одномерный массив как объект, к котрому будет применен фильтр:
(Используя пример из википедии: https://ru.wikipedia.org/wiki/%D0%9C%D0%B5%D0%B4%D0%B8%D0%B0%D0%BD%D0%BD%D1%8B%D0%B9_%D1%84%D0%B8%D0%BB%D1%8C%D1%82%D1%80)
<p>
Пусть есть одномерный массив x = [2 80 6 3]
Пусть окно проверки будет размером 3, обозначено круглыми скобками

1 итерация: (2 80 6)
упорядочить
(2 6 80) = медианное значение 6 = выход итерации = 6

2 итерация: (80 6 3)
упорядочить
(3 6 80) = медианное значение 6 = выход итерации 6

Алогритм выполнен, выход фильтра [6 6], потеряны 2 элемента. Тренд сохраняется и при других размерах "окна". Таким образом: (длина окна - 1)/2 = количество потерянных элементов с одного края. То есть, в нашем случае были потеряны первый и последний элементы исходного массива. Продублируем элементы. Получаем:
[2 2 80 6 3 3]
Применим к исправленному исходному массиву медианный фильтр

1 итерация: (2 2 80)
упорядочить
(2 2 80) = медианное значение 2 = выход итерации = 2

2 итерация: (2 80 6)
упорядочить
(2 6 80) = медианное значение 6 = выход итерации = 6

3 итерация: (80 6 3)
упорядочить
(3 6 80) = медианное значение 6 = выход итерации = 6

4 итерация: (6 3 3)
упорядочить
(3 3 6) = медианное значение 3 = выход итерации = 3

Выход функции [2 6 6 3]. Значения были существенно сглажены.

Стоит упомянуть, что размер окна так же может быть четным. Но даже в одномерных массивах возникают определенные трудности по его применению. Например, можно брать левое медианное значение в окне, можно брать правое, существует путь с применением среднего арифметического обоих чисел и округленное для целого числа. Для проверки алгоритмических способностей фильтра в текущей задаче это будет избыточно. </p>



```python
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
```

Проверим на примере из википедии


```python
print(median_1d([2,80,6,3], 3))
```

Результат совпал. Проверим при окне размером 1.


```python
print(median_1d([2,80,6,3], 1))
```

<p>
Смысла делать с окном ноль нет, не берется медианы.
<p>
Рассмотрим двумерный массив.
<p>
По сути алгоритм тот же, но в двумерном пространстве.
<p>
Берем окно квадратного размера, по сути матрицу меньшего порядка, чем изначальную. При этом рекомендация брать нечетную размерность аргументируется схожим образом как для одномерных массивов. Проходим этим окном по изображению. На каждой итерации разворачиваем текущий минор в ряд и применяем медианный фильтр. После этого заменяем центральный элемент минора на медианное значение от всех элеемнтов внутри этого минора - то есть, стоящее по середине отсортированного ряда.
<p>
Зададим матрицу:


```python
sample_matrix = []
for i in range(100):
    temp_row = []
    for j in range(100):
        temp_row.append(random.randint(1,100))
    sample_matrix.append(temp_row)
for i in sample_matrix:
    print(i)
```

Первый минор, размер 3:


```python
sample_matrix = np.array(sample_matrix)
minor_size = 3
sample_matrix_minor = sample_matrix[0:minor_size, 0:minor_size]
for i in sample_matrix_minor:
    print(i)
```

    [ 3 41 84]
    [67 92 88]
    [20 21 41]


Развернем и возьмем медианное значение


```python
unfolded_minor = np.reshape(sample_matrix_minor, (1,minor_size**2))
print(unfolded_minor)
print("Сортируем")
print(np.sort(unfolded_minor))
median_value = int(np.median(unfolded_minor))
print("Медианное значение =", median_value)
    
```

    [[ 3 41 84 67 92 88 20 21 41]]
    Сортируем
    [[ 3 20 21 41 41 67 84 88 92]]
    Медианное значение = 41


Присвоим центральному элементу минора его же медианное значение


```python
sample_matrix_minor[(minor_size//2)][(minor_size//2)] = median_value
print(minor_size//2)
for i in sample_matrix_minor:
    print(i)
```

    1
    [ 3 41 84]
    [67 41 88]
    [20 21 41]


Отлично, все работает. Теперь попробуем применить фильтр ко всей этой матрице:


```python
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
```


```python
f, axarr = plt.subplots(1,2)
axarr[0].imshow(sample_matrix, cmap='gray')
axarr[0].set_title('Исходная матрица')

axarr[1].imshow(sample_matrix_median, cmap='gray')
axarr[1].set_title('После медианного фильтра')
```




    Text(0.5, 1.0, 'После медианного фильтра')




    
![png](README_files/README_26_1.png)
    



```python
f, axarr = plt.subplots(1,2)
axarr[0].imshow(sample_matrix)
axarr[0].set_title('Исходная матрица')

axarr[1].imshow(sample_matrix_median)
axarr[1].set_title('После медианного фильтра')
```




    Text(0.5, 1.0, 'После медианного фильтра')




    
![png](README_files/README_27_1.png)
    


Медианный фильтр применен. Попробуем на изображении. Поскольку входное изображение имеет цвета, его можно представить как матрицу, где кол-во строк = высота изображения, кол-во столбцов = ширина, и каждый пиксель является одномерным массивом из трех элементов = интенсивность red, green, blue соотвественно. Получается 3-х ранговый тензор. Поскольку мы применяем фильтр сейчас к двумерному массиву, нам нужно преобразовать изображение в карту интенсивностей. То есть, мы потеряем цвет, но получим карту интенсивностей изображения в градациях серого. Самый простой способ - взять сумму всех интенсивностей по каналам и разделить на количество каналов.
<p>
Попробуем:

Изначальное изображение


```python
plt.imshow(image1_RGB)
```




    <matplotlib.image.AxesImage at 0x758be7476600>




    
![png](README_files/README_30_1.png)
    



```python
height, width, _ = image1_RGB.shape
image1_RGB_intensity = np.zeros((height, width), dtype='uint8')
print(image1_RGB.shape)
for rows in range(image1_RGB.shape[0]):
    for columns in range(image1_RGB.shape[1]):
        image1_RGB_intensity[rows][columns] = sum(image1_RGB[rows][columns]) // 3
plt.imshow(image1_RGB_intensity, cmap='gray')
print(image1_RGB_intensity)
```

    (741, 1153, 3)


    /tmp/ipykernel_5855/4285366573.py:6: RuntimeWarning: overflow encountered in scalar add
      image1_RGB_intensity[rows][columns] = sum(image1_RGB[rows][columns]) // 3



    
![png](README_files/README_31_2.png)
    


Теперь соберем это все в медианный фильтр для двумерного массива:


```python
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
    
```




    Text(0.5, 1.0, 'После медианного фильтра')




    
![png](README_files/README_33_1.png)
    


Последнее - завернем перевод в карту интенсивностей в функцию


```python
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
```

    /tmp/ipykernel_5855/713769101.py:7: RuntimeWarning: overflow encountered in scalar add
      image_intensity[rows][columns] = sum(image[rows][columns]) // 3





    Text(0.5, 1.0, 'Карта инстенсивностей')




    
![png](README_files/README_35_2.png)
    


Посмотрим на результаты применения медианного фильтра к изображению 2


```python
image = intensity(image2)

f, axarr = plt.subplots(1,2, figsize = (12,6))
axarr[0].imshow(image, cmap='gray')
axarr[0].set_title('Исходная матрица')

axarr[1].imshow(median_2d(image, 13), cmap='gray')
axarr[1].set_title('После медианного фильтра')
```

    /tmp/ipykernel_5855/713769101.py:7: RuntimeWarning: overflow encountered in scalar add
      image_intensity[rows][columns] = sum(image[rows][columns]) // 3





    Text(0.5, 1.0, 'После медианного фильтра')




    
![png](README_files/README_37_2.png)
    



```python
f, axarr = plt.subplots(1,2, figsize = (12,6))
axarr[0].imshow(image, cmap='gray')
axarr[0].set_title('Исходная матрица')

axarr[1].imshow(median_2d(image, 13), cmap='gray')
axarr[1].set_title('После медианного фильтра')
```




    Text(0.5, 1.0, 'После медианного фильтра')




    
![png](README_files/README_38_1.png)
    


А что насчет цветных изображений, как поступить там? наша функция работает же лишь для 2х мерных массивов. Решить эту проблему можно применим медианный фильтр для каждого из каналов отдельно. Рассмотрим изображение 3:


```python
image3_RGB = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB) #изначально opencv видит как BGR, переведем в RGB
plt.imshow(image3_RGB)
```




    <matplotlib.image.AxesImage at 0x758be6e4b4d0>




    
![png](README_files/README_40_1.png)
    



```python
height, width, _ = image3_RGB.shape
image3_R, image3_G, image3_B = np.split(image3_RGB, 3, axis=2)

image3_R_median = median_2d(image3_R, 13)
image3_G_median = median_2d(image3_G, 13)
image3_B_median = median_2d(image3_B, 13)
image3_R_median.shape

image3_RGB_median = np.dstack((image3_R_median, image3_G_median, image3_B_median))
plt.imshow(image3_RGB_median)
```




    <matplotlib.image.AxesImage at 0x758be6edd100>




    
![png](README_files/README_41_1.png)
    


Получилось! Сравним:


```python
image = image3_RGB
image1 = image3_RGB_median
f, axarr = plt.subplots(1,2, figsize = (12,6))
axarr[0].imshow(image)
axarr[0].set_title('Исходная матрица')

axarr[1].imshow(image1)
axarr[1].set_title('После медианного фильтра')

```




    Text(0.5, 1.0, 'После медианного фильтра')




    
![png](README_files/README_43_1.png)
    


Фоформиим как функцию:


```python
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
```




    <matplotlib.image.AxesImage at 0x758be6c22960>




    
![png](README_files/README_45_1.png)
    



```python
image = image3_RGB
image1 = image3_RGB_median
image2 = median_2d_RGB(image1, 39)
```


```python
f, axarr = plt.subplots(1,3, figsize = (15,9))
axarr[0].imshow(image)
axarr[0].set_title('Исходная матрица')

axarr[1].imshow(image1)
axarr[1].set_title('После медианного фильтра')

axarr[2].imshow(image2)
axarr[2].set_title('Результат фильтра после медианного фильтра')
```




    Text(0.5, 1.0, 'Результат фильтра после медианного фильтра')




    
![png](README_files/README_47_1.png)
    


Поскольку значения на изображении остались теми же, фильтровать повторно там нечего. Соответственно, результаты 2 и 3 абсолютно идентичны.

### 1.2 Фильтр гаусса

По определению: Размытие по Гауссу в цифровой обработке изображений — способ размытия изображения с помощью функции Гаусса, названной в честь немецкого математика Карла Фридриха Гаусса.

Этот эффект широко используется в графических редакторах для уменьшения шума изображения и снижения детализации. Визуальный эффект этого способа размытия напоминает эффект просмотра изображения через полупрозрачный экран, и отчётливо отличается от эффекта боке, создаваемого расфокусированным объективом или тенью объекта при обычном освещении. 

Математика: Поскольку преобразование Фурье функции Гаусса само является функцией Гаусса, применение размытия по Гауссу приводит к уменьшению высокочастотных компонентов изображения. Таким образом, размытие по Гауссу является фильтром нижних частот. 
В этом способе размытия функция Гаусса (которая также используется для описания нормального распределения в теории вероятностей) используется для вычисления преобразования, применяемого к каждому пикселю изображения. Формула функции Гаусса в одном измерении: 


```python
img = cv2.imread('./gaussfunc.png')
plt.imshow(img)
```




    <matplotlib.image.AxesImage at 0x758be6bf5610>




    
![png](README_files/README_52_1.png)
    


Сигма в этой функции это среднеквадратическое отклонение нормального распределения. Визуализируем в desmos.


```python
img = Image('./output.gif')
img
```




    <IPython.core.display.Image object>



В двухмерном пространстве по определению это произведение двух функций Гаусса, для каждого измерения. Зададим функцию Гаусса как функцию в коде:


```python
def Gauss(x, sigma):
    return (1/math.sqrt(2*math.pi*(sigma**2)))*(math.e**(-(x**2)/(2*(sigma**2))))

print(Gauss(0, 0.7))
```

    0.5699175434306182


Визуализируем


```python
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
```


    
![png](README_files/README_58_0.png)
    


Попробуем задать второе измерение и визуализировать


```python
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
```

    (600, 600)



    
![png](README_files/README_60_1.png)
    


Как видим, образуется своебразный колокол. Интенсивность в центре выше. В этом строится основная идея применения фильтра. Если мы представим изображение, по которому будем проходиться апертурой, то станет ясно что то, что окажется в ее центре имеет больший вес. А то что к краям - там значение меньше, важность меньше. Соответственно, поэтому фильтр Гаусса является фильтром низких частот. Практически это значит, что если есть изображения с маленькими шумами, то фильтр их должен убрать, попробуем:


```python
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

```


    
![png](README_files/README_62_0.png)
    


    (200, 200)



```python
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

```


```python
f, axarr = plt.subplots(1,2, figsize = (12,6))
axarr[0].imshow(image1_RGB_intensity, cmap='gray')
axarr[0].set_title('Исходная матрица')

image1_RGB_intensity_gaussian = gaussian_2d(image1_RGB_intensity, 3.7, 121)
axarr[1].imshow(image1_RGB_intensity_gaussian, cmap='gray')
axarr[1].set_title('После фильтра Гаусса')

print(image1_RGB_intensity.shape, image1_RGB_intensity_gaussian.shape)
```

    limits for Z kernel 1.0 1.0
    limits for Z kernel after normalization 7.259019782378444e-117 0.011625634995755686
    limits for gaussian output 0.010995363459556876 82.94163571012916
    (741, 1153) (741, 1153)



    
![png](README_files/README_64_1.png)
    


Супер, все работает. А что делать с RGB? То же что и с медианным. Сделаем все ровно то же самое, просто по каналам.


```python
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

```

Отлично, можно визуализировать:


```python
image = image3_RGB
image1 = gaussian_2d_RGB(image3_RGB, 3.7, 121)
f, axarr = plt.subplots(1,2, figsize = (12,6))
axarr[0].imshow(image)
axarr[0].set_title('Исходная матрица')

axarr[1].imshow(image1)
axarr[1].set_title('После фильтра Гаусса')
```




    Text(0.5, 1.0, 'После фильтра Гаусса')




    
![png](README_files/README_68_1.png)
    


# 2. Морфологические операции

По опредению: Морфология является широким набором операций обработки изображений, которые процесс отображает на основе форм. Морфологические операции применяют элемент структурирования к входному изображению, создавая выходное изображение, одного размера. В морфологической операции значение каждого пикселя в выходном изображении основано на сравнении соответствующего пикселя во входном изображении с его соседями. Источник: https://docs.exponenta.ru/images/morphological-dilation-and-erosion.html


### 2.1 Эрозия

Значение выходного пикселя является минимальным значением всех пикселей в окружении. В бинарном изображении пиксель установлен в 0 если какой-либо из соседних пикселей имеет значение 0.

Морфологическая эрозия удаляет острова и маленькие объекты так, чтобы только независимые объекты остались.

То есть, мы смотрим на соседей в окрестности апертуры относительно таргета - целевого пикселя. Если хоть один выполняет условие - таргет приобретает значение из условия. Импортируем новое изображение для этой главы:


```python
image4 = cv2.imread('sample_image4.jpg')
image4 = cv2.cvtColor(image4, cv2.COLOR_BGR2RGB)
plt.imshow(image4)
```




    <matplotlib.image.AxesImage at 0x758bf6eb2960>




    
![png](README_files/README_73_1.png)
    


Так же, как и раньше для простоты переведем в чернобелое изображений исходник. В этот раз я захотел использовать не карту инстенсивности, а реальный перевод в черно-белый формат с полутонами. По сути, та же карта интенсивности. Только в этот раз мы не просто будем усреднять значения по каналам, а использовать корректную фотограмметрическую формулу для такого перевода: Result = 0.299 R + 0.587 G + 0.114 B


```python
def intensity_grayscale(image):
    height, width, _ = image.shape
    image_intensity = np.zeros((height, width), dtype='uint8')

    for rows in range(image.shape[0]):
        for columns in range(image.shape[1]):
            #отталкиваемся от того, что исходник RGB и порядок каналов сохранен
            image_intensity[rows][columns] = 0.299*image[rows][columns][0] + 0.587*image[rows][columns][1] + 0.114*image[rows][columns][2]

    return image_intensity

```


```python
image = image4
image4_gray = intensity_grayscale(image)
```


```python
f, axarr = plt.subplots(1,2, figsize = (12,6))
axarr[0].imshow(image)
axarr[0].set_title('Исходная матрица')

axarr[1].imshow(image4_gray, cmap='gray')
axarr[1].set_title('После перевода')
```




    Text(0.5, 1.0, 'После перевода')




    
![png](README_files/README_77_1.png)
    



```python
print(image4_gray.dtype)
```

    uint8


Теперь можем приступать к тому же алгоритму! У нас будет кернел - апертура, в окрестности которой мы будем сравнивать пиксели. Как проходить мы знаем, как добавлять паддинг для избежания потери краев изображения при свертке знаем. Принципиально один вопрос - как сравнивать. Наше изображение, как показано выше, закодировано в формате uint8. То бишь, глубина цвета 8 бит. Unsigned = нет знака. Соответственно максимум у нас каждый цвет кодируется 8 битами, а не 7. Значит значение интенсивности цвета лежит в промежутке от 0 до 2^8. 0...255. Где 256 - его нет, так как отчет мы ведем с нуля.

Отлично, мы будем сравнивать значения пикселей под апертурой в диапазоне от 0 до 255. Соответственно, нужно то, с чем мы будем сравнивать, условие. Для этих целей вводим понятие - порог, threshold. Пускай этот порог будет иметь некоторое значение. Например, 122. Значит то, что будет в окрестности апертуры ниже или равно получит логически ноль. Если такие пиксели в окрестности есть - такое значение получит и таргет. Все что больше - на результат не повлияет.

Стоит отметить, что та же логика будет применена к главе 2.2. Но в обратную сторону. То есть если больше или равно условию/порогу - повлияет на таргет. Это будет рассмотрено в следующей главе. 

Приступим к реализации:


```python
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

```


```python
image = image4_gray
image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4), 
                         interpolation=cv2.INTER_AREA) #зачем? исходное было очень большое, программа долго выполнялась. Хотите без потери времени работать с большими - делайте многопоток и кидайте PR, вообще капитальными красавчиками будете!

image4_gray_eroded = eroded_threshold(image, 42, 3)

```

    limits for eroded output 42.0 254.0



```python
f, axarr = plt.subplots(1,2, figsize = (12,6))
axarr[0].imshow(image, cmap='gray')
axarr[0].set_title('Исходная матрица')

axarr[1].imshow(image4_gray_eroded, cmap='gray')
axarr[1].set_title('После перевода')
```




    Text(0.5, 1.0, 'После перевода')




    
![png](README_files/README_85_1.png)
    


Отлично! Как мы видим, при текущих параметрах, тонкие паутинки стали пропадать. При этом сам паук остался. Попробуем вообще избавиться от паутинок:


```python
image = image4_gray
image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4), 
                         interpolation=cv2.INTER_AREA) #зачем? исходное было очень большое, программа долго выполнялась. Хотите без потери времени работать с большими - делайте многопоток и кидайте PR, вообще капитальными красавчиками будете!

image4_gray_eroded = eroded_threshold(image, 68, 3)

f, axarr = plt.subplots(1,2, figsize = (12,6))
axarr[0].imshow(image, cmap='gray')
axarr[0].set_title('Исходная матрица')

axarr[1].imshow(image4_gray_eroded, cmap='gray')
axarr[1].set_title('После перевода')
```

    limits for eroded output 68.0 254.0





    Text(0.5, 1.0, 'После перевода')




    
![png](README_files/README_87_2.png)
    


Получилось! Теперьв центре мы в состоянии выденить лишь самого паука, без паутины. То есть логика простая. Маленькие штуки - фильтруются. Чем тоньше цель - тем легче ее "съесть", проверяя то насколько она тонкая засчет соседей. Соотвественно размер апертуры - насколько хирургически мы действуем. А порог - таргет с которым мы сравниваем. Попробуем обратную операцию, расширение/диляция/дилатация.

P.S. А как обстоят дела с RGB? Так же как с прошлыми. Но там мы применяем эту операцию отдельно к каждому из каналов. 
Кстати, эрозия так же активно применяется к бинаризованным изображениям. О них упомянуто в главе 3.1. Посколько там порог задается и применяется на этапе бинаризации, функция эрозии или диляции уже не потребует задания какого либо порога. Там либо 0 у соседей ищем, либо 1 соответственно.

### 2.2 Дилатация a.k.a Диляция, Расширение


Значение выходного пикселя является максимальным значением всех пикселей в окружении. В бинарном изображении пиксель установлен в 1 если какой-либо из соседних пикселей имеет значение 1.

Морфологическое расширение делает объекты более видимыми и заполняет маленькие отверстия в объектах.

Используем прошлый код функции и перевернем логику. Гипотеза - паутинки должны стать толще. Пробуем:


```python
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
    

```


```python
image = image4_gray
image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4), 
                         interpolation=cv2.INTER_AREA) #зачем? исходное было очень большое, программа долго выполнялась. Хотите без потери времени работать с большими - делайте многопоток и кидайте PR, вообще капитальными красавчиками будете!

image4_gray_dilated = dilated_threshold(image, 122, 3)

f, axarr = plt.subplots(1,2, figsize = (12,6))
axarr[0].imshow(image, cmap='gray')
axarr[0].set_title('Исходная матрица')

axarr[1].imshow(image4_gray_dilated, cmap='gray')
axarr[1].set_title('После перевода')
```

    limits for dilated output 0.0 122.0





    Text(0.5, 1.0, 'После перевода')




    
![png](README_files/README_93_2.png)
    


Выводы те же самые с проправкой на то, что операция обратная. Гипотеза верна, ч.т.д.

Ради эксперимента в главе 3.1. применим диляцию к бинаризованному изображению для наглядности.

# 3. Прочие операции

### 3.1 Пороговая бинаризация (для rgb и grayscale изображения)

По определению:
Процесс бинаризации – это перевод цветного (или в градациях серого) изображения в двухцветное черно-белое. Главным параметром такого преобразования является порог t – значение, с которым сравнивается яркость каждого пикселя. По результатам сравнения, пикселю присваивается значение 0 или 1. Существуют различные методы бинаризации, которые можно условно разделить на две группы – глобальные и локальные. В первом случае величина порога остается неизменной в течение всего процесса бинаризации. Во втором изображение разбивается на области, в каждой из которых вычисляется локальный порог.
Источник: https://habr.com/ru/articles/278435/

То есть, есть некоторый порог с которым мы сравниваем каждый пиксель изображения. Наш случай - первый, глобальный, простой. Меньше порога - присваиваем пикселю ноль, больше порога - единицу. Стоит отметить сразу, что для RGB логика та же, просто отдельно по каналам мы проверяем попиксельно интенсивности. 

Начнем импортируя новое изображение и выделяя интересующую нас область:


```python
image5 = cv2.imread('sample_image5.jpg')
image5 = cv2.cvtColor(image5, cv2.COLOR_BGR2RGB)
plt.imshow(image5)
```




    <matplotlib.image.AxesImage at 0x758beac59940>




    
![png](README_files/README_101_1.png)
    



```python
image5_gray = intensity_grayscale(image5)
image5_gray_cropped = image5_gray[300:600, 200:600]

plt.imshow(image5_gray_cropped, cmap='gray')
plt.show()
```


    
![png](README_files/README_102_0.png)
    


Супер! Теперь попробуем пройтись по изображению с порогом в 125. Реализуем функцию:


```python
def binarized(image, threshold):
    image1 = image.copy()
    
    for rows in range (image1.shape[0]):
            for columns in range(image1.shape[1]):
                if image1[rows][columns] <= threshold:
                    image1[rows][columns] = 0
                else:
                    image1[rows][columns] = 1

    return image1
```


```python
image = image5_gray_cropped
image_binarized = binarized(image, 125)

f, axarr = plt.subplots(1,2, figsize = (12,6))
axarr[0].imshow(image, cmap='gray')
axarr[0].set_title('Исходная матрица')

axarr[1].imshow(image_binarized, cmap='gray')
axarr[1].set_title('После перевода')
```




    Text(0.5, 1.0, 'После перевода')




    
![png](README_files/README_105_1.png)
    


Как видим, то что было светлее 125 стало единицей - белым - неважным. То, что было равно или темнее 125 стало выразительным. Отдельные элементы лица, с которыми нам, возможно, пришлось бы работать стали отдельными и их отделить от остального изображения станет легче.

Как обещал, попробуем избавиться от неточностей при помощи морфологических операций. Внимание на нос - я хочу оставить ноздрю ии попробовать избавиться от остальных ненужных частей. Применим эрозию:

Для этого поменяем функцию эрозии для бинаризованных изображений:


```python
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

```


```python
image_binarized_eroded = eroded_bin(image_binarized, 5)

f, axarr = plt.subplots(1,2, figsize = (12,6))
axarr[0].imshow(image_binarized, cmap='gray')
axarr[0].set_title('Исходная матрица')

axarr[1].imshow(image_binarized_eroded, cmap='gray')
axarr[1].set_title('После перевода')
```

    limits for eroded output 0.0 1.0





    Text(0.5, 1.0, 'После перевода')




    
![png](README_files/README_110_2.png)
    


Получилось! Но мы слишком сильно избавились от частей носа. Наш таргет - ноздря тоже пострадала. Попробуем применить диляцию:


```python
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

```


```python
image_binarized_eroded_dilated = dilated_bin(image_binarized_eroded, 3)

f, axarr = plt.subplots(1,2, figsize = (12,6))
axarr[0].imshow(image_binarized_eroded, cmap='gray')
axarr[0].set_title('Исходная матрица')

axarr[1].imshow(image_binarized_eroded_dilated, cmap='gray')
axarr[1].set_title('После перевода')
```

    limits for eroded output 0.0 1.0





    Text(0.5, 1.0, 'После перевода')




    
![png](README_files/README_113_2.png)
    


Отлично. Ноздря на месте и мы значительно избавились от лишних деталей. Вернемся к RGB:

Попробуем сделать так же, но для RGB. Пускай для каждого канала мы зададим отдельно свой порог

Определим для этого функцию:


```python
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

```


```python
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
```

    (1200, 1200, 3)





    Text(0.5, 1.0, 'После перевода')




    
![png](README_files/README_118_2.png)
    


### 3.2 Выравнивание гистограммы

По определению:
Операция выравнивания гистограмм (увеличение контраста) часто используется для увеличения качества изображения.
Гистограмма представляет из себя функцию h(x), которая возвращает суммарное количество пикселей, яркость которых равна x.

Гистограмма h полутонового изображения I задается выражением:


```python
img = cv2.imread('./histfunc.png')
plt.imshow(img)
```




    <matplotlib.image.AxesImage at 0x758be64b3a40>




    
![png](README_files/README_121_1.png)
    


, где m соответствует интервалам значений яркости

Визуально гистограмма представляет из себя прямоугольник, ширина которого равна максимально возможному значению яркости точки на исходном изображении. Для полутоновых изображений мы будем работать с диапазоном яркостей точек от 0 до 255, а значит и ширина гистограммы будет равна 256. Высота гистограммы может быть любой, но для наглядности мы будем работать с прямоугольными гистограммами.

С точки зрения программиста, гистограмма — это одномерный массив размерностью 256 (в нашем случае), где каждый элемент массива хранит в себе суммарное количество точек соответствующей яркостью. 

Надо визуализировать. Сначала попробуем вывести гистограмму изображения:


```python
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
```

    0 255



    
![png](README_files/README_125_1.png)
    


Видно, что функция распределения распределена не равномерно.

В начале стабильные значения. Примерно вторую четверть графика она близка в нулю (относительно вертикальной координаты), потом идет небольшой рост и в последней четверти графика все спокойно за исключением конца, там опять резкий рост. Процедура выравнивания гистограммы заключается в том, чтобы сделать функцию распределения более равномерной, чтобы она возрастала примерно одинаково во всем своем диапазоне.

Посчитаем PDF(вероятности) для каждой интенсивности. Напоминаю, что у нас 256 интенсивностей, а промежуток значений от 0 до 255.


```python
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


```


    
![png](README_files/README_128_0.png)
    


Теперь считаем кумулятиву (CDF)


```python
cdf = [0] * 256
cumulative = 0.0

for i in range(256):
    cumulative += probabilities[i]
    cdf[i] = cumulative

names = list(counts.keys())
values = cdf
    
plt.bar(names, values)
plt.show()

```


    
![png](README_files/README_130_0.png)
    


Все по теории, все отлично, финальная кумулятива = 1. Теперь попробуем выравнять. То есть пиксель с интенсивностью умножаем на соответствующую этой интенсивности кумулятиву.


```python
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
```




    Text(0.5, 1.0, 'После перевода')




    
![png](README_files/README_132_1.png)
    


как видим, изображение было чересчур темным, а стало достаточно сбланасированно ярким. Посмотрим на гистограммы:


```python
counts = image_hist(image)
total_pixels = image.shape[0]*image.shape[1]

names = list(counts.keys())
values = list(counts.values())

names = list(counts.keys())
values = list(counts.values())
    
plt.bar(names, values)
plt.show()
```


    
![png](README_files/README_134_0.png)
    



```python
image = image_hist_equalized
counts = image_hist(image)
total_pixels = image.shape[0]*image.shape[1]

names = list(counts.keys())
values = list(counts.values())

names = list(counts.keys())
values = list(counts.values())
    
plt.bar(names, values)
plt.show()
```


    
![png](README_files/README_135_0.png)
    


Как видно, цель выполнена, гистограмма стала значительно равнее. Попробуем на RGB:


```python
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

```


```python
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
```


```python
image = image4
image_hist_equalized = hist_equalize_RGB(image)
f, axarr = plt.subplots(1,2, figsize = (12,6))
axarr[0].imshow(image)
axarr[0].set_title('Исходная матрица')

axarr[1].imshow(image_hist_equalized)
axarr[1].set_title('После перевода')
```




    Text(0.5, 1.0, 'После перевода')




    
![png](README_files/README_139_1.png)
    


### 3.3 поворот изображений на угол кратный 90 градусов

Что такое поворот? По своей сути это смена рядов и столбцов исходной матрицы местами. То есть, транспонирование в каком то смысле. Давайте попробуем...


```python
plt.imshow(image4_gray, cmap='gray')
```




    <matplotlib.image.AxesImage at 0x758be5be9730>




    
![png](README_files/README_142_1.png)
    



```python
image4_gray_transposed = np.transpose(image4_gray)
plt.imshow(image4_gray_transposed, cmap='gray')
```




    <matplotlib.image.AxesImage at 0x758be5773320>




    
![png](README_files/README_143_1.png)
    


Получилось! Мы повернули картинку на -90 градусов. Попробуем повращать дальше:


```python
image4_gray_transposed = np.transpose(image4_gray_transposed)
plt.imshow(image4_gray_transposed, cmap='gray')
```




    <matplotlib.image.AxesImage at 0x758be6455220>




    
![png](README_files/README_145_1.png)
    


Не вышло. Функция транспонирования реверсивна. Если ее применить повторно, то ряды и столбцы просто поменяются местами обратно. Как быть?

Вообще, подсказка кроется в первом результате транспонирования. Если бы паучок смотрел в противоположную сторону, то это был бы поворот на +90 градусов, то есть, по часовой. Попробуем еще раз:


```python
image4_gray_transposed = np.transpose(image4_gray)
for rows in range(len(image4_gray_transposed)):
    curr_row = image4_gray_transposed[rows]
    rev_row = curr_row[::-1]
    image4_gray_transposed[rows] = rev_row
    
plt.imshow(image4_gray_transposed, cmap='gray')
```




    <matplotlib.image.AxesImage at 0x758be5635490>




    
![png](README_files/README_148_1.png)
    


Что-то пошло не так... изображение отзеркалено по горизонтали. Паучок должен был смотреть наверх. Получается, мы перепутали порядок операций. Надо было разворачивать ряды, а мы развернули "столбцы" изначального изображения, потому что работали над транспонированным. попробуем изменить порядок действий и провести проверку:


```python
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
```


```python
f, axarr = plt.subplots(1,3, figsize = (12,6))

axarr[0].imshow(image, cmap='gray')
axarr[0].set_title('изначально')

axarr[1].imshow(image_90_cw, cmap='gray')
axarr[1].set_title('по часовой')

axarr[2].imshow(image_90_ccw, cmap='gray')
axarr[2].set_title('против часовой')
```




    Text(0.5, 1.0, 'против часовой')




    
![png](README_files/README_151_1.png)
    


Супер, получилось! Получается, применив реверс рядов и транспонируя результат мы вращаем результат на 90 градусов по часовой стрелке. Сколько раз это будет сделано можно задать целым числом - как параметр для этой функции. Оформим как функцию для grayscale. Для RGB так же, как и везде делается ровно то же самое, просто по каждому из каналов...


```python
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
    
```




    <matplotlib.image.AxesImage at 0x758be58f7ef0>




    
![png](README_files/README_153_1.png)
    



```python
plt.imshow(rotate_90_cw(image, 4), cmap='gray')
```




    <matplotlib.image.AxesImage at 0x758be55e1610>




    
![png](README_files/README_154_1.png)
    



```python
plt.imshow(rotate_90_cw(image, 3), cmap='gray')
```




    <matplotlib.image.AxesImage at 0x758be5450e00>




    
![png](README_files/README_155_1.png)
    



```python
plt.imshow(rotate_90_cw(image, 2), cmap='gray')
```




    <matplotlib.image.AxesImage at 0x758be54fba10>




    
![png](README_files/README_156_1.png)
    



```python
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
```




    Text(0.5, 1.0, 'вверх ногами')




    
![png](README_files/README_157_1.png)
    


# Вывод

Все получилось. Кто сдох - тот лох. Кто дочитал - красавчик. Передохни и иди читай вторую лабу.

Костин Арсений, 8Е21, вариант 3.

Лабораторная работа №2. Визуальная одометрия (навигация)
Цель: Разработать систему визуальной одометрии (навигации) по группе фотографий.
Ход работы: сделайте не менее 8 фото с переносом камеры или ноутбука по квадрату (то есть двиньте сначала вправо, потом вперед, потом влево, потом назад и обратно в начальную точку). Используя данные фотографии реализуйте следующее:
<p> 1.	Определите на каждой фотографии ключевые точки </p>
<p>2.	Отфильтруйте самые наилучшие применяю адаптивный радиус и локальные максимумы, не забудьте так же выровнять по яркости изображения.</p>
<p>3.	Постройте по каждой точке дескриптор (можете использовать любой, рекомендуется SIFT)</p>
<p>4.	Сопоставьте два соседних изображения на предмет соответствия ключевых точек. То есть определите пары одинаковых точек.</p>
<p>5.	Постройте модель преобразования изображений, учитывайте только поворот и сдвиг.</p>
<p>6.	С учетом полученных моделей постройте траекторию движения камеры.</p>
<p>Проверка работоспособности: будет осуществляться на специальной группе фото, предоставленных преподавателем. Траектория движения, для которых недоступна.</p>
<p>В процессе выполнения вы можете использовать готовые функции по погрузке данных, перевода в цветовые пространства, фильтрации, для построения прямых и траекторий. Функции 1-6 описанные выше должны быть реализованы самостоятельно.</p>



```python
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import random
#from PIL import Image
from IPython.display import Image
%matplotlib inline
import math
import lab1_functions as lb1
import os
print(os.getcwd())
print(os.listdir())
```

    /home/ars/cv-labs-sem8/lab1
    ['sequence5.jpeg', 'lab1_functions.py', 'sample_image2.png', 'lab2.py', 'sequence4.jpeg', '__pycache__', 'doodles.ipynb', 'lab1.py', 'lab1.ipynb', 'sequence6.jpeg', 'sample_image3.png', 'sequence8.jpeg', 'sample_image.jpg', 'histfunc.png', 'gpt-stripfunctions.py', 'sequence3.jpeg', 'sample_image4.jpg', 'output.gif', 'sequence7.jpeg', 'sample_image5.jpg', 'gaussfunc.png', 'sequence1.jpeg', 'gradient.png', 'harris1.png', 'sequence2.jpeg', 'lab2.ipynb']


# 2.1 Загрузить изображения

Приступим, импортируем сделанные изображения:


```python
image1=cv2.cvtColor(cv2.imread('sequence1.jpeg'), cv2.COLOR_BGR2RGB)
image2=cv2.cvtColor(cv2.imread('sequence2.jpeg'), cv2.COLOR_BGR2RGB)
image3=cv2.cvtColor(cv2.imread('sequence3.jpeg'), cv2.COLOR_BGR2RGB)
image4=cv2.cvtColor(cv2.imread('sequence4.jpeg'), cv2.COLOR_BGR2RGB)
image5=cv2.cvtColor(cv2.imread('sequence5.jpeg'), cv2.COLOR_BGR2RGB)
image6=cv2.cvtColor(cv2.imread('sequence6.jpeg'), cv2.COLOR_BGR2RGB)
image7=cv2.cvtColor(cv2.imread('sequence7.jpeg'), cv2.COLOR_BGR2RGB)
image8=cv2.cvtColor(cv2.imread('sequence8.jpeg'), cv2.COLOR_BGR2RGB)

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
```




    <matplotlib.image.AxesImage at 0x78ddea78b230>




    
![png](README_files/README_165_1.png)
    


Лирическое отступление - чтобы не размазывать отчет, работа будет вестись над grayscale изображениями. То что мы будем использовать - не зависит от цветов, как видно из первой лабы. Все что возможно - можно сделать для RGB повторяя те же операций и преобразования, просто трижды - по разу для каждого цветового канала. Это не цель лабораторной работы. Приступим, переведем изображения в черно-белый формат, используя функцию из прошлой лабы.


```python
images_sequence_gray = []
for img in images_sequence:
    images_sequence_gray.append(lb1.intensity_grayscale(img))
```


```python
f, axarr = plt.subplots(2,4, figsize = (12,6))

axarr[0,0].imshow(images_sequence_gray[0], cmap='gray')
axarr[0,1].imshow(images_sequence_gray[1], cmap='gray')
axarr[0,2].imshow(images_sequence_gray[2], cmap='gray')
axarr[0,3].imshow(images_sequence_gray[3], cmap='gray')

axarr[1,0].imshow(images_sequence_gray[4], cmap='gray')
axarr[1,1].imshow(images_sequence_gray[5], cmap='gray')
axarr[1,2].imshow(images_sequence_gray[6], cmap='gray')
axarr[1,3].imshow(images_sequence_gray[7], cmap='gray')
```




    <matplotlib.image.AxesImage at 0x78ddd93675f0>




    
![png](README_files/README_168_1.png)
    


То есть, нам нужно:

загрузить изображения, привести их к одинаковой яркости / grayscale, найти ключевые точки, отфильтровать точки, построить дескрипторы (SIFT), сопоставить точки между соседними кадрами, вычислить преобразование (поворот + сдвиг), накопить преобразования и построить траекторию камеры

Исправим проблемы с яркостью, применив реализованный модуль для выравнивания гистограммы:

# 2.2 Привести к одинаковой яркости / grayscale

для нашего удобства и ментального здоровья зададим функцию готовую для отображения картинок из серии:


```python
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
```


```python
show_images(images_sequence_gray, 2, 4)
```


    
![png](README_files/README_174_0.png)
    


Супер. Вернемся к идее применить выравнивание гистограммы на всех изображениях:


```python
images_hist_equalized = images_sequence_gray
for img in images_hist_equalized:
    img = lb1.hist_equalize(img)
    
show_images(images_sequence_gray, 2, 4)
```


    
![png](README_files/README_176_0.png)
    


# 2.3 Найти ключевые точки

Приступаем. На изображениях у нас есть два магнита рядом, но нам могут значительно помешать: блики, тень от фотографа, другие нерелватные в этом контексте детали. Это визуальный <b>шум</b>. От шума надо избавиться, рассмотрим изображение 1, применим для подавления шумов фильтр Гаусса.


```python
images_temp = [images_hist_equalized[0], lb1.gaussian_2d(images_hist_equalized[0], 1.2, 21)]
show_images(images_temp, 1, 2)
```

    limits for Z kernel 1.0 1.0
    limits for Z kernel after normalization 7.658057422279819e-32 0.11052426603583844
    limits for gaussian output 2.4991448589236085 234.7124602763381



    
![png](README_files/README_179_1.png)
    


Стало слегка лучше. Теперь перейдем к теории:

Ключевые точки: это такие места, по которым сравнивая две картинки можно отследить движение.

Если бы мы смотрели на голубое небо и взяли его кусочек как ключевую точку, то распределение интенсивностей было бы +- одинаковое между ним и другим кусочком неба. Это плохая ключевая точка.

Если бы мы смотрели на фото прикроватной тумбочки, то могли бы предположить, что край тумбочки - хорошая ключевая точка, т.к. интенсивность прыгает на моменте перехода от края тумбочки к ее боковой части в тени. По идее - уже неплохо, но если двигаться вдоль этого края - ситуация не поменяется при сравнении двух кадров.

Из этого следует, что лучший вариант - когда меняются по двум направлениям тренды. Например, угол тумбочки. Вдоль него не подвигаться, то есть изменение интенсивности слева и спереди (перед стеной) достаточно легко отслеживаются.  

Из математики следует, что производная функции показывает скорость изменения ее значения. А градиент - вектор, показывающий <b>НАПРАВЛЕНИЕ</b> наибыстрейшего увеличения функции. Это то, что надо нам. Формула для градиента в общей форме выглядит так:


```python
img = cv2.imread('./gradient.png')
plt.imshow(img)
```




    <matplotlib.image.AxesImage at 0x78ddc7ffaa20>




    
![png](README_files/README_186_1.png)
    


Исходя из вышесказанного, понятно, что алгоритмы поиска ключевых точек ищут точки, где изменение яркости происходит во многих направлениях одновременно.

Математически - ищем градиенты изображения, по x и y координатам. 

Пусть I_x = изменение по х, I_y = изменение по y.
<p>Значит, место, где I_x = 0, I_y = 0 это однородная область.
<p>Если I_x большое, I_y маленькое = это край.
<p>Если I_x большое, I_y большое = это угол (ключевая точка).

Из опыта прошлой лабы мы понимаем, что смотреть на сам один пиксель недостаточно. Нужно брать апертуру/кернел/область/окно. Так и сделаем. Идейно по определению подходит фильтр Хариса. Источник: https://docs.exponenta.ru/R2021a/visionhdl/ug/corner-detection.html

### 2.3.1 Разберем этот фильтр

Фактически у нас есть исходное изображение, minor_size, k, threshold_ratio.

minor_size - так же как в прошлой лабе, размер апертуры/кернел/окно. маленькое окно = чувствительность к мелким деталям, большое окно = реагирует только на крупные структуры

источник: https://docs.opencv.org/3.4/dc/d0d/tutorial_py_features_harris.html
<p> k - коэффициент, испольуемый в формуле Хариса:


```python
img = cv2.imread('./harris1.png')
plt.imshow(img)
```




    <matplotlib.image.AxesImage at 0x78dda6851070>




    
![png](README_files/README_195_1.png)
    


Этот коэффициент регулирует насколько алгоритм строго смотрит края.
<p>Маленький = алгоритм более терпим к краям, может принимать некоторые края за углы
<p>Большой = алгоритм строгий, оставляет только очень выраженные углы

threshold_ratio

После вычисления Harris response R нужно решить какие точки считать ключевыми.

Для этого берётся максимум:
R_max = max(R)

и строится порог:

threshold = threshold_ratio * R_max

Если threshold_ratio = 0.01, то берутся точки у которых R > 1% от максимального

Если увеличить до 0.1 = останутся только самые сильные углы.


```python
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
```

Что тут происходит? Сначала вычисляются упомянутые градиенты изображения.
Градиент показывает, насколько быстро меняется яркость:

Ix — изменение яркости по горизонтали

Iy — изменение яркости по вертикали

Это делается с помощью центральной разности: берётся разница между соседними пикселями.

После этого алгоритм начинает рассматривать каждую точку изображения и маленькое окно вокруг неё (minor_size). Внутри этого окна суммируются значения градиентов:

квадрат горизонтального градиента

квадрат вертикального градиента

произведение двух градиентов

Эти суммы используются для вычисления величины R. Она показывает, насколько вероятно, что точка является углом.

Дальше выбираются только точки, у которых R достаточно большое (больше порога).
После этого выполняется проверка локального максимума: точка должна быть больше всех своих соседей. Это нужно, чтобы оставить только самые сильные углы и убрать лишние точки вокруг них.

В итоге функция возвращает:

keypoints — координаты найденных углов

Ix — карту горизонтальных градиентов

Iy — карту вертикальных градиентов


```python
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
```

Функция draw_keypoints рисует найденные ключевые точки на изображении. Сначала создаётся копия изображения. Если изображение чёрно-белое, оно превращается в трёхканальное (RGB), чтобы на нём можно было рисовать цветные элементы.


Для каждой найденной точки:

на изображении рисуется маркер (точка)

если включён параметр show_vectors, дополнительно рисуется стрелка

Стрелка показывает направление градиента в этой точке. Она берётся из Ix и Iy и показывает, в каком направлении яркость изменяется сильнее всего.

Параметр vector_scale просто увеличивает длину стрелок, чтобы их было лучше видно.


```python
gray = images_temp[1]

keypoints, Ix, Iy = harris_keypoints(gray, minor_size=9)

draw_keypoints(gray, keypoints)
```

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [2.4991448589236085..234.7124602763381].



    
![png](README_files/README_204_1.png)
    





    array([[[255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
            ...,
            [255, 255, 255],
            [255, 255, 255],
            [255, 255, 255]],
    
           [[255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
            ...,
            [255, 255, 255],
            [255, 255, 255],
            [255, 255, 255]],
    
           [[255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
            ...,
            [255, 255, 255],
            [255, 255, 255],
            [255, 255, 255]],
    
           ...,
    
           [[255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
            ...,
            [255, 255, 255],
            [255, 255, 255],
            [255, 255, 255]],
    
           [[255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
            ...,
            [255, 255, 255],
            [255, 255, 255],
            [255, 255, 255]],
    
           [[255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
            ...,
            [255, 255, 255],
            [255, 255, 255],
            [255, 255, 255]]], shape=(480, 640, 3), dtype=uint8)




    <Figure size 640x480 with 0 Axes>


Как видим, ключевые точки отлично отобразились. Но здесь, думаю, сыграло хорошее качество изображения и правильно подобранный наугад размер апертуры. Попробуем с более быстрым вариантом, с апертурой поменьше:


```python
gray = images_temp[1]

keypoints, Ix, Iy = harris_keypoints(gray, minor_size=3)

draw_keypoints(gray, keypoints)
```

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [2.4991448589236085..234.7124602763381].



    
![png](README_files/README_206_1.png)
    





    array([[[255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
            ...,
            [255, 255, 255],
            [255, 255, 255],
            [255, 255, 255]],
    
           [[255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
            ...,
            [255, 255, 255],
            [255, 255, 255],
            [255, 255, 255]],
    
           [[255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
            ...,
            [255, 255, 255],
            [255, 255, 255],
            [255, 255, 255]],
    
           ...,
    
           [[255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
            ...,
            [255, 255, 255],
            [255, 255, 255],
            [255, 255, 255]],
    
           [[255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
            ...,
            [255, 255, 255],
            [255, 255, 255],
            [255, 255, 255]],
    
           [[255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
            ...,
            [255, 255, 255],
            [255, 255, 255],
            [255, 255, 255]]], shape=(480, 640, 3), dtype=uint8)




    <Figure size 640x480 with 0 Axes>


Точек меньше, но очевидно ошибочной можно назвать лишь одну, сверху. Откуда она взялась? На изображении в этом месте блик от окна. Похоже, что это светлое пятно и было обнаружено. Для анализа была написана функция рисующая стрелки направлений для обнаруженных градиентов, посмотрим:


```python
draw_keypoints(gray, keypoints, Ix, Iy, show_vectors=True)
```

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [2.4991448589236085..234.7124602763381].



    
![png](README_files/README_208_1.png)
    





    array([[[255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
            ...,
            [255, 255, 255],
            [255, 255, 255],
            [255, 255, 255]],
    
           [[255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
            ...,
            [255, 255, 255],
            [255, 255, 255],
            [255, 255, 255]],
    
           [[255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
            ...,
            [255, 255, 255],
            [255, 255, 255],
            [255, 255, 255]],
    
           ...,
    
           [[255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
            ...,
            [255, 255, 255],
            [255, 255, 255],
            [255, 255, 255]],
    
           [[255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
            ...,
            [255, 255, 255],
            [255, 255, 255],
            [255, 255, 255]],
    
           [[255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
            ...,
            [255, 255, 255],
            [255, 255, 255],
            [255, 255, 255]]], shape=(480, 640, 3), dtype=uint8)




    <Figure size 640x480 with 0 Axes>


Интересно. Блик на самом деле левее. При ближайшем рассмотрении оказалось, что на холодильнике была точка с грязью. Она темная и поэтому была обнаружена. Получается, нам нужно быть готовым к ошибочным точкам и как то их фильтровать. Рассмотрим это в следующей части. А пока предлагаю для наглядности отобразить градиенты и сравнить с изначальным изображением.


```python
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
```


    
![png](README_files/README_210_0.png)
    


# 2.4 Отфильтровать точки

Перед поиском точек мы заранее задумались о том, чтобы выравнять освещение изображений и избавиться от жестких бликов и прочих плохих факторов, которые могли испортить нам обнаружение. Поэтому мы обогнали задачи для этой работы. Но в итоге у нас все равно остался хоть один, но вброс. Душить его дальше размытием по Гауссу - можно, но неинтересно. А что если грязь была бы побольше по площади? а если бы темнее? Такой расклад сделал бы повторное размытие наприменимым. Более того, с дополнительным размытиеммы уменьшаем эффективность поиска градиентов. Нужен альтернативный метод. Для такие задач используются разные методы. Некоторые из них: DBSCAN, RANSAC.

RANSAC - популярно, круто, но сложно. DBSCAN - тоже круто и популярно, но понятнее. Применяем кластеризацию и потом по соседям фильтруем. Возьмем как основу этот метод, но выкинем из него класетиразацию для упрощения. Будем по количеству соседей проверять точки "в лоб". В нашем случае - отличное решение. Для очень загруженных пятнами изображений уже пригодится упомянутая кластеризация. А пока оставим так:


```python
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
```

Что происходит:

берём точку

смотрим сколько других точек ближе чем radius

если их меньше min_neighbors, считаем её шумом

Твоя точка сверху просто исчезнет, потому что рядом с ней нет соседей.


```python
keypoints_filtered = filter_isolated_points(keypoints)
```


```python
draw_keypoints(gray, keypoints_filtered)
```

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [2.4991448589236085..234.7124602763381].



    
![png](README_files/README_217_1.png)
    





    array([[[255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
            ...,
            [255, 255, 255],
            [255, 255, 255],
            [255, 255, 255]],
    
           [[255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
            ...,
            [255, 255, 255],
            [255, 255, 255],
            [255, 255, 255]],
    
           [[255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
            ...,
            [255, 255, 255],
            [255, 255, 255],
            [255, 255, 255]],
    
           ...,
    
           [[255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
            ...,
            [255, 255, 255],
            [255, 255, 255],
            [255, 255, 255]],
    
           [[255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
            ...,
            [255, 255, 255],
            [255, 255, 255],
            [255, 255, 255]],
    
           [[255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
            ...,
            [255, 255, 255],
            [255, 255, 255],
            [255, 255, 255]]], shape=(480, 640, 3), dtype=uint8)




    <Figure size 640x480 with 0 Axes>


Мы избавились от вброса но с этим потеряли много нужных точек! Настраиваем наш фильтр:


```python
keypoints_filtered = filter_isolated_points(keypoints, 30, 3)
```


```python
draw_keypoints(gray, keypoints_filtered)
```

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [2.4991448589236085..234.7124602763381].



    
![png](README_files/README_220_1.png)
    





    array([[[255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
            ...,
            [255, 255, 255],
            [255, 255, 255],
            [255, 255, 255]],
    
           [[255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
            ...,
            [255, 255, 255],
            [255, 255, 255],
            [255, 255, 255]],
    
           [[255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
            ...,
            [255, 255, 255],
            [255, 255, 255],
            [255, 255, 255]],
    
           ...,
    
           [[255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
            ...,
            [255, 255, 255],
            [255, 255, 255],
            [255, 255, 255]],
    
           [[255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
            ...,
            [255, 255, 255],
            [255, 255, 255],
            [255, 255, 255]],
    
           [[255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
            ...,
            [255, 255, 255],
            [255, 255, 255],
            [255, 255, 255]]], shape=(480, 640, 3), dtype=uint8)




    <Figure size 640x480 with 0 Axes>


Гораздо лучше! Займемся дескрипторами.

 # 2.5 Построить дескрипторы (SIFT)

Дескриптор — это числовое описание окрестности ключевой точки. Он должен быть устойчив к изменению освещения, небольшому повороту и сдвигу — чтобы одна и та же точка на двух разных кадрах давала похожий дескриптор, а разные точки — непохожие.
<p>SIFT (Scale-Invariant Feature Transform) — один из самых известных алгоритмов для этого. Он строит дескриптор из гистограмм градиентов в окрестности точки. Алгоритм состоит из четырёх этапов, разберём каждый.

Подготовим материалы для дальнейшей работы по аналогии с прошлыми шагами:


```python
working_images = []
for i in images_hist_equalized:
    working_images.append(lb1.gaussian_2d(i, 1.2, 21))
    
show_images(working_images)
```


    
![png](README_files/README_225_0.png)
    


Применили размытие на всех изображениях. Это подавит мелкий шум перед вычислением градиентов. Теперь найдём и отобразим ключевые точки для всей серии:


```python
working_images_keypoints = []
working_images_visualised = []
for i in working_images:
    keypoints, Ix, Iy = harris_keypoints(i, minor_size=3)
    keypoints = filter_isolated_points(keypoints, 30, 3)
    working_images_keypoints.append(keypoints)
    working_images_visualised.append(draw_keypoints(i, keypoints))
```

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [2.4991448589236085..234.7124602763381].



    
![png](README_files/README_227_1.png)
    


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [1.7053560773225729..230.01169692807747].



    <Figure size 640x480 with 0 Axes>



    
![png](README_files/README_227_4.png)
    


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [1.7084885130797884..227.01658055830052].



    <Figure size 640x480 with 0 Axes>



    
![png](README_files/README_227_7.png)
    


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [1.3283362435732422..228.96061634182445].



    <Figure size 640x480 with 0 Axes>



    
![png](README_files/README_227_10.png)
    


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [1.5241928455509115e-08..228.15299918746342].



    <Figure size 640x480 with 0 Axes>



    
![png](README_files/README_227_13.png)
    


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [2.195662967528683..234.32658106268428].



    <Figure size 640x480 with 0 Axes>



    
![png](README_files/README_227_16.png)
    


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [1.6660146816096204..238.43859877259345].



    <Figure size 640x480 with 0 Axes>



    
![png](README_files/README_227_19.png)
    


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [4.0161608308704375..235.52770214907088].



    <Figure size 640x480 with 0 Axes>



    
![png](README_files/README_227_22.png)
    



    <Figure size 640x480 with 0 Axes>


Отлично. Ключевые точки найдены на всех кадрах. Заметно, что точки кластеризуются вокруг объектов — магнитов, — что и ожидается: именно там происходят резкие изменения яркости в обоих направлениях. Переходим к теории SIFT.

Так и что мы делаем сейчас? Harris дал точки, но он не умеет их узнавать на другом изображении. Если повернуть картинку, изменить масштаб или освещение — координаты точек изменятся.

Именно поэтому появился алгоритм SIFT. Его задача: для каждой точки построить уникальное числовое описание (дескриптор), которое можно сравнивать между кадрами.

SIFT — идея алгоритма

Алгоритм делает две вещи: находит устойчивые точки интереса и строит для каждой точки вектор признаков, который описывает локальную структуру изображения

Этот вектор потом можно сравнивать между изображениями.

Классический SIFT состоит из 4 этапов:

## 2.5.1 Пирамиды Гаусса

Первый шаг — создать несколько размытых версий изображения.

Это нужно, чтобы точки находились независимо от масштаба.
Мелкие детали исчезают при сильном размытии, а крупные остаются.

Алгоритм строит так называемую пирамиду Гаусса.


```python
def gaussian_pyramid(image, sigmas=[1,2,4,8]):
    
    pyramid = []
    
    for sigma in sigmas:
        blurred = lb1.gaussian_2d(image, sigma, minor_size=17)
        pyramid.append(blurred)
        
    return pyramid
```


```python
pyramid1 = gaussian_pyramid(working_images[0])
show_images(pyramid1, 1, 4)
```


    
![png](README_files/README_234_0.png)
    


На каждом следующем уровне пирамиды изображение размывается сильнее: мелкие детали пропадают, крупные структуры остаются. Это позволяет находить точки на разных масштабах. В нашей задаче камера движется примерно на одном расстоянии от объекта, поэтому масштабная инвариантность не критична: пирамида строится для полноты алгоритма.

## 2.5.2 Разница гауссиан DoG (АХТУНГ!!!)

DoG (Difference of Gaussians) — это разница между соседними уровнями пирамиды Гаусса. Математически это приближение лапласиана гауссиана (LoG), который хорошо реагирует на точки и края.
<p>В оригинальном SIFT именно в DoG-пространстве ищутся экстремумы — точки, которые являются максимумом или минимумом среди 26 соседей (8 в своём слое + 9 выше + 9 ниже). Мы эту функцию реализуем, но использовать для детектирования не будем — у нас уже есть Харис.


```python
def difference_of_gaussians(pyramid):
    
    dogs = []
    
    for i in range(len(pyramid)-1):
        dog = pyramid[i+1] - pyramid[i]
        dogs.append(dog)
        
    return dogs
```

Обычно SIFT ищет точки в DoG, но мы уже реализовали алгоритм Хариса, поэтому использовать я буду его. Пирамида тоже не нужна для детектирования: только для масштабной инвариантности, которая в данной лабе не приоритет, у нас камера +- на одном расстоянии от холодильника.

## 2.5.3 Экстремумы

Шаг 2.5.3 важен — он нужен не для поиска новых точек, а для того чтобы назначить каждой точке доминирующий угол по локальной гистограмме градиентов. Без этого дескриптор не будет инвариантен к повороту.


```python
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
```

Что происходит внутри:
<p>Для каждой точки берётся окно orientation_window_size * orientation_window_size пикселей.
В каждом пикселе считается магнитуда и угол градиента. Вклад каждого пикселя взвешивается на магнитуду и на гауссов вес — пиксели в центре окна важнее, чем на краях.
<p>Вклады накапливаются в гистограмму из 36 бинов (шаг 10°, покрывают 0..360°). Бин с максимальным значением даёт доминирующую ориентацию точки.
<p>На выходе каждая точка (r, c) превращается в тройку (r, c, angle_rad) — теперь дескриптор будет строиться относительно этого угла и станет инвариантен к повороту камеры.


```python
oriented_kp = compute_keypoint_orientations(keypoints, Ix, Iy)
print(f"Точек после ориентации: {len(oriented_kp)}")
```

    Точек после ориентации: 919


Точки получили ориентацию. Переходим к построению самого дескриптора.

## 2.5.4 Построение дескриптора

Для каждой ориентированной точки берём патч 16x16 пикселей и делим его на сетку 4x4 блока (каждый 4x4 пикселя).
<p>В каждом блоке строится гистограмма градиентов по 8 направлениям (бины по 45 градусов). Ключевой момент: угол каждого пикселя считается относительно доминирующей ориентации точки — это и даёт инвариантность к повороту.
<p>16 блоков x 8 бинов = вектор из 128 чисел. Он нормализуется, затем значения обрезаются на уровне 0.2 (это стандартный трюк SIFT для подавления нелинейностей освещения) и нормализуются снова


```python
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
```

Дескриптор готов. Проверим на одном изображении:


```python
valid_kp, descs = compute_sift_descriptors(oriented_kp, Ix, Iy)
print(f"Дескрипторов: {len(descs)}, форма вектора: {descs[0].shape}")
```

    Дескрипторов: 919, форма вектора: (128,)


Супер! 128-мерный вектор для каждой точки получен. Теперь применим пайплайн ко всей серии изображений:


```python
all_keypoints = []
all_descriptors = []

for img in working_images:
    kp, Ix, Iy = harris_keypoints(img, minor_size=3)
    kp = filter_isolated_points(kp, 30, 3)
    oriented = compute_keypoint_orientations(kp, Ix, Iy)
    valid_kp, descs = compute_sift_descriptors(oriented, Ix, Iy)
    all_keypoints.append(valid_kp)
    all_descriptors.append(descs)
```

Дескрипторы построены для всех кадров. Количество точек может отличаться между кадрами — это нормально, часть точек отсеивается у края изображения.

Мы это сделали, теперь приступаем к работе с ними - сопоставим соседние кадры и провизуализируем это!

За одно я решил разобраться с визуализацией картинок, так как теперь придется работать с несколькими сразу. Оформим функцию show_images_any, которая уже и с grayscale и с rgb справится. Почему не редактировал ту? мне лень, а еще в этой версии все будет "плотно" рядом отображено.


```python
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
```

# 2.6 Сопоставить точки между соседними кадрами

Идея: Для каждого дескриптора из кадра A ищем ближайший дескриптор в кадре B по евклидовому расстоянию.

Тест Лоу (Lowe's ratio test): берём два ближайших соседа (best и second_best). Если dist(best) < ratio * dist(second_best), матч считается надёжным. Стандартное значение ratio = 0.75. Смысл: если лучший матч явно лучше второго — он скорее всего правильный. Если они близки по расстоянию — скорее всего оба неправильные.

P.S. Далее "матч" = match. С англ. совпадение.


```python
def euclidean_distance(a, b):
    diff = a - b
    return math.sqrt(float(np.sum(diff * diff)))
```


```python
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
```

Функция euclidean_distance считает расстояние между двумя дескрипторами-векторами.
<p>match_descriptors для каждой точки кадра A перебирает все точки кадра B и находит два ближайших дескриптора. Тест Лоу отсеивает неоднозначные матчи: если лучший и второй по качеству кандидат похожи по расстоянию — значит точка неуникальная и лучше её отбросить. Остаются только чёткие, уверенные совпадения.


```python
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
```

draw_matches склеивает два кадра горизонтально и рисует цветные линии между сопоставленными точками. Каждая пара: своим цветом для наглядности.
<p>Запускаем матчинг для всех соседних пар кадров:


```python
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
```

    
    Кадр 0 → Кадр 1
      Найдено матчей: 475



    
![png](README_files/README_265_1.png)
    


    
    Кадр 1 → Кадр 2
      Найдено матчей: 492



    
![png](README_files/README_265_3.png)
    


    
    Кадр 2 → Кадр 3
      Найдено матчей: 453



    
![png](README_files/README_265_5.png)
    


    
    Кадр 3 → Кадр 4
      Найдено матчей: 506



    
![png](README_files/README_265_7.png)
    


    
    Кадр 4 → Кадр 5
      Найдено матчей: 555



    
![png](README_files/README_265_9.png)
    


    
    Кадр 5 → Кадр 6
      Найдено матчей: 594



    
![png](README_files/README_265_11.png)
    


    
    Кадр 6 → Кадр 7
      Найдено матчей: 615



    
![png](README_files/README_265_13.png)
    


Матчи найдены. Видно, что точки на магнитах уверенно сопоставляются между кадрами.

# 2.7 Вычислить преобразование (поворот + сдвиг)

Модель преобразования:
   У нас только поворот и сдвиг (без масштаба и проективных искажений).
   Это называется rigid body transformation (жёсткое тело):
   <p>
   [x']   [cos θ  -sin θ] [x]   [tx]
   <p>
   [y'] = [sin θ   cos θ] [y] + [ty]
   <p>
   Как считаем? По парам матчей вычисляем угол поворота и вектор сдвига.

   Шаги:
   1. Центрируем обе группы точек (вычитаем центроид)
   2. Для каждой пары считаем угол: atan2(y_b - y_centroid_b, x_b - ...) и т.д.
      Точнее — используем cross и dot product между центрированными векторами,
      это даёт угол поворота по каждой паре.
   3. Усредняем углы (через sin/cos, иначе проблемы с переходом через 0/360).
   4. Вычисляем сдвиг: tx, ty = centroid_b - R @ centroid_a

 P.S. функция также реализует упрощённый RANSAC —
   повторяем выборку случайных пар N раз, берём модель с наибольшим консенсусом.
   Это защищает от неправильных матчей (outliers), которые всегда есть.


```python
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
```

Супер, смотрим трансформации для всей выборки.


```python
transforms = []

for i, matches in enumerate(all_matches):
    print(f"\nТрансформация {i} → {i+1}:")
    angle, tx, ty, inliers = estimate_rotation_translation(matches)
    transforms.append((angle, tx, ty))

    draw_matches(working_images[i], working_images[i+1], inliers)
```


    
![png](README_files/README_272_0.png)
    



    
![png](README_files/README_272_1.png)
    



    
![png](README_files/README_272_2.png)
    



    
![png](README_files/README_272_3.png)
    



    
![png](README_files/README_272_4.png)
    



    
![png](README_files/README_272_5.png)
    



    
![png](README_files/README_272_6.png)
    


После RANSAC остались только матчи, согласующиеся с моделью поворот+сдвиг. Количество inliers относительно общего числа матчей показывает качество сопоставления: чем выше доля тем лучше.

# 2.8 Накопить преобразования и построить траекторию камеры

Теперь у нас есть список трансформаций для каждой соседней пары кадров. Осталось накопить их и получить траекторию.
<p>Первый подход — накапливать (angle, tx, ty) последовательно. Позиция камеры на шаге i+1 вычисляется через позицию на шаге i с учётом накопленного угла. Этот метод работает, но ошибки накапливаются от кадра к кадру.


```python
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
```

build_trajectory идёт по списку трансформаций и на каждом шаге прибавляет к позиции камеры инвертированный сдвиг: если объект уехал вправо на tx камера уехала влево. Угол накапливается отдельно и используется только для отрисовки стрелок направления.

Отлично! Визуализируем:


```python
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
```


```python
positions, angles = build_trajectory(transforms)

labels = [f'img{i+1}' for i in range(len(positions))]
draw_trajectory(positions, angles, image_labels=labels)

print("\nСводная таблица трансформаций:")
print(f"{'Пара':<12} {'Угол (°)':<12} {'tx (px)':<12} {'ty (px)':<12}")
for i, (angle, tx, ty) in enumerate(transforms):
    print(f"{i}→{i+1:<9} {math.degrees(angle):<12.2f} {tx:<12.1f} {ty:<12.1f}")
```


    
![png](README_files/README_280_0.png)
    


Траектория выглядит разумно по форме, однако при сравнении с реальным маршрутом камеры видно, что ошибка замыкания велика: накопленные неточности в трансформациях уводят финишную точку далеко от старта. Попробуем более точный метод — через центроиды ключевых точек.
<p>Идея: вместо того чтобы накапливать трансформации, мы для каждого кадра независимо считаем центр масс всех ключевых точек. Это позиция объекта в пикселях кадра. Сдвиг объекта между кадрами, это и есть движение в системе координат изображения. Ошибки не накапливаются, каждый кадр независим.
 


```python
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
```

build_trajectories_from_keypoints считает центроид всех ключевых точек для каждого кадра.
Смещение относительно первого кадра даёт траекторию объекта. Камера движется строго противоположно: если объект уехал на (dx, dy) в пикселях кадра, камера переместилась на (-dx, -dy) в мировых координатах.


```python
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
```

draw_trajectory_generic это универсальная функция отрисовки. Принимает любой список позиций, заголовок и цвет. Рисует траекторию, подписывает точки, отмечает старт и финиш, показывает пунктиром ошибку замыкания.


```python
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
```

draw_both_trajectories выводит обе траектории рядом для сравнения. Теперь запустим:


```python
labels = [f'img{i+1}' for i in range(len(working_images))]
obj_positions, cam_positions, centroids = build_trajectories_from_keypoints(all_keypoints)
```

Центроиды посчитаны. Рисуем траектории по отдельности:


```python
draw_trajectory_generic(obj_positions, labels, 'Траектория объекта', 'darkorange')
draw_trajectory_generic(cam_positions, labels, 'Траектория камеры',  'steelblue')
```


    
![png](README_files/README_290_0.png)
    



    
![png](README_files/README_290_1.png)
    


И рядом для сравнения:


```python
draw_both_trajectories(obj_positions, cam_positions, labels)
```


    
![png](README_files/README_292_0.png)
    


# Вывод
Задачи лабораторной работы выполнены в полном объеме

Костин Арсений, 8Е21, вариант 3.

# Лабораторная работа №3. Работа с видеопотоком

<p>Цель: Научиться анализировать видеопоток.
<p>Ход работы: получить видеопоток с Web-камеры и определить перемещающийся в кадре объект. Используя данные видеопотока реализуйте следующее:
<p> 1. Реализуйте получение данных с Web-камеры
<p> 2. Реализуйте алгоритм вычитания фона
<p> 3. Реализуйте определение движущегося предмета
<p> 4. Постройте траекторию движения объекта.
<p> 5. Проведите тестирование на тестовом видео.
<p> Проверка работоспособности: будет осуществляться на специальном видео, предоставленным преподавателем. Траектория движения, для которых недоступна.


```python
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import Image
%matplotlib inline
import math
import time
import lab1_functions as lb1
import lab2_functions as lb2
from collections import deque
import os
print(os.getcwd())
print(os.listdir())
```

    /Users/arseniikostin/cv-labs-sem8/labs
    ['sample_image2.png', 'sample_image3.png', 'gradient.png', 'lab3.py', 'histfunc.png', 'lab2.py', 'output.gif', 'sample_image4.jpg', 'sample_image5.jpg', 'lab1_functions.py', 'sequence1.jpeg', 'sequence6.jpeg', 'sequence7.jpeg', '__pycache__', 'doodles.ipynb', 'sequence8.jpeg', 'lab2.ipynb', 'sequence4.jpeg', 'harris1.png', 'sequence5.jpeg', 'gpt-stripfunctions.py', 'gaussfunc.png', 'lab1.py', 'lab3.ipynb', 'sequence2.jpeg', 'clean.ipynb', 'stitch.py', 'lab1.ipynb', 'lab3_v3.ipynb', 'clearoutput.py', 'sample_image.jpg', 'sequence3.jpeg', 'lab2_functions.py', 'combined.ipynb']


# 3.1 Получить видеопоток с веб-камеры

Запись разбита на два отдельных шага — фон и движение — чтобы можно было переснять каждый независимо.

### Шаг 1 — запись фона

Запускаем ячейку, убираем всё из кадра и ждём 3 секунды. Консоль тикает каждую секунду. После записи показываем первый и последний кадр — проверяем что в кадре пусто.


```python
BG_SECONDS  = 3
FPS_APPROX  = 10
N_BG_FRAMES = BG_SECONDS * FPS_APPROX

cap = cv2.VideoCapture(2)

print('ФОН')
print(f'Держите кадр ПУСТЫМ в течение {BG_SECONDS} секунд...')

bg_frames = []
for i in range(N_BG_FRAMES):
    ret, frame = cap.read()
    if ret:
        bg_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if i % FPS_APPROX == 0:
        print('...')
    cv2.waitKey(100)

cap.release()
print(f'Записано кадров фона: {len(bg_frames)}')

f, axarr = plt.subplots(1, 2, figsize=(12, 5))
axarr[0].imshow(bg_frames[0])
axarr[0].set_title('Фон начало')
axarr[1].imshow(bg_frames[-1])
axarr[1].set_title('Фон конец')
for ax in axarr: ax.axis('off')
plt.suptitle('в кадре не должно быть лишних объектов')
plt.show()
```

    === ЗАПИСЬ ФОНА ===
    Держите кадр ПУСТЫМ в течение 3 секунд...
    ...
    ...
    ...
    Записано кадров фона: 30



    
![png](combined345_files/combined345_4_1.png)
    


### Шаг 2 — запись движения

После старта идёт обратный отсчёт 3 ... 1, вносим объект и плавно двигаем его по кадру 5 секунд.


```python
COUNTDOWN_SEC  = 3
MOTION_SECONDS = 5
N_MOT_FRAMES   = MOTION_SECONDS * FPS_APPROX

cap = cv2.VideoCapture(2)

print('запись')
for s in range(COUNTDOWN_SEC, 0, -1):
    print(f'  Старт через {s}...')
    time.sleep(1)
print('двигаем')

motion_frames = []
for i in range(N_MOT_FRAMES):
    ret, frame = cap.read()
    if ret:
        motion_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if i % FPS_APPROX == 0:
        print(f'...')
    cv2.waitKey(100)

cap.release()
frames_sequence = bg_frames + motion_frames
print(f'Записано кадров движения: {len(motion_frames)}')

step = max(1, len(motion_frames) // 4)
f, axarr = plt.subplots(1, 4, figsize=(18, 4))
for i, ax in enumerate(axarr):
    idx = min(i * step, len(motion_frames) - 1)
    ax.imshow(motion_frames[idx])
    ax.set_title(f'Кадр движения #{idx}')
    ax.axis('off')
plt.suptitle('объект должен быть виден и двигаться')
plt.show()
```

    запись
      Старт через 3...
      Старт через 2...
      Старт через 1...
    двигаем
    ...
    ...
    ...
    ...
    ...
    Записано кадров движения: 50



    
![png](combined345_files/combined345_6_1.png)
    


# 3.2 Инициализировать вычитатель фона

Берём все кадры фона и считаем попиксельное среднее — это и есть наша модель фона. Логика та же, что и при любом усреднении: случайные отклонения из-за шума камеры компенсируют друг друга, остаётся стабильная картина пустой сцены.


```python
def build_background(frames):
    bg = np.zeros_like(frames[0], dtype=float)
    for f in frames:
        bg += f.astype(float)
    bg /= len(frames)
    return bg

background = build_background(bg_frames)

plt.figure(figsize=(6, 4))
plt.imshow(background.astype(np.uint8))
plt.title('Модель фона')
plt.axis('off')
plt.show()
```


    
![png](combined345_files/combined345_8_0.png)
    


# 3.3 Применить вычитание фона к кадру

Для каждого кадра считаем абсолютную разность с фоном по каждому каналу RGB. Потом усредняем разность по трём каналам — получаем одноканальное изображение, где яркость пикселя = насколько сильно он отличается от фона.


```python
THRESHOLD = 10

def subtract_background(frame, bg, threshold):
    diff      = np.abs(frame.astype(float) - bg)
    diff_gray = (diff[:, :, 0] + diff[:, :, 1] + diff[:, :, 2]) / 3.0
    mask      = np.zeros(diff_gray.shape, dtype=np.uint8)
    mask[diff_gray > threshold] = 255
    return diff_gray, mask
```

# 3.4 Получить маску переднего плана (foreground mask)

Применяем вычитание к тестовому кадру. Всё что ярче порога — бинаризуется в белый (255), остальное — чёрный (0). Белые пиксели — кандидаты в «движущийся объект».


```python
frame_test = motion_frames[len(motion_frames) // 2]

diff_gray_test, mask_test = subtract_background(frame_test, background, THRESHOLD)

f, axarr = plt.subplots(1, 3, figsize=(15, 5))
axarr[0].imshow(frame_test)
axarr[0].set_title('Кадр с движением')
axarr[1].imshow(diff_gray_test, cmap='gray')
axarr[1].set_title('Разность |кадр - фон|')
axarr[2].imshow(mask_test, cmap='gray')
axarr[2].set_title(f'Бинарная маска (порог = {THRESHOLD})')
for ax in axarr: ax.axis('off')
plt.show()
```


    
![png](combined345_files/combined345_12_0.png)
    


# 3.5 Очистить маску (морфология, шумоподавление)

Сырая маска шумная — на ней много мелких белых пятен от перепадов освещения и шума камеры. Применяем морфологическое открытие: сначала эрозия уничтожает мелкие пятна, потом дилатация возвращает размер оставшимся (настоящим) объектам.

В лабе 1 эрозия и дилатация были реализованы через тройные питоновские циклы — на одном изображении это нормально, но на 50 кадрах видео работало бы более 5 минут. Поэтому здесь делаем то же самое, но через numpy-операции: скользящее минимальное/максимальное по окну через `np.lib.stride_tricks`. Результат идентичный — только быстро.


```python
def fast_erode(mask_bin, size=5):
    pad = size // 2
    padded = np.pad(mask_bin, pad, mode='constant', constant_values=1)
    h, w = mask_bin.shape

    windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size))
    return windows.min(axis=(-2, -1)).astype(np.uint8)

def fast_dilate(mask_bin, size=5):
    pad = size // 2
    padded = np.pad(mask_bin, pad, mode='constant', constant_values=0)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size))
    return windows.max(axis=(-2, -1)).astype(np.uint8)

def clean_mask(mask_uint8, morph_size=5):
    mask_bin = (mask_uint8 // 255).astype(np.uint8)
    eroded   = fast_erode(mask_bin,  morph_size)
    dilated  = fast_dilate(eroded,   morph_size)
    return (dilated * 255).astype(np.uint8)

mask_cleaned = clean_mask(mask_test, morph_size=5)

f, axarr = plt.subplots(1, 2, figsize=(10, 5))
axarr[0].imshow(mask_test,    cmap='gray')
axarr[0].set_title('Маска до очистки')
axarr[1].imshow(mask_cleaned, cmap='gray')
axarr[1].set_title('Маска после эрозии + дилатации')
for ax in axarr: ax.axis('off')
plt.show()
```


    
![png](combined345_files/combined345_14_0.png)
    


# 3.6 Найти контуры движущихся объектов

Контурный пиксель — белый пиксель маски, у которого хотя бы один из четырёх соседей чёрный. То есть он стоит на границе объекта. Проходим по всей маске и собираем такие пиксели в список.


```python
def find_contour_pixels(mask_uint8):
    m    = mask_uint8
    inner = m[1:-1, 1:-1]
    is_white    = inner == 255
    has_black_neighbor = (
        (m[0:-2, 1:-1] == 0) |
        (m[2:,   1:-1] == 0) |
        (m[1:-1, 0:-2] == 0) |
        (m[1:-1, 2:]   == 0)
    )
    contour_map = is_white & has_black_neighbor
    rows, cols  = np.where(contour_map)
    return list(zip(rows + 1, cols + 1))

contour_pixels = find_contour_pixels(mask_cleaned)

frame_with_contour = frame_test.copy()
for (r, c) in contour_pixels:
    frame_with_contour[r, c] = [255, 0, 0]

plt.figure(figsize=(7, 5))
plt.imshow(frame_with_contour)
plt.title(f'Контур объекта (найдено {len(contour_pixels)} пикселей)')
plt.axis('off')
plt.show()
```


    
![png](combined345_files/combined345_16_0.png)
    


# 3.7 Определить и отфильтровать объекты по размеру

На маске может быть несколько белых областей — часть из них шум, который не убрала морфология. Чтобы найти отдельные объекты, используем обход в ширину (BFS): стартуем из непосещённого белого пикселя, обходим все связные с ним — это один объект. Повторяем для всех оставшихся. Компоненты с площадью меньше `min_area` отбрасываем как шум.


```python
def connected_components(mask_uint8, min_area=300):
    visited    = np.zeros_like(mask_uint8, dtype=bool)
    h, w       = mask_uint8.shape
    components = []

    for r in range(h):
        for c in range(w):
            if mask_uint8[r, c] == 255 and not visited[r, c]:
                queue     = deque([(r, c)])
                visited[r, c] = True
                component = []
                while queue:
                    cr, cc = queue.popleft()
                    component.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if mask_uint8[nr, nc] == 255 and not visited[nr, nc]:
                                visited[nr, nc] = True
                                queue.append((nr, nc))
                if len(component) >= min_area:
                    components.append(component)

    return components

components = connected_components(mask_cleaned, min_area=300)
print(f'Найдено объектов после фильтрации по площади: {len(components)}')
for i, comp in enumerate(components):
    print(f'  Объект {i}: {len(comp)} пикселей')
```

    Найдено объектов после фильтрации по площади: 4
      Объект 0: 39430 пикселей
      Объект 1: 2696 пикселей
      Объект 2: 384 пикселей
      Объект 3: 1549 пикселей


# 3.8 Вычислить центроид и bounding box объекта

Центроид — «центр масс» компоненты, то есть среднее по строкам и столбцам всех её пикселей. Bounding box — минимальный охватывающий прямоугольник, находим через min/max строк и столбцов.


```python
def get_centroid_and_bbox(component):
    px = np.array(component)
    centroid_r = px[:, 0].mean()
    centroid_c = px[:, 1].mean()
    r0, c0 = px[:, 0].min(), px[:, 1].min()
    r1, c1 = px[:, 0].max(), px[:, 1].max()
    return (centroid_c, centroid_r), (r0, c0, r1, c1)

vis_frame = frame_test.copy()
for comp in components:
    (cx, cy), (r0, c0, r1, c1) = get_centroid_and_bbox(comp)
    vis_frame[r0, c0:c1] = [255, 0, 0]
    vis_frame[r1, c0:c1] = [255, 0, 0]
    vis_frame[r0:r1, c0] = [255, 0, 0]
    vis_frame[r0:r1, c1] = [255, 0, 0]
    cr, cc = int(cy), int(cx)
    for d in range(-6, 7):
        if 0 <= cr+d < vis_frame.shape[0]: vis_frame[cr+d, cc] = [0, 255, 0]
        if 0 <= cc+d < vis_frame.shape[1]: vis_frame[cr, cc+d] = [0, 255, 0]

plt.figure(figsize=(8, 6))
plt.imshow(vis_frame)
plt.title('Bounding box (красный) и центроид (зелёный)')
plt.axis('off')
plt.show()
```


    
![png](combined345_files/combined345_20_0.png)
    


# 3.9 Накопить координаты центроида и построить траекторию

Запускаем полный конвейер по всем кадрам движения. На каждом кадре: вычитание фона → очистка маски → поиск компонент → берём самую большую → запоминаем центроид. Если объект не найден — пишем `None`.

Траекторию строим через `lb2.draw_trajectory_generic` из второй лабы.


```python
MIN_AREA   = 300
trajectory = []

for idx, frame in enumerate(motion_frames):
    _, mask = subtract_background(frame, background, THRESHOLD)
    mask_cl = clean_mask(mask, morph_size=5)
    comps   = connected_components(mask_cl, min_area=MIN_AREA)

    if comps:
        largest     = max(comps, key=lambda c: len(c))
        (cx, cy), _ = get_centroid_and_bbox(largest)
        trajectory.append((cx, cy))
    else:
        trajectory.append(None)

detected = sum(1 for t in trajectory if t is not None)
print(f'Кадров движения обработано: {len(motion_frames)}')
print(f'Кадров с обнаруженным объектом: {detected}')

if detected == 0:
    print('Объект не найден')

valid_traj  = [(i, p) for i, p in enumerate(trajectory) if p is not None]
pos_list    = [p for p in trajectory if p is not None]
labels_list = [str(i) for i, _ in valid_traj]

if pos_list:
    lb2.draw_trajectory_generic(
        pos_list,
        image_labels=labels_list,
        title='Траектория движущегося объекта',
        color='darkorange'
    )
```

    Кадров движения обработано: 50
    Кадров с обнаруженным объектом: 50



    
![png](combined345_files/combined345_22_1.png)
    


# 3.10 Отобразить результаты (оригинал + маска + траектория + bounding box)

Финальная сводная визуализация. Берём первый кадр, где объект обнаружен, и показываем четыре этапа рядом. На последнем — рисуем накопленную траекторию линиями через алгоритм Брезенхема (рисует прямую попиксельно, без `cv2.line`).


```python
def bresenham_line(img, x0, y0, x1, y1, color):
    dx, dy = abs(x1-x0), abs(y1-y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        if 0 <= y0 < img.shape[0] and 0 <= x0 < img.shape[1]:
            img[y0, x0] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy: err -= dy; x0 += sx
        if e2 <  dx: err += dx; y0 += sy

best_idx   = next((i for i, t in enumerate(trajectory) if t is not None), 0)
best_frame = motion_frames[best_idx]

_, mask_best = subtract_background(best_frame, background, THRESHOLD)
mask_best_cl = clean_mask(mask_best, morph_size=5)
comps_best   = connected_components(mask_best_cl, min_area=MIN_AREA)

frame_result = best_frame.copy()
for comp in comps_best:
    (cx, cy), (r0, c0, r1, c1) = get_centroid_and_bbox(comp)
    frame_result[r0, c0:c1] = [255, 0, 0]
    frame_result[r1, c0:c1] = [255, 0, 0]
    frame_result[r0:r1, c0] = [255, 0, 0]
    frame_result[r0:r1, c1] = [255, 0, 0]
    cr, cc = int(cy), int(cx)
    for d in range(-6, 7):
        if 0 <= cr+d < frame_result.shape[0]: frame_result[cr+d, cc] = [0, 255, 0]
        if 0 <= cc+d < frame_result.shape[1]: frame_result[cr, cc+d] = [0, 255, 0]

prev_pt = None
for pt in trajectory:
    if pt is None:
        prev_pt = None
        continue
    px, py = int(pt[0]), int(pt[1])
    if prev_pt is not None:
        bresenham_line(frame_result, int(prev_pt[0]), int(prev_pt[1]), px, py, [255, 220, 0])
    if 0 <= py < frame_result.shape[0] and 0 <= px < frame_result.shape[1]:
        frame_result[py, px] = [255, 255, 0]
    prev_pt = pt

f, axarr = plt.subplots(1, 4, figsize=(22, 5))
axarr[0].imshow(best_frame)
axarr[0].set_title(f'Оригинал (кадр #{best_idx})')
axarr[1].imshow(mask_best, cmap='gray')
axarr[1].set_title('Маска (до очистки)')
axarr[2].imshow(mask_best_cl, cmap='gray')
axarr[2].set_title('Маска (после морфологии)')
axarr[3].imshow(frame_result)
axarr[3].set_title('Bbox + траектория')
for ax in axarr: ax.axis('off')
plt.tight_layout()
plt.show()
```


    
![png](combined345_files/combined345_24_0.png)
    


# Вывод

В ходе лабораторной работы реализован полный конвейер обнаружения и трекинга движущегося объекта без использования готовых алгоритмов OpenCV.

Модель фона строится как попиксельное среднее по кадрам пустой сцены. Вычитание фона выполняется через абсолютную разность с последующей бинаризацией по порогу. Морфологическое открытие (эрозия + дилатация) реализовано через скользящие numpy-окна (`stride_tricks`) — это сохраняет логику операций из лабы 1, но работает на всём видео за секунды вместо минут. Поиск объектов выполнен BFS по связным компонентам с фильтрацией по площади. Центроид и bounding box вычисляются аналитически через min/max/mean по координатам пикселей. Траектория строится через `lb2.draw_trajectory_generic`, отрисовка линий — алгоритмом Брезенхема.

Костин Арсений, 8Е21, вариант 3.

# Лабораторная работа №4. Разработка алгоритма определения лиц.

<p>Цель: на практике закрепить полученные в ходе курса знания, в том числе по машинному обучению и нейронным сетям для решения задачи детектирования лиц и классификации лиц на мужчин и женщин.
<p>Ход работы: в ходе первой и второй лабораторной каждый из студентов собрал свои фотографии. Данную выборку можно использовать в качестве обучающей выборки для синтеза алгоритмов. Разметку данных каждый студент проводит сам. Алгоритм детектирования и классификации может быть любым.

<p><b>Выбранный метод: HOG + Linear SVM</b>

<p>HOG (Histogram of Oriented Gradients) описывает форму объекта через распределение направлений градиентов — ровно тот же принцип, что использовался в детекторе Харриса в лабе 2, только там мы считали $I_x$, $I_y$ для поиска углов, а здесь строим из них гистограммы для описания внешнего вида патча.

<p>SVM (Support Vector Machine) ищет гиперплоскость, максимально разделяющую два класса в пространстве HOG-признаков.


```python
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import Image
%matplotlib inline
import math
import os
import pickle

import lab1_functions as lb1
import lab2_functions as lb2
import lab3_functions as lb3
from collections import deque

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.datasets import fetch_lfw_people

# Настройка графиков по ГОСТ: шрифт с засечками, 14pt
plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
    'font.size':        14,
    'axes.titlesize':   14,
    'axes.labelsize':   14,
    'xtick.labelsize':  12,
    'ytick.labelsize':  12,
    'legend.fontsize':  12,
    'figure.dpi':       100,
})

print(os.getcwd())
print(os.listdir())
```

    /Users/arseniikostin/cv-labs-sem8/labs
    ['sample_image2.png', 'sample_image3.png', 'gradient.png', 'histfunc.png', 'lab2.py', 'output.gif', 'sample_image4.jpg', 'sample_image5.jpg', 'lab1_functions.py', 'sequence1.jpeg', 'sequence6.jpeg', 'sequence7.jpeg', '__pycache__', 'detector_model.pkl', 'doodles.ipynb', 'sequence8.jpeg', 'lb5cv.png', 'lab2.ipynb', 'sequence4.jpeg', 'gender_model.pkl', 'harris1.png', 'sequence5.jpeg', 'lab4test.py', 'lab4.ipynb', 'gpt-stripfunctions.py', 'live_camera.py', 'gaussfunc.png', 'lab1.py', 'lab4_styled.ipynb', 'lab3.ipynb', 'sequence2.jpeg', 'signals.csv', 'clean.ipynb', 'lab3_functions.py', 'stitch.py', 'lab4_functions.py', 'lab1.ipynb', 'lab5.ipynb', 'lab5_styled.ipynb', 'clearoutput.py', 'sample_image.jpg', 'sequence3.jpeg', 'lab2_functions.py', 'combined.ipynb']


# 4.1 Загрузка датасета

Используем LFW (Labeled Faces in the Wild) — открытый датасет из 13 000+ фотографий лиц публичных людей. Параметр `min_faces_per_person=20` оставляет только тех, кого достаточно много — так модель лучше обобщается и не переобучается на единичные примеры.

Разметку по полу делаем вручную через словарь `GENDER_LABELS` — это и есть ручная разметка данных согласно условию лабы.


```python
print('Загружаем датасет LFW...')
lfw = fetch_lfw_people(min_faces_per_person=20, resize=0.5, color=True)

print(f'Изображений: {lfw.images.shape[0]}')
print(f'Размер патча: {lfw.images.shape[1]}x{lfw.images.shape[2]}')
print(f'Персон: {len(lfw.target_names)}')
print('Имена:', lfw.target_names)
```

    Загружаем датасет LFW...
    Изображений: 3023
    Размер патча: 62x47
    Персон: 62
    Имена: ['Alejandro Toledo' 'Alvaro Uribe' 'Amelie Mauresmo' 'Andre Agassi'
     'Angelina Jolie' 'Ariel Sharon' 'Arnold Schwarzenegger'
     'Atal Bihari Vajpayee' 'Bill Clinton' 'Carlos Menem' 'Colin Powell'
     'David Beckham' 'Donald Rumsfeld' 'George Robertson' 'George W Bush'
     'Gerhard Schroeder' 'Gloria Macapagal Arroyo' 'Gray Davis'
     'Guillermo Coria' 'Hamid Karzai' 'Hans Blix' 'Hugo Chavez' 'Igor Ivanov'
     'Jack Straw' 'Jacques Chirac' 'Jean Chretien' 'Jennifer Aniston'
     'Jennifer Capriati' 'Jennifer Lopez' 'Jeremy Greenstock' 'Jiang Zemin'
     'John Ashcroft' 'John Negroponte' 'Jose Maria Aznar'
     'Juan Carlos Ferrero' 'Junichiro Koizumi' 'Kofi Annan' 'Laura Bush'
     'Lindsay Davenport' 'Lleyton Hewitt' 'Luiz Inacio Lula da Silva'
     'Mahmoud Abbas' 'Megawati Sukarnoputri' 'Michael Bloomberg' 'Naomi Watts'
     'Nestor Kirchner' 'Paul Bremer' 'Pete Sampras' 'Recep Tayyip Erdogan'
     'Ricardo Lagos' 'Roh Moo-hyun' 'Rudolph Giuliani' 'Saddam Hussein'
     'Serena Williams' 'Silvio Berlusconi' 'Tiger Woods' 'Tom Daschle'
     'Tom Ridge' 'Tony Blair' 'Vicente Fox' 'Vladimir Putin' 'Winona Ryder']


Присваиваем метки пола вручную. 0 — мужчина, 1 — женщина. Это разметка данных.


```python
GENDER_LABELS = {
    'Ariel Sharon': 0, 'Colin Powell': 0, 'Donald Rumsfeld': 0,
    'George W Bush': 0, 'Gerhard Schroeder': 0, 'Hugo Chavez': 0,
    'Tony Blair': 0, 'Junichiro Koizumi': 0, 'Jean Chretien': 0,
    'John Ashcroft': 0, 'Vladmir Putin': 0, 'Hamid Karzai': 0,
    'Luiz Inacio Lula da Silva': 0, 'Jacques Chirac': 0, 'Jiang Zemin': 0,
    'Vicente Fox': 0, 'Silvio Berlusconi': 0, 'Alejandro Toledo': 0,
    'John Snow': 0, 'Arnold Schwarzenegger': 0,
    'Lleyton Hewitt': 0, 'Andre Agassi': 0, 'Tiger Woods': 0,
    'Jennifer Aniston': 1, 'Halle Berry': 1, 'Laura Bush': 1,
    'Serena Williams': 1, 'Winona Ryder': 1,
    'Gloria Macapagal Arroyo': 1, 'Condoleezza Rice': 1,
}

gender_labels = []
valid_indices = []

for i, target_id in enumerate(lfw.target):
    name = lfw.target_names[target_id]
    if name in GENDER_LABELS:
        gender_labels.append(GENDER_LABELS[name])
        valid_indices.append(i)

images_valid  = lfw.images[valid_indices]
gender_labels = np.array(gender_labels)

print(f'Изображений с разметкой пола: {len(images_valid)}')
print(f'Мужчин: {(gender_labels == 0).sum()}, Женщин: {(gender_labels == 1).sum()}')

f, axes = plt.subplots(2, 8, figsize=(16, 5))
for row, gender in enumerate([0, 1]):
    idxs = np.where(gender_labels == gender)[0][:8]
    for col, idx in enumerate(idxs):
        axes[row, col].imshow(images_valid[idx])
        name = lfw.target_names[lfw.target[valid_indices[idx]]]
        axes[row, col].set_title(name.split()[-1], fontsize=7)
        axes[row, col].axis('off')
axes[0, 0].set_ylabel('Мужчины', fontsize=10)
axes[1, 0].set_ylabel('Женщины', fontsize=10)
plt.suptitle('Примеры из датасета LFW с ручной разметкой пола')
plt.tight_layout()
plt.show()
```

    Изображений с разметкой пола: 2026
    Мужчин: 1844, Женщин: 182



    
![png](combined345_files/combined345_33_1.png)
    


# 4.2 HOG-дескриптор

HOG — Histogram of Oriented Gradients. Алгоритм пошагово:

**Шаг 1 — градиенты.** Для каждого пикселя считаем $I_x$ и $I_y$ через центральные разности — ровно так же, как в лабе 2 при вычислении детектора Харриса. Из градиентов получаем магнитуду и угол:

$$M = \sqrt{I_x^2 + I_y^2}, \quad \theta = \arctan\!\left(\frac{I_y}{I_x}\right) \bmod 180°$$

Угол берём по модулю 180° (неориентированный) — нам не важно смотрит ли граница вверх или вниз.

**Шаг 2 — ячейки.** Делим изображение на клетки 8×8 пикселей. В каждой ячейке строим гистограмму из 9 бинов по углам (0–180°), взвешенную по магнитуде.

**Шаг 3 — блоки.** Объединяем соседние 2×2 ячейки в блок и нормируем его вектор на L2-норму. Нормировка делает дескриптор устойчивым к изменению освещения.

**Результат** — конкатенация всех нормированных блоков.

### Шаг 1: градиенты

Перевод в grayscale через взвешенную сумму каналов, потом центральные разности.


```python
def to_gray(image_float):
    if len(image_float.shape) == 3:
        return (0.299*image_float[:,:,0] + 0.587*image_float[:,:,1] + 0.114*image_float[:,:,2]) * 255.0
    return image_float * 255.0

def compute_gradients(gray):
    Ix = np.zeros_like(gray)
    Iy = np.zeros_like(gray)
    Ix[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) / 2.0
    Iy[1:-1, :] = (gray[2:, :] - gray[:-2, :]) / 2.0
    magnitude = np.sqrt(Ix**2 + Iy**2)
    angle     = np.degrees(np.arctan2(Iy, Ix)) % 180.0
    return Ix, Iy, magnitude, angle

sample = images_valid[0]
gray   = to_gray(sample)
Ix, Iy, magnitude, angle = compute_gradients(gray)

f, axarr = plt.subplots(1, 4, figsize=(16, 4))
axarr[0].imshow(sample)
axarr[0].set_title('Исходное фото')
axarr[1].imshow(gray, cmap='gray')
axarr[1].set_title('Grayscale')
axarr[2].imshow(magnitude, cmap='hot')
axarr[2].set_title('Магнитуда градиента')
axarr[3].imshow(angle, cmap='hsv')
axarr[3].set_title('Угол градиента, °')
for ax in axarr: ax.axis('off')
plt.tight_layout()
plt.show()
```


    
![png](combined345_files/combined345_36_0.png)
    


### Шаг 2: гистограммы по ячейкам

Сетка ячеек 8×8 пикселей. В каждой — 9-бинная гистограмма, где вес пикселя равен его магнитуде. Посмотрим на гистограммы нескольких ячеек — у ячеек с выраженными краями (брови, контур лица) будут пики в конкретных бинах.


```python
CELL_SIZE = 8
NUM_BINS  = 9
BIN_WIDTH = 180.0 / NUM_BINS

def build_cell_histograms(magnitude, angle, cell_size=8, num_bins=9):
    h, w  = magnitude.shape
    n_cy  = h // cell_size
    n_cx  = w // cell_size
    hists = np.zeros((n_cy, n_cx, num_bins))
    for cy in range(n_cy):
        for cx in range(n_cx):
            r0, r1 = cy*cell_size, (cy+1)*cell_size
            c0, c1 = cx*cell_size, (cx+1)*cell_size
            hist, _ = np.histogram(angle[r0:r1, c0:c1], bins=num_bins,
                                   range=(0, 180), weights=magnitude[r0:r1, c0:c1])
            hists[cy, cx] = hist
    return hists

cell_hists = build_cell_histograms(magnitude, angle, CELL_SIZE, NUM_BINS)
print(f'Сетка ячеек: {cell_hists.shape[0]}×{cell_hists.shape[1]}, бинов на ячейку: {NUM_BINS}')

bins_x = np.arange(NUM_BINS) * BIN_WIDTH
f, axes = plt.subplots(3, 4, figsize=(14, 8))
for i, ax in enumerate(axes.flat):
    cy = (i // 4) + 2
    cx = (i %  4) + 1
    ax.bar(bins_x, cell_hists[cy, cx], width=BIN_WIDTH*0.9, color='steelblue')
    ax.set_title(f'Ячейка [{cy},{cx}]', fontsize=9)
    ax.set_xlabel('Угол, °', fontsize=8)
    ax.set_xticks(bins_x)
    ax.set_xticklabels([f'{int(b)}' for b in bins_x], fontsize=7)
plt.suptitle('Гистограммы ориентированных градиентов по ячейкам')
plt.tight_layout()
plt.show()
```

    Сетка ячеек: 7×5, бинов на ячейку: 9



    
![png](combined345_files/combined345_38_1.png)
    


### Шаг 3: блочная нормировка и финальный дескриптор

Объединяем соседние 2×2 ячейки в блок, нормируем вектор блока:

$$\mathbf{v}_{\text{norm}} = \frac{\mathbf{v}}{\sqrt{\|\mathbf{v}\|^2 + \varepsilon}}$$

Маленький $\varepsilon = 10^{-6}$ защищает от деления на ноль в пустых ячейках. После нормировки все блоки конкатенируем в один вектор — это и есть HOG-дескриптор изображения.


```python
BLOCK_SIZE = 2

def normalize_blocks(cell_hists, block_size=2):
    n_cy, n_cx, num_bins = cell_hists.shape
    descriptor = []
    for by in range(n_cy - block_size + 1):
        for bx in range(n_cx - block_size + 1):
            block = cell_hists[by:by+block_size, bx:bx+block_size].flatten()
            norm  = np.sqrt(np.sum(block**2) + 1e-6)
            descriptor.append(block / norm)
    return np.concatenate(descriptor)

def hog_descriptor(image_float, cell_size=8, num_bins=9, block_size=2):
    gray  = to_gray(image_float)
    _, _, magnitude, angle = compute_gradients(gray)
    hists = build_cell_histograms(magnitude, angle, cell_size, num_bins)
    return normalize_blocks(hists, block_size)

desc_test = hog_descriptor(sample)
print(f'Размер HOG-дескриптора для патча {sample.shape[0]}×{sample.shape[1]}: {len(desc_test)}')
```

    Размер HOG-дескриптора для патча 62×47: 864


# 4.3 Извлечение признаков

Вычисляем HOG для каждого изображения датасета. Каждое изображение → вектор признаков. Все векторы складываем в матрицу `X_faces` (изображений × признаков).


```python
print('Вычисляем HOG для лиц...')
X_faces = []
for i, img in enumerate(images_valid):
    X_faces.append(hog_descriptor(img))
    if (i+1) % 200 == 0:
        print(f'  {i+1}/{len(images_valid)}')

X_faces  = np.array(X_faces)
y_gender = gender_labels.copy()

print(f'Матрица признаков: {X_faces.shape}')
```

    Вычисляем HOG для лиц...
      200/2026
      400/2026
      600/2026
      800/2026
      1000/2026
      1200/2026
      1400/2026
      1600/2026
      1800/2026
      2000/2026
    Матрица признаков: (2026, 864)


## Негативные примеры для детектора лицо/не-лицо

Детектору нужно видеть оба класса — и лица, и не-лица. Берём те же фотографии LFW и создаём из них патчи, которые заведомо не являются лицом:

- **вертикальный переворот** — нос смотрит вверх, лоб внизу, структура нарушена
- **сдвиг вниз + шум сверху** — лоб обрезан, вместо него случайный шум
- **поворот на 90°** — ориентация полностью нарушена
- **поворот на 180°** — перевёрнутое лицо, иная градиентная структура
- **угловой кроп** — угол кадра LFW, там фон или плечи
- **случайный шум** — никакой структуры вообще

Каждый тип берёт разные исходные изображения, поэтому в train и test попадают разные патчи и SVM не может их просто запомнить.


```python
img_h, img_w = lfw.images.shape[1], lfw.images.shape[2]
N_NEG    = len(images_valid)
all_imgs = lfw.images
np.random.seed(42)

def make_negatives(all_imgs, img_h, img_w, n_total):
    n_imgs   = len(all_imgs)
    per_type = n_total // 6 + 1
    negatives = []
    idx = np.random.permutation(n_imgs)

    for i in range(per_type):
        patch = all_imgs[idx[i % n_imgs]].copy()
        negatives.append(patch[::-1, :, :])

    for i in range(per_type):
        patch   = all_imgs[idx[(i + per_type) % n_imgs]].copy()
        shift   = max(1, img_h // 3)
        shifted = np.zeros_like(patch)
        shifted[shift:, :, :]  = patch[:img_h - shift, :, :]
        shifted[:shift, :, :]  = np.random.rand(shift, img_w, 3).astype(np.float32)
        negatives.append(shifted)

    for i in range(per_type):
        patch   = all_imgs[idx[(i + 2*per_type) % n_imgs]].copy()
        rotated = np.transpose(patch, (1, 0, 2))
        if rotated.shape[0] < img_h or rotated.shape[1] < img_w:
            rotated = np.pad(rotated, ((0, max(0, img_h-rotated.shape[0])),
                                       (0, max(0, img_w-rotated.shape[1])),
                                       (0, 0)), mode='edge')
        negatives.append(rotated[:img_h, :img_w, :])

    for i in range(per_type):
        patch = all_imgs[idx[(i + 3*per_type) % n_imgs]].copy()
        negatives.append(patch[::-1, ::-1, :])

    for i in range(per_type):
        src = all_imgs[idx[(i + 4*per_type) % n_imgs]]
        big = np.kron(src, np.ones((2, 2, 1)))
        h_b, w_b = big.shape[:2]
        negatives.append(np.clip(big[h_b-img_h:, w_b-img_w:, :], 0, 1).astype(np.float32))

    for i in range(per_type):
        negatives.append(np.random.rand(img_h, img_w, 3).astype(np.float32))

    return negatives[:n_total]

print('Генерируем негативные примеры (6 типов трансформаций)...')
neg_patches = make_negatives(all_imgs, img_h, img_w, N_NEG)

print('Вычисляем HOG для негативов...')
X_neg = []
for i, patch in enumerate(neg_patches):
    X_neg.append(hog_descriptor(patch))
    if (i+1) % 500 == 0:
        print(f'  {i+1}/{N_NEG}')

X_neg = np.array(X_neg)
y_neg = np.full(N_NEG, -1)

X_detect = np.vstack([X_faces, X_neg])
y_detect = np.concatenate([np.ones(len(X_faces), dtype=int), y_neg])

print(f'Всего для детектора: {X_detect.shape}')
print(f'  лиц: {(y_detect==1).sum()},  не-лиц: {(y_detect==-1).sum()}')

per_type = N_NEG // 6
labels_types = ['вертик. флип', 'сдвиг вниз', 'поворот 90°', 'поворот 180°', 'угловой кроп', 'шум']
f, axes = plt.subplots(6, 4, figsize=(10, 14))
for t in range(6):
    for j in range(4):
        axes[t, j].imshow(np.clip(neg_patches[t * per_type + j], 0, 1))
        axes[t, j].axis('off')
    axes[t, 0].set_ylabel(labels_types[t], fontsize=9)
plt.suptitle('Негативные примеры — 6 типов трансформаций')
plt.tight_layout()
plt.show()
```

    Генерируем негативные примеры (6 типов трансформаций)...
    Вычисляем HOG для негативов...
      500/2026
      1000/2026
      1500/2026
      2000/2026
    Всего для детектора: (4052, 864)
      лиц: 2026,  не-лиц: 2026



    
![png](combined345_files/combined345_44_1.png)
    


# 4.4 Обучение SVM

Обучаем два независимых классификатора.

**Детектор** — бинарный: лицо (+1) или не-лицо (−1). Принимает решение в каждой позиции скользящего окна.

**Классификатор пола** — бинарный: мужчина (0) или женщина (1). Применяется только к патчам, которые детектор уже признал лицом.

Перед обучением масштабируем признаки через `StandardScaler` — вычитаем среднее и делим на стандартное отклонение. Без этого бины с большими значениями будут доминировать и SVM будет работать хуже.

### Детектор лицо/не-лицо


```python
X_tr, X_te, y_tr, y_te = train_test_split(
    X_detect, y_detect, test_size=0.2, random_state=42, stratify=y_detect
)

print('Обучаем детектор...')
detector_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm',    LinearSVC(C=0.1, max_iter=2000))
])
detector_pipe.fit(X_tr, y_tr)

y_pred = detector_pipe.predict(X_te)
print(f'Точность на тесте: {accuracy_score(y_te, y_pred):.3f}')
print(classification_report(y_te, y_pred, target_names=['не-лицо', 'лицо']))
```

    Обучаем детектор...
    Точность на тесте: 0.994
                  precision    recall  f1-score   support
    
         не-лицо       1.00      0.99      0.99       406
            лицо       0.99      1.00      0.99       405
    
        accuracy                           0.99       811
       macro avg       0.99      0.99      0.99       811
    weighted avg       0.99      0.99      0.99       811
    


### Классификатор пола: мужчина / женщина


```python
X_tr_g, X_te_g, y_tr_g, y_te_g = train_test_split(
    X_faces, y_gender, test_size=0.2, random_state=42, stratify=y_gender
)

print('Обучаем классификатор пола...')
gender_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm',    LinearSVC(C=1.0, max_iter=2000))
])
gender_pipe.fit(X_tr_g, y_tr_g)

y_pred_g = gender_pipe.predict(X_te_g)
print(f'Точность на тесте: {accuracy_score(y_te_g, y_pred_g):.3f}')
print(classification_report(y_te_g, y_pred_g, target_names=['мужчина', 'женщина']))

with open('detector_model.pkl', 'wb') as f:
    pickle.dump(detector_pipe, f)
with open('gender_model.pkl', 'wb') as f:
    pickle.dump(gender_pipe, f)
print('Модели сохранены.')
```

    Обучаем классификатор пола...
    Точность на тесте: 0.943
                  precision    recall  f1-score   support
    
         мужчина       0.97      0.96      0.97       370
         женщина       0.67      0.72      0.69        36
    
        accuracy                           0.94       406
       macro avg       0.82      0.84      0.83       406
    weighted avg       0.95      0.94      0.94       406
    
    Модели сохранены.


    /Users/arseniikostin/cv-labs-sem8/venv/lib/python3.14/site-packages/sklearn/svm/_base.py:1258: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      warnings.warn(


# 4.5 Скользящее окно и пирамида масштабов

Детектор обучен на патчах фиксированного размера. Чтобы находить лица разного размера на любом изображении, используем две идеи.

**Пирамида масштабов** — уменьшаем изображение с коэффициентом 0.85 на каждом шаге. На каждом масштабе запускаем скользящее окно. Маленькое окно таким образом «видит» и большие лица — просто уменьшенные.

**Скользящее окно** — перемещаем окно с шагом `step` по строкам и столбцам, для каждой позиции считаем HOG и спрашиваем детектор.

**Non-Maximum Suppression (NMS)** — одно лицо даёт десятки срабатываний в соседних позициях. NMS оставляет только прямоугольник с наибольшей уверенностью из всех перекрывающихся. Перекрытие измеряем через IoU (Intersection over Union):

$$\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|}$$


```python
def sliding_window(image_uint8, win_h, win_w, step=16):
    h, w = image_uint8.shape[:2]
    for r in range(0, h - win_h + 1, step):
        for c in range(0, w - win_w + 1, step):
            yield r, c, image_uint8[r:r+win_h, c:c+win_w]

def image_pyramid(image_uint8, scale=0.85, min_size=64):
    img    = image_uint8.copy()
    factor = 1.0
    while True:
        yield img, factor
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        if new_h < min_size or new_w < min_size:
            break
        img    = cv2.resize(img, (new_w, new_h))
        factor *= scale

def iou(boxA, boxB):
    r0 = max(boxA[0], boxB[0]); c0 = max(boxA[1], boxB[1])
    r1 = min(boxA[2], boxB[2]); c1 = min(boxA[3], boxB[3])
    inter = max(0, r1-r0) * max(0, c1-c0)
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0

def nms(detections, iou_thresh=0.3):
    if not detections:
        return []
    detections = sorted(detections, key=lambda x: x[0], reverse=True)
    kept = []
    while detections:
        best = detections.pop(0)
        kept.append(best)
        detections = [d for d in detections if iou(best[1:], d[1:]) < iou_thresh]
    return kept

def detect_and_classify(image_uint8, detector, gender_clf,
                        win_h, win_w, step=16, scale=0.85,
                        det_threshold=0.5, iou_thresh=0.3):
    detections = []
    for img_scaled, factor in image_pyramid(image_uint8, scale=scale):
        for r, c, patch in sliding_window(img_scaled, win_h, win_w, step):
            desc  = hog_descriptor(patch.astype(np.float32) / 255.0).reshape(1, -1)
            score = detector.decision_function(desc)[0]
            if score > det_threshold:
                r0, c0 = int(r/factor), int(c/factor)
                r1, c1 = int((r+win_h)/factor), int((c+win_w)/factor)
                detections.append((score, r0, c0, r1, c1))

    detections = nms(detections, iou_thresh)

    results = []
    for score, r0, c0, r1, c1 in detections:
        crop = image_uint8[r0:r1, c0:c1]
        if crop.size == 0:
            continue
        face_f = cv2.resize(crop, (win_w, win_h)).astype(np.float32) / 255.0
        gender = gender_clf.predict(hog_descriptor(face_f).reshape(1, -1))[0]
        results.append((r0, c0, r1, c1, gender))

    return results

WIN_H = images_valid.shape[1]
WIN_W = images_valid.shape[2]
print(f'Размер окна детектора: {WIN_H}×{WIN_W} пикселей')
```

    Размер окна детектора: 62×47 пикселей


# 4.6 Тестирование на фотографиях

Берём случайные изображения из тестовой выборки (те, что модель не видела при обучении) и классифицируем пол. Рамка красная — мужчина, синяя — женщина.


```python
test_imgs_idx = np.random.choice(len(X_te_g), 4, replace=False)

f, axes = plt.subplots(1, 4, figsize=(18, 5))
for i, idx in enumerate(test_imgs_idx):
    orig_idx = valid_indices[idx]
    image_u8 = (lfw.images[orig_idx] * 255).astype(np.uint8)

    pred_g = gender_pipe.predict(X_faces[idx].reshape(1, -1))[0]
    true_g = y_gender[idx]

    color_border = [255, 0, 0] if pred_g == 0 else [0, 0, 255]
    img_show = image_u8.copy()
    img_show[0:4, :]  = color_border
    img_show[-4:, :]  = color_border
    img_show[:, 0:4]  = color_border
    img_show[:, -4:]  = color_border

    label_pred = 'Муж' if pred_g == 0 else 'Жен'
    label_true = 'Муж' if true_g == 0 else 'Жен'

    axes[i].imshow(img_show)
    axes[i].set_title(f'Предсказание: {label_pred}\nИстина: {label_true}', fontsize=10)
    axes[i].axis('off')

plt.suptitle('Классификация пола (красный — мужчина, синий — женщина)')
plt.tight_layout()
plt.show()
```


    
![png](combined345_files/combined345_53_0.png)
    


# 4.7 Детектирование в реальном времени с веб-камеры

Живая камера вынесена в отдельный скрипт `live_camera.py` — он открывает окно cv2 и работает пока не нажать **Q**. Перед запуском убедитесь, что ячейка 4.4 выполнена и файлы `detector_model.pkl`, `gender_model.pkl` сохранены в папке с лабами.

**Запуск из терминала:**
```bash
python live_camera.py
```

Параметры в начале скрипта: `CAM_INDEX`, `DET_THRESHOLD`, `STEP`, `SCALE_DOWN`.

# 4.8 Оценка качества

Строим матрицы ошибок для обоих классификаторов на тестовой выборке.


```python
from sklearn.metrics import confusion_matrix

y_pred_det = detector_pipe.predict(X_te)
cm_det     = confusion_matrix(y_te, y_pred_det, labels=[-1, 1])

y_pred_gen = gender_pipe.predict(X_te_g)
cm_gen     = confusion_matrix(y_te_g, y_pred_gen, labels=[0, 1])

f, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, cm, title, labels in [
    (axes[0], cm_det, 'Детектор лицо / не-лицо', ['не-лицо', 'лицо']),
    (axes[1], cm_gen, 'Классификатор пола',       ['мужчина', 'женщина'])
]:
    ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(title)
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_ylabel('Истина')
    ax.set_xlabel('Предсказание')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=14)

plt.tight_layout()
plt.show()

print(f'Точность детектора:           {accuracy_score(y_te, y_pred_det):.3f}')
print(f'Точность классификатора пола: {accuracy_score(y_te_g, y_pred_gen):.3f}')
```


    
![png](combined345_files/combined345_56_0.png)
    


    Точность детектора:           0.994
    Точность классификатора пола: 0.943


# Вывод

В ходе лабораторной работы реализован полный конвейер детектирования лиц и классификации пола на основе HOG + Linear SVM.

HOG-дескриптор вычисляется вручную: градиенты через центральные разности (как в лабе 2), гистограммы ориентаций по ячейкам, блочная L2-нормировка. Детектор на LinearSVC разделяет патчи на «лицо» и «не-лицо», второй SVM классифицирует пол. Для локализации лиц применяется пирамида масштабов и скользящее окно, дублирующиеся срабатывания убираются NMS по порогу IoU. Негативные примеры синтезируются из самого датасета через 6 типов трансформаций, что обеспечивает реальную разнообразность и корректные метрики.

Костин Арсений, 8Е21, вариант 3.

# Лабораторная работа №5. Фильтрация сигналов

<p>Цель: на практике закрепить полученные в ходе курса знания о методах фильтрации сигналов при помощи цифровых фильтров, таких как: экспоненциальное скользящее среднее, медианный фильтр, фильтр Гауса, фильтр Калмана.
<p>Ход работы:
<p>1. Создайте зашумленный тестовый сигнал в ППП Matlab или в среде Matlab Simulink, на основе чистого сигнала с добавлением белого шума.
<p>2. Разработайте и протестируйте следующие алгоритмы фильтрации:
<p>a) Фильтр на основе скользящего среднего;
<p>b) Медианный фильтр;
<p>c) Фильтр Гауса;
<p>В качестве дополнительных фильтров можно реализовать фильтр Калмана.
<p>3. Проведите сравнение работы фильтров, с использованием графической информации (качественная оценка), а также на основе количественной информации (количественная оценка), в качестве которой используйте интегральную квадратичную ошибку.
<p>4. Сделайте выводы о работе фильтров, их особенностях и применимости.


```python
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import os

import lab1_functions as lb1
import lab2_functions as lb2
import lab3_functions as lb3

plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
    'font.size':        14,
    'axes.titlesize':   14,
    'axes.labelsize':   14,
    'xtick.labelsize':  12,
    'ytick.labelsize':  12,
    'legend.fontsize':  12,
    'figure.dpi':       100,
})

print(os.getcwd())
print(os.listdir())
```

    /Users/arseniikostin/cv-labs-sem8/labs
    ['sample_image2.png', 'sample_image3.png', 'gradient.png', 'histfunc.png', 'lab2.py', 'output.gif', 'sample_image4.jpg', 'sample_image5.jpg', 'lab1_functions.py', 'sequence1.jpeg', 'sequence6.jpeg', 'sequence7.jpeg', '__pycache__', 'detector_model.pkl', 'doodles.ipynb', 'sequence8.jpeg', 'lb5cv.png', 'lab2.ipynb', 'sequence4.jpeg', 'gender_model.pkl', 'harris1.png', 'sequence5.jpeg', 'lab4test.py', 'lab4.ipynb', 'gpt-stripfunctions.py', 'live_camera.py', 'gaussfunc.png', 'lab1.py', 'lab3.ipynb', 'sequence2.jpeg', 'signals.csv', 'clean.ipynb', 'lab3_functions.py', 'stitch.py', 'lab4_functions.py', 'lab1.ipynb', 'lab5.ipynb', 'lab5_styled.ipynb', 'clearoutput.py', 'sample_image.jpg', 'sequence3.jpeg', 'lab2_functions.py', 'combined.ipynb']


# 5.0 Создание сигнала

В ПО Matlab Simulink были использованы блоки: Signal Generator (пилообразный сигнал, частота 3, амплитуда 5), Band-Limited White Noise (шум). Выход записан в `signals.csv`.


```python
from IPython.display import Image
Image('lb5cv.png')
```




    
![png](combined345_files/combined345_63_0.png)
    



# 5.1 Загрузка сигнала

Читаем CSV с тремя столбцами: `Time`, `ClearSignal`, `NoisySignal`. Парсим вручную — определяем разделитель (таб или запятая), читаем заголовок, потом строки.


```python
def load_csv(path):
    with open(path, 'r') as f:
        lines = f.read().strip().split('\n')
    sep     = '\t' if '\t' in lines[0] else ','
    headers = lines[0].split(sep)
    columns = {h: [] for h in headers}
    for line in lines[1:]:
        for h, val in zip(headers, line.split(sep)):
            columns[h].append(float(val))
    return {h: np.array(v) for h, v in columns.items()}

data  = load_csv('signals.csv')
t     = data['Time']
clean = data['ClearSignal']
noisy = data['NoisySignal']

print(f'Загружено точек: {len(t)}')
print(f'Временной диапазон: {t[0]:.2f} — {t[-1]:.2f} с')
print(f'Шаг дискретизации dt = {t[1]-t[0]:.4f} с')

plt.figure(figsize=(12, 4))
plt.plot(t, noisy, color='tomato',    alpha=0.6, linewidth=0.8, label='Зашумлённый')
plt.plot(t, clean, color='steelblue', linewidth=1.5,             label='Чистый')
plt.title('Исходный сигнал')
plt.xlabel('Время, с')
plt.ylabel('Амплитуда')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

    Загружено точек: 101
    Временной диапазон: 0.00 — 10.00 с
    Шаг дискретизации dt = 0.1000 с



    
![png](combined345_files/combined345_65_1.png)
    


# 5.2 Метрика качества — интегральная квадратичная ошибка (ИКО)

ИКО показывает насколько отфильтрованный сигнал близок к чистому. Чем меньше — тем лучше:

$$\text{ИКО} = \sum_{i=1}^{N} \left(\hat{x}_i - x_i\right)^2 \cdot \Delta t$$

где $\hat{x}_i$ — отфильтрованное значение, $x_i$ — чистое, $\Delta t$ — шаг дискретизации. Умножение на $\Delta t$ делает метрику независимой от частоты дискретизации — это фактически численный интеграл.


```python
dt = t[1] - t[0]

def ise(filtered, reference, dt):
    return np.sum((filtered - reference) ** 2) * dt

ise_noisy = ise(noisy, clean, dt)
print(f'ИКО зашумлённого сигнала (базовая линия): {ise_noisy:.4f}')
```

    ИКО зашумлённого сигнала (базовая линия): 92.1645


# 5.3 Фильтр скользящего среднего

Скользящее среднее — простейший сглаживающий фильтр. Для каждой точки берём окно из `window` соседних значений и заменяем точку их средним.

Почему это работает: высокочастотный шум случайно колеблется то вверх, то вниз. При усреднении эти отклонения компенсируют друг друга и стремятся к нулю. Чем шире окно — тем сильнее сглаживание, но тем больше задержка и тем сильнее размываются острые перепады.

Сначала проверим идею на одной точке, потом соберём в функцию.


```python
window = 5
i      = 10
half   = window // 2

neighbors = noisy[i - half : i + half + 1]
print(f'Точка i={i}, зашумлённое значение: {noisy[i]:.4f}')
print(f'Соседи в окне {window}: {neighbors}')
print(f'Среднее = {np.mean(neighbors):.4f}')
print(f'Чистое значение: {clean[i]:.4f}')
```

    Точка i=10, зашумлённое значение: 1.6318
    Соседи в окне 5: [ 2.60715991  1.29361908  1.63181353  1.1965239  -0.99470561]
    Среднее = 1.1469
    Чистое значение: 0.1861


Теперь применим ко всему сигналу. На краях используем усечённое окно — не выходим за границы массива.


```python
def moving_average(signal, window):
    n    = len(signal)
    result = np.zeros(n)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        result[i] = np.mean(signal[lo:hi])
    return result

ma_3  = moving_average(noisy, window=3)
ma_7  = moving_average(noisy, window=7)
ma_15 = moving_average(noisy, window=15)

f, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
for ax, filtered, w in zip(axes, [ma_3, ma_7, ma_15], [3, 7, 15]):
    ax.plot(t, noisy,    color='tomato',     alpha=0.4, linewidth=0.8, label='Зашумлённый')
    ax.plot(t, clean,    color='steelblue',  linewidth=1.5,            label='Чистый')
    ax.plot(t, filtered, color='darkorange', linewidth=1.5,            label=f'MA, окно={w}')
    ax.set_title(f'Скользящее среднее, окно = {w}')
    ax.set_xlabel('Время, с')
    ax.legend()
    ax.grid(alpha=0.3)
axes[0].set_ylabel('Амплитуда')
plt.suptitle('Скользящее среднее при разных размерах окна')
plt.tight_layout()
plt.show()

print('ИКО скользящего среднего:')
for filtered, w in zip([ma_3, ma_7, ma_15], [3, 7, 15]):
    print(f'  окно={w:2d}: {ise(filtered, clean, dt):.4f}')
```


    
![png](combined345_files/combined345_71_0.png)
    


    ИКО скользящего среднего:
      окно= 3: 38.8673
      окно= 7: 27.7220
      окно=15: 42.6430


# 5.4 Медианный фильтр

Вместо среднего берём **медиану** окна — значение, которое делит отсортированный набор соседей пополам. Медиана нечувствительна к выбросам: даже если один пиксель в окне случайно улетел в 50 раз от нормы — медиана его проигнорирует.

Это та же логика, что в `lb1.median_2d` из лабы 1, только для одного измерения. Проверим на примере с выбросом:


```python
example = np.array([1.0, 1.2, 1.1, 50.0, 0.9, 1.3, 1.0])
print(f'Данные с выбросом: {example}')
print(f'Среднее: {np.mean(example):.2f}  — сильно искажено выбросом')
print(f'Медиана: {np.median(example):.2f} — выброс не влияет')
```

    Данные с выбросом: [ 1.   1.2  1.1 50.   0.9  1.3  1. ]
    Среднее: 8.07  — сильно искажено выбросом
    Медиана: 1.10 — выброс не влияет


Теперь применим к сигналу. Сортируем окно, берём центральный элемент — как в лабе 1.


```python
def median_filter_1d(signal, window):
    n    = len(signal)
    result = np.zeros(n)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        sorted_vals = np.sort(signal[lo:hi])
        result[i]   = sorted_vals[len(sorted_vals) // 2]
    return result

med_3  = median_filter_1d(noisy, window=3)
med_7  = median_filter_1d(noisy, window=7)
med_15 = median_filter_1d(noisy, window=15)

f, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
for ax, filtered, w in zip(axes, [med_3, med_7, med_15], [3, 7, 15]):
    ax.plot(t, noisy,    color='tomato',   alpha=0.4, linewidth=0.8, label='Зашумлённый')
    ax.plot(t, clean,    color='steelblue',linewidth=1.5,            label='Чистый')
    ax.plot(t, filtered, color='seagreen', linewidth=1.5,            label=f'Медиана, окно={w}')
    ax.set_title(f'Медианный фильтр, окно = {w}')
    ax.set_xlabel('Время, с')
    ax.legend()
    ax.grid(alpha=0.3)
axes[0].set_ylabel('Амплитуда')
plt.suptitle('Медианный фильтр при разных размерах окна')
plt.tight_layout()
plt.show()

print('ИКО медианного фильтра:')
for filtered, w in zip([med_3, med_7, med_15], [3, 7, 15]):
    print(f'  окно={w:2d}: {ise(filtered, clean, dt):.4f}')
```


    
![png](combined345_files/combined345_75_0.png)
    


    ИКО медианного фильтра:
      окно= 3: 46.5027
      окно= 7: 30.7003
      окно=15: 44.2246


# 5.5 Фильтр Гаусса

Гауссовский фильтр — взвешенное скользящее среднее, где вес каждого соседа определяется функцией Гаусса:

$$w_k = \frac{1}{\sigma\sqrt{2\pi}}\, e^{-\frac{k^2}{2\sigma^2}}$$

Ближние соседи влияют больше, далёкие — меньше. Это делает фильтр мягче обычного скользящего среднего: он сглаживает шум, но лучше сохраняет форму сигнала. Та же функция `lb1.Gauss` использовалась в лабе 1 для размытия изображений.

Посмотрим как выглядит ядро при разных $\sigma$:


```python
def gauss_kernel_1d(window, sigma):
    half   = window // 2
    coords = np.arange(-half, half + 1)
    kernel = np.array([lb1.Gauss(k, sigma) for k in coords])
    kernel /= kernel.sum()
    return coords, kernel

plt.figure(figsize=(10, 4))
for sigma in [1.0, 2.0, 4.0]:
    coords, kernel = gauss_kernel_1d(window=15, sigma=sigma)
    plt.plot(coords, kernel, marker='o', markersize=3, label=f'$\\sigma={sigma}$')
plt.title('Гауссово ядро при разных $\\sigma$ (окно = 15)')
plt.xlabel('Позиция в окне')
plt.ylabel('Вес')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print('Сумма весов (должна быть 1):')
for sigma in [1.0, 2.0, 4.0]:
    _, k = gauss_kernel_1d(15, sigma)
    print(f'  sigma={sigma}: {k.sum():.6f}')
```


    
![png](combined345_files/combined345_77_0.png)
    


    Сумма весов (должна быть 1):
      sigma=1.0: 1.000000
      sigma=2.0: 1.000000
      sigma=4.0: 1.000000


Применяем как взвешенную свёртку. На краях перенормируем — делим на сумму весов только тех соседей, что попали в границы сигнала.


```python
def gaussian_filter_1d(signal, window, sigma):
    _, kernel = gauss_kernel_1d(window, sigma)
    n    = len(signal)
    half = window // 2
    result = np.zeros(n)
    for i in range(n):
        acc   = 0.0
        w_sum = 0.0
        for ki, offset in enumerate(range(-half, half + 1)):
            j = i + offset
            if 0 <= j < n:
                acc   += signal[j] * kernel[ki]
                w_sum += kernel[ki]
        result[i] = acc / w_sum
    return result

gauss_s1 = gaussian_filter_1d(noisy, window=15, sigma=1.0)
gauss_s2 = gaussian_filter_1d(noisy, window=15, sigma=2.0)
gauss_s4 = gaussian_filter_1d(noisy, window=15, sigma=4.0)

f, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
for ax, filtered, sigma in zip(axes, [gauss_s1, gauss_s2, gauss_s4], [1.0, 2.0, 4.0]):
    ax.plot(t, noisy,    color='tomato',   alpha=0.4, linewidth=0.8, label='Зашумлённый')
    ax.plot(t, clean,    color='steelblue',linewidth=1.5,            label='Чистый')
    ax.plot(t, filtered, color='purple',   linewidth=1.5,            label=f'Гаусс, $\\sigma={sigma}$')
    ax.set_title(f'Фильтр Гаусса, $\\sigma = {sigma}$')
    ax.set_xlabel('Время, с')
    ax.legend()
    ax.grid(alpha=0.3)
axes[0].set_ylabel('Амплитуда')
plt.suptitle('Фильтр Гаусса при разных $\\sigma$ (окно = 15)')
plt.tight_layout()
plt.show()

print('ИКО фильтра Гаусса:')
for filtered, sigma in zip([gauss_s1, gauss_s2, gauss_s4], [1.0, 2.0, 4.0]):
    print(f'  sigma={sigma}: {ise(filtered, clean, dt):.4f}')
```


    
![png](combined345_files/combined345_79_0.png)
    


    ИКО фильтра Гаусса:
      sigma=1.0: 33.1898
      sigma=2.0: 24.2179
      sigma=4.0: 29.7750


# 5.6 Фильтр Калмана (дополнительный)

Фильтр Калмана — рекуррентный оптимальный фильтр. В отличие от предыдущих он не просто усредняет прошлое — он **предсказывает** следующее состояние и **корректирует** предсказание по новому измерению.

Для скалярного случая:

**Предсказание:**
$$\hat{x}_{k|k-1} = \hat{x}_{k-1}, \qquad P_{k|k-1} = P_{k-1} + Q$$

**Коэффициент Калмана** — баланс между доверием к модели и к измерению:
$$K_k = \frac{P_{k|k-1}}{P_{k|k-1} + R}$$

**Коррекция:**
$$\hat{x}_k = \hat{x}_{k|k-1} + K_k(z_k - \hat{x}_{k|k-1}), \qquad P_k = (1 - K_k)\,P_{k|k-1}$$

где $Q$ — дисперсия шума процесса, $R$ — дисперсия шума измерений. При $Q \ll R$ фильтр доверяет модели и сильно сглаживает. При $Q \gg R$ — доверяет измерениям и почти не фильтрует.

Разберём один шаг вручную:


```python
Q = 0.01
R = 1.0

x_est = noisy[0]
P     = 1.0
z     = noisy[1]

x_pred = x_est
P_pred = P + Q
K      = P_pred / (P_pred + R)
x_new  = x_pred + K * (z - x_pred)

print(f'Измерение z          = {z:.4f}')
print(f'Предсказание         = {x_pred:.4f}')
print(f'Коэфф. Калмана K     = {K:.4f}  (0 = доверяем модели, 1 = доверяем измерению)')
print(f'Скорректир. оценка   = {x_new:.4f}')
print(f'Чистое значение      = {clean[1]:.4f}')
```

    Измерение z          = 10.4861
    Предсказание         = 5.1384
    Коэфф. Калмана K     = 0.5025  (0 = доверяем модели, 1 = доверяем измерению)
    Скорректир. оценка   = 7.8256
    Чистое значение      = 3.7356


Теперь запустим по всему сигналу и посмотрим как меняется поведение при разных $Q$:


```python
def kalman_filter(signal, Q=0.01, R=1.0):
    n      = len(signal)
    result = np.zeros(n)
    x_est  = signal[0]
    P      = 1.0
    result[0] = x_est
    for i in range(1, n):
        P_pred    = P + Q
        K         = P_pred / (P_pred + R)
        x_est     = x_est + K * (signal[i] - x_est)
        P         = (1 - K) * P_pred
        result[i] = x_est
    return result

kalman_tight = kalman_filter(noisy, Q=0.001, R=1.0)
kalman_mid   = kalman_filter(noisy, Q=0.01,  R=1.0)
kalman_loose = kalman_filter(noisy, Q=0.1,   R=1.0)

f, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
for ax, filtered, q in zip(axes, [kalman_tight, kalman_mid, kalman_loose], [0.001, 0.01, 0.1]):
    ax.plot(t, noisy,    color='tomato',    alpha=0.4, linewidth=0.8, label='Зашумлённый')
    ax.plot(t, clean,    color='steelblue', linewidth=1.5,            label='Чистый')
    ax.plot(t, filtered, color='goldenrod', linewidth=1.5,            label=f'Калман, Q={q}')
    ax.set_title(f'Фильтр Калмана, $Q={q}$')
    ax.set_xlabel('Время, с')
    ax.legend()
    ax.grid(alpha=0.3)
axes[0].set_ylabel('Амплитуда')
plt.suptitle('Фильтр Калмана при разных $Q$ ($R=1.0$)')
plt.tight_layout()
plt.show()

print('ИКО фильтра Калмана:')
for filtered, q in zip([kalman_tight, kalman_mid, kalman_loose], [0.001, 0.01, 0.1]):
    print(f'  Q={q}: {ise(filtered, clean, dt):.4f}')
```


    
![png](combined345_files/combined345_83_0.png)
    


    ИКО фильтра Калмана:
      Q=0.001: 61.7831
      Q=0.01: 55.9111
      Q=0.1: 43.3380


# 5.7 Сравнительный анализ

Перебираем параметры каждого фильтра и находим оптимальные по ИКО.


```python
best_ma_ise, best_ma, best_ma_w = float('inf'), None, None
for w in range(3, 30, 2):
    f_ = moving_average(noisy, w)
    e  = ise(f_, clean, dt)
    if e < best_ma_ise:
        best_ma_ise, best_ma, best_ma_w = e, f_, w
print(f'Лучшее MA:      окно={best_ma_w}, ИКО={best_ma_ise:.4f}')

best_med_ise, best_med, best_med_w = float('inf'), None, None
for w in range(3, 30, 2):
    f_ = median_filter_1d(noisy, w)
    e  = ise(f_, clean, dt)
    if e < best_med_ise:
        best_med_ise, best_med, best_med_w = e, f_, w
print(f'Лучший медиан.: окно={best_med_w}, ИКО={best_med_ise:.4f}')

best_gauss_ise, best_gauss, best_gauss_p = float('inf'), None, None
for w in [9, 13, 17, 21]:
    for sigma in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
        f_ = gaussian_filter_1d(noisy, w, sigma)
        e  = ise(f_, clean, dt)
        if e < best_gauss_ise:
            best_gauss_ise, best_gauss, best_gauss_p = e, f_, (w, sigma)
print(f'Лучший Гаусс:   окно={best_gauss_p[0]}, sigma={best_gauss_p[1]}, ИКО={best_gauss_ise:.4f}')

best_kal_ise, best_kal, best_kal_p = float('inf'), None, None
for Q in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
    for R in [0.5, 1.0, 2.0, 5.0]:
        f_ = kalman_filter(noisy, Q=Q, R=R)
        e  = ise(f_, clean, dt)
        if e < best_kal_ise:
            best_kal_ise, best_kal, best_kal_p = e, f_, (Q, R)
print(f'Лучший Калман:  Q={best_kal_p[0]}, R={best_kal_p[1]}, ИКО={best_kal_ise:.4f}')
```

    Лучшее MA:      окно=9, ИКО=27.4124
    Лучший медиан.: окно=5, ИКО=26.7342
    Лучший Гаусс:   окно=21, sigma=2.0, ИКО=24.2170
    Лучший Калман:  Q=0.5, R=2.0, ИКО=41.4477


Теперь все лучшие фильтры на одном графике:


```python
plt.figure(figsize=(14, 5))
plt.plot(t, noisy,      color='tomato',    alpha=0.35, linewidth=0.8, label='Зашумлённый')
plt.plot(t, clean,      color='black',     linewidth=2.0,             label='Чистый')
plt.plot(t, best_ma,    color='darkorange',linewidth=1.5,
         label=f'MA, окно={best_ma_w} (ИКО={best_ma_ise:.3f})')
plt.plot(t, best_med,   color='seagreen',  linewidth=1.5,
         label=f'Медиана, окно={best_med_w} (ИКО={best_med_ise:.3f})')
plt.plot(t, best_gauss, color='purple',    linewidth=1.5,
         label=f'Гаусс, $\\sigma={best_gauss_p[1]}$ (ИКО={best_gauss_ise:.3f})')
plt.plot(t, best_kal,   color='goldenrod', linewidth=1.5,
         label=f'Калман, Q={best_kal_p[0]} (ИКО={best_kal_ise:.3f})')
plt.title('Сравнение фильтров (лучшие параметры по ИКО)')
plt.xlabel('Время, с')
plt.ylabel('Амплитуда')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```


    
![png](combined345_files/combined345_87_0.png)
    


Количественное сравнение — таблица ИКО и столбчатая диаграмма:


```python
filters = [
    ('Зашумлённый (без фильтра)',                                noisy),
    (f'Скользящее среднее (окно={best_ma_w})',                  best_ma),
    (f'Медианный (окно={best_med_w})',                          best_med),
    (f'Гаусс (окно={best_gauss_p[0]}, sigma={best_gauss_p[1]})', best_gauss),
    (f'Калман (Q={best_kal_p[0]}, R={best_kal_p[1]})',          best_kal),
]

print(f'{"Фильтр":<48} {"ИКО":>10} {"Улучшение":>12}')
print('-' * 73)
for name, f_ in filters:
    e    = ise(f_, clean, dt)
    impr = (1 - e / ise_noisy) * 100
    print(f'{name:<48} {e:10.4f} {impr:11.1f}%')

names  = [r[0].split('(')[0].strip() for r in filters]
values = [ise(r[1], clean, dt) for r in filters]
colors = ['tomato', 'darkorange', 'seagreen', 'purple', 'goldenrod']

plt.figure(figsize=(11, 5))
bars = plt.bar(names, values, color=colors)
for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.01 * max(values),
             f'{val:.3f}', ha='center', va='bottom', fontsize=11)
plt.title('Интегральная квадратичная ошибка по фильтрам')
plt.ylabel('ИКО')
plt.xticks(rotation=12, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
```

    Фильтр                                                  ИКО    Улучшение
    -------------------------------------------------------------------------
    Зашумлённый (без фильтра)                           92.1645         0.0%
    Скользящее среднее (окно=9)                         27.4124        70.3%
    Медианный (окно=5)                                  26.7342        71.0%
    Гаусс (окно=21, sigma=2.0)                          24.2170        73.7%
    Калман (Q=0.5, R=2.0)                               41.4477        55.0%



    
![png](combined345_files/combined345_89_1.png)
    


# 5.8 Анализ остатков

Остатки — разность между отфильтрованным сигналом и чистым. У хорошего фильтра остатки должны выглядеть как белый шум без видимой формы — это значит, что никакая структура сигнала не была потеряна и не осталась в шуме.


```python
residuals  = [('MA', best_ma - clean), ('Медиана', best_med - clean),
              ('Гаусс', best_gauss - clean), ('Калман', best_kal - clean)]
res_colors = ['darkorange', 'seagreen', 'purple', 'goldenrod']

f, axes = plt.subplots(2, 2, figsize=(14, 7), sharey=True)
for ax, (name, res), color in zip(axes.flat, residuals, res_colors):
    ax.plot(t, res, color=color, linewidth=0.9)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title(f'{name}  (std = {np.std(res):.4f})')
    ax.set_xlabel('Время, с')
    ax.set_ylabel('Остаток')
    ax.grid(alpha=0.3)
plt.suptitle('Остатки: отфильтрованный $-$ чистый сигнал')
plt.tight_layout()
plt.show()
```


    
![png](combined345_files/combined345_91_0.png)
    


# Вывод

В ходе лабораторной работы реализованы и сравнены четыре алгоритма фильтрации одномерного сигнала.

**Скользящее среднее** — простейший фильтр. Хорошо подавляет равномерный шум, но размывает острые перепады сигнала и вносит задержку пропорционально ширине окна.

**Медианный фильтр** — устойчив к выбросам: единичный шумовой всплеск полностью подавляется, потому что медиана нечувствительна к крайним значениям. На гладких участках проигрывает гауссу.

**Фильтр Гаусса** — взвешенное усреднение с экспоненциально убывающими весами. Сглаживает мягче скользящего среднего, лучше сохраняет форму сигнала. Параметр $\sigma$ удобно контролирует полосу пропускания.

**Фильтр Калмана** — рекуррентный фильтр, балансирует между предсказанием модели и текущим измерением. При правильно подобранных $Q$ и $R$ даёт наименьшую ИКО. Принципиальное отличие — работает без задержки, используя только текущее и предыдущее состояние, что делает его пригодным для реального времени.
