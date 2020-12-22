'''
https://colab.research.google.com/drive/1oPOkmqTZhQx2Gr6Su6jVoM3nnqwP1Qsa#scrollTo=kksxdqD5ztoE
'''
from typing import Dict, List, Set, Tuple

import numpy
import numpy as np
import matplotlib.pyplot as plt
import random

#1
def price_statistics(shop_price_dict: Dict) -> Dict:
    result = {}
    result.update({'min' : min(shop_price_dict.values())})
    result.update({'max' : max(shop_price_dict.values())})
    result.update({'mean' : int(np.mean(list(shop_price_dict.values())))})
    result.update({'median' : int(np.median(list(shop_price_dict.values())))})
    return result

#2
def filter_shops_by_price(shop_price_dict: Dict, price_min: float, price_max: float) -> Set[str]:
    resultSet = set()
    for (key, value) in shop_price_dict.items():
        if value > price_min and value < price_max:
            resultSet.add(key)
    return resultSet

#3
def dict_inv(input_dict: Dict) -> Dict:
    result = {} #as good: shopsList
    for (shop, goodsList) in input_dict.items():
        for good in goodsList:
            if good not in result.keys():
                result.update({good: [shop]})
            else:
                shopsList = list(result.get(good))
                shopsList.append(shop)
                result.update({good: shopsList})
    return result

#4
def happy_b_day(a: List[int]) -> List[int]:
    accumulator = []
    result = []
    for i in a:
        if i not in accumulator:
            accumulator.append(i)
        else:
            result.append(i)
    result.sort()
    return result

#5
'''
HERE IS
n - 365 days in a year
k - pupils
WRONG!
https://klikunov-nd.livejournal.com/152859.html
Probability = n*(n-1)*(n-2)*…*(n-k+1) / n^(k-1)



https://ru.wikipedia.org/wiki/%D0%9F%D0%B0%D1%80%D0%B0%D0%B4%D0%BE%D0%BA%D1%81_%D0%B4%D0%BD%D0%B5%D0%B9_%D1%80%D0%BE%D0%B6%D0%B4%D0%B5%D0%BD%D0%B8%D1%8F

'''
def collision_probability(d: List[List[int]]) -> float:
    DAYS_IN_YEAR = 365
    kAsClasses = len(d) # The number of Classes (k aka Classes)

    avgPupilsList = []
    for i in d:
        avgPupilsList.append(len(i))
    nAsPupilsInAverage = int(numpy.mean(avgPupilsList, axis=0)) # The average number of pupils in all classes (n aka Pupils)

    probabilityList = []
    for pupils in d:
        pupilsCount = len(pupils)
        if pupilsCount > DAYS_IN_YEAR:
            probabilityList.append(1.0)
            continue

        allRepetitionsCount = numpy.math.factorial(DAYS_IN_YEAR)
        possibleRepetitionsCount = numpy.math.factorial((DAYS_IN_YEAR - pupilsCount))

        probability = 1 - allRepetitionsCount / possibleRepetitionsCount / pow(DAYS_IN_YEAR, pupilsCount)
        # print("Prob: " + str(probability))

        probabilityList.append(probability)

    result = numpy.mean(probabilityList)
    return float(result)



#6
def matrix_multiplication(a, b):
    return a.dot(b)

##########################################
##########################################
##########################################
##########################################
##########################################

shop_price_dict =  {'citrus':      47999,
                    'newtime':     37530,
                    'buyua':       39032,
                    'storeinua':   37572,
                    'allo':        48499,
                    'istore':      39999,
                    'tehnokrat':   39340,
                    'estore':      40169,
                    'gstore':      40792,
                    'touch':       39330,
                    'bigmag':      37900,}
# 1
''' 'min': 37530, 'max': 48499, 'mean': 40742, 'median': 39340'''
# print(price_statistics(shop_price_dict))

# 2
''' (39000, 40000) -> {'istore', 'buyua', 'touch', 'tehnokrat'}'''
# print(filter_shops_by_price(shop_price_dict, 39000, 40000))

# 3
'''
input
{'rozetka': ['iphone', 'macbook', 'ipad'],
'fua': ['macbook', 'ipad'],
'citrus': ['iphone', 'macbook', 'earpods'],
'allo': ['earpods', 'iphone']}

output:
{'earpods': ['allo', 'citrus'],
 'ipad': ['rozetka', 'fua'],
 'iphone': ['rozetka', 'citrus', 'allo'],
 'macbook': ['rozetka', 'fua', 'citrus']}
'''
input_dict =   {'rozetka': ['iphone', 'macbook', 'ipad'],
                'fua': ['macbook', 'ipad'],
                'citrus': ['iphone', 'macbook', 'earpods'],
                'allo': ['earpods', 'iphone']}
# print(dict_inv(input_dict))


#4
'''
a=[196,171,119,86,198,296,354,146,284,142,197,296,5,128,339,10,225,80,77,164,86,132,336,354,318] 
Output:  [86,296,354]
'''
random.seed(41)
a = [random.randint(1, 365) for _ in range(25)]
a=[196,171,119,86,198,296,354,146,284,142,197,296,5,128,339,10,225,80,77,164,86,132,336,354,318]
# print(happy_b_day(a))



#5
'''
Напишите функцию, принимающую датасет  List[List[int]]  и возвращающую вероятность  float , 
посчитанную по "k" классам того, что в рандомном классе размером  n  будет хотя бы один особый день рождения с коллизиями.

В нашем примере,  n=6,k=3  коллизий нет. Но если мы возьмем классы по 25ть человек, 
и при этом соберем относительно большую статистику, например, по 10000 классам результат будет более интересный. 
Найдите вероятность коллизии дней рождений для такого класса. Вы можете получить разные результаты, но при  k=10000 
погрешность не будет незначительной.

d=[[196,171,119,86,198,296],[354,146,284,142,197,296],[5,128,339,10,225,80]]
'''
random.seed(41)
# n = 6
# k = 3
n = 50
k = 10000
def new_class(n: int) -> List[int]:
    return [random.randint(1, 365) for _ in range(n)]

def new_class_dataset(n: int, k: int) -> List[List[int]]:
    return [new_class(n) for _ in range(k)]

# d = new_class_dataset(n, k)
# d=[[196,171,119,86,198,296], [354,146,284,142,197,296], [5,128,339,10,225,80]]
# print(d)
# print(collision_probability(d))



#6
'''
Matrix multiplication
Point on plane frawing

LaTeX
$$ \left[
  \begin{array}{ c c }
     0.7 & -0.7 \\
     0.7 & 0.7
  \end{array} \right]
  \cdot
  \left[
  \begin{array}{ c }
     2 \\
     3
  \end{array} \right]
  =
    \left[
  \begin{array}{ c }
     3.5 \\
     0.7
  \end{array} \right]
$$
'''
a = np.array([[0.7, 0.7],
             [-0.7, 0.7]])
b = np.array([[2.0], [3.0]])
c = matrix_multiplication(a, b)
print(c)
plt.plot(c)
plt.ylabel('Y axis')
plt.xlabel('X axis')
plt.show()


#7
'''
\begin{align}
E &= \frac{\partial}{\partial x} e^{2x} + z^4x + x^3 - 2z + 3;
\end{align}
'''


#8
'''
https://www.matburo.ru/tvart_sub.php?p=calc_gg_ball

********************************

**Решение и рассуждения согласно закона распределения случайной величины**


Общее число возможных элементарных исходов для данных испытаний равно числу способов, которыми можно извлечь 2 шаров из 7:
\begin{align}
С_7^2 &= \frac{7!}{2!(7-2)!} &= \frac{7!}{2!5!} &= 21
\end{align}

Вероятность что вынуто 0 белых шаров из 5:
\begin{align}
С_5^0 &= \frac{5!}{0!(5-0)!} &= \frac{5!}{0!5!} &= \frac{1}{1} &= 1
\end{align}

Вероятность что вынуто 2 черных шара из 2:
\begin{align}
С_2^2 &= \frac{2!}{2!(2-2)!} &= \frac{2!}{2!0!} &= \frac{1}{1} &= 1
\end{align}

Вероятность того что 2 выбранных шара черные:
\begin{align}
P(2) &= \frac{С_5^0 \cdot С_2^2}{С_7^2} &= \frac{1\cdot1} {21} &= 0.04761
\end{align}
'''