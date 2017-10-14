from datetime import timedelta, datetime

from matplotlib import pyplot as mp
import numpy as np

from .math import random_parameters, number_of_demands_per_day, gaussian_d

# Number of authors 3
# author #1 20      80 10       Fiction
# author #2 30      10 5        Math, Fiction
# author #3 70      50 10000    Math, Fiction
# author #4 70      50 1        Math
# author #5 70      50 10000    Classics
authors = [
    [80, 10],
    [10, 5],
    [50, 10000],
    [50, 1],
    [50, 10000]
]
author_types = [
    [1],
    [0, 1],
    [1, 0],
    [0],
    [2]
]
# Number books 100
# Number types 3
# type #1 Math      80 10
# type #2 Fiction   40 20
# type #3 Classics  50 10000
types = [
    [80, 10],
    [40, 20],
    [50, 10000]
]
def gen():
    for i in range(100):
        author = np.random.randint(0, 5, 1)[0]
        author_type = author_types[author]
        if len(author_type) == 1:
            book_type = author_type[0]
        else:
            random_type_i = np.random.randint(0, 1, 1)[0]
            book_type = author_type[random_type_i]
        g_author = authors[author]
        g_type = types[book_type]
        type_mu, type_sig = random_parameters(g_type[0], g_type[1])
        author_mu, author_sig = random_parameters(g_author[0], g_author[1])
        demands = np.random.randint(2000, 3000, 1)[0]
        array_for_weeks = number_of_demands_per_day(
            demands, type_mu, type_sig, author_mu, author_sig
        )
        current_day_of_week = 1
        current_week = 1
        for array_for_days in array_for_weeks:
            for per_day in array_for_days:
                per_day = int(per_day)
                if not per_day:
                    continue
                day_distribution = gaussian_d(11, 5, per_day, 24)
                day_distribution = [int(x) for x in day_distribution]
                for current_hour in day_distribution:
                    print([
                        author,
                        i,
                        current_week,
                        current_day_of_week,
                        current_hour,
                        book_type
                    ])
                current_day_of_week += 1
            current_week += 1



def main():
    # number_of_points = 8736
    # # 8736
    # d = gaussian_w(50, 10, 7, 20)
    # # d = gaussian_d(11, 5, 5, 24)
    # print(np.sum(d))
    # mp.plot(d)
    # mp.show()
    gen()

if __name__ == "__main__":
    main()
