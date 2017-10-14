from datetime import timedelta, datetime

from matplotlib import pyplot as mp
import numpy as np

from gen.math import (
    random_parameters, number_of_demands_per_day, gaussian_d, normalize_day
)
from gen.file import write

# Number of authors 5
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
    data = []
    for i in range(100):
        # Get a random author
        author = np.random.randint(0, 5, 1)[0]
        # Get list of types of books this author writes
        author_type = author_types[author]
        # Get type of the book from the list randomly
        # If the length of the list is 1 -> no need to randomize
        if len(author_type) == 1:
            book_type = author_type[0]
        else:
            random_type_i = np.random.randint(0, 1, 1)[0]
            book_type = author_type[random_type_i]
        # Get this authors gaussian params
        g_author = authors[author]
        # Get this book type gaussian params
        g_type = types[book_type]
        # Randomize gaussian params
        type_mu, type_sig = random_parameters(g_type[0], g_type[1])
        author_mu, author_sig = random_parameters(g_author[0], g_author[1])
        # Randomize the demand
        demands = np.random.randint(2000, 3000, 1)[0]
        # Generate the demand hours through the year
        hours = generate_hours(
            demands, type_mu, type_sig, author_mu, author_sig
        )
        for j in range(len(hours)-1):
            hour = hours[j]
            solution = hours[j+1] - hour
            current_hour = hour % 24
            day = ((hour - current_hour) / 24)
            current_day_of_week += day % 7
            current_week += (day - current_day_of_week) / 7
            line = [
                author,
                i,
                current_week,
                current_day_of_week,
                current_hour,
                book_type,
                solution
            ]
            data.append(line)
    write("data/input.data", data)



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
