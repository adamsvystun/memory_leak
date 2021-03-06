from matplotlib import pyplot as mp
import numpy as np
import pickle

from gen.math import random_parameters, generate_hours, gaussian_w
from gen.file import write
from gen.parameters import (
    AUTHORS, AUTHOR_TYPES, TYPES, NUMBER_OF_BOOKS, FILENAME,
    NUMBER_OF_TYPES
)

<<<<<<< HEAD
=======
import pickle
import json
>>>>>>> 06317ec826569e6a10df335da4e8d35626cd1af8

def main():
    data = []
    all_demands = 0
    all_hours = 0
    rental_history = []
    for i in range(NUMBER_OF_BOOKS):
        # Get a random author
        author = np.random.randint(0, 5, 1)[0]
        # Get list of types of books this author writes
        author_type = AUTHOR_TYPES[author]
        # Get type of the book from the list randomly
        # If the length of the list is 1 -> no need to randomize
        if len(author_type) == 1:
            book_type = author_type[0]
        else:
            random_type_i = np.random.randint(0, len(author_type), 1)[0]
            book_type = author_type[random_type_i]
        # Get this authors gaussian params
        g_author = AUTHORS[author]
        # Get this book type gaussian params
        g_type = TYPES[book_type]
        # Randomize gaussian params
        type_mu, type_sig = random_parameters(g_type[0], g_type[1])
        type2_mu, type2_sig = None, None
        if len(g_type) == 4:
            type2_mu, type2_sig = random_parameters(g_type[2], g_type[3])
        author_mu, author_sig = random_parameters(g_author[0], g_author[1])
        # Randomize the demand
        demands = np.random.randint(2000, 3000, 1)[0]
        # Generate the demand hours through the year
        all_demands += demands
        hours = generate_hours(
            demands, type_mu, type_sig, author_mu, author_sig, 50, 10,
            type2_mu, type2_sig
        )
        all_hours += len(hours)
        sols = []
        for j in range(len(hours)-1):
            hour = hours[j]
            solution = hours[j+1] - hour
            sols.append(solution)
            current_hour = hour % 24
            day = ((hour - current_hour) / 24)
            current_day_of_week = int(day % 7)
            current_week = int((day - current_day_of_week) / 7)
            # solution = 1 if solution == 0 else solution
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
        # mp.plot(sols)
        # mp.show()
        plot_weeks = [0] * 53
        plot_days = [0] * 7
        for hour in hours:
            solution = hours[j+1] - hour
            sols.append(solution)
            current_hour = hour % 24
            day = ((hour - current_hour) / 24)
            current_day_of_week = int(day % 7)
            current_week = int((day - current_day_of_week) / 7)
            plot_weeks[current_week] += 1
        for hour in hours:
            solution = hours[j+1] - hour
            sols.append(solution)
            current_hour = hour % 24
            day = ((hour - current_hour) / 24)
            current_day_of_week = int(day % 7)
            plot_days[current_day_of_week] += 1
        # rental_history.append(
        #     [author, i, book_type, plot_weeks, plot_days, hours, sols]
        # )
        rental_history.append([int(author), int(i), int(book_type), hours, sols, plot_days, plot_weeks])
    print(len(data))
    with open("test.json", "w") as fp:   #Pickling
        fp.write(json.dumps(rental_history))

    write(FILENAME, data)


if __name__ == "__main__":
    main()
