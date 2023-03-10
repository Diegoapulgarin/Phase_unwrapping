import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from random import randint, choice
#%%


def operate_terms(x, y, cross_term, c, function_list, x_terms=True, y_terms=True, cross_terms=True, multiply=True):
    if multiply:
        mod = 1
    else:
        mod = 0

    if x_terms:
        img1 = function_list[c % len(function_list)](x)
        c += 1
    else:
        img1 = mod

    if y_terms:
        img2 = function_list[c % len(function_list)](y)
        c += 1
    else:
        img2 = mod

    if cross_terms:
        img3 = function_list[c % len(function_list)](cross_term)
        c += 1
    else:
        img3 = mod

    if multiply:
        img = img1 * img2 * img3
    else:
        img = img1 + img2 + img3

    return img, c


def generate_image(x, y, nmax, operation, function_list, multiply=True, **kwargs):
    operations = {
        0: x + y,
        1: x - y,
        2: x * y,
        3: x / y
    }

    c = 0
    img, c = operate_terms(x, y, operations[operation], c, function_list, **kwargs, multiply=multiply)

    while c < len(function_list):
        new_img, c = operate_terms(x, y, operations[operation], c, function_list, **kwargs, multiply=multiply)
        if multiply:
            img *= new_img
        else:
            img += new_img

    img = img / abs(img).max() * nmax
    return img


if __name__ == '__main__':
    data_folder = 'data'
    real_folder = 'real_simple'
    wrapped_folder = 'wrapped_simple'
    n_images = 5000
    start_number = 1
    img_size = 128
    size = complex(f'{img_size}j')

    if data_folder not in os.listdir('.'):
        os.mkdir(data_folder)

    if real_folder not in os.listdir(data_folder):
        os.mkdir(os.path.join(data_folder, real_folder))

    if wrapped_folder not in os.listdir(data_folder):
        os.mkdir(os.path.join(data_folder, wrapped_folder))

    x, y = np.mgrid[-1:1:size, -1:1:size]
    nmax = 2 * np.pi
    # nmax = 11 * np.pi  # 5 phase wrappings

    function_list = [
        np.sin,
        np.cos
    ]

    x_variations = [
        x,
        x ** 2,
    ]

    y_variations = [
        y,
        y ** 2,
    ]

    for n in range(n_images):
        # x_factor = randint(-5, 5)
        x_factor = randint(-3, 3)
        if x_factor == 0:
            x_factor = 1

        # y_factor = randint(-5, 5)
        y_factor = randint(-3, 3)
        if y_factor == 0:
            y_factor = 1

        nm = nmax
        # nm = nmax / randint(1, 5)

        f_list = []
        for i in range(randint(2, 3)):
            f_list.append(choice(function_list))

        # multiply = bool(randint(0, 1))
        multiply = True
        x_terms = bool(randint(0, 1))
        y_terms = bool(randint(0, 1))
        cross_terms = False
        # cross_terms = bool(randint(0, 1))
        operation = 2
        # operation = randint(0, 3)

        while not x_terms and not y_terms and not cross_terms:
            x_terms = bool(randint(0, 1))
            y_terms = bool(randint(0, 1))
            cross_terms = bool(randint(0, 1))

        img = generate_image(choice(x_variations) * x_factor, choice(y_variations) * y_factor, nm, operation, f_list,
                             multiply=multiply, x_terms=x_terms, y_terms=y_terms, cross_terms=cross_terms)

        # img = generate_image(x * x_factor, y * y_factor, nm, operation, f_list,
        #                      multiply=multiply, x_terms=x_terms, y_terms=y_terms, cross_terms=cross_terms)

        filename = f'{n + start_number}.png'
        plt.imsave(os.path.join(data_folder, real_folder, filename), img, cmap='gray')
        plt.close()

        wrapped_img = np.angle(np.exp(1j * img))
        plt.imsave(os.path.join(data_folder, wrapped_folder, filename), wrapped_img, cmap='gray')
        plt.close()
