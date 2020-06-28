"""
Example showing how to estimate the value of PI 
using Monte-Carlo approximation on equation of circle
"""
import argparse
import math
import os
import matplotlib
matplotlib.use('Agg')

from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()


def circle_equation(x, y, x_i, y_i):
    # Implementation of circle equation that returns
    # r = sqrt( (x-x_i)**2 + (y-y_i)**2
    return np.sqrt((x-x_i) ** 2 + (y-y_i) ** 2)


def estimate_pi(n_samples=100, r=100, save_path=None):
    
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    # define our sampling space
    h, w = 2 * r, 2 * r

    y_i, x_i = r, r

    # create image to display 
    display_image = np.ones((h, w, 3)).astype(np.uint8) * 255

    outside_count = 0
    inside_count = 0
    for ns in range(n_samples):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)

        r_i = circle_equation(x, y, x_i, y_i)

        if r_i > r:
            # outside circle
            outside_count += 1
            # put a red dot in image
            display_image[y, x, :] = [244,67,54]
            
        elif r_i <= r:
            inside_count += 1
            # put a green dot in image
            display_image[y, x, :] = [0,230,118]

        if save_path is not None and ns % 1000 == 0:
            Image.fromarray(display_image).save(
                os.path.join(save_path, 'image_%.9d.png' % ns))

    # (pi * r * r) / (2r * 2r)
    return 4*inside_count/(outside_count+inside_count)

def Gaussian1D(x, mu, sigma):
    ATerm = 1/(sigma * np.sqrt(2 * math.pi))
    BTerm = np.exp(-0.5 * ((x-mu)/sigma) ** 2)
    return ATerm * BTerm

def parse_args():
    parser = argparse.ArgumentParser(
        "Estimating PI with multiple Monte-Carlo simulations on equation of circle")
    parser.add_argument('--n_samples', type=int, default=10000,
                        help='number of samples to use', required=False)
    parser.add_argument('--n_exp', type=int, default=1000,
                        help='number of repeated experiments', required=False)
    parser.add_argument('--radius', type=int, default=100,
                        help='radius of the circle used in circle equation', required=False)
    parser.add_argument('--save_path', type=str, default=None,
                        help='path to save output images', required=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # create folder if needed
    if args.save_path != None:
        os.makedirs(args.save_path, exist_ok=True)

    
    results = np.zeros((args.n_exp))
    
    for n in range(args.n_exp):
        print('%d/%d' % (n, args.n_exp))
        results[n] = estimate_pi(args.n_samples, args.radius, args.save_path)

    # if we ran more than 1 experiment
    if args.n_exp > 1:
        freq, edges = np.histogram(results, bins=100)

        # normalise histogram
        norm_factor = np.sum(freq) * (edges[1] - edges[0])
        freq = freq/norm_factor

        plt.bar(edges[:-1], freq, width=np.diff(edges))

        # find mean and variance
        mean_val = np.mean(results)
        var_val = np.var(results)

        # calculate gaussian

        # define eval grid
        x = np.linspace(np.min(results), np.max(results), 100)
        
        # evaluate a 1D gaussian
        y = Gaussian1D(x, mean_val, math.sqrt(var_val))
        plt.plot(x, y, 'r')

        os.makedirs('figure', exist_ok=True)
        plt.savefig('figure/montecarlo_error_gaus.png')

    print(mean_val)