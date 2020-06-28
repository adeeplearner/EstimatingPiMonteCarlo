## Estimating value of $\pi$ using only equation of circle

The code is provided as implementation to estimate value of $\pi$ using Monte Carlo on the following equation of circle only:

$ r^2 =  x^2 + y^2 $

In the sampling-based Monte Carlo approach, we sample a random point in a square grid of length $2r$, where $r$ is radius of a circle that fits inside the square. The sampled point $(x_i, y_i)$ is used in equation $\sqrt{x_i^2 + y_i^2}$ and the result is compared with $r$. If the result is greater than $r$ then the point is identified as out-lier (red points), otherwise it is considered as in-lier (green points). 


Here is output from an experiment, showing in-liers and out-liers:

![animation pi](figure/montecarlo.gif)

We know that the area of square is:
$ area_{sq} = 2r\times2r = 4r^2$

Similarly we know the area of circle is:
$ area_{circ} = \pi r ^ 2$

Then value of $\pi$ can be calculated as:

$\pi = 4 \times \frac{area_{circ}}{area_{sq}} = 4 \times \frac{\pi r^2}{4r^2}$

In our experiments, using n_samples=10000 and n_exp=1000, we found the estimated value of $\pi=3.1410628$

The following shows distributions of results from our experiments with a Gaussian representing the mean and std of the experiments overlayed.

![Geometric pattern with fading gradient](figure/montecarlo_error_gaus.png | width = 200)
