import numpy as np
from scipy.integrate import fixed_quad, quad
# import quadpy
import time
from matplotlib import pyplot as plt
from scipy import interpolate


def f(x, n):
    return(np.cos(x)**n)

def df(x, n):
    return(-n*np.cos(x)**(n-1)*np.sin(x))

def ds(x, n):
    return(np.sqrt(f(x, n)**2+df(x, n)**2))

# def s(x, n, r=1-1e-5):
#     if x <= -np.pi/2*r:


r = 1-1e-5
print(r)
print(f(-np.pi/2*r, 0.5))
# print(ds(-np.pi/2*r, 0.1))
n = 0.5
func = lambda x: ds(x, n)
print(
    fixed_quad(
        func, 
        -np.pi/2*r, 
        -np.pi/2*0.9, 
        n=1000
    )[0]+f(-np.pi/2*r, n)
)
exit

def get_spline():
    arrx = np.linspace(-np.pi/2, np.pi/2, 1000)
    arrn = np.linspace(0.1, 1.0, 100)
    arrs = []
    for n in arrn:
        func = lambda x: ds(x, n)
        arrs.append(
            np.array(
                [fixed_quad(func, -np.pi/2, x, n=1000)[0] for x in arrx]
            )
        )
    arrs = np.array(arrs)
    return(interpolate.interp2d(arrx, arrn, arrs))


# def s(n):
#     x = np.linspace(-np.pi/2, np.pi/2, 1000)
    
    


    # plt.show()

    # plt.plot([n], [fixed_quad(func, -np.pi/2, 0, n=10000)[0]], '+')

    # start = time.clock()
    # for i in range(10):
    #     s1 = fixed_quad(func, -np.pi/2, np.pi/2, n=10000)
    # print(s1, time.clock()-start)
    
    # start = time.clock()
    # for i in range(10):
    #     s2 = quadpy.line_segment.integrate(
    #         func,
    #         np.array([[-np.pi/2], [np.pi/2]]),
    #         quadpy.line_segment.GaussKronrod(1)
    #     )
    # print(s2, time.clock()-start)

    # start = time.clock()
    # for i in range(10):
    #     s4 = quadpy.line_segment.integrate(
    #         func,
    #         np.array([[-np.pi/2], [np.pi/2]]),
    #         quadpy.line_segment.NewtonCotesOpen(5)
    #     )
    # print(s4, time.clock()-start)

s = get_spline()

x = arrx = np.linspace(-np.pi/2, np.pi/2, 1000)

plt.plot(x, s(x, 0.1), label="0.1")
plt.plot(x, s(x, 0.3), label="0.3")
plt.plot(x, s(x, 0.4), label="0.4")
plt.plot(x, s(x, 0.5), label="0.5")
plt.plot(x, s(x, 0.6), label="0.6")
plt.plot(x, s(x, 0.7), label="0.7")
plt.plot(x, s(x, 0.8), label="0.8")
plt.plot(x, s(x, 0.9), label="0.9")
plt.plot(x, s(x, 1.0), label="1.0")
plt.legend()

plt.show()
