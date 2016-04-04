
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

COUNT = 20000

def measure(n):
    "Measurement model, return two coupled measurements."
    m1 = np.random.normal(size=n)
    m2 = np.random.normal(size=n)
    return m1+m2, m1-m2

m1, m2 = measure(COUNT)
m1 += 10

r = np.sqrt(m1**2+m2**2)
phi = np.arctan2(m2, m1)

phi *= 4.0

m1 = r*np.cos(phi)
m2 = r*np.sin(phi)

xmin = m1.min()
xmax = m1.max()
ymin = m2.min()
ymax = m2.max()

X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([m1, m2])
kernel = stats.gaussian_kde(values)
Z = np.reshape(kernel(positions).T, X.shape)

fig, ax = plt.subplots()
kde = ax.contourf(
    np.rot90(Z), 100, 
    cmap=plt.cm.inferno_r,
    extent=[xmin, xmax, ymin, ymax]
)
# kde v= ax.imshow(
#     np.rot90(Z), 
#     cmap=plt.cm.jet,
#     extent=[xmin, xmax, ymin, ymax]
# )
plt.colorbar(kde)
# ax.plot(m1, m2, '.k')
# ax.set_xlim([xmin, xmax])
# ax.set_ylim([ymin, ymax])
plt.show()
