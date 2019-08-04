from ai.fri3d import lib
from scipy import LowLevelCallable, integrate


print(lib)

height = LowLevelCallable.from_cython(lib, "vanilla_axis_height")
res = integrate.quad(height, -0.4, 0.4, (1.0, 0.4, 0.5))

print(res)
