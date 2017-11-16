
class A:
    def __init__(self):
        self._x = None
        self._props = [
            p for p in dir(A)
            if isinstance(getattr(A, p),property)
        ]

    @property
    def x(self):
        return self._x
    @x.setter
    def x(self, val):
        self._x = val

class B(A):
    def __init__(self):
        super(B, self).__init__()
        self._y = None
    @A.x.setter
    def x(self, val):
        self._x = 1
    @property
    def y(self):
        return self._y

b = B()
print(b._props)
