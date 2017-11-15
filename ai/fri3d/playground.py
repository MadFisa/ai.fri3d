
class SomeClass:
    def __init__(self):
        self.test()
        setattr(self, '_attr', 1)

    @property
    def prop(self):
        return self._attr
    @prop.setter
    def prop(self, val):
        self._attr = val
    
    def test(self):
        props = [
            p for p in dir(self.__class__)
            if isinstance(getattr(self.__class__, p), property)
        ]
        print(props)



a = SomeClass()

class A:
    def __init__(self):
        self._x = None
    @property
    def x(self):
        return self._x
    @x.setter
    def x(self, val):
        self._x = val
    
    def smth(self):
        do_smth()

# a = A()
# a.x

class B:
    def __init__(self):
        self._x = lambda t: None
    @property
    def x(self):
        return self._x
    @x.setter
    def x(self, func):
        self._x = func
    def do_smth(self, t):
        

# b = B()
# b.x(t)
