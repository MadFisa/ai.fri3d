
from setuptools import setup, find_packages

setup(
    name='cme',
    
    version='1.0.0.dev1',
    
    description='The model of Coronal Mass Ejection (CME)',

    author='Alexey Isavnin',
    author_email='alexey.isavnin@gmail.com',
    
    license='MIT',
    
    keywords='cme coronal mass ejection sun solar physics model',
    
    packages=find_packages(exclude=['tests*']),
)
