from prepareData import clock
import numpy as np
import array
import pandas as pd


@clock()
def a():
    with open('fm20170403.txt') as f:
        for line in f:
            pass


@clock()
def b():
     f= pd.read_csv('fm20170403.txt', sep=' ')
     for line in f:
         pass


@clock()
def c():
    f = np.loadtxt('fm20170403.txt')
    for line in f:
        pass


a()
b()
c()

