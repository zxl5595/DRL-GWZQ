# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 05:14:18 2019

@author: Gaoxiang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

q_table = pd.read_pickle('q_table.pickle')
agent = pd.read_pickle('agent.pickle')
obs = pd.read_pickle('observer.pickle')

def show(table):
    
    df = pd.DataFrame(np.zeros((9,9)), index = range(9), columns = range(9), dtype = np.float64)
    
    for state in table.index:
        x = int(list(state)[1])
        y = int(list(state)[4])
        sur = [[x, y-1, 2], [x, y+1, 3], [x-1, y, 0], [x+1, y, 1]]
        for i in sur:
            if '[{0}, {1}]'.format(i[0], i[1]) in table.index:
                df.loc[x][y] += table.loc['[{0}, {1}]'.format(i[0], i[1])][i[2]]
            else:
                pass
    print(df)
    
    
    fig = plt.figure()
    ax = Axes3D(fig)
    
    X = np.arange(0, 9)
    Y = np.arange(0, 9)
    X, Y = np.meshgrid(X, Y)
    
    Z = np.zeros((9,9))
    for x in range(9):
        for y in range(9):
            Z[x][y] = df.loc[x, y]
    
    ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, cmap=plt.get_cmap('rainbow'))
    ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap=plt.get_cmap('rainbow'))
    plt.show()


show(q_table)
show(agent)
show(obs)