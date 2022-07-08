import numpy as np
from matplotlib import pyplot as plt
import os 

def get_quality1(mask_max):
    return 8 

def get_quality2(mask_max):
    return (mask_max * 16) // 1 

def get_quality3(mask_max):
    return 8 + (mask_max * 8) // 1 

def get_quality4(mask_max):
    return (16 * (mask_max ** (1/3))) // 1
   
def get_quality5(mask_max):
    return (mask_max * 7) // 1 

def get_quality6(mask_max):
    return (16 * (mask_max ** (4))) // 1
   
def get_quality7(mask_max):
    if mask_max > 0.8:
        return 15
    else:
        return 0 


fs = [
    get_quality1,
    get_quality2,
    get_quality3,
    get_quality4,
    get_quality5,
    get_quality6,
    get_quality7,
]

x = np.linspace(0, 1 , 10000)

i = 0

for f in fs:

    i += 1
    if i == 1:
        plt.plot(x, [8]*10000, color='red')  
    elif i == 7: 
        plt.plot(x, [0]*8000 + [16]*2000, color='red')  
    else: 
        plt.plot(x, f(x), color='red')  

    ax = plt.gca()

    ax.set_ylim([-1, 17])

    # os.mkdir("plots/functions")
    plt.savefig('plots/functions/{}.png'.format(i))

    plt.clf()