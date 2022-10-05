from projector import Projector , Stimuli
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    proj = Projector()
    proj.stop()
    stim = Stimuli(proj)
    exp = lambda x , speed : speed**x
    n = 1000
    for i in range(5):
        arr1 = np.linspace(2,5,n)
        exp_arr = np.array([exp(j,i) for j in arr1])
        exp_arr /= exp_arr[-1]
        arr = stim._create_shadow(i,n)
        plt.plot(range(n),exp_arr,label=f'{i}^x')
        plt.plot(range(len(arr)),arr,label=f'x^{i}')

    plt.legend()
    plt.show()