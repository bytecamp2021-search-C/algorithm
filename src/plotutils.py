import matplotlib.pyplot as plt
import numpy as np
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']


def draw(coef,coverage,diversity,file='test.PNG'):
    # plt.figure(figsize=(15,7))
    # plt.suptitle('random sensitivity result',fontweight='bold')
    #分别
    # plt.subplot(131)
    plt.figure()
    plt.title('seperate random sensitivity')
    plt.plot(coef,coverage,color='r',label='coverage')
    plt.plot(coef,diversity,color='b',label='diversity')
    plt.xlabel('randomness')
    plt.ylabel('proportion')
    plt.xlim((-0.1,1))
    plt.ylim((0,100))
    plt.legend()
    #局部
    # plt.subplot(132)
    plt.figure()
    plt.title('total random sensitivity')
    plt.xlabel('randomness')
    plt.ylabel('proportion')
    plt.plot(coef,coverage+diversity)
    #总体
    # plt.subplot(133)
    plt.figure()
    plt.title('relevance between coverage and sensitivity')
    plt.plot(coverage,diversity)
    plt.xlabel('coverage')
    plt.ylabel('diversity')
    #save
    plt.show()
    plt.savefig(file)

def draw_multiline():
    pass


if __name__ =='__main__':
    coef = np.arange(0,1,0.1)
    diversity = np.random.uniform(0,100,(10))
    coverage = np.random.uniform(0,100,(10))
    print(coef.shape,diversity.shape,coverage.shape)
    draw(coef,coverage,diversity)
    # print(coef)