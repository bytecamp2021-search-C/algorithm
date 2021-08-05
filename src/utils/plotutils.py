import matplotlib.pyplot as plt
import pickle
from src.utils.bcutils import load_object, save_object
import numpy as np


# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']


def draw_coef(coef, coverage, diversity, file='test.PNG'):
    # plt.figure(figsize=(15,7))
    # plt.suptitle('random sensitivity result',fontweight='bold')
    # 分别
    # plt.subplot(131)
    coef, coverage, diversity = np.array(coef), np.array(coverage), np.array(diversity)
    plt.figure()
    plt.title('seperate random sensitivity')
    plt.plot(coef, coverage, color='r', label='coverage')
    plt.plot(coef, diversity, color='b', label='diversity')
    plt.xlabel('randomness')
    plt.ylabel('proportion')
    plt.savefig('random_sensitivity_seperate.png')
    plt.legend()
    # 局部
    # plt.subplot(132)
    plt.figure()
    plt.title('total random sensitivity')
    plt.xlabel('randomness')
    plt.ylabel('proportion')
    plt.plot(coef, 0.7 * coverage + 0.3 * diversity)
    plt.savefig('random_sensitivity_total.png')
    # 总体
    # plt.subplot(133)
    plt.figure()
    plt.title('relevance between coverage and sensitivity')
    plt.xlabel('coverage')
    plt.ylabel('diversity')
    plt.plot(coverage, diversity)
    plt.savefig('diversity_coverage.png')
    plt.show()


def draw_multiline():
    pass


if __name__ == '__main__':
    obj = load_object('../results/eval_results_hnsw_random_0804.obj')
    # obj['diversity']  = np.array(obj['diversity'])/100
    # obj['diversity_coef'] = np.array(obj['diversity_coef'])
    # obj['coverage'] = np.array(obj['coverage'])
    # save_object(obj,'../results/eval_results_hnsw_random_0804.obj')

    draw_coef(obj['diversity_coef'], obj['coverage'], obj['diversity'])
    # print()
