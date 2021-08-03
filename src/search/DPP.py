import numpy as np
import math
import time
from bcutils import load_input_data
from metrics import diversity,coverage


def raw_dpp(kernel_matrix, max_length, epsilon=1E-10):
    """
    Our proposed fast implementation of the greedy algorithm
    :param kernel_matrix: 2-d array
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list
    """
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        di2s[selected_item] = -np.inf
        selected_item = np.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)
    return selected_items
def wrapresult(results,dis,items,user):
    wrap_result = [0.0,[]]
    for result in results:
        wrap_result[1].append((result,
                               dis(items[result],user)))
    return  wrap_result

def mydpp(recall_number, user, items, similarity = lambda u,v:np.dot(u,v), dis = lambda u,v:np.dot(u,v), epsilon=1E-10):
    """
    max_length:要找回元素个数
    user:候选用户embed
    items:候选商品个数
    similarity(u,v):评估相似性的函数
    dis(u,v):评估不相似性的函数
    """
    item_size = items.shape[0]
    cis = np.zeros((recall_number, item_size))
    di2s = np.array([similarity(user,items[i])**2*dis(items[i], items[i]) for i in range(item_size)])
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    while len(selected_items) < recall_number:
        k = len(selected_items) - 1
        print(k)
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = np.array([similarity(user,items[i])*similarity(user,items[selected_item])
                             *dis(items[i], items[selected_item])
                             for i in range(item_size)])
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        di2s[selected_item] = -np.inf
        selected_item = np.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)
    return wrapresult(selected_items,dis,items,user)
if __name__ =='__main__':
    print('cool')
    items, users, tags, neighbors = load_input_data()
    # theta = 0.9
    # similarity = lambda u,v: np.exp(0.9/(2-2*0.9)*np.log(np.linalg.norm(u-v)))
    # similarity = lambda  u,v:np.linalg.norm(u-v)**(4.5)
    similarity = lambda u,v:np.linalg.norm(u-v)
    dis = lambda u,v:np.dot(u/np.linalg.norm(u),v/np.linalg.norm(v))
    start = time.time()
    results = []
    for id,user in enumerate(users[:1]):
        results.append(mydpp(5,user,items,similarity,dis))
    end = time.time()
    print(results)
    print(f'use time {end-start}')
    print(f'diversity is:{diversity(results,tags)},coverage is:{coverage(results,neighbors)}')

