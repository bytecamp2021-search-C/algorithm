import numpy as np


def set_cover_plus(recall_number, dis, tags, kinds, penalty=0.2):
    item_num = len(dis)
    covers = []
    cost = []
    RU = set(kinds)
    U = set(kinds)
    for i in range(len(dis)):
        covers.append(set(tags[i].tolist()))
        count = len(U & covers[i])
        if count == 0:
            count2 = len(RU & covers[i]) / len(covers[i])  # *penalty
            cost.append(np.inf if count2 == 0 else dis[i] / count2)
        else:
            cost.append(dis[i] / count)

    cost = np.array(cost)
    # print(cost)
    select_items = []
    for j in range(recall_number):
        select_item = np.argmin(cost)
        select_items.append(select_item)
        cost[select_item] += np.inf
        U = U - covers[select_item]
        for i in range(item_num):
            if i in select_items:
                continue
            count = len(U & covers[i])
            if count == 0:
                count2 = len(RU & covers[i]) / len(covers[i])  # *penalty
                cost[i] = np.inf if count2 == 0 else dis[i] / count2
            else:
                cost[i] = dis[i] / count
    return select_items


# '''
def approximate_set_cover(recall_number, dis, tags, kinds):
    item_num = len(dis)
    covers = []
    cost = []
    RU = set(kinds)  #
    U = set(kinds)
    for i in range(len(dis)):
        covers.append(set(tags[i].tolist()))
        cost.append(dis[i] / len(covers[i]))
    cost = np.array(cost)
    # print(cost)
    select_items = []
    for j in range(recall_number):
        select_item = np.argmin(cost)
        select_items.append(select_item)
        cost[select_item] += np.inf
        U = U - covers[select_item]
        for i in range(item_num):
            if i in select_items:
                continue
            count = len(U & covers[i])
            if count == 0:
                cost[i] = np.inf
            else:
                cost[i] = dis[i] / count
    return select_items


# '''

if __name__ == '__main__':
    x = [i for i in range(30)]
    dis = [1] * 5
    tags = []
    for i in range(5):
        tags.append(np.random.choice(x, np.random.randint(1, 10), replace=False))
    print(np.array(tags))
    print(approximate_set_cover(5, dis, tags, 30))
