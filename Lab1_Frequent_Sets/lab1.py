import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from apyori import apriori
import time
import pyfpgrowth


class Eclat:
    def __init__(self, min_support=0.01, max_items=5, min_items=1):
        self.min_support = min_support
        self.max_items = max_items
        self.min_items = min_items
        self.item_lst = list()
        self.item_len = 0
        self.item_dict = dict()
        self.final_dict = dict()
        self.data_size = 0

    def read_data(self, dataset):
        for index, row in enumerate(dataset):
            row_wo_na = set(row)
            for item in row_wo_na:
                if item in self.item_dict:
                    self.item_dict[item][0] += 1
                else:
                    self.item_dict.setdefault(item, []).append(1)
                self.item_dict[item].append(index)
        self.data_size = len(dataset)
        self.item_lst = list(self.item_dict.keys())
        self.item_len = len(self.item_lst)
        self.min_support = self.min_support * self.data_size

    def recur_eclat(self, item_name, tids_array, minsupp, num_items, k_start):
        if tids_array[0] >= minsupp and num_items <= self.max_items:
            for k in range(k_start + 1, self.item_len):
                if self.item_dict[self.item_lst[k]][0] >= minsupp:
                    new_item = str(item_name) + " | " + str(self.item_lst[k])
                    new_tids = np.intersect1d(tids_array[1:], self.item_dict[self.item_lst[k]][1:])
                    new_tids_size = new_tids.size
                    new_tids = np.insert(new_tids, 0, new_tids_size)
                    if new_tids_size >= minsupp:
                        if num_items >= self.min_items: self.final_dict.update({new_item: new_tids})
                        self.recur_eclat(new_item, new_tids, minsupp, num_items + 1, k)

    def fit(self, dataset):
        i = 0
        self.read_data(dataset)
        for w in self.item_lst:
            self.recur_eclat(w, self.item_dict[w], self.min_support, 2, i)
            i += 1
        return self

    def transform(self):
        return {k: "{0:.4f}%".format((v[0] + 0.0) / self.data_size * 100) for k, v in self.final_dict.items()}


def get_data_from_dat_file(filename, rows):
    data = pd.read_table(filename)
    records = []
    for i in range(1, min(data.shape[0], rows)):
        records.append(data.values[i][0].split())
    return records


def apriori_frequent_sets(data, min_support=0.1):
    result = list(apriori(data, min_support=min_support))
    sets = []
    for fset in result:
        sets.append(list(fset.items))
    return sets


start = time.time()
retail_data = get_data_from_dat_file('retail.dat.txt', 500)       # 88161
tdk_data = get_data_from_dat_file('T10I4D100K.dat.txt', 100)         # 340182
kosarak_data = get_data_from_dat_file('kosarak.dat.txt', 2000)    # 990001
print('Getting data > ', time.time() - start)
#
# s0 = time.time()
# tdk_apriori_res = apriori_frequent_sets(tdk_data, 0.05)
# print(tdk_apriori_res)
# print(time.time() - s0)
#
# s1 = time.time()
# retail_apriori_res = apriori_frequent_sets(retail_data, 0.2)
# print(retail_apriori_res)
# print(time.time() - s1)
#
# s2 = time.time()
# kosarak_apriori_res = apriori_frequent_sets(kosarak_data, 0.2)
# print(kosarak_apriori_res)
# print(time.time() - s2)
#
#
supports = [0.05, 0.1, 0.2]
times = [[], [], []]
lengths = [[], [], []]
max_lengths = [[], [], []]
for s in supports:
    t = time.time()
    patterns = pyfpgrowth.find_frequent_patterns(retail_data, len(retail_data) * s)
    times[0].append(time.time() - t)
    lengths[0].append(len(patterns))
    max_lengths[0].append(max([len(item) for item in patterns]))

    t = time.time()
    retail_apriori_res = apriori_frequent_sets(retail_data, s)
    times[1].append(time.time() - t)
    lengths[1].append(len(retail_apriori_res))
    max_lengths[1].append(len(retail_apriori_res[-1]))

    t = time.time()
    model = Eclat(min_support=s)
    model.fit(retail_data)
    res = model.transform()
    times[2].append(time.time() - t)
    lengths[2].append(len(res))
    max_lengths[2].append(max([len(item.split('|')) for item in res]))


f, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.plot(supports, times[0], color='g')
ax1.plot(supports, times[1], color='r')
ax1.plot(supports, times[2], color='b')
ax1.legend(['FP-Growth', 'Apriori', 'Eclat'], loc=2)

ax2.plot(supports, lengths[0], color='g')
ax2.plot(supports, lengths[1], color='r')
ax2.plot(supports, lengths[2], color='b')
ax2.legend(['FP-Growth', 'Apriori', 'Eclat'], loc=2)

ax3.plot(supports, max_lengths[0], color='g')
ax3.plot(supports, max_lengths[1], color='r')
ax3.plot(supports, max_lengths[2], color='b')
ax3.legend(['FP-Growth', 'Apriori', 'Eclat'], loc=2)
plt.show()
