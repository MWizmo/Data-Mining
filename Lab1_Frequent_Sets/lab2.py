import matplotlib.pyplot as plt
import time
import pyfpgrowth
from lab1 import get_data_from_dat_file


def get_associations_rules(patterns, minconf):
    rules = pyfpgrowth.generate_association_rules(patterns, minconf)
    res = []
    for r in rules:
        res.append((','.join(r), ','.join(rules[r][0])))
    return res


data = get_data_from_dat_file('kosarak.dat.txt', 200000)
minconfs = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
patterns = pyfpgrowth.find_frequent_patterns(data, len(data) * 0.05)
times = []
lengths = []
max_lengths = []
for c in minconfs:
    start = time.time()
    rules = get_associations_rules(patterns, c)
    time.sleep(0.1)
    times.append(time.time() - start)
    print(rules)
    lengths.append(len(rules))
    a = [len(item[0]) + len(item[1]) for item in rules]
    max_lengths.append(max([len(item[0].split(',')) + len(item[1].split(',')) for item in rules]))
print(times)
f, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.plot(minconfs, times, color='r')
ax1.legend(['Время работы'])
ax2.plot(minconfs, lengths, color='g')
ax2.legend(['Число шаблонов'])
ax3.plot(minconfs, max_lengths, color='b')
ax3.legend(['Max длина шаблона'])
plt.show()



