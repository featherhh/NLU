# import numpy as np
# X = [1, 3, 4,5]
#
# index = np.arange(len(X))
# np.random.shuffle(index)
# print(index)

dict1 ={"salary": 200, "jiangjin": 20, "buzhu": 40}
print(dict1.items())
print(sorted(dict1.items(), key=lambda x: x[0]))

