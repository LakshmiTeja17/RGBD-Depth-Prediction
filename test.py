from numpy.random import randint
import numpy as np
import os

l1 = os.listdir("./rgb")
l2 = os.listdir("./depth")
l1 = np.array(l1)
l2 = np.array(l2)

a1 = np.setdiff1d(l1,l2)
a2 = np.setdiff1d(l2,l1)

print(a1)
print(a2)

for i in range(len(a1)):
    path = os.path.join("./rgb", a1[i])
    os.remove(path)


for i in range(len(a2)):
    path = os.path.join("./depth", a2[i])
    os.remove(path)


# a = np.random.choice(l1, size=150, replace=False, p=None)
# for i in range(150):
# 	print(a[i])
