import matplotlib.pyplot as plt
import numpy as np

# Plot alpha vs error validation
plt.figure(0)

x = np.array([2,3,1,0])
n_1 = np.array([10, 20, 50, 70, 100, 300, 500, 700, 1000, 2000, 3000, 5000, 7000, 10000, 11000, 12800])
a_1 = np.array([32.480740, 32.604006, 32.141757, 32.881356, 33.559322, 39.907550, 41.879815, 44.530046, 46.317411, 50.878274, 56.147920, 61.633282, 63.235747, 67.026194, 68.351310, 69.922958])

n_2 = np.array([10, 20, 50, 70, 100, 200,300,380])
a_2 = np.array([33.775039, 35.130971, 39.044684, 41.325116, 45.701079, 52.449923, 55.993837, 64.067797])


one = plt.plot(n_1, a_1,marker='.', label='Random Selection')
two = plt.plot(n_2, a_2,marker='.', label='Mutual Information Selection')
plt.xlabel('Number of Features')
plt.ylabel('Accuracy %')
plt.legend(loc='lower right', shadow=True)
plt.savefig('featurenum_accuracy.png')
plt.show()