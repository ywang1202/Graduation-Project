import scipy.io as scio
import matplotlib.pyplot as plt


def smooth(scalar,weight=0.85):
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    return smoothed

data = scio.loadmat('/home/ywang/毕设/Code/3/our/models/loss/Epsilon1_1.mat')
# y = data['val_pixel'][0]
y = data['epsilon_1'][0]
y = smooth(y, 0.6)
x = [i for i in range(29571)]

plt.figure(figsize=[30,10])
plt.xlabel('Step')
plt.ylabel('Loss')
plt.plot(x, y)
plt.show()
