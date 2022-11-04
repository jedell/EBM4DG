import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader
import torch


def generate_hypersphere(center, radius, n_points):
    data = np.zeros((n_points, 3))
    for i in range(0, n_points):
        # generate random points inside and on hypersphere
        data[i, :] = np.random.randn(3) * radius + center
        
    return data
    

def generate_data():
    # generate 3 hyperspheres with random centers and radii
    data = np.zeros((300, 3))
    labels = np.zeros((300, 1))
    centers = np.zeros((3, 3))
    r = np.zeros((3, 1))
    for i in range(0, 3):
        # random center between -3 and 3
        center = np.random.rand(3) * 12 - 6
        centers[i, :] = center
        # random radius between 1 and 2
        radius = np.random.rand(1) + 1
        r[i, 0] = radius
        data[i*100:(i+1)*100, :] = generate_hypersphere(center, radius, 100)
    # assign labels to data points
    labels = assign_labels(data, labels, centers, r)
    return data, labels


def assign_labels(data, labels, centers, r):
    for i in range(0, 300):
        labels[i] = -np.log(1/(1 + np.exp(-np.linalg.norm(data[i, :] - centers[i//100, :])))) + (r[i//100, 0] - np.linalg.norm(data[i, :] - centers[i//100, :]))
        # add noise from normal distribution with mean 0 and standard deviation 0.1 to labels
        labels[i] = labels[i] + np.random.normal(0, r[i//100, 0]/2)
        # use activation function to assign binary labels
        if labels[i] > 0.4:
            labels[i] = 1
        else:
            labels[i] = 0

    return labels

def plot_data(data, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels)
    plt.show()

class TestingData(torch.utils.data.IterableDataset):
    
    def __init__(self, n=300, num_centers=3, rng_of_centers=6, rng_of_radii=1, hardcoded=false):
        self.n = n
        self.centers = self.__gen_centers(num_centers, rng_of_centers)
        self.radii = self.__gen_radii(num_centers, rng_of_radii)
        self.hardcoded = hardcoded
    
    def __len__(self):
        return self.n
    
    def __iter__(self):
        for i in range(self.n):
            index = i % len(self.centers)
            center = self.centers[index]
            radius = self.radii[index]
            x = np.random.randn(3) * radius + center
            target = -np.log(1/(1 + np.exp(-np.linalg.norm(x - center)))) + (radius - np.linalg.norm(x - center))
            target = target + np.random.normal(0, radius/2)
            if target > 0.4:
                target = 1
            else:
                target = 0
            yield x, target

    def __gen_centers(self, n, rng):
        centers = np.zeros((n, 3))
        if hardcoded:
            return centers
        for i in range(0, n):
            # random center between -3 and 3
            center = np.random.rand(3) * (rng * 2) - rng
            centers[i, :] = center
        return centers
    
    def __gen_radii(self, n, rng):
        radii = np.zeros((3, 1))
        for i in range(0, n):
            radius = np.random.rand(1) + 1 # fix to increase rang of centers
            radii[i, :] = radius
            # do cutoff
        return radii
    
    
np.random.seed(0)
# np.random.rand(3)
train = TestingData(1000, 2, 6)
train_dataloader = DataLoader(train, batch_size=3000)
images, labels = next(iter(train_dataloader))
unique, counts = np.unique(labels, return_counts=True)
d = dict(zip(unique, counts))
print(d[1]/ d[0])
# plot_data(images,labels)

# # generate data
# data, labels = generate_data()
# # plot data
# plot_data(data, labels)

# plot_data(data[:200], labels[:200])
# plot_data(data[200:], labels[200:])

# # export data and labels to csv
# np.savetxt("X_train.csv", data[:200], delimiter=",")
# np.savetxt("y_train.csv", labels[:200], delimiter=",")

# np.savetxt("X_val.csv", data[200:], delimiter=",")
# np.savetxt("y_val.csv", labels[200:], delimiter=",")