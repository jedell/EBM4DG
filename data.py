import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
        if labels[i] > 0.5:
            labels[i] = 1
        else:
            labels[i] = 0

    return labels

def plot_data(data, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels)
    plt.show()

# generate data
data, labels = generate_data()
# plot data
plot_data(data, labels)
