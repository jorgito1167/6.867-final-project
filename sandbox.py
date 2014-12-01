import data_extraction as de
import visualize_data as vis
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import get_test_data


def test():
    # Twice as wide as it is tall.
    fig = plt.figure(figsize=plt.figaspect(0.5))

    #---- First subplot
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R)
    print type(X)
    print X.shape
    print Y.shape
    print Z.shape
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
            linewidth=0, antialiased=False)
    ax.set_zlim3d(-1.01, 1.01)

    fig.colorbar(surf, shrink=0.5, aspect=10)

    #---- Second subplot
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    X, Y, Z = get_test_data(0.05)
    ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

    plt.show()
    
if __name__ == '__main__':
    #test()
    df = de.read_clean()[0]
    vis.plot_var_3d(df,['time', 'humidity'])
