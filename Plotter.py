import matplotlib.pyplot as plt
import numpy as np

def plot_segment(compressed_segment: int, x_axes: list, y_axes: list):
    for i in range(len(y_axes)):
        plot_x = np.reshape(x_axes[i], (-1))
        #plot_y = y_axes[i].transpose()[compressed_segment]
        plot_y = y_axes[i]
        #plot_y *= MAX_SPEED

        # Create the plot
        plt.plot(plot_x, plot_y, label=f'{compressed_segment}')

    # plot_y = y.transpose()[1]
    # plot_y *= MAX_SPEED
    # plt.plot(plot_x, plot_y, label=r'1')

    # Add labels and title
    plt.xlabel('timestamp')
    plt.ylabel('speed')
    plt.title(f'Segments: [{compressed_segment}]')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()