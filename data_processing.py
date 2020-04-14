"""
Graphing Polar Functions v1.0
4/11/2020
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
Graphs all frequencies in MHz
Assumes frequency increments are of even size
Example input:
filename = "sample_output.csv
total_freqs = 1, 5, 10, ext.
start_freq = 1000 MHz
freq_inc = 100 MHz
angle = 1, .5, .25, ext.
'''


def graph_all_freqs(filename, total_freqs=10, start_freq=1000, freq_inc=100, angle=1):
    # read the csv into pandas dataframe
    df = pd.read_csv(filename)

    # Sort the data by frequency, phi, and theta
    df = df.sort_values(by=['freq', 'phi', 'theta'])

    # Define the number of rows per frequency
    theta_max = (360 // angle) - 1

    # Starting point in dataframe
    index = 0
    index_next = (360 // angle)

    # Isolate phi column and convert to radians
    theta = df.iloc[index:theta_max, [3]]
    phi = theta['phi'] * np.pi / 180

    # Isolate value column
    r = df.iloc[index:theta_max, [4]]
    value = r['value']

    # Create polar plot
    current_freq = start_freq
    current_freq_string = str(start_freq)
    ax = plt.subplot(111, projection='polar', autoscale_on=True)
    for x in range(0, total_freqs):
        ax.plot(phi, value, label=current_freq_string + ' MHz')
        index += index_next
        current_freq += freq_inc
        theta = df.iloc[index:index + theta_max, [3]]
        phi = theta['phi'] * np.pi / 180
        r = df.iloc[index:index + theta_max, [4]]
        value = r['value']
        current_freq_string = str(current_freq)
    ax.grid(True)
    plt.title('2D Plane (XY axis)')
    plt.legend(loc='upper right')
    plt.show()


'''
Graphs one frequency in MHz
Assumes frequency increments are of even size
Example input:
filename = "sample_output.csv"
req_freq = 1000 MHz
start_freq = 1000 MHz
freq_inc = 100 MHz
angle = 1, .5, .25, ext.
'''


def graph_one_freq(filename, req_freq=1000, start_freq=1000, freq_inc=100, angle=1):
    # read the csv into pandas dataframe
    df = pd.read_csv(filename)

    # Sort the data by frequency, phi, and theta
    df = df.sort_values(by=['freq', 'phi', 'theta'])

    # Define the number of rows per frequency
    theta_max = (360 // angle) - 1

    # Find the starting point in dataframe
    index_next = (360 // angle)
    index = ((req_freq - start_freq) // freq_inc) * index_next

    # Isolate phi column and convert to radians
    theta = df.iloc[index:index + theta_max, [3]]
    phi = theta['phi'] * np.pi / 180

    # Isolate value column
    r = df.iloc[index:index + theta_max, [4]]
    value = r['value']

    # Create polar plot
    req_freq_string = str(req_freq)
    ax = plt.subplot(111, projection='polar', autoscale_on=True)
    ax.plot(phi, value, label=req_freq_string + ' MHz')
    ax.grid(True)
    plt.title('2D Plane (XY axis)')
    plt.legend(loc='upper right')
    plt.show()
