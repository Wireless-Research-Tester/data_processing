"""
Graphing Polar Functions v7.0
9/21/2020
Updated buttons
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import CheckButtons
import os


plt.style.use('seaborn-white')

lines = []
labels = []
visibility = []


def graph_all_polar(df, num_of_freqs, resolution):
    """
    This function graphs all frequencies from a list or linear sweep
    Polar plot - azimuth vs. amplitude
    Assumes frequencies are in MHz

    :param df: Sorted DataFrame
    :param num_of_freqs: Total number of different frequencies in Data Frame
    :param resolution: Angle between measurements (1-5 degrees) (passed by JSON)
    :return: None
    """
    # Starting point in dataframe
    index = 0

    global lines
    global labels
    global visibility
    global Live

    # Number of rows
    number_of_rows = len(df.index)

    # Number of points per freq
    points_per_freq = number_of_rows // num_of_freqs

    # Define the number of rows per frequency
    rows_per_freq = (points_per_freq // resolution)

    # Isolate phi column and convert to radians
    phi_val_set = np.radians(df.iloc[index:rows_per_freq, [3]])

    # Isolate magnitude column and convert to relative zero
    max_magnitude = df['magnitude'].max()
    magnitude_val_set = df.iloc[index:rows_per_freq, [4]]
    if max_magnitude > 0:
        magnitude_val_set = magnitude_val_set['magnitude'] + max_magnitude
    else:
        magnitude_val_set = magnitude_val_set['magnitude'] - max_magnitude

    # Create a string of requested frequency for legend
    current_freq = (df['freq'].values[index])
    current_freq_string = str(current_freq)
    # labels.append(current_freq_string + ' MHz')

    # Create polar plot
    ax = plt.subplot(111, projection='polar', autoscale_on=True)

    # Add freqs to plot
    ax.plot(phi_val_set, magnitude_val_set, label=current_freq_string + ' MHz')
    for x in range(1, num_of_freqs):
        index += rows_per_freq
        phi_val_set = np.radians(df.iloc[index:index + rows_per_freq, [3]])
        magnitude_val_set = df.iloc[index:index + rows_per_freq, [4]]
        if max_magnitude > 0:
            magnitude_val_set = magnitude_val_set['magnitude'] + max_magnitude
        else:
            magnitude_val_set = magnitude_val_set['magnitude'] - max_magnitude
        current_freq = (df['freq'].values[index])
        current_freq_string = str(current_freq)
        # labels.append(current_freq_string + ' MHz')
        ax.plot(phi_val_set, magnitude_val_set, label=current_freq_string + ' MHz')

    # Customize Plot
    ax.set_rlabel_position(0)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rmin(-40.0)
    ax.grid(True)
    plt.thetagrids(range(0, 360, 15))
    plt.subplots_adjust(top=0.924, bottom=0.042, left=0.047, right=0.961)
    plt.title('Amplitude vs. Azimuth')
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
    if not Live:
        rax = plt.axes([0.005, 0.1, 0.12, 0.4])
        lines, labels = ax.get_legend_handles_labels()
        visibility = [line.get_visible() for line in ax.lines]
        check = CheckButtons(rax, labels, visibility)
        check.on_clicked(set_visible)
    plt.show()


def graph_one_polar(df, req_freq, num_of_freqs, resolution):
    """
    This function graphs one frequency from a list or linear sweep
    Polar plot - azimuth vs. amplitude
    Assumes frequencies are in MHz

    :param df: Sorted DataFrame
    :param req_freq: Frequency requested from program user (passed by JSON)
    :param num_of_freqs: Total number of different frequencies in Data Frame
    :param resolution: Angle between measurements (1-5 degrees) (passed by JSON)
    :return: None
    """
    # Number of rows
    number_of_rows = len(df.index)

    # Number of points per freq
    points_per_freq = number_of_rows // num_of_freqs

    # Define the number of rows per frequency
    rows_per_freq = (points_per_freq // resolution)

    # Find the requested frequency index in dataframe
    index = freq_index(df, req_freq, num_of_freqs, resolution)

    # Isolate phi column and convert to radians
    phi_val_set = np.radians(df.iloc[index:index + rows_per_freq, [3]])

    # Isolate magnitude column
    magnitude_val_set = df.iloc[index:index + rows_per_freq, [4]]
    max_magnitude = int(magnitude_val_set.max())
    if max_magnitude > 0:
        magnitude_val_set = magnitude_val_set['magnitude'] + max_magnitude
    else:
        magnitude_val_set = magnitude_val_set['magnitude'] - max_magnitude

    # Create a string of requested frequency for legend
    req_freq_string = str(req_freq)

    # Create polar plot
    ax = plt.subplot(111, projection='polar', autoscale_on=True)
    ax.plot(phi_val_set, magnitude_val_set, label=req_freq_string + ' MHz')
    ax.set_rlabel_position(0)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rmin(-40)
    ax.grid(True)
    plt.thetagrids(range(0, 360, 15))
    plt.subplots_adjust(top=0.924, bottom=0.042, left=0.047, right=0.961)
    plt.title('Amplitude vs. Azimuth')
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
    plt.show()


def graph_all_rect(df, num_of_freqs, resolution):
    """
    This function graphs all frequencies from a list or linear sweep
    Polar plot - azimuth vs. amplitude
    Assumes frequencies are in MHz

    :param df: Sorted DataFrame
    :param num_of_freqs: Total number of different frequencies in Data Frame
    :param resolution: Angle between measurements (1-5 degrees) (passed by JSON)
    :return: None
    """
    # Starting point in dataframe
    index = 0

    global lines
    global labels
    global visibility

    # Number of rows
    number_of_rows = len(df.index)

    # Number of points per freq
    points_per_freq = number_of_rows // num_of_freqs

    # Define the number of rows per frequency
    rows_per_freq = (points_per_freq // resolution)

    # Isolate phi column and convert to radians
    phi_val_set = df.iloc[index:rows_per_freq, [3]]

    # Isolate magnitude column and convert to relative zero
    # max_magnitude is the largest magnitude in the file
    max_magnitude = df['magnitude'].max()
    magnitude_val_set = df.iloc[index:rows_per_freq, [4]]
    if max_magnitude > 0:
        magnitude_val_set = magnitude_val_set['magnitude'] + max_magnitude
    else:
        magnitude_val_set = magnitude_val_set['magnitude'] - max_magnitude

    # Create a string of requested frequency for legend
    current_freq = (df['freq'].values[index])
    current_freq_string = str(current_freq)

    # Create polar plot
    ax = plt.subplot()

    # Add freqs to plot
    ax.plot(phi_val_set, magnitude_val_set, label=current_freq_string + ' MHz')
    for x in range(1, num_of_freqs):
        index += rows_per_freq
        phi_val_set = df.iloc[index:index + rows_per_freq, [3]]
        magnitude_val_set = df.iloc[index:index + rows_per_freq, [4]]
        if max_magnitude > 0:
            magnitude_val_set = magnitude_val_set['magnitude'] + max_magnitude
        else:
            magnitude_val_set = magnitude_val_set['magnitude'] - max_magnitude
        current_freq = (df['freq'].values[index])
        current_freq_string = str(current_freq)
        ax.plot(phi_val_set, magnitude_val_set, label=current_freq_string + ' MHz')

    # Customize Plot
    ax.grid(True)
    ax.set_xlim(left=0, right=360)
    ax.set_ylim(top=0, bottom=-40)
    plt.subplots_adjust(left=0.16, right=.90)
    plt.title('Amplitude vs. Azimuth', fontweight="bold", fontsize=35, verticalalignment='top')
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
    lines, labels = ax.get_legend_handles_labels()
    visibility = [line.get_visible() for line in ax.lines]
    rax = plt.axes([0.005, 0.1, 0.12, 0.4])
    check = CheckButtons(rax, labels, visibility)
    check.on_clicked(set_visible)
    plt.show()


def set_visible(label):
    index = labels.index(label)
    lines[index].set_visible(not lines[index].get_visible())
    plt.draw()


def is_file_empty(file_location):
    """
    This function reads a file into a Pandas DataFrame

    :param file_location: Location of .csv file where measurement control wrote data into
    :return: Bool True if and only if file exists and is its size is 0 bytes
    """
    return os.path.exists(file_location) and os.path.getsize(file_location) == 0


def read_file(file_location):
    """
    This function reads a file into a Pandas DataFrame

    :param file_location: Location of .csv file where measurement control wrote data into
    :return: if not empty returns DataFrame
    """
    df = pd.read_csv(file_location)
    return df


def sort_file(df):
    """
    This function sorts the DataFrame rows in ascending order
    by frequency, phi, and theta; in that order

    :param df: Unsorted DataFrame
    :return: Sorted DataFrame
    """
    # Sort the data by frequency, phi, and theta
    df = df.sort_values(by=['freq', 'phi'])
    return df


def tot_freqs(df):
    """
    This function determines how many frequencies were included in the sweep or list

    :param df: Unsorted DataFrame
    :return: Number of different frequencies in DataFrame
    """
    if df.empty:
        return False
    start_freq = (df['freq'].values[0])
    num_of_freqs = 1
    for x in range(1, len(df.index)):
        if start_freq == df['freq'].values[x]:
            return num_of_freqs
        elif start_freq != df['freq'].values[x]:
            num_of_freqs += 1
    return num_of_freqs


def freq_index(df, req_freq, num_of_freqs, resolution):
    """
    This function finds the starting index of a specific frequency within the DataFrame

    :param df: Sorted DataFrame
    :param req_freq: Frequency requested from program user (passed by JSON)
    :param num_of_freqs: Total number of different frequencies in Data Frame
    :param resolution: Angle between measurements (1-5 degrees) (passed by JSON)
    :return: if freq is in DataFrame return index
    """
    data_points = (360 // resolution)
    index = 0
    for x in range(0, num_of_freqs):
        if req_freq == (df['freq'].values[index]):
            return index
        else:
            index += data_points
    return False


def start_graphing(individual_freq, polar, resolution, req_freq=None):
    """
    This function takes the arguments passed by JSON and calls initial functions

    :param individual_freq: Flag set if program user wants one frequency plotted
    :param polar: if polar -> graphs polar
    :param resolution: Angle between measurements (1-5 degrees) (passed by JSON)
    :param req_freq: Frequency requested from program user (passed by JSON)
    :return: None
    """
    fig.clf()
    file_location = 'sample_output.csv'
    # check if file exist and it is empty
    is_empty = is_file_empty(file_location)
    if is_empty:
        print('File is empty')
        return
    data_frame = read_file(file_location)
    if data_frame.empty:
        print('File only contains column headers')
        return
    number_of_freqs = tot_freqs(data_frame)
    if number_of_freqs:
        data_frame = sort_file(data_frame)
        if polar:
            if individual_freq:
                graph_one_polar(data_frame, req_freq, number_of_freqs, resolution)
            else:
                graph_all_polar(data_frame, number_of_freqs, resolution)
        else:
            graph_all_rect(data_frame, number_of_freqs, resolution)
    return


def animate(i):
    start_graphing(IndividualFreq, Polar, Resolution, ReqFreq)


# Values will come from user input
IndividualFreq = False
ReqFreq = 1000  # Values in MHz
Resolution = 1
Polar = 1
# Flag to set up live plotting
Live = 0

fig = plt.figure(figsize=(15, 10), dpi=80)  # width = 10, height = 10 figsize=(10, 10), dpi=70

if Live:
    ani = animation.FuncAnimation(fig, animate, interval=1000)  # Updates every second
else:
    start_graphing(IndividualFreq, Polar, Resolution, ReqFreq)

plt.show()
