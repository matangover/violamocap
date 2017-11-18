import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from svdt import svdt

def load_qualisys_data(filename):
    with open(filename) as input_file:
        reader = csv.reader(input_file, delimiter="\t")
        # Skip Qualisys header
        for line in reader:
            if line[0] == "MARKER_NAMES":
                marker_names = line[1:]
                break
        
        raw_frames = [map(float, frame) for frame in reader]
    
    columns = pd.MultiIndex.from_product([marker_names, ["x", "y", "z"]])
    return pd.DataFrame(raw_frames, columns=columns)

def plot_marker_trajectories(frames):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for marker in frames.columns.levels[0]:
        marker_frames = frames.loc[:, marker]
        ax.plot(marker_frames["x"], marker_frames["y"], marker_frames["z"], label=marker)
    ax.set_title('Marker Trajectories')
    ax.legend()
    plt.show()
    
