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
    
    marker_count = len(raw_frames[0]) / 3
    frame_count = len(raw_frames)
    #return np.array(raw_frames).reshape(frame_count, marker_count, 3)
    columns = pd.MultiIndex.from_product([marker_names, ["x", "y", "z"]])
    return pd.DataFrame(raw_frames, columns=columns)

    #frames = [[map(float, marker) for marker in chunks(frame, 3)] for frame in raw_frames]
    #return frames
    #np.array(raw_frames)

def plot_marker_trajectories(frames):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for marker in frames.columns.levels[0]:
        marker_frames = frames.loc[:, marker]
        ax.plot(marker_frames["x"], marker_frames["y"], marker_frames["z"], label=marker)
    ax.set_title('Marker Trajectories')
    ax.legend()
    plt.show()
    
def chunks(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def get_reference_markers(frame, reference_marker_indices, virtual_marker_indices):
    # return [
    #     frame[]
    # ]
    pass

def get_viola_body_markers(frame):
    return np.append(frame[6], [frame[7], frame[8]])

def current(orig, pos, rot):
    return pos+np.dot(rot, orig)

def get_virtual_markers(frame, reference_frame):
    rotation, translation, rmse = svdt(reference_pos_bow, marker_pos_bow)

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, projection='3d')

    # leftbow, rightbow, frogbody, frogtip, finetuner, mic, scroll, bodybottom, bodytop
    marker_indices = range(4) #+ range(6, 9)
    markers = [frame[marker_index] for marker_index in marker_indices]

        
    bow_hair_current = [
        current(marker, translation, rotation)[0]
        for marker in bow_hair
    ]

    virtual_markers_current = bow_hair_current #+ viola_markers_current
    marker_data = np.array(markers + virtual_markers_current)
    ax3.scatter(marker_data[:,0], marker_data[:,1], marker_data[:,2])


    markers_orig = np.concatenate(([data[marker_index][frame] for marker_index in marker_indices], bow_hair))
    ax3.scatter(markers_orig[:,0], markers_orig[:,1], markers_orig[:,2], c='r')

    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
# 
# def get_bow_markers(frame_num):
#     return np.append(data[0][frame_num], [data[2][frame_num], data[3][frame_num]])

# test_frame = 
# marker_pos = np.array([get_viola_body_markers(test_frame)])
# reference_pos = get_viola_body_markers(frame)
