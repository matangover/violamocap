import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from svdt import svdt
from collections import namedtuple

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

# RigidBody = namedtuple("RigidBody", ["reference_markers", "virtual_markers", "calibration_filename"])
class RigidBody:
    def __init__(self, reference_markers, virtual_markers, calibration_filename):
        self.reference_markers = reference_markers
        self.virtual_markers = virtual_markers
        calibration_frames = load_qualisys_data("../Experiment data/mocap/" + calibration_filename)
        # Arbitrarily choose the first frame as calibration frame
        reference_frame = calibration_frames.iloc[0]
        self.reference_pos = reference_frame[reference_markers]
        self.reference_pos_virtual = reference_frame[virtual_markers]

rigid_bodies = {
    "viola": RigidBody(
        reference_markers=["bodybottom", "bodytop", "scroll"],
        virtual_markers=["Astring", "Dstring", "Gstring", "Cstring", "Abridge", "Dbridge", "Gbridge", "Cbridge"],
        calibration_filename="Calibration0004.tsv"
    ),
    "bow": RigidBody(
        reference_markers=["frogtip", "frogbody", "rightbow"],
        virtual_markers=["frog_hair", "tip_hair"],
        calibration_filename="Calibration_bow hair0005.tsv"
    )
}

def calculate_rigid_body(frames, body):
    reference_pos_reindexed = body.reference_pos.reindex(frames[body.reference_markers].columns)
    rotation, translation, rmse = svdt(reference_pos_reindexed, frames[body.reference_markers])
    # Maybe this is possible with matrix operations instead of a for loop.
    virtual_markers = pd.DataFrame([
        get_current_marker_positions(body.reference_pos_virtual, rotation[i], translation[i])
        for i in range(len(frames))
    ])
    return virtual_markers, rmse

def current(orig, rot, pos):
    return pos+np.dot(rot, orig)

def get_current_marker_positions(reference_frame, rotation, translation):
    markers = pd.Series(index=reference_frame.index)
    # Use unique() because the index may include columns that have already been filtered out. (Hack...)
    marker_names = reference_frame.index.unique().levels[0]
    for marker in marker_names:
        markers[marker] = current(reference_frame[marker], rotation, translation)
    return markers

    
def extract_bow_parameters(participant, trial):
    filename = "../Experiment data/mocap/%s000%s.tsv" % (participant, trial)
    frames = load_qualisys_data(filename)
    # extract trajectory plot image also
    virtual_markers = {}
    for body_name, body in rigid_bodies.items():
        virtual_markers[body_name], rmse = calculate_rigid_body(frames, body)
        
    virtual_markers_bow = virtual_markers["bow"]
    virtual_markers = virtual_markers["viola"]
    # TODO: Faster to pre-compute vectors for bow and strings for each frame.
    skewness = pd.Series(get_skewness(participant, trial, i, virtual_markers_bow, virtual_markers) for i in range(len(frames)))
    velocity = pd.Series(get_velocity(participant, trial, i,  virtual_markers_bow, virtual_markers) for i in range(len(frames)))
    return skewness, velocity

def unit_vector(vector):
    """ Returns the unit vector of the vector. """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2' """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def get_skewness(participant, trial, frame, virtual_markers_bow, virtual_markers):
    bow = virtual_markers_bow.iloc[frame]
    bow_vec = bow['tip_hair'] - bow['frog_hair']
    string = get_string_for_frame(participant, trial, frame)
    viola = virtual_markers.iloc[frame]
    string_vec = viola[string + 'bridge'] - viola[string + 'string']
    skewness_radians = np.pi / 2 - angle_between(string_vec, bow_vec)
    skewness_degrees = skewness_radians / np.pi * 180
    return skewness_degrees

def get_velocity(participant, trial, frame, virtual_markers_bow, virtual_markers):
    if frame == 0:
        return np.nan
    
    bow = virtual_markers_bow.iloc[frame]
    bow_prev = virtual_markers_bow.iloc[frame-1]
    bow_vec = unit_vector(bow['tip_hair'] - bow['frog_hair'])
    
    string = get_string_for_frame(participant, trial, frame)
    string_bridge = virtual_markers.iloc[frame][string + 'bridge']
    # Using the previous string creates a big spike in velocity.
    # Probably the moment of bow change should be nan bow velocity.
    # string_prev = get_string_for_frame(frame-1)
    # string_bridge_prev = virtual_markers.iloc[frame-1][string_prev + 'bridge']
    difference = (bow['tip_hair'] - string_bridge) - (bow_prev['tip_hair'] - string_bridge)
    velocity = np.dot(difference, bow_vec)
    # The units of this are Millimeters per frame:
    # mm/frame = mm/(sec/100) = 100mm/sec = 0.1m/sec
    # frame = sec/100
    # so dividing the result by 10 should give m/sec.
    # avg should be ~1m/4sec = 0.25m/sec --> works.
    return velocity

# TODO: cache this in a dataframe (string per participant trial)
def get_string_for_frame(participant, trial, frame):
    changes = bow_changes[participant][trial]
    for i in range(3):
        if frame < changes.iloc[0, i + 1]:
            return changes.columns[i]
        
    return changes.columns[-1]

def load_bow_changes_for_participant(participant):
    return {
        trial: pd.read_csv(
            "../Experiment data/bow_changes/%s %s.csv" % (participant, trial),
            index_col=0
        )
        for trial in trials
    }

def load_bow_changes():
    return {
        participant: load_bow_changes_for_participant(participant)
        for participant in participants
    }

participants = ["Marina", "Matan"]
trials = range(1, 4)
bow_changes = load_bow_changes()

def get_frame_range(participant, trial):
    changes = bow_changes[participant][trial]
    first_frame = int(changes.iloc[0, 0])
    last_frame = int(changes.iloc[-1, -1])
    return slice(first_frame, last_frame)

viola_strings = ['C', 'G', 'D', 'A']

def add_guide_lines(ax):
    # fig = plt.figure(figsize=(10,4))
    # ax = fig.add_subplot(111)
    # plot = ax.plot(series.iloc[first_frame:last_frame])
    # Add horizontal line on zero.
    ax.axhline(0, linewidth=1, linestyle='-.', color='pink')
    # Add vertical line for every bow change.
    # 1 stroke = 4 beats * 1 sec/beat * * 100 frame/sec = 400 frames
    # 1 string = 8 strokes = 3200 frames
    # 4 strings = 12800 frames
    for string_index, string in enumerate(viola_strings):
        string_start_frame = string_index * 3200
        ax.axvline(string_start_frame, linewidth=1, linestyle='--', color='green')
        for change in range(1,8):
            stroke_start_frame = string_start_frame + change * 400
            ax.axvline(stroke_start_frame, linewidth=0.5, linestyle='--', color='gray')
    ax.axvline(12800, linewidth=1, linestyle='--', color='green')
    ticks = plt.xticks(range(0, 12800, 3200), viola_strings)
    #plt.title("%s #%s - %s" % (participant, trial, data_type))    
    #ylim = {"velocity": (-7, 7), "skewness": (-30, 30)}
    #ax.set_ylim(*ylim[data_type])
#    return ax
 #   plt.savefig()
#    plt.close()

def normalize_frame_numbers(data, participant, trial):
    series = data[participant][trial]
    frame_range = get_frame_range(participant, trial)
    series = series.iloc[frame_range]
    frame_count = frame_range.stop - frame_range.start
    series.index = np.arange(frame_count)
    # Fill in missing values from end, if there weren't enough frames
    desired_frame_count = 4 * 8 * 4 * 100
    series.reindex(np.arange(desired_frame_count))
    return series

def load_data(data_type, participant, trial):
    filename = "%s_%s_%s.csv" % (data_type, participant, trial)
    return pd.read_csv("../extracted_motion_parameters/data/" + filename, index_col=0, header=None, squeeze=True)

def load_normalized_data(data_type):
    from collections import defaultdict
    data = defaultdict(dict)
    for participant in participants:
        for trial in trials:
            data[participant][trial] = load_data(data_type, participant, trial)
    
    normalized = [
        normalize_frame_numbers(data, participant, trial)
        for participant in participants
        for trial in trials
    ]
    normalized_dataframe = pd.concat(normalized, axis='columns')
    normalized_dataframe.columns = pd.MultiIndex.from_product(
        [participants, trials],
        names=["Participant", "Trial"]
    )
    return normalized_dataframe
