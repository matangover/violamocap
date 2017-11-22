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
