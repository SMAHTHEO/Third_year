import numpy as np
from mujoco_py import load_model_from_xml, MjSim, MjViewer

xmlpath = "assets/dp_env_v2.xml"


def play_raw_mocap(mocap):
    with open(xmlpath) as fin:
        MODEL_XML = fin.read()

    model = load_model_from_xml(MODEL_XML)
    sim = MjSim(model)
    viewer = MjViewer(sim)

    phase_offset = np.array([0.0, 0.0, 0.0])

    # import cv2
    # from VideoSaver import VideoSaver
    # width = 640
    # height = 480

    # vid_save = VideoSaver(width=width, height=height)
    while True:
        # for i in range(5):
        for config in mocap:
            tmp_val = config
            sim_state = sim.get_state()
            sim_state.qpos[:] = tmp_val[:]
            sim_state.qpos[:3] += phase_offset[:]
            sim.set_state(sim_state)
            sim.forward()
            viewer.render()
            # vid_save.addFrame(viewer.read_pixels(width, height, depth=False))

        sim_state = sim.get_state()
        phase_offset = sim_state.qpos[:3]
        phase_offset[2] = 0

    # vid_save.close()


def play_mocap_np_file(mocap_filepath):
    with open(xmlpath) as fin:
        MODEL_XML = fin.read()

    model = load_model_from_xml(MODEL_XML)
    sim = MjSim(model)
    viewer = MjViewer(sim)

    # Load the mocap
    mocap = np.load(mocap_filepath)
    # print(mocap.shape)
    # print(mocap[0])
    from time import sleep

    phase_offset = np.array([0.0, 0.0, 0.0])

    # import cv2
    # from VideoSaver import VideoSaver
    # width = 640
    # height = 480

    # vid_save = VideoSaver(width=width, height=height)
    while True:
        print(mocap.shape)
        # for i in range(5):
        for config in mocap:
            tmp_val = config
            sim_state = sim.get_state()
            sim_state.qpos[:] = tmp_val[:]
            sim_state.qpos[:3] += phase_offset[:]
            sim.set_state(sim_state)
            sim.forward()
            viewer.render()
            # vid_save.addFrame(viewer.read_pixels(width, height, depth=False))

        sim_state = sim.get_state()
        phase_offset = sim_state.qpos[:3]
        phase_offset[2] = 0

    # vid_save.close()


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("motion_file", help="Path to the motion file")
    args = parser.parse_args()

    play_mocap_np_file(args.motion_file)
