import numpy as np
import mujoco
import glfw
import argparse
import time

# python3.10 mocap_player_new.py (logs/0-test/test2.npy)


# 模型文件路径
xmlpath = "assets/dp_env_v2.xml"

# 定义渲染视图参数，可根据需求进行调整
VIEWPORT_WIDTH = 640     # 渲染窗口宽度
VIEWPORT_HEIGHT = 480    # 渲染窗口高度
FONT_SCALE = mujoco.mjtFontScale.mjFONTSCALE_150  # 字体比例
CAMERA_LOOKAT = [0, 0, 1]  # 摄像机目标位置（模型坐标系中的点）
CAMERA_DISTANCE = 3.0      # 摄像机距离
CAMERA_ELEVATION = -20     # 摄像机俯仰角
CAMERA_AZIMUTH = 180       # 摄像机方位角

def play_mocap_np_file(mocap_filepath):
    """
    播放并渲染动作捕捉数据文件。

    参数:
        mocap_filepath (str): 动作捕捉数据文件的路径 (.npy 文件格式)。
    """
    # 加载 Mujoco 模型和数据
    model = mujoco.MjModel.from_xml_path(xmlpath)
    data = mujoco.MjData(model)

    # 加载动作捕捉数据
    mocap = np.load(mocap_filepath)
    num_frames = mocap.shape[0]  # 动作捕捉数据的帧数

    # 初始化 GLFW 库以创建窗口
    if not glfw.init():
        print("Could not initialize GLFW")
        return
    
    # 创建渲染窗口
    window = glfw.create_window(VIEWPORT_WIDTH, VIEWPORT_HEIGHT, "Mujoco Simulation", None, None)
    if not window:
        glfw.terminate()
        print("Could not create GLFW window")
        return
    glfw.make_context_current(window)  # 设置当前 OpenGL 上下文

    # 设置渲染器的摄像机、选项和场景
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scene = mujoco.MjvScene(model, maxgeom=1000)
    context = mujoco.MjrContext(model, FONT_SCALE)
    viewport = mujoco.MjrRect(0, 0, VIEWPORT_WIDTH, VIEWPORT_HEIGHT)

    # 设置摄像机参数
    cam.lookat[:] = CAMERA_LOOKAT  # 摄像机目标位置
    cam.distance = CAMERA_DISTANCE  # 摄像机到目标的距离
    cam.elevation = CAMERA_ELEVATION  # 摄像机俯仰角
    cam.azimuth = CAMERA_AZIMUTH  # 摄像机方位角

    # 初始化位移补偿，用于处理模型在循环播放中的平移
    phase_offset = np.zeros(3)
    frame = 0  # 初始化帧计数

    # 主渲染循环
    while not glfw.window_should_close(window):
        glfw.poll_events()  # 处理窗口事件

        # 设置视口为当前窗口大小
        window_width, window_height = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, window_width, window_height)


        # 获取当前帧的关节配置
        tmp_val = mocap[frame % num_frames]
        data.qpos[:] = tmp_val[:]  # 设置关节位置
        data.qpos[:3] += phase_offset[:]  # 增加位移偏移

        # 更新模拟状态
        mujoco.mj_forward(model, data)

        # 更新并渲染场景
        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scene)
        mujoco.mjr_render(viewport, scene, context)
        glfw.swap_buffers(window)  # 切换缓冲区，显示新帧

        frame += 1  # 更新帧计数

        # 控制播放速度 (可选)
        # time.sleep(0.033)

    # 终止 GLFW
    glfw.terminate()

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Play and render mocap data in Mujoco")
    parser.add_argument("motion_file", help="Path to the motion file (.npy format)")
    args = parser.parse_args()

    # 调用播放函数
    play_mocap_np_file(args.motion_file)
