#!/usr/bin/env python3
"""
Final Corrected Motion Viewer for BVH and DeepMimic

This definitive viewer correctly implements forward kinematics for both data formats,
ensuring accurate, stable, and comparable visualizations. This version fixes the
'End Site' parsing bug and refactors the renderer for stability.
"""
import numpy as np
import json
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import argparse
from typing import Dict, List, Tuple

# --- BVH Parser (Corrected to handle End Sites properly) ---
class BVHParser:
    def __init__(self):
        self.joints = {}
        self.joint_order = []
        self.frames = []
        self.frame_time = 0.0

    def parse(self, bvh_file: str) -> Dict:
        with open(bvh_file, 'r') as f:
            lines = f.readlines()
        hierarchy_end = self._parse_hierarchy(lines)
        self._parse_motion(lines[hierarchy_end:])
        return {
            'joints': self.joints,
            'joint_order': self.joint_order,
            'frames': self.frames,
            'frame_time': self.frame_time
        }

    def _parse_hierarchy(self, lines: List[str]) -> int:
        joint_stack = []
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if not parts: continue

            keyword = parts[0]

            if keyword == 'MOTION':
                return i

            elif keyword in ['ROOT', 'JOINT']:
                joint_name = parts[1]
                parent_name = joint_stack[-1] if joint_stack else None
                self.joints[joint_name] = {
                    'parent': parent_name, 'children': [], 'offset': [0,0,0],
                    'channels': [], 'rot_order': ''
                }
                if parent_name:
                    self.joints[parent_name]['children'].append(joint_name)
                self.joint_order.append(joint_name)
                joint_stack.append(joint_name)

            elif keyword == 'End':
                parent_name = joint_stack[-1]
                end_name = f"{parent_name}_End"
                self.joints[end_name] = {'parent': parent_name, 'children': [], 'offset': [0,0,0], 'channels': []}
                self.joints[parent_name]['children'].append(end_name)
                self.joint_order.append(end_name) # FIX: Add End Site to joint_order
                joint_stack.append(end_name)

            elif keyword == 'OFFSET':
                self.joints[joint_stack[-1]]['offset'] = [float(v) for v in parts[1:]]

            elif keyword == 'CHANNELS':
                channels = parts[2:]
                self.joints[joint_stack[-1]]['channels'] = channels
                rot_order = ''.join([c[0] for c in channels if 'rotation' in c.lower()]).upper()
                self.joints[joint_stack[-1]]['rot_order'] = rot_order

            elif keyword == '}':
                joint_stack.pop()
        return 0

    def _parse_motion(self, motion_lines: List[str]):
        for line in motion_lines:
            line = line.strip()
            if line.startswith('Frames:'):
                self.num_frames = int(line.split(':')[1].strip())
            elif line.startswith('Frame Time:'):
                self.frame_time = float(line.split(':')[1].strip())
            elif line and (line[0].isdigit() or line[0] == '-'):
                self.frames.append([float(x) for x in line.split()])

# --- Math Utilities ---
def euler_to_matrix(euler_rad: List[float], order: str) -> np.ndarray:
    order = order or 'ZXY' # Default to ZXY if order is missing
    mats = {
        'X': lambda x: np.array([[1,0,0],[0,math.cos(x),-math.sin(x)],[0,math.sin(x),math.cos(x)]]),
        'Y': lambda y: np.array([[math.cos(y),0,math.sin(y)],[0,1,0],[-math.sin(y),0,math.cos(y)]]),
        'Z': lambda z: np.array([[math.cos(z),-math.sin(z),0],[math.sin(z),math.cos(z),0],[0,0,1]])
    }
    r = np.identity(3)
    for i, axis in enumerate(order):
        r = r @ mats[axis](euler_rad['XYZ'.index(axis)])
    return r

def quaternion_to_matrix(q: List[float]) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]])

# --- Renderer Base Class ---
class BaseRenderer:
    def __init__(self, ax, title):
        self.ax = ax
        self.ax.set_title(title)
        self.lines = []
        self.points = None

    def clear(self):
        for line in self.lines:
            line.remove()
        if self.points:
            self.points.remove()
        self.lines.clear()

# --- Corrected BVH Renderer (Refactored) ---
class BVHRenderer(BaseRenderer):
    def __init__(self, ax, bvh_data):
        super().__init__(ax, "BVH Animation (Z-up)")
        self.data = bvh_data
        self.root_name = bvh_data['joint_order'][0]
        self.scale = 0.01

    def _get_pose_for_frame(self, frame_data):
        """Parses a single frame's flat data into a dict of joint poses."""
        pose = {}
        channel_idx = 0
        for joint_name in self.data['joint_order']:
            joint_info = self.data['joints'][joint_name]
            num_channels = len(joint_info['channels'])
            values = frame_data[channel_idx : channel_idx + num_channels]

            rotation = {'X': 0, 'Y': 0, 'Z': 0}
            position = {'X': 0, 'Y': 0, 'Z': 0}

            for i, chan in enumerate(joint_info['channels']):
                axis = chan[0].upper()
                if 'position' in chan.lower():
                    position[axis] = values[i]
                elif 'rotation' in chan.lower():
                    rotation[axis] = math.radians(values[i])

            pose[joint_name] = {'rotation': [rotation['X'], rotation['Y'], rotation['Z']], 'position': [position['X'], position['Y'], position['Z']]}
            channel_idx += num_channels
        return pose

    def render_frame(self, frame_idx):
        self.clear()
        frame = self.data['frames'][frame_idx]
        pose_data = self._get_pose_for_frame(frame)

        positions = {}

        def fk(joint_name, parent_pos, parent_rot):
            joint = self.data['joints'][joint_name]
            joint_pose = pose_data[joint_name]

            local_rot_mat = euler_to_matrix(joint_pose['rotation'], joint.get('rot_order'))
            world_rot = parent_rot @ local_rot_mat

            offset = np.array(joint['offset']) * self.scale
            world_pos = parent_pos + parent_rot @ offset

            if joint_name == self.root_name:
                world_pos = np.array(joint_pose['position']) * self.scale

            positions[joint_name] = world_pos

            for child in joint['children']:
                fk(child, world_pos, world_rot)

        fk(self.root_name, np.zeros(3), np.identity(3))

        points = np.array([[pos[0], pos[2], pos[1]] for name, pos in positions.items() if "End" not in name])
        if points.size > 0:
            self.points = self.ax.scatter(points[:,0], points[:,1], points[:,2], c='r', s=10)

        for parent, joint_data in self.data['joints'].items():
            for child in joint_data['children']:
                if parent in positions and child in positions:
                    p1, p2 = positions[parent], positions[child]
                    line, = self.ax.plot([p1[0], p2[0]], [p1[2], p2[2]], [p1[1], p2[1]], 'b-')
                    self.lines.append(line)
        return positions

# --- Corrected DeepMimic Renderer ---
class DeepMimicRenderer(BaseRenderer):
    def __init__(self, ax, dm_data):
        super().__init__(ax, "DeepMimic Animation (Z-up)")
        self.data = dm_data
        self._setup_skeleton()

    def _setup_skeleton(self):
        self.joint_order = ['root', 'chest', 'neck', 'right_hip', 'right_knee', 'right_ankle', 'right_shoulder', 'right_elbow', 'left_hip', 'left_knee', 'left_ankle', 'left_shoulder', 'left_elbow']
        self.parents = {'chest': 'root', 'neck': 'chest', 'right_hip': 'root', 'right_knee': 'right_hip', 'right_ankle': 'right_knee', 'right_shoulder': 'chest', 'right_elbow': 'right_shoulder', 'left_hip': 'root', 'left_knee': 'left_hip', 'left_ankle': 'left_knee', 'left_shoulder': 'chest', 'left_elbow': 'left_shoulder'}
        self.offsets = {'chest': [0, 0.1, 0], 'neck': [0, 0.4, 0], 'right_hip': [0.1, -0.1, 0], 'right_knee': [0, -0.4, 0], 'right_ankle': [0, -0.4, 0], 'right_shoulder': [0.2, 0.35, 0], 'right_elbow': [0.3, 0, 0], 'left_hip': [-0.1, -0.1, 0], 'left_knee': [0, -0.4, 0], 'left_ankle': [0, -0.4, 0], 'left_shoulder': [-0.2, 0.35, 0], 'left_elbow': [-0.3, 0, 0]}
        self.dof_axis = {'chest': 'y', 'neck': 'y', 'right_hip': 'x', 'right_knee': 'x', 'right_ankle': 'x', 'right_shoulder': 'z', 'right_elbow': 'x', 'left_hip': 'x', 'left_knee': 'x', 'left_ankle': 'x', 'left_shoulder': 'z', 'left_elbow': 'x'}
        self.json_joint_order = ['chest', 'neck', 'right_hip', 'right_knee', 'right_ankle', 'right_shoulder', 'right_elbow', 'left_hip', 'left_knee', 'left_ankle', 'left_shoulder', 'left_elbow']

    def render_frame(self, frame_idx):
        self.clear()
        frame = self.data['Frames'][frame_idx]

        root_pos = np.array([frame[1], frame[3], frame[2]]) # X, Z, Y
        root_quat = [frame[4], frame[6], frame[5], frame[7]] # X, Z, Y, W

        positions = {'root': root_pos}
        rotations = {'root': quaternion_to_matrix(root_quat)}

        joint_angles = {name: angle for name, angle in zip(self.json_joint_order, frame[8:])}

        for joint_name in self.joint_order:
            if joint_name == 'root': continue

            parent_name = self.parents[joint_name]
            angle = joint_angles.get(joint_name, 0.0)
            axis = self.dof_axis.get(joint_name, 'x')

            rot_vec = [angle if axis == 'x' else 0, angle if axis == 'y' else 0, angle if axis == 'z' else 0]
            local_rot = euler_to_matrix(rot_vec, 'XYZ')

            world_rot = rotations[parent_name] @ local_rot

            offset = np.array(self.offsets[joint_name])
            offset_zup = np.array([offset[0], offset[2], offset[1]])

            positions[joint_name] = positions[parent_name] + rotations[parent_name] @ offset_zup
            rotations[joint_name] = world_rot

        points = np.array(list(positions.values()))
        if points.size > 0:
            self.points = self.ax.scatter(points[:,0], points[:,1], points[:,2], c='orange', s=15)

        for child, parent in self.parents.items():
            if child in positions and parent in positions:
                p1, p2 = positions[parent], positions[child]
                line, = self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'g-')
                self.lines.append(line)
        return positions

# --- Main Application ---
class MotionViewerApp:
    def __init__(self, bvh_file, dm_file):
        self.fig = plt.figure(figsize=(16, 8))
        self.ax_bvh = self.fig.add_subplot(1, 2, 1, projection='3d')
        self.ax_dm = self.fig.add_subplot(1, 2, 2, projection='3d')

        self.bvh_data = BVHParser().parse(bvh_file)
        with open(dm_file, 'r') as f:
            self.dm_data = json.load(f)

        self.bvh_renderer = BVHRenderer(self.ax_bvh, self.bvh_data)
        self.dm_renderer = DeepMimicRenderer(self.ax_dm, self.dm_data)

        self.num_frames = min(len(self.bvh_data['frames']), len(self.dm_data['Frames']))
        self.current_frame = 0
        self.animation = None
        self.playing = False

        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.update_frame(0)

    def on_key_press(self, event):
        if event.key == ' ': self.toggle_animation()
        elif event.key == 'right': self.set_frame(self.current_frame + 1)
        elif event.key == 'left': self.set_frame(self.current_frame - 1)
        elif event.key == 'r': self.set_frame(0)

    def set_frame(self, frame_idx):
        self.current_frame = max(0, min(frame_idx, self.num_frames - 1))
        if not self.playing:
            self.update_frame(self.current_frame)

    def toggle_animation(self):
        self.playing = not self.playing
        if self.playing:
            self.animation = animation.FuncAnimation(self.fig, self.animate, frames=range(self.current_frame, self.num_frames), interval=int(self.bvh_data['frame_time'] * 1000), repeat=False)
            self.fig.canvas.draw_idle()
        else:
            if self.animation:
                self.animation.event_source.stop()

    def animate(self, i):
        return self.update_frame(i)

    def update_frame(self, i):
        self.current_frame = i
        bvh_pos = self.bvh_renderer.render_frame(i)
        dm_pos = self.dm_renderer.render_frame(i)

        all_points = np.array(list(bvh_pos.values()) + list(dm_pos.values()))
        if all_points.size > 0:
            center = all_points.mean(axis=0)
            max_range = (all_points.max(axis=0) - all_points.min(axis=0)).max() / 1.5

            for ax in [self.ax_bvh, self.ax_dm]:
                ax.set_xlim(center[0] - max_range, center[0] + max_range)
                ax.set_ylim(center[2] - max_range, center[2] + max_range)
                ax.set_zlim(center[1] - max_range, center[1] + max_range)
                ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z (UP)')

        self.fig.suptitle(f'Aligned Motion Viewer (Z-up) | Frame: {i}/{self.num_frames-1}\nControls: SPACE=Play/Pause, LEFT/RIGHT=Step Frame, R=Reset')
        self.fig.canvas.draw_idle()
        return self.bvh_renderer.lines + [self.bvh_renderer.points] + self.dm_renderer.lines + [self.dm_renderer.points]

def main():
    parser = argparse.ArgumentParser(description="Final Aligned Motion Viewer.")
    parser.add_argument('--bvh', required=True, help='Path to the input BVH file.')
    parser.add_argument('--deepmimic', required=True, help='Path to the converted DeepMimic JSON file.')
    args = parser.parse_args()

    viewer = MotionViewerApp(args.bvh, args.deepmimic)
    plt.show()

if __name__ == '__main__':
    main()
