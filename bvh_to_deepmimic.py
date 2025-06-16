#!/usr/bin/env python3
"""
Robust BVH to DeepMimic Converter (Mathematically Correct Version)

This script provides a more reliable conversion by:
1.  Focusing only on the primary motion-driving joints from the BVH file.
2.  Correctly handling the combination of rotations from multiple source joints
    using hierarchical matrix multiplication for mathematical accuracy.
3.  Using the most significant rotational axis for each joint type.
"""

import numpy as np
import json
import math
import argparse
from typing import Dict, List, Tuple, Optional

class BVHParser:
    """Parses BVH files to extract skeleton and motion data."""

    def __init__(self):
        self.joints = {}
        self.joint_order = []
        self.frames = []
        self.frame_time = 0.0
        self.num_frames = 0

    def parse(self, bvh_file: str) -> Dict:
        """Main parsing function."""
        with open(bvh_file, 'r') as f:
            lines = f.readlines()
        hierarchy_end = self._parse_hierarchy(lines)
        self._parse_motion(lines[hierarchy_end:])
        return {
            'joints': self.joints,
            'joint_order': self.joint_order,
            'frames': self.frames,
            'frame_time': self.frame_time,
            'num_frames': self.num_frames
        }

    def _parse_hierarchy(self, lines: List[str]) -> int:
        """Parses the HIERARCHY section of the BVH file."""
        joint_stack = []
        current_joint = None
        line_idx = 0
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('MOTION'):
                line_idx = i
                break
            elif line.startswith('ROOT') or line.startswith('JOINT'):
                parts = line.split()
                joint_name = parts[1]
                current_joint = {
                    'name': joint_name,
                    'parent': joint_stack[-1] if joint_stack else None,
                    'children': [],
                    'offset': [0.0, 0.0, 0.0],
                    'channels': [],
                    'rot_order': ''.join([c[0] for c in parts[2:] if 'rotation' in c.lower()]).upper() # Extract rotation order
                }
                if joint_stack:
                    self.joints[joint_stack[-1]]['children'].append(joint_name)
                self.joints[joint_name] = current_joint
                self.joint_order.append(joint_name)
                joint_stack.append(joint_name)
            elif line.startswith('End Site'):
                end_name = f"{current_joint['name']}_End"
                self.joints[end_name] = {'name': end_name, 'parent': current_joint['name'], 'children': [], 'offset': [0.0, 0.0, 0.0], 'channels': []}
                current_joint['children'].append(end_name)
                joint_stack.append(end_name)
            elif line.startswith('OFFSET'):
                self.joints[joint_stack[-1]]['offset'] = [float(p) for p in line.split()[1:]]
            elif line.startswith('CHANNELS'):
                parts = line.split()
                self.joints[joint_stack[-1]]['channels'] = parts[2:]
                # Infer rotation order from channel names
                self.joints[joint_stack[-1]]['rot_order'] = ''.join([c[0] for c in parts[2:] if 'rotation' in c.lower()]).upper()
            elif line.startswith('}'):
                if joint_stack:
                    joint_stack.pop()
        return line_idx

    def _parse_motion(self, motion_lines: List[str]):
        """Parses the MOTION section of the BVH file."""
        for line in motion_lines:
            line = line.strip()
            if line.startswith('Frames:'):
                self.num_frames = int(line.split(':')[1].strip())
            elif line.startswith('Frame Time:'):
                self.frame_time = float(line.split(':')[1].strip())
            elif line and not line.startswith('MOTION'):
                self.frames.append([float(x) for x in line.split()])

class BVHToDeepMimicConverter:
    """
    A converter that correctly maps complex BVH arm rotations to a simplified
    DeepMimic format.
    """

    def __init__(self):
        self.joint_mapping = {
            'CC_Base_Hip': 'root',
            'CC_Base_Waist': 'chest',
            'CC_Base_Spine01': 'chest',
            'CC_Base_NeckTwist01': 'neck',
            'CC_Base_L_Thigh': 'left_hip',
            'CC_Base_L_Calf': 'left_knee',
            'CC_Base_L_Foot': 'left_ankle',
            'CC_Base_R_Thigh': 'right_hip',
            'CC_Base_R_Calf': 'right_knee',
            'CC_Base_R_Foot': 'right_ankle',
            'CC_Base_L_Upperarm': 'left_shoulder',
            'CC_Base_R_Upperarm': 'right_shoulder',
            'CC_Base_L_Forearm': 'left_elbow',
            'CC_Base_L_ElbowShareBone': 'left_elbow',
            'CC_Base_R_Forearm': 'right_elbow',
            'CC_Base_R_ElbowShareBone': 'right_elbow',
        }

    # --- NEW: Matrix Math Utilities ---
    def _euler_to_matrix(self, euler_rad: List[float], order: str = 'ZXY') -> np.ndarray:
        """Converts Euler angles (in radians) to a 3x3 rotation matrix."""
        order = order if order else 'ZXY'
        mats = {
            'X': lambda x: np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]]),
            'Y': lambda y: np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]]),
            'Z': lambda z: np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
        }
        r = np.identity(3)
        for axis in order:
            idx = 'XYZ'.index(axis)
            r = r @ mats[axis](euler_rad[idx])
        return r

    def _matrix_to_euler(self, matrix: np.ndarray, order: str = 'ZXY') -> List[float]:
        """Converts a 3x3 rotation matrix to Euler angles (in radians), handling gimbal lock."""
        if order == 'ZXY':
            # Check for gimbal lock at the North Pole
            if np.isclose(matrix[2, 1], 1.0):
                x = math.pi / 2
                y = 0
                z = math.atan2(matrix[1, 0], matrix[0, 0])
            # Check for gimbal lock at the South Pole
            elif np.isclose(matrix[2, 1], -1.0):
                x = -math.pi / 2
                y = 0
                z = math.atan2(-matrix[1, 0], -matrix[0, 0])
            # General case
            else:
                x = math.asin(matrix[2, 1])
                y = math.atan2(-matrix[2, 0], matrix[2, 2])
                z = math.atan2(-matrix[0, 1], matrix[1, 1])
            return [x, y, z]
        else:
            raise NotImplementedError(f"Rotation order {order} not supported for matrix to euler conversion.")

    def convert(self, bvh_data: Dict, output_file: str, loop: bool = True, fps: int = 30) -> Dict:
        """Main conversion function."""
        print("Starting conversion with the final robust method.")
        deepmimic_frames = []
        target_dt = 1.0 / fps

        for i, bvh_frame in enumerate(bvh_data['frames']):
            try:
                dm_frame = self._convert_frame(bvh_frame, bvh_data, i * target_dt)
                if dm_frame and len(dm_frame) == 36:
                    deepmimic_frames.append(dm_frame)
            except Exception as e:
                print(f"Error converting frame {i}: {e}")

        motion_data = {"Loop": "wrap" if loop else "none", "Frames": deepmimic_frames}
        with open(output_file, 'w') as f:
            json.dump(motion_data, f, indent=2)

        print(f"\nConversion complete. Saved {len(deepmimic_frames)} frames to {output_file}")
        self._verify_motion(bvh_data, deepmimic_frames)
        return motion_data

    def _convert_frame(self, bvh_frame: List[float], bvh_data: Dict, time: float) -> List[float]:
        """Converts a single BVH frame to the DeepMimic format."""
        dm_frame = [time]

        # Root Position and Rotation
        root_pos = [p * 0.01 for p in bvh_frame[0:3]]  # Scale from cm to m
        root_euler = bvh_frame[3:6]
        root_quat = self._euler_to_quaternion(root_euler)
        dm_frame.extend(root_pos)
        dm_frame.extend(root_quat)

        # Extract and combine rotations from all relevant BVH joints
        joint_rotations = self._extract_and_combine_rotations(bvh_frame, bvh_data)

        # Build the 28 joint values for DeepMimic
        joint_data = []
        dm_joint_order = [
            'chest', 'neck', 'right_hip', 'right_knee', 'right_ankle',
            'right_shoulder', 'right_elbow', 'left_hip', 'left_knee',
            'left_ankle', 'left_shoulder', 'left_elbow'
        ]

        for joint_name in dm_joint_order:
            if joint_name in joint_rotations:
                value = self._get_primary_rotation(joint_name, joint_rotations[joint_name])
                joint_data.append(value)
            else:
                joint_data.append(0.0)

        while len(joint_data) < 28:
            joint_data.append(0.0)
        dm_frame.extend(joint_data[:28])

        return dm_frame

    # --- REVISED: This function now uses matrix multiplication ---
    def _extract_and_combine_rotations(self, bvh_frame: List[float], bvh_data: Dict) -> Dict[str, List[float]]:
        """
        Extracts 3D rotations and combines them using hierarchical matrix
        multiplication for mathematical accuracy.
        """
        # Initialize a dictionary to hold the final combined rotation matrix for each DeepMimic joint.
        # The value is an identity matrix, which represents "no rotation".
        dm_joint_names = set(self.joint_mapping.values())
        combined_matrices = {name: np.identity(3) for name in dm_joint_names}

        data_idx = 0
        # Iterate through the BVH joints in their defined hierarchical order.
        for joint_name in bvh_data['joint_order']:
            joint_info = bvh_data['joints'][joint_name]
            num_channels = len(joint_info['channels'])

            # Check if this BVH joint is one we need to map.
            if joint_name in self.joint_mapping:
                dm_joint_name = self.joint_mapping[joint_name]

                # Extract this joint's local rotation from the frame data.
                joint_rotation_euler = [0.0, 0.0, 0.0]  # X, Y, Z
                channel_offset = 0
                for chan_name in joint_info['channels']:
                    if 'rotation' in chan_name.lower():
                        value = math.radians(bvh_frame[data_idx + channel_offset])
                        if 'x' in chan_name.lower(): joint_rotation_euler[0] = value
                        elif 'y' in chan_name.lower(): joint_rotation_euler[1] = value
                        elif 'z' in chan_name.lower(): joint_rotation_euler[2] = value
                    channel_offset += 1

                # Convert the local Euler rotation to a rotation matrix.
                local_matrix = self._euler_to_matrix(joint_rotation_euler, joint_info.get('rot_order', 'ZXY'))

                # Multiply the existing combined matrix by this new local matrix.
                # The order (Total @ Local) is crucial for correctly chaining rotations.
                combined_matrices[dm_joint_name] = combined_matrices[dm_joint_name] @ local_matrix

            data_idx += num_channels

        # After combining all matrices, convert the final matrices back to Euler angles.
        final_rotations = {}
        for name, matrix in combined_matrices.items():
            final_rotations[name] = self._matrix_to_euler(matrix, order='ZXY')

        return final_rotations

    def _get_primary_rotation(self, joint_name: str, rotation_vec: List[float]) -> float:
        """
        Selects the most representative single rotation value from a 3D vector
        based on the joint type.
        """
        x, y, z = rotation_vec

        if 'elbow' in joint_name or 'knee' in joint_name:
            return y if 'elbow' in joint_name else z
        elif 'shoulder' in joint_name:
            return x if abs(x) > abs(z) else z
        else:
            return max(rotation_vec, key=abs)

    def _euler_to_quaternion(self, euler_deg: List[float]) -> List[float]:
        """Converts Euler angles (in degrees) to a quaternion [x, y, z, w]."""
        roll, pitch, yaw = [math.radians(d) for d in euler_deg]
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return [x, y, z, w]

    def _verify_motion(self, bvh_data: Dict, dm_frames: List[List[float]]):
        """Prints a comparison for a key frame to verify the conversion."""
        print("\n--- Motion Verification (Frame 196) ---")
        frame_idx = 196
        if frame_idx >= len(bvh_data['frames']) or frame_idx >= len(dm_frames):
            print("Frame 196 not available for verification.")
            return

        bvh_rotations = self._extract_and_combine_rotations(bvh_data['frames'][frame_idx], bvh_data)
        dm_frame = dm_frames[frame_idx]
        dm_arm_indices = {'right_shoulder': 13, 'right_elbow': 14, 'left_shoulder': 18, 'left_elbow': 19}

        print(f"{'Joint':<15} | {'BVH Combined 3D (deg)':<30} | {'DM Final Value (deg)':<20}")
        print("-" * 75)

        for name, index in dm_arm_indices.items():
            bvh_rot_deg = [math.degrees(r) for r in bvh_rotations.get(name, [0,0,0])]
            dm_val_deg = math.degrees(dm_frame[index])

            bvh_str = f"[{bvh_rot_deg[0]:6.1f}, {bvh_rot_deg[1]:6.1f}, {bvh_rot_deg[2]:6.1f}]"
            print(f"{name:<15} | {bvh_str:<30} | {dm_val_deg:20.1f}")


def main():
    parser = argparse.ArgumentParser(description="Robust BVH to DeepMimic Converter")
    parser.add_argument('--bvh', required=True, help='Input BVH file')
    parser.add_argument('--output', required=True, help='Output DeepMimic JSON file')
    args = parser.parse_args()

    bvh_parser = BVHParser()
    bvh_data = bvh_parser.parse(args.bvh)

    converter = BVHToDeepMimicConverter()
    converter.convert(bvh_data, args.output)

if __name__ == "__main__":
    main()
