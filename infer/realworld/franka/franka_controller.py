# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
import sys
import time

import geometry_msgs.msg as geom_msg
import numpy as np
import psutil
import rospy
from dynamic_reconfigure.client import Client as ReconfClient
from franka_gripper.msg import GraspActionGoal, MoveActionGoal
from franka_msgs.msg import ErrorRecoveryActionGoal, FrankaState
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from serl_franka_controllers.msg import ZeroJacobian

from ..common.ros import ROSController
from .franka_robot_state import FrankaRobotState


class _ImmediateResult:
    """Small compatibility wrapper to mimic RLinf WorkerRef.wait() API."""

    def __init__(self, value):
        self._value = value

    def wait(self):
        return [self._value]


class FrankaController:
    """Local Franka robot arm controller."""

    @staticmethod
    def launch_controller(
        robot_ip: str,
        env_idx: int = 0,
        node_rank: int = 0,
        worker_rank: int = 0,
        ros_pkg: str = "serl_franka_controllers",
    ):
        del env_idx, node_rank, worker_rank
        return FrankaController(robot_ip=robot_ip, ros_pkg=ros_pkg)

    def __init__(self, robot_ip: str, ros_pkg: str = "serl_franka_controllers"):
        self._logger = logging.getLogger(__name__)
        self._robot_ip = robot_ip
        self._ros_pkg = ros_pkg

        self._state = FrankaRobotState()
        self._ros = ROSController()
        self._init_ros_channels()

        self._impedance: psutil.Process | None = None
        self._joint: psutil.Process | None = None

        self.start_impedance()

        self._reconf_client = ReconfClient(
            "cartesian_impedance_controllerdynamic_reconfigure_compliance_param_node"
        )

    def _init_ros_channels(self):
        self._arm_equilibrium_channel = "/cartesian_impedance_controller/equilibrium_pose"
        self._arm_reset_channel = "/franka_control/error_recovery/goal"
        self._arm_jacobian_channel = "/cartesian_impedance_controller/franka_jacobian"
        self._arm_state_channel = "franka_state_controller/franka_states"

        self._ros.create_ros_channel(
            self._arm_equilibrium_channel, geom_msg.PoseStamped, queue_size=10
        )
        self._ros.create_ros_channel(
            self._arm_reset_channel, ErrorRecoveryActionGoal, queue_size=1
        )
        self._ros.connect_ros_channel(
            self._arm_jacobian_channel, ZeroJacobian, self._on_arm_jacobian_msg
        )
        self._ros.connect_ros_channel(
            self._arm_state_channel, FrankaState, self._on_arm_state_msg
        )

        self._gripper_move_channel = "/franka_gripper/move/goal"
        self._gripper_grasp_channel = "/franka_gripper/grasp/goal"
        self._gripper_state_channel = "/franka_gripper/joint_states"
        self._ros.create_ros_channel(
            self._gripper_move_channel, MoveActionGoal, queue_size=1
        )
        self._ros.create_ros_channel(
            self._gripper_grasp_channel, GraspActionGoal, queue_size=1
        )
        self._ros.connect_ros_channel(
            self._gripper_state_channel, JointState, self._on_gripper_state_msg
        )

    def _on_arm_jacobian_msg(self, msg: ZeroJacobian):
        self._state.arm_jacobian = np.array(list(msg.zero_jacobian)).reshape(
            (6, 7), order="F"
        )

    def _on_arm_state_msg(self, msg: FrankaState):
        tmatrix = np.array(list(msg.O_T_EE)).reshape(4, 4).T
        r = R.from_matrix(tmatrix[:3, :3].copy())
        self._state.tcp_pose = np.concatenate([tmatrix[:3, -1], r.as_quat()])

        self._state.arm_joint_velocity = np.array(list(msg.dq)).reshape((7,))
        self._state.arm_joint_position = np.array(list(msg.q)).reshape((7,))
        self._state.tcp_force = np.array(list(msg.K_F_ext_hat_K)[:3])
        self._state.tcp_torque = np.array(list(msg.K_F_ext_hat_K)[3:])
        try:
            self._state.tcp_vel = (
                self._state.arm_jacobian @ self._state.arm_joint_velocity
            )
        except Exception as exc:  # noqa: BLE001
            self._state.tcp_vel = np.zeros(6)
            self._logger.warning(
                "Jacobian not set, end-effector velocity temporarily unavailable: %s",
                exc,
            )

    def _on_gripper_state_msg(self, msg: JointState):
        self._state.gripper_position = np.sum(msg.position)

    def _wait_robot(self, sleep_time: int = 1):
        time.sleep(sleep_time)

    def _wait_for_joint(self, target_pos: list[float], timeout: int = 30):
        wait_time = 0.01
        waited_time = 0
        target_pos = np.array(target_pos)

        while (
            not np.allclose(
                target_pos, self._state.arm_joint_position, atol=1e-2, rtol=1e-2
            )
            and waited_time < timeout
        ):
            time.sleep(wait_time)
            waited_time += wait_time

        if waited_time >= timeout:
            self._logger.warning("Joint position wait timeout exceeded")
        else:
            self._logger.debug(
                "Joint position reached %s", self._state.arm_joint_position
            )

    def reconfigure_compliance_params(self, params: dict[str, float]):
        self._reconf_client.update_configuration(params)
        self._logger.debug("Reconfigure compliance parameters: %s", params)
        return _ImmediateResult(None)

    def is_robot_up(self):
        arm_state_status = self._ros.get_input_channel_status(self._arm_state_channel)
        gripper_state_status = self._ros.get_input_channel_status(
            self._gripper_state_channel
        )
        return _ImmediateResult(arm_state_status and gripper_state_status)

    def get_state(self):
        return _ImmediateResult(self._state)

    def start_impedance(self):
        self._impedance = psutil.Popen(
            [
                "roslaunch",
                self._ros_pkg,
                "impedance.launch",
                "robot_ip:=" + self._robot_ip,
                "load_gripper:=true",
            ],
            stdout=sys.stdout,
            stderr=sys.stdout,
        )

        self._wait_robot()
        self._logger.debug(
            "Start Impedance controller: %s",
            self._impedance.status() if self._impedance else "none",
        )
        return _ImmediateResult(None)

    def stop_impedance(self):
        if self._impedance:
            self._impedance.terminate()
            self._impedance = None
            self._wait_robot()
        self._logger.debug("Stop Impedance controller")
        return _ImmediateResult(None)

    def clear_errors(self):
        self._ros.put_channel(self._arm_reset_channel, ErrorRecoveryActionGoal())
        return _ImmediateResult(None)

    def reset_joint(self, reset_pos: list[float]):
        self.stop_impedance()
        self.clear_errors()

        self._wait_robot()
        self.clear_errors()

        assert len(reset_pos) == 7, (
            f"Invalid reset position, expected 7 dimensions but got {len(reset_pos)}"
        )

        rospy.set_param("/target_joint_positions", reset_pos)
        self._joint = psutil.Popen(
            [
                "roslaunch",
                self._ros_pkg,
                "joint.launch",
                "robot_ip:=" + self._robot_ip,
                "load_gripper:=true",
            ],
            stdout=sys.stdout,
        )
        self._wait_robot()
        self._logger.debug("Joint reset begins")
        self.clear_errors()

        self._wait_for_joint(reset_pos)

        self._joint.terminate()
        self._wait_robot()
        self.clear_errors()
        self.start_impedance()
        return _ImmediateResult(None)

    def move_arm(self, position: np.ndarray):
        assert len(position) == 7, (
            f"Invalid position, expected 7 dimensions but got {len(position)}"
        )
        pose_msg = geom_msg.PoseStamped()
        pose_msg.header.frame_id = "0"
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.pose.position = geom_msg.Point(position[0], position[1], position[2])
        pose_msg.pose.orientation = geom_msg.Quaternion(
            position[3], position[4], position[5], position[6]
        )

        self._ros.put_channel(self._arm_equilibrium_channel, pose_msg)
        self._logger.debug("Move arm to position: %s", position)
        return _ImmediateResult(None)

    def move_gripper(self, position: int, speed: float = 0.3):
        assert 0 <= position <= 255, (
            f"Invalid gripper position {position}, must be between 0 and 255"
        )
        move_msg = MoveActionGoal()
        move_msg.goal.width = float(position / (255 * 10))
        move_msg.goal.speed = speed

        self._ros.put_channel(self._gripper_move_channel, move_msg)
        self._logger.debug("Move gripper to position: %s", position)
        return _ImmediateResult(None)

    def open_gripper(self):
        move_msg = MoveActionGoal()
        move_msg.goal.width = 0.09
        move_msg.goal.speed = 0.3

        self._ros.put_channel(self._gripper_move_channel, move_msg)
        self._state.gripper_open = True
        self._logger.debug("Open gripper")
        return _ImmediateResult(None)

    def close_gripper(self):
        grasp_msg = GraspActionGoal()
        grasp_msg.goal.width = 0.01
        grasp_msg.goal.speed = 0.3
        grasp_msg.goal.epsilon.inner = 1
        grasp_msg.goal.epsilon.outer = 1
        grasp_msg.goal.force = 130

        self._ros.put_channel(self._gripper_grasp_channel, grasp_msg)
        self._state.gripper_open = False
        self._logger.debug("Close gripper")
        return _ImmediateResult(None)

