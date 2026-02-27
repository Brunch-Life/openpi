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

import logging
import os
import pathlib
import sys
import time
from typing import Callable, Optional

import psutil
import rospy
from filelock import FileLock


class ROSController:
    """Controller for ROS communication. One controller corresponds to one robot."""

    def __init__(self, ros_version: int = 1):
        self._logger = logging.getLogger(__name__)
        self._ros_version = ros_version
        assert self._ros_version == 1, "Currently only ROS 1 is supported."

        ros_lock_file = "/tmp/.ros.lock"
        if not os.path.exists(os.path.dirname(ros_lock_file)):
            ros_lock_file = os.path.join(pathlib.Path.home(), ".ros.lock")
        self._ros_lock = FileLock(ros_lock_file)

        if self._ros_version == 1:
            with self._ros_lock:
                self._ros_core = None
                for proc in psutil.process_iter():
                    if proc.name() == "roscore":
                        self._ros_core = proc
                        break

                if self._ros_core is None:
                    self._ros_core = psutil.Popen(
                        ["roscore"], stdout=sys.stdout, stderr=sys.stdout
                    )
                    time.sleep(1)

        rospy.init_node("franka_controller", anonymous=True)

        self._output_channels: dict[str, rospy.Publisher] = {}
        self._input_channels: dict[str, rospy.Subscriber] = {}
        self._input_channel_status: dict[str, bool] = {}

    def get_input_channel_status(self, name: str) -> bool:
        if name not in self._input_channel_status:
            return False
        return self._input_channel_status.get(name, False)

    def create_ros_channel(
        self, name: str, data_class: rospy.Message, queue_size: Optional[int] = None
    ):
        self._output_channels[name] = rospy.Publisher(
            name, data_class, queue_size=queue_size
        )

    def connect_ros_channel(
        self, name: str, data_class: rospy.Message, callback: Callable
    ):
        def callback_wrapper(*args, **kwargs):
            self._input_channel_status[name] = True
            return callback(*args, **kwargs)

        self._input_channel_status[name] = False
        self._input_channels[name] = rospy.Subscriber(
            name, data_class, callback_wrapper
        )

    def put_channel(self, name: str, data: rospy.Message):
        if name in self._output_channels:
            assert isinstance(data, self._output_channels[name].data_class), (
                f"Invalid data type for ROS channel '{name}'. "
                f"Expected {self._output_channels[name].data_class}, got {type(data)}."
            )
            self._output_channels[name].publish(data)
        else:
            self._logger.warning("ROS channel '%s' is not created.", name)

