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

import queue
import threading
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class CameraInfo:
    name: str
    serial_number: str
    resolution: tuple[int, int] = (640, 480)
    fps: int = 15
    enable_depth: bool = False


class Camera:
    """Camera class for Intel RealSense capture.

    Transplanted from RLinf realworld camera utility.
    """

    def __init__(
        self,
        camera_info: CameraInfo,
    ):
        import pyrealsense2 as rs

        self._camera_info = camera_info
        self._device_info = {}
        for device in rs.context().devices:
            self._device_info[device.get_info(rs.camera_info.serial_number)] = device
        available_serials = sorted(self._device_info.keys())
        if not available_serials:
            raise RuntimeError(
                "No Intel RealSense camera detected. "
                "Check USB connection/power/permissions and run `rs-enumerate-devices`."
            )
        if camera_info.serial_number not in self._device_info:
            raise RuntimeError(
                "Requested RealSense serial not found. "
                f"requested={camera_info.serial_number}, available={available_serials}"
            )

        self._serial_number = camera_info.serial_number
        self._device = self._device_info[self._serial_number]
        self._enable_depth = camera_info.enable_depth

        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._config.enable_device(self._serial_number)
        self._config.enable_stream(
            rs.stream.color,
            camera_info.resolution[0],
            camera_info.resolution[1],
            rs.format.bgr8,
            camera_info.fps,
        )
        if self._enable_depth:
            self._config.enable_stream(
                rs.stream.depth,
                camera_info.resolution[0],
                camera_info.resolution[1],
                rs.format.z16,
                camera_info.fps,
            )
        self.profile = self._pipeline.start(self._config)

        self._align = rs.align(rs.stream.color)

        self._frame_queue = queue.Queue()
        self._frame_capturing_thread = threading.Thread(
            target=self._capture_frames, daemon=True
        )
        self._frame_capturing_start = False
        self._last_capture_error: str | None = None

    @property
    def name(self):
        return self._camera_info.name

    def open(self):
        self._frame_capturing_start = True
        self._frame_capturing_thread.start()

    def close(self, join_timeout: float = 2.0):
        self._frame_capturing_start = False

        # Stop the pipeline first so wait_for_frames() unblocks promptly.
        try:
            self._pipeline.stop()
        except Exception:  # noqa: BLE001
            pass

        if self._frame_capturing_thread.is_alive():
            self._frame_capturing_thread.join(timeout=join_timeout)

        try:
            self._config.disable_all_streams()
        except Exception:  # noqa: BLE001
            pass

    def get_frame(self, timeout: float = 5.0):
        assert self._frame_capturing_start, (
            "Frame capturing is not started. Cannot get frame."
        )
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty as exc:
            raise RuntimeError(
                f"Timed out after {timeout}s waiting for camera frame "
                f"(thread_alive={self._frame_capturing_thread.is_alive()}, "
                f"last_error={self._last_capture_error})"
            ) from exc

    def _capture_frames(self):
        consecutive_failures = 0
        max_failures = max(5, self._camera_info.fps)
        while self._frame_capturing_start:
            time.sleep(1 / self._camera_info.fps)
            try:
                has_frame, frame = self._read_frame()
            except Exception as exc:  # noqa: BLE001
                # Temporary read failures can happen due USB jitter.
                self._last_capture_error = f"{type(exc).__name__}: {exc}"
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    break
                continue
            if not has_frame:
                self._last_capture_error = "Received an invalid/non-video frame from camera."
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    break
                continue
            consecutive_failures = 0
            if not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self._frame_queue.put(frame)

    def _read_frame(self):
        frames = self._pipeline.wait_for_frames()
        aligned_frames = self._align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        if self._enable_depth:
            depth_frame = aligned_frames.get_depth_frame()

        if color_frame.is_video_frame():
            frame = np.asarray(color_frame.get_data())
            if self._enable_depth and depth_frame.is_depth_frame():
                depth = np.expand_dims(np.asarray(depth_frame.get_data()), axis=2)
                return True, np.concatenate((frame, depth), axis=-1)
            return True, frame
        return False, None

    @staticmethod
    def list_connected_devices() -> list[dict[str, str]]:
        import pyrealsense2 as rs

        devices = []
        for device in rs.context().devices:
            serial_number = "unknown"
            name = "unknown"
            if device.supports(rs.camera_info.serial_number):
                serial_number = device.get_info(rs.camera_info.serial_number)
            if device.supports(rs.camera_info.name):
                name = device.get_info(rs.camera_info.name)
            devices.append({"name": name, "serial_number": serial_number})
        return devices

    @property
    def last_capture_error(self) -> str | None:
        return self._last_capture_error
