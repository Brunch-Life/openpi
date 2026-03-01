
import pyrealsense2 as rs


ctx = rs.context()
devices = ctx.query_devices()

if len(devices) == 0:
    raise RuntimeError("No RealSense devices connected")

for i, dev in enumerate(devices):
    sn = dev.get_info(rs.camera_info.serial_number)
    name = dev.get_info(rs.camera_info.name)
    fw = dev.get_info(rs.camera_info.firmware_version)
    print(f"[{i}] {name}  SN={sn}  FW={fw}")