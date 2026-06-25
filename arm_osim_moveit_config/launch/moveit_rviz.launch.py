from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_moveit_rviz_launch


def generate_launch_description():
    moveit_config = MoveItConfigsBuilder("upper_arm_osim_shoulder_mesh", package_name="arm_osim_moveit_config").to_moveit_configs()
    return generate_moveit_rviz_launch(moveit_config)
