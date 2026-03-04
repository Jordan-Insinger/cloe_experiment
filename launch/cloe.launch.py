from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
import os

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='cloe_experiment',
            namespace='agent_1',
            executable='cloe',
            name='cloe_experiment_node'
        )
    ])
