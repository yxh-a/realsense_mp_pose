from setuptools import find_packages, setup
import os
import glob

package_name = 'image_pose_tracking'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), 
            ['config/right_arm.urdf.xacro', 
             'config/right_arm_osim_shoulder.urdf.xacro',
             'config/right_arm_osim_shoulder_mesh.urdf.xacro',
             'config/right_arm.urdf',
             'config/right_arm.rviz',
             'config/subject_arm_lengths.yaml']),
        (os.path.join('share', package_name, 'meshes', 'human_subject_right_arm'),
            glob.glob('meshes/human_subject_right_arm/*.stl')),
    ],
    install_requires=[
        'setuptools',
        'opencv-python',
        'matplotlib',
        'numpy',
        'scipy',
        'rclpy'
        ],
    zip_safe=True,
    maintainer='yuxuan_cor',
    maintainer_email='alanhyxszic@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'forward_solver = image_pose_tracking.arm_solver_forward:main',
            'keypoint_extraction = image_pose_tracking.mediapipe_kp:main',
            'pointcloud_extraction = image_pose_tracking.pointcloud_extraction:main',
            'keypoint_distribution_extraction = image_pose_tracking.keypoint_distribution_extraction:main',
            'arm_segment_pointcloud_extraction = image_pose_tracking.arm_segment_pointcloud_extraction:main',
            'urdf_xacro_viewer = image_pose_tracking.urdf_xacro_viewer:main',
        ],
    },
)
