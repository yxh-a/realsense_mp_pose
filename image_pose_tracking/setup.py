from setuptools import find_packages, setup
import os

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
            ['config/right_arm.urdf', 
             'config/right_arm.rviz',
             'config/right_arm_red.urdf']),
    ],
    install_requires=[
        'setuptools',
        'opencv-python',
        'mediapipe',
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
            'pose_tracking = image_pose_tracking.pose_tracking:main',
            'forward_solver = image_pose_tracking.arm_solver_forward:main',
        ],
    },
)

