from setuptools import find_packages, setup
import os
import glob
package_name = 'face_recognition_zed'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # (os.path.join('share', package_name, 'output'), glob('output/*')),
        # (os.path.join('share', package_name, 'training'), glob('training/*')),
        # (os.path.join('share', package_name, 'output'), glob('output/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='olagh',
    maintainer_email='olaghattas@hotmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detect_zed = face_recognition_zed.zed_head_box:main',
        ],
    },
)
