from setuptools import find_packages, setup

package_name = 'cloe_experiment'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='UFLAutonomyPark',
    maintainer_email='uflautonomypark@gmail.com',
    description='ros2 python node for cloe experiment',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
                'cloe = cloe_experiment.cloe_node:main',
        ],
    },
)
