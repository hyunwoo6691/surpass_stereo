from setuptools import find_packages, setup

package_name = 'surpass_stereo'

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
    maintainer='brendan',
    maintainer_email='brendan.f.burkhart@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={},
    entry_points={
        'console_scripts': [
            'surpass_stereo = surpass_stereo.surpass_stereo:main'
        ],
    },
)
