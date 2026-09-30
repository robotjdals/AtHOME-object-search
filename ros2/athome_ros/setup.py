from setuptools import setup

package_name = "athome_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="robotjdals",
    maintainer_email="jmsm1378@kw.ac.kr",
    description="ROS2 adapters and nodes for the AtHOME search system",
    license="TODO",
    entry_points={
        "console_scripts": [
            "run_visit = athome_ros.run_visit:main",
            "search_server = athome_ros.search_server:main",
            "command_cli = athome_ros.command_cli:main",
            "fake_motion_server = athome_ros.fake_motion_server:main",
            "fake_perception = athome_ros.fake_perception:main",
            "preflight = athome_ros.preflight:main",
            "run_report = athome_ros.run_report:main",
        ],
    },
)
