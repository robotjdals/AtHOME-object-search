from setuptools import setup

package_name = "athome_perception"

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
    description="Real-time perception for the AtHOME search",
    license="TODO",
    entry_points={
        "console_scripts": [
            "realtime_perception = athome_perception.realtime_perception:main",
        ],
    },
)
