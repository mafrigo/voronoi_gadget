from setuptools import setup


setup(
    name="voronoi_gadget",
    version="1.0",
    packages=['voronoi_gadget'],
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib>=3.6.3",
        "pyyaml",
        "vorbin"
    ],
    package_data={'voronoi_gadget': ['voronoi_gadget/config/*.yaml']},
    include_package_data=True,
)