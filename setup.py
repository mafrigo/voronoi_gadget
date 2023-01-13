from setuptools import setup


setup(
    name="voronoi_gadget",
    version="1.0",
    packages=['voronoi_gadget'],
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pyyaml",
        # "pathlib",
        "vorbin"
    ]
)