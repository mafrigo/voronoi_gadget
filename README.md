# voronoi_gadget
![Alt text](example_output.png?raw=true "Galaxy Kinematics example")

## Description
Creates mock integral field unit (IFU) galaxy images from galaxy simulations, like the one shown above. 
Each Voronoi bin (spaxel) contains roughly the same number of simulation particles in projection. 
This simulates the signal-to-noise criterion used in real IFU maps to determine the bin shape and size.

The code uses the [pygad](https://bitbucket.org/broett/pygad) library to open simulation snapshots, 
meaning it supports natively Gadget and Arepo simulations, but it can also be used on 
other formats with some tweaking.

It also uses the [vorbin](https://pypi.org/project/vorbin/) package by Michele Cappellari for creating the Voronoi grid, 
therefore any paper using this should cite [Cappellari & Copin 2003](http://adsabs.harvard.edu/abs/2003MNRAS.342..345C).

## Usage example:
see example_script.py.

## Output examples:
See figure above, or figures 4,5,6,10,11 in Frigo et al. 2019 (https://arxiv.org/pdf/1811.11059.pdf).
