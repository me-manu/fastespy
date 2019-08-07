# fastespy

A python module for analysing and simulating data from a transition edge sensor.

The module is currently under heavy development for analysis of the ALPS II experiment. 
Conda and pip support will hopefully be added in the near future. 

For the time being, you will need to clone the repository and manually add its path to your python path. 
In bash this would be done like this:
```export PYTHONPATH=".:/dir/to/fastespy/:$PYTHONPATH"```

You will also need the following python packages:

- numpy
- scipy
- iminuit

Furthermore, if you have aquired data with `ROOT`, you will need to install it in order to convert 
the data to numpy files.
You can do this by installing the `ROOT` conda environment, see https://anaconda.org/conda-forge/root

## License 

This project is licensed under a 3-clause BSD style license - see the `LICENSE.md` file.
