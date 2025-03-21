**Characterizing Dynamic Functional Connectivity Subnetwork Contributions in Narrative Classification with Shapley Values**

This repository contains the code for the paper "Characterizing Dynamic Functional Connectivity Subnetwork Contributions in Narrative Classification with Shapley Values".

### Repository Structure
The repository is organized as follows:
- `data/`: contains the data used in the paper  modelled as temporal networks, saved as tensor in the .npy format.
- `src/`: contains the code for the experiments in the paper.
- `4_classification/`: contains the code for the 4 class combined classification experiments and plots.
- `movie_audio_classification/`: contains the code for the 2 class modality classification experiments and plots.
- `airport_restaurant_classification/`: contains the code for the 2 class content classification experiments and plots.
- `plots/`: contains the code and plots for the figures in the paper.
- `files/`: contains the files related to the content labels and name of the network files.
- `utils.jl`: contains the utility functions used in the experiments.

The code is written mainly in Julia, excpet for the part of the cleaning of the raw fMRI signal and the creation of the timeseries.  The code is tested with Julia 1.10.1 and the required packages are listed in the `Project.toml` file. 

### How to run the Julia code interactively

To run the code, you need to have Julia installed on your machine. You can follow the instruction of the [`juliaup` repository](https://github.com/JuliaLang/juliaup).

After having installed Julia 1.10.1, you can open the Julia-REPL from this repository and run the following commands to install the required packages:

```julia
] activate .
] instantiate
```

Then you can run the experiments by running the following command in the Julia-REPL:

```julia
include("file_you_want_to_run.jl")
```

### How to run the Python code 

The code for the cleaning of the raw fMRI signal and the creation of the timeseries is written in Python. You can run the code by running the following command in the terminal:

```bash
python3 create_timeseries.py
```

after having installed the required packages listed in the `requirements.txt` file.


