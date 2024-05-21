This repository contains the code for the paper "Characterizing Dynamic Functional Connectivity Subnetwork Contributions in Narrative Classification with Shapley Values" submitted to NeurIPS 2024.

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

