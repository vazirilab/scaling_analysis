# scaling_analysis
[![Documentation Status](https://readthedocs.org/projects/scaling-analysis/badge/?version=latest)](https://scaling-analysis.readthedocs.io/en/latest/?badge=latest)

`scaling_analysis` enables estimation of the reliable dimensionality of neuronal population dynamics and its scaling as a function of the number of sampled neurons, as described by [Manley et al. Neuron 2024](https://doi.org/10.1016/j.neuron.2024.02.011).

The most important functions include:

- [svca.run_SVCA_partition](https://scaling-analysis.readthedocs.io/en/latest/scaling_analysis.html#scaling_analysis.svca.run_SVCA_partition) samples a given number of neurons and performs shared variance component analysis (SVCA) using [PopulationCoding.dimred.SVCA](https://scaling-analysis.readthedocs.io/projects/PopulationCoding/en/latest/PopulationCoding.html#PopulationCoding.dimred.SVCA).
- [predict.predict_from_behavior](https://scaling-analysis.readthedocs.io/en/latest/scaling_analysis.html#scaling_analysis.predict.predict_from_behavior) performs SVCA on a sampling of a specified number of neurons and then predicts the neural SVCs from behavioral variables.

Note that the analysis modules each contain a command line interface (CLI) which is described in the `main()` function within each module in the [API](https://scaling-analysis.readthedocs.io/en/latest/scaling_analysis.html).

### Usage

`pip install scaling_analysis`

Check out the [demos](https://scaling-analysis.readthedocs.io/en/latest/demo_single_hemisphere.html) for examples of the analyses described in [Manley et al. 2024](https://doi.org/10.1016/j.neuron.2024.02.011).

Example datasets are freely available at https://doi.org/10.5281/zenodo.10403684.

Check out the full API in the [documentation](https://scaling-analysis.readthedocs.io/).

### Citation

If you use this package, please cite the [paper](https://doi.org/10.1016/j.neuron.2024.02.011):

> Manley, J., Lu, S., Barber, K., Demas, J., Kim, H., Meyer, D., Martínez Traub, F., & Vaziri, A. (2024). Simultaneous, cortex-wide dynamics of up to 1 million neurons reveal unbounded scaling of dimensionality with neuron number. Neuron. [https://doi.org/10.1016/j.neuron.2024.02.011](https://doi.org/10.1016/j.neuron.2024.02.011).