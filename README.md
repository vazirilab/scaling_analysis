# 📈 scaling_analysis 📈

### Estimating the "reliable" dimensionality of neuronal population dynamics, and how it scales with the number of recorded neurons

Source code accompanying the article:

> Manley, J., Lu, S., Barber, K., Demas, J., Kim, H., Meyer, D., Martínez Traub, F., & Vaziri, A. (2024). Simultaneous, cortex-wide dynamics of up to 1 million neurons reveal unbounded scaling of dimensionality with neuron number. Neuron. [https://doi.org/10.1016/j.neuron.2024.02.011](https://doi.org/10.1016/j.neuron.2024.02.011).

This codebase is split into two packages: [`PopulationCoding`](PopulationCoding) and [`scaling_analysis`](scaling_analysis).

## PopulationCoding
[![Documentation Status](https://readthedocs.org/projects/populationcoding/badge/?version=latest)](https://scaling-analysis.readthedocs.io/projects/PopulationCoding/en/latest/?badge=latest)

[`PopulationCoding`](PopulationCoding) includes some more general purpose functions for dimensionality reduction and other analysis of neurobehavioral data. Check out the full API in the [documentation](https://scaling-analysis.readthedocs.io/projects/PopulationCoding).

## scaling_analysis
[![Documentation Status](https://readthedocs.org/projects/scaling-analysis/badge/?version=latest)](https://scaling-analysis.readthedocs.io/en/latest/?badge=latest)

[`scaling_analysis`](scaling_analysis) enables estimation of the reliable dimensionality of neuronal population dynamics and its scaling as a function of the number of sampled neurons, as described by [Manley et al. Neuron 2024](https://doi.org/10.1016/j.neuron.2024.02.011). Check out the [demo](https://scaling-analysis.readthedocs.io/en/latest/demo.html) for examples!

## Example data

Interested in large-scale neuronal population dynamics? Example datasets are freely available at https://doi.org/10.5281/zenodo.10403684.