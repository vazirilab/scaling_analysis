# PopulationCoding
[![Documentation Status](https://readthedocs.org/projects/populationcoding/badge/?version=latest)](https://scaling-analysis.readthedocs.io/projects/PopulationCoding/en/latest/?badge=latest)

`PopulationCoding` includes some functions for dimensionality reduction and other analysis of neurobehavioral data.
A few functions of particular relevance include:

- [dimred.SVCA](https://scaling-analysis.readthedocs.io/projects/PopulationCoding/en/latest/PopulationCoding.html#PopulationCoding.dimred.SVCA): SVCA, originally described by [Stringer et al. 2019](https://doi.org/10.1126/science.aav7893), with a few extra conveniences. Utilized for reliable dimensionality estimation in [scaling_analysis](../scaling_analysis/scaling_analysis/svca.py)
- [predict.canonical_cov](https://scaling-analysis.readthedocs.io/projects/PopulationCoding/en/latest/PopulationCoding.html#PopulationCoding.predict.canonical_cov): canonical covariance analysis, utilized for prediction of neural activity from behvaior in [scaling_analysis.predict.predict_from_behavior](../scaling_analysis/scaling_analysis/predict.py)
- [corr.sig_stim_corr](https://scaling-analysis.readthedocs.io/projects/PopulationCoding/en/latest/PopulationCoding.html#PopulationCoding.corr.sig_stim_corr): identify neurons significantly correlated to a stimulus according to the protocol we used in [Demas et al. 2021](https://doi.org/10.1038/s41592-021-01239-8)

### Usage

`pip install PopulationCoding`

Check out the full API in the [documentation](https://scaling-analysis.readthedocs.io/projects/PopulationCoding).

### Citation

If you use this package, please cite the [paper](https://doi.org/10.1016/j.neuron.2024.02.011):

> Manley, J., Lu, S., Barber, K., Demas, J., Kim, H., Meyer, D., Martínez Traub, F., & Vaziri, A. (2024). Simultaneous, cortex-wide dynamics of up to 1 million neurons reveal unbounded scaling of dimensionality with neuron number. Neuron. [https://doi.org/10.1016/j.neuron.2024.02.011](https://doi.org/10.1016/j.neuron.2024.02.011).