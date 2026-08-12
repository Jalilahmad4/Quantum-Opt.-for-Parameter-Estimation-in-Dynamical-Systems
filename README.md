# Quantum Optimization for Parameter Estimation in Dynamical Systems

This repository contains sample code accompanying the paper  
**“A Quantum Optimization Framework for Data-Assimilation-Augmented Parameter Estimation.”**

The code provided here demonstrates the proposed hybrid classical–quantum parameter estimation framework using the **Susceptible–Infected–Susceptible (SIS) epidemiological model**.

## Method

The computational workflow is

**Data Assimilation → Coarse Parameter Grid → Quadratic Surrogate → QUBO → Ising Hamiltonian → Quantum Optimization**

The differential equations and data-assimilation calculations are performed classically. Model evaluations are computed only on a coarse parameter grid. A quadratic surrogate of the resulting cost functional is then constructed and evaluated on a refined parameter grid.

The refined optimization problem is encoded as a **Quadratic Unconstrained Binary Optimization (QUBO)** problem and mapped to an **Ising Hamiltonian** for quantum optimization.

The quantum optimization stage searches for the binary-encoded parameter values that minimize the surrogate cost functional; the quantum computer is not used to solve the SIS differential equations directly.

## SIS Model

The model is

$$
\frac{dS}{dt}=-\beta SI+\gamma I,
$$

$$
\frac{dI}{dt}=\beta SI-\gamma I.
$$

where:

- $\beta$ is the transmission rate,
- $\gamma$ is the recovery rate,
- $S(t)$ is the susceptible population fraction,
- $I(t)$ is the infected population fraction.

The sample implementation estimates $\beta$ and $\gamma$ using observations of $I(t)$.

## Quantum Optimization

The parameter-estimation cost functional is evaluated on a coarse parameter grid and used to construct a quadratic surrogate. The surrogate is evaluated on a finer grid and represented as a QUBO problem.

The binary optimization problem is then mapped to an Ising Hamiltonian and solved using a quantum optimization method.

The sample code demonstrates the complete workflow from SIS data generation and data assimilation to QUBO construction and quantum-assisted parameter recovery.

## Citation

If you use this code in your research, please cite:

> M. J. Ahmad, M. Mohammadisiahroudi, A. Biswas, and K. Hoffman,  
> *A Quantum Optimization Framework for Data-Assimilation-Augmented Parameter Estimation*, 2026.  
> DOI: [10.13140/RG.2.2.26744.81923/1](https://doi.org/10.13140/RG.2.2.26744.81923/1)

### BibTeX

```bibtex
@misc{ahmad2026quantum,
  author = {Ahmad, Muhammad Jalil and Mohammadisiahroudi, Mohammadhossein and Biswas, Animikh and Hoffman, Kathleen},
  title = {A Quantum Optimization Framework for Data-Assimilation-Augmented Parameter Estimation},
  year = {2026},
  doi = {10.13140/RG.2.2.26744.81923/1},
  url = {https://doi.org/10.13140/RG.2.2.26744.81923/1}
}
