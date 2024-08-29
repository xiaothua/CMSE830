# CMSE830 Midterm Project

## Exploring Efficient Computational Strategies through Interaction and Visualization

### Project Overview

My research project focuses on the development of a 3D phase field model to simulate crystal growth using the Chemical Vapor Deposition (CVD) method. The primary computational technique employed is the Finite Element Method (FEM), which discretizes continuous models to enable numerical simulations. A key aspect of this project is the use of Adaptive Mesh Refinement (AMR) to repeatedly refine the elements in the surface regions in the domain, leading to variations in computational load as the surface area of the model changes. This dynamic refinement presents challenges in efficiently allocating computational resources.

### Objectives

The goal of this project is to compare CPU and memory usage as well as the corresponding computation time across different model sizes and other parameters. Additionally, the project will evaluate the computational efficiency of different CPUs available in the HPCC. Given the numerous parameters that need to be considered, employing interactive and visual methods for data analysis and interpretation is expected to be an effective approach for deriving conclusions.

### Key Features

- **3D Phase Field Modeling**: Simulation of crystal growth using FEM.
- **Adaptive Mesh Refinement (AMR)**: Dynamic adjustment of mesh elements to accurately model surface changes, influencing computational demands.
- **Performance Comparison**: Analysis of CPU and memory consumption, along with computation time, across various model sizes and configurations.
- **Cross-Platform Efficiency Evaluation**: Comparison of different CPU resources that HPCC provided to identify the most efficient computational strategy.
- **Interactive Visualization**: Utilization of interactive visualizations to explore the complex relationships between model parameters, computational resources, and performance outcomes.

### Usage

This project is implemented using [Streamlit](https://streamlit.io/), allowing users to interact with the simulation data and explore various scenarios. The app provides real-time visualizations and analysis tools to help users identify the most efficient computational strategies for their specific needs.

### Conclusion
