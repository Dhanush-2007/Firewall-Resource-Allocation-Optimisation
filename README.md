# Firewall Resource Allocation Optimization

A machine learning and optimization project that performs dynamic CPU resource allocation for next-generation firewalls using the Penalty Method. The project analyzes network traffic from the CIC-DDoS2019 dataset, clusters attack patterns using K-Means, and allocates firewall CPU resources based on traffic risk levels.

---

## Features

- Dynamic firewall CPU resource allocation
- Penalty Method constrained optimization
- Network traffic clustering using K-Means
- Traffic preprocessing and feature scaling
- Risk-based resource allocation
- Convergence visualization
- Export of optimization results
- Performance analysis on CIC-DDoS2019 traffic

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- K-Means Clustering
- Penalty Method Optimization

---

## Project Structure

- `firewall_optimizer.py` – Main optimization and clustering pipeline
- `requirements.txt` – Python dependencies

---

## Dataset

This project uses the **CIC-DDoS2019** dataset provided by the Canadian Institute for Cybersecurity (CIC).

Due to the dataset's large size, it is not included in this repository.

Download the dataset from:

https://www.unb.ca/cic/datasets/ddos-2019.html

After downloading, place the required CSV file in the project directory and rename it to:

```
Dataset.csv
```

---

## Methodology

The project performs the following steps:

1. Load and preprocess the CIC-DDoS2019 dataset.
2. Extract network traffic features.
3. Normalize features using StandardScaler.
4. Cluster traffic using K-Means.
5. Assign risk weights to traffic clusters.
6. Optimize CPU allocation using the Penalty Method.
7. Visualize optimization convergence.
8. Export optimized firewall resource allocation.

---

## Optimization Objective

The optimization minimizes firewall risk while satisfying CPU allocation constraints.

Subject to:

- Total CPU allocation = 100%
- Positive CPU allocation for every traffic stream

---

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python firewall_optimizer.py
```

---

## Future Improvements

- Implement additional constrained optimization techniques
- Compare against Genetic Algorithms and Particle Swarm Optimization
- Evaluate additional clustering algorithms
- Support real-time traffic stream allocation
