"""
Dynamic Resource Allocation for Next-Generation Firewalls
Using Constrained Optimization (Penalty Method)

Optimization - Question 2, Option A
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class FirewallOptimizer:
    """
    Implements the Penalty Method for firewall CPU resource allocation.
    
    
    """
    
    def __init__(self, weights: List[float], stream_names: List[str]):
        self.weights = np.array(weights, dtype=float)
        self.stream_names = stream_names
        self.n_streams = len(weights)
        
        # Carefully tuned parameters
        self.max_iterations = 300
        self.learning_rate = 0.02      # Small for stability
        self.penalty_init = 0.5
        self.penalty_growth = 1.5
        self.tolerance = 1e-5
        self.scale = 20.0
        
        self.history = []
        self.optimal_allocation = None
        
    def objective(self, x: np.ndarray) -> float:
        """Compute objective: sum(w_i * exp(-x_i/scale))"""
        return np.sum(self.weights * np.exp(-x / self.scale))
    
    def constraint(self, x: np.ndarray) -> float:
        """Compute constraint: sum(x_i) - 100"""
        return np.sum(x) - 100
    
    def gradient_objective(self, x: np.ndarray) -> np.ndarray:
        """Gradient of objective"""
        return -(self.weights / self.scale) * np.exp(-x / self.scale)
    
    def gradient_constraint(self, x: np.ndarray) -> np.ndarray:
        """Gradient of constraint (all ones)"""
        return np.ones(self.n_streams)
    
    def penalized_objective(self, x: np.ndarray, penalty: float) -> float:
        """Penalized objective: f(x) + (penalty/2) * g(x)^2"""
        g = self.constraint(x)
        return self.objective(x) + (penalty / 2.0) * g**2
    
    def gradient_penalized(self, x: np.ndarray, penalty: float) -> np.ndarray:
        """Gradient of penalized objective"""
        grad_f = self.gradient_objective(x)
        g = self.constraint(x)
        grad_g = self.gradient_constraint(x)
        return grad_f + penalty * g * grad_g
    
    def optimize(self, verbose: bool = True) -> Dict:
        """Run Penalty Method with proper convergence"""
        
        # Initialize proportional to weights, sum to 100
        x = (self.weights / np.sum(self.weights)) * 100.0
        
        penalty = self.penalty_init
        best_x = x.copy()
        best_violation = abs(self.constraint(x))
        
        if verbose:
            print("="*70)
            print("PENALTY METHOD OPTIMIZATION")
            print("="*70)
            print(f"Streams: {self.n_streams}")
            print(f"Weights: {self.weights}")
            print(f"Initial allocation: {x}")
            print(f"Initial sum: {np.sum(x):.2f}%")
            print(f"Learning rate: {self.learning_rate}")
            print(f"Initial penalty: {penalty}")
            print("-"*70)
        
        for iteration in range(self.max_iterations):
            # Compute metrics
            obj_val = self.objective(x)
            constraint_val = self.constraint(x)
            violation = abs(constraint_val)
            penalized_obj = self.penalized_objective(x, penalty)
            
            # Store history
            self.history.append({
                'iteration': iteration,
                'objective': obj_val,
                'constraint': constraint_val,
                'violation': violation,
                'penalized_objective': penalized_obj,
                'penalty': penalty,
                'allocation': x.copy()
            })
            
            # Track best solution
            if violation < best_violation:
                best_violation = violation
                best_x = x.copy()
            
            # Compute gradient
            grad = self.gradient_penalized(x, penalty)
            
            # Gradient descent step
            x_new = x - self.learning_rate * grad
            
            # Project to positive range
            x_new = np.maximum(1.0, x_new)  # Minimum 1% per stream
            
            # Compute change
            change = np.linalg.norm(x_new - x)
            
            if verbose and iteration % 15 == 0:
                print(f"Iter {iteration:3d} | Obj: {obj_val:7.2f} | "
                      f"g(x): {constraint_val:6.2f} | "
                      f"|g|: {violation:6.2f} | "
                      f"μ: {penalty:7.2f} | "
                      f"Δ: {change:6.4f}")
            
            # Update
            x = x_new
            
            # Increase penalty every 30 iterations
            if iteration > 0 and iteration % 30 == 0:
                penalty *= self.penalty_growth
            
            # Check convergence
            if change < self.tolerance and violation < 0.5:
                if verbose:
                    print(f"\n✓ Converged at iteration {iteration}")
                    print(f"  Change: {change:.6f}")
                    print(f"  Violation: {violation:.4f}")
                break
        
        # Use best solution found
        if best_violation < abs(self.constraint(x)):
            x = best_x
            if verbose:
                print(f"\n✓ Using best solution (violation: {best_violation:.4f})")
        
        self.optimal_allocation = x
        
        total = np.sum(x)
        final_risk = self.objective(x)
        final_constraint = self.constraint(x)
        
        if verbose:
            print("="*70)
            print("OPTIMIZATION RESULTS")
            print("="*70)
            print(f"Total CPU: {total:.2f}%")
            print(f"Final risk: {final_risk:.4f}")
            print(f"Constraint: {final_constraint:.4f}")
            print(f"Converged: {abs(final_constraint) <= 1.0}")
            print("-"*70)
        
        return {
            'optimal_allocation': x,
            'total_allocation': total,
            'final_risk': final_risk,
            'converged': abs(final_constraint) <= 1.0,
            'iterations': iteration + 1,
            'history': self.history
        }
    
    def print_allocation_table(self):
        """Print formatted allocation table"""
        if self.optimal_allocation is None:
            print("Run optimize() first!")
            return
        
        x = self.optimal_allocation
        total = np.sum(x)
        
        print("\n" + "="*85)
        print("CPU ALLOCATION TABLE")
        print("="*85)
        print(f"{'Stream Name':<30} {'Weight':<10} {'CPU %':<12} "
              f"{'% of Total':<12} {'Risk Score':<12}")
        print("-"*85)
        
        for i in range(self.n_streams):
            risk = self.weights[i] * np.exp(-x[i] / self.scale)
            percentage = (x[i] / total) * 100
            print(f"{self.stream_names[i]:<30} {self.weights[i]:<10.0f} "
                  f"{x[i]:<12.2f} {percentage:<12.1f} {risk:<12.4f}")
        
        print("-"*85)
        print(f"{'TOTAL':<30} {'':<10} {total:<12.2f} {100.0:<12.1f} "
              f"{self.objective(x):<12.4f}")
        print("="*85 + "\n")
        
        # Interpretation
        max_idx = np.argmax(self.weights)
        min_idx = np.argmin(self.weights)
        print("INTERPRETATION:")
        print(f"• Highest-risk stream (w={self.weights[max_idx]:.0f}): {x[max_idx]:.2f}% CPU")
        print(f"• Lowest-risk stream (w={self.weights[min_idx]:.0f}): {x[min_idx]:.2f}% CPU")
        print(f"• Allocation ratio: {x[max_idx]/x[min_idx]:.2f}:1 (high:low risk)")
        print(f"• Total resources utilized: {total:.2f}%\n")
    
    def plot_convergence(self):
        """Plot convergence history"""
        if not self.history:
            print("Run optimize() first!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Penalty Method Convergence Analysis', 
                     fontsize=16, fontweight='bold')
        
        iterations = [h['iteration'] for h in self.history]
        objectives = [h['objective'] for h in self.history]
        violations = [h['violation'] for h in self.history]
        penalties = [h['penalty'] for h in self.history]
        
        # Plot 1: Objective
        axes[0, 0].plot(iterations, objectives, 'b-', linewidth=2.5)
        axes[0, 0].set_xlabel('Iteration', fontsize=12)
        axes[0, 0].set_ylabel('Objective Value (Risk)', fontsize=12)
        axes[0, 0].set_title('Risk Score Minimization', fontsize=13, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Constraint violation
        axes[0, 1].plot(iterations, violations, 'r-', linewidth=2.5)
        axes[0, 1].axhline(y=1.0, color='orange', linestyle='--', linewidth=2, 
                          label='Tolerance (|g|=1)')
        axes[0, 1].axhline(y=0, color='g', linestyle='--', linewidth=2, 
                          label='Perfect (|g|=0)')
        axes[0, 1].set_xlabel('Iteration', fontsize=12)
        axes[0, 1].set_ylabel('|Constraint Violation|', fontsize=12)
        axes[0, 1].set_title('Constraint Satisfaction', fontsize=13, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')
        
        # Plot 3: Penalty parameter
        axes[1, 0].plot(iterations, penalties, 'g-', linewidth=2.5, 
                       marker='o', markersize=4, markevery=max(1, len(iterations)//15))
        axes[1, 0].set_xlabel('Iteration', fontsize=12)
        axes[1, 0].set_ylabel('Penalty Parameter (μ)', fontsize=12)
        axes[1, 0].set_title('Penalty Parameter Growth', fontsize=13, fontweight='bold')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3, which='both')
        
        # Plot 4: CPU allocation
        allocations = np.array([h['allocation'] for h in self.history])
        colors = plt.cm.viridis(np.linspace(0, 0.9, self.n_streams))
        
        for i in range(self.n_streams):
            axes[1, 1].plot(iterations, allocations[:, i], 
                          label=self.stream_names[i], linewidth=2.5, 
                          color=colors[i])
        
        equal = 100 / self.n_streams
        axes[1, 1].axhline(y=equal, color='gray', linestyle=':', 
                          alpha=0.6, linewidth=2, label=f'Equal ({equal:.1f}%)')
        
        axes[1, 1].set_xlabel('Iteration', fontsize=12)
        axes[1, 1].set_ylabel('CPU Allocation (%)', fontsize=12)
        axes[1, 1].set_title('Resource Allocation Evolution', fontsize=13, fontweight='bold')
        axes[1, 1].legend(fontsize=9, loc='best')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('firewall_convergence.png', dpi=300, bbox_inches='tight')
        print("✓ Convergence plot saved as 'firewall_convergence.png'")
        plt.show()


def load_cic_ddos_dataset(dataset_path: str):
    """Load and preprocess CIC-DDoS2019 dataset"""
    print("\n" + "="*70)
    print("LOADING CIC-DDoS2019 DATASET")
    print("="*70)
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    try:
        print(f"Loading: {dataset_path}")
        df = pd.read_csv(dataset_path, nrows=50000)
        print(f"Loaded: {len(df):,} records")
        
        label_col = next((col for col in df.columns if 'label' in col.lower()), None)
        if not label_col:
            raise ValueError("Label column not found")
        
        print(f"\nAttack distribution:")
        print(df[label_col].value_counts())
        
        df_attacks = df[~df[label_col].str.contains('BENIGN', case=False, na=False)].copy()
        print(f"\nAttack records: {len(df_attacks):,}")
        
        # Features for clustering
        feature_cols = [
            ' Flow Duration', ' Total Fwd Packets', ' Total Backward Packets',
            ' Flow Bytes/s', ' Flow Packets/s', ' Fwd Packet Length Mean',
            ' Bwd Packet Length Mean', ' Flow IAT Mean', ' Fwd IAT Mean',
            ' Bwd IAT Mean', ' Fwd Header Length', ' Bwd Header Length'
        ]
        
        available = [c for c in feature_cols if c in df_attacks.columns]
        
        if len(available) >= 5:
            print(f"\nClustering with {len(available)} features")
            
            X = df_attacks[available].fillna(0).replace([np.inf, -np.inf], 0)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            df_attacks['cluster'] = kmeans.fit_predict(X_scaled)
            
            # Create streams with MORE VARIED weights for better visualization
            streams = []
            risk_weights = [950, 850, 800, 750, 650]
            
            for cid in range(5):
                cluster_data = df_attacks[df_attacks['cluster'] == cid]
                count = len(cluster_data)
                
                pattern = "Pattern"
                if ' Flow Duration' in cluster_data.columns:
                    mean_dur = cluster_data[' Flow Duration'].mean()
                    if mean_dur > 100000:
                        pattern = "Long_Duration"
                    elif mean_dur < 10000:
                        pattern = "Short_Burst"
                    else:
                        pattern = "Medium_Flow"
                
                streams.append({
                    'name': f'Cluster_{cid}_{pattern}',
                    'packet_volume': int(count),
                    'risk_weight': risk_weights[cid]
                })
            
            df_streams = pd.DataFrame(streams)
        else:
            # Fallback
            print("\nUsing fallback streams")
            total = len(df_attacks)
            streams = [
                {'name': 'High_Volume_Flooding', 'packet_volume': int(total*0.35), 'risk_weight': 950},
                {'name': 'Protocol_Exploit', 'packet_volume': int(total*0.25), 'risk_weight': 850},
                {'name': 'Application_Layer', 'packet_volume': int(total*0.20), 'risk_weight': 800},
                {'name': 'Stealthy_Attack', 'packet_volume': int(total*0.15), 'risk_weight': 900},
                {'name': 'Mixed_Traffic', 'packet_volume': int(total*0.05), 'risk_weight': 700}
            ]
            df_streams = pd.DataFrame(streams)
        
        print("\n" + "="*70)
        print("TRAFFIC STREAMS")
        print("="*70)
        print(df_streams.to_string(index=False))
        print("="*70)
        
        df_streams.to_csv('dataset_preprocessing_summary.csv', index=False)
        print("\n✓ Saved preprocessing summary")
        
        return df_streams
        
    except Exception as e:
        print(f"\nERROR: {e}")
        raise


def main():
    """Main execution"""
    
    print("="*70)
    print("FIREWALL RESOURCE ALLOCATION - PENALTY METHOD")
    print("AIT 203: Optimization - Question 2, Option A")
    print("="*70)
    
    DATASET_PATH = "Dataset.csv"
    
    df_streams = load_cic_ddos_dataset(DATASET_PATH)
    
    stream_names = df_streams['name'].tolist()
    risk_weights = df_streams['risk_weight'].tolist()
    
    print(f"\n{'='*70}")
    print(f"Optimizing {len(stream_names)} streams")
    print(f"Weights: {risk_weights}")
    print(f"{'='*70}\n")
    
    optimizer = FirewallOptimizer(weights=risk_weights, stream_names=stream_names)
    results = optimizer.optimize(verbose=True)
    
    optimizer.print_allocation_table()
    optimizer.plot_convergence()
    
    # Export
    results_df = pd.DataFrame({
        'Stream': stream_names,
        'Risk_Weight': risk_weights,
        'Packet_Volume': df_streams['packet_volume'].tolist(),
        'Optimal_CPU_Allocation': results['optimal_allocation'],
        'Percentage_of_Total': (results['optimal_allocation'] / 
                               np.sum(results['optimal_allocation'])) * 100,
        'Risk_Contribution': risk_weights * np.exp(-results['optimal_allocation'] / 20.0)
    })
    
    results_df.to_csv('firewall_optimization_results.csv', index=False)
    
    print("\n" + "="*70)
    print("REPORT SUMMARY")
    print("="*70)
    print(f"✓ Dataset: CIC-DDoS2019 ({df_streams['packet_volume'].sum():,} attacks)")
    print(f"✓ Method: Penalty Method with quadratic penalty")
    print(f"✓ Converged: {results['converged']}")
    print(f"✓ Iterations: {results['iterations']}")
    print(f"✓ Total CPU: {results['total_allocation']:.2f}%")
    print(f"✓ Final Risk: {results['final_risk']:.4f}")
    print(f"\nFormulation:")
    print(f"  min  f(x) = Σ w_i × exp(-x_i/20)")
    print(f"  s.t. g(x) = Σ x_i - 100 = 0")
    print(f"  Penalty: P(x,μ) = f(x) + (μ/2)×g(x)²")
    print(f"  Update: x_k+1 = x_k - α∇P(x_k,μ_k)")
    print(f"  Parameters: α={optimizer.learning_rate}, μ₀={optimizer.penalty_init}, γ={optimizer.penalty_growth}")
    print("="*70 + "\n")
    
    return results


if __name__ == "__main__":
    results = main()