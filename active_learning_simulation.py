
import warnings
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")


                                            
               
                                            

@dataclass
class ActiveLearningConfig:
                              
    n_initial: int = 200

                                            
    k_per_round: int = 50

                                      
    n_rounds: int = 1                                        

                                      
    random_seeds: List[int] = None

                   
    test_size: float = 0.2

                                           
    use_group_split: bool = True

                                                           
    target_value: Optional[float] = 1.34

                                                            
    acquisition_beta: float = 1.0

                                   
    n_estimators: int = 400

    def __post_init__(self):
        if self.random_seeds is None:
            self.random_seeds = list(range(10))                       


                                            
                            
                                            

class UncertaintyEnsemble:

    def __init__(self, n_estimators: int = 400, random_state: int = 42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=25,
            min_samples_split=3,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=random_state,
            n_jobs=-1,
            bootstrap=True
        )
        self.is_fitted = False

    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_with_uncertainty(self, X) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

                                        
        all_predictions = np.stack([
            tree.predict(X) for tree in self.model.estimators_
        ], axis=0)

                                   
        mu = all_predictions.mean(axis=0)
        sigma = all_predictions.std(axis=0, ddof=1)

        return mu, sigma

    def predict(self, X):
        mu, _ = self.predict_with_uncertainty(X)
        return mu


                                            
                       
                                            

class AcquisitionFunction:

    @staticmethod
    def uncertainty_sampling(mu: np.ndarray, sigma: np.ndarray,
                            k: int, **kwargs) -> np.ndarray:
        scores = sigma
        return np.argsort(scores)[-k:]

    @staticmethod
    def target_oriented_uncertainty(mu: np.ndarray, sigma: np.ndarray,
                                   k: int, target: float = 1.34,
                                   beta: float = 1.0, **kwargs) -> np.ndarray:
                                          
                                                    
        scores = sigma - beta * np.abs(mu - target)
        return np.argsort(scores)[-k:]

    @staticmethod
    def expected_improvement(mu: np.ndarray, sigma: np.ndarray,
                           k: int, current_best: float, **kwargs) -> np.ndarray:
        from scipy.stats import norm

                                
        sigma_safe = np.maximum(sigma, 1e-6)

                 
        z = (mu - current_best) / sigma_safe

                              
        ei = (mu - current_best) * norm.cdf(z) + sigma_safe * norm.pdf(z)

        return np.argsort(ei)[-k:]

    @staticmethod
    def random_sampling(mu: np.ndarray, sigma: np.ndarray,
                       k: int, rng: np.random.Generator, **kwargs) -> np.ndarray:
        return rng.choice(len(mu), size=k, replace=False)


                                            
                           
                                            

class ActiveLearningSimulator:

    def __init__(self, config: ActiveLearningConfig):
        self.config = config
        self.results = []

    def prepare_data_split(self, X, y, formulas: List[str],
                          groups: Optional[np.ndarray] = None,
                          seed: int = 42) -> Dict:
        if self.config.use_group_split and groups is not None:
                                                     
            gss = GroupShuffleSplit(
                n_splits=1,
                test_size=self.config.test_size,
                random_state=seed
            )
            train_pool_idx, test_idx = next(gss.split(X, groups=groups))
        else:
                                   
            train_pool_idx, test_idx = train_test_split(
                np.arange(len(X)),
                test_size=self.config.test_size,
                random_state=seed
            )

        return {
            'train_pool_idx': train_pool_idx,
            'test_idx': test_idx,
            'X_train_pool': X[train_pool_idx],
            'y_train_pool': y[train_pool_idx],
            'formulas_train_pool': [formulas[i] for i in train_pool_idx],
            'X_test': X[test_idx],
            'y_test': y[test_idx],
            'formulas_test': [formulas[i] for i in test_idx]
        }

    def run_single_trial(self,
                        X, y, formulas: List[str],
                        groups: Optional[np.ndarray] = None,
                        acquisition_fn: Callable = None,
                        acquisition_name: str = "Uncertainty",
                        seed: int = 42) -> Dict:
        rng = np.random.default_rng(seed)

                            
        data_split = self.prepare_data_split(X, y, formulas, groups, seed)

        train_pool_size = len(data_split['train_pool_idx'])
        X_pool = data_split['X_train_pool']
        y_pool = data_split['y_train_pool']
        X_test = data_split['X_test']
        y_test = data_split['y_test']

                                           
        all_pool_indices = np.arange(train_pool_size)

                                                
        n_init = min(self.config.n_initial, train_pool_size)
        L_indices = rng.choice(all_pool_indices, size=n_init, replace=False)
        U_indices = np.setdiff1d(all_pool_indices, L_indices)

                       
        round_results = []

                                
        X_L = X_pool[L_indices]
        y_L = y_pool[L_indices]

        model = UncertaintyEnsemble(
            n_estimators=self.config.n_estimators,
            random_state=seed
        )
        model.fit(X_L, y_L)

                              
        y_pred_test = model.predict(X_test)
        metrics = {
            'round': 0,
            'n_labeled': len(L_indices),
            'mae': mean_absolute_error(y_test, y_pred_test),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'r2': r2_score(y_test, y_pred_test)
        }
        round_results.append(metrics)

                                
        for round_num in range(1, self.config.n_rounds + 1):
                                       
            X_U = X_pool[U_indices]
            mu_U, sigma_U = model.predict_with_uncertainty(X_U)

                                  
            k = min(self.config.k_per_round, len(U_indices))

            if acquisition_fn is None:
                acquisition_fn = AcquisitionFunction.uncertainty_sampling

                                    
            acq_kwargs = {
                'target': self.config.target_value,
                'beta': self.config.acquisition_beta,
                'current_best': y_L.min() if 'improvement' in acquisition_name.lower() else None,
                'rng': rng
            }

                              
            selected_in_U = acquisition_fn(mu_U, sigma_U, k, **acq_kwargs)
            queried_indices = U_indices[selected_in_U]

                                                   
            L_indices = np.concatenate([L_indices, queried_indices])
            U_indices = np.setdiff1d(U_indices, queried_indices)

                           
            X_L = X_pool[L_indices]
            y_L = y_pool[L_indices]

            model = UncertaintyEnsemble(
                n_estimators=self.config.n_estimators,
                random_state=seed + round_num
            )
            model.fit(X_L, y_L)

                                  
            y_pred_test = model.predict(X_test)
            metrics = {
                'round': round_num,
                'n_labeled': len(L_indices),
                'mae': mean_absolute_error(y_test, y_pred_test),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'r2': r2_score(y_test, y_pred_test)
            }
            round_results.append(metrics)

        return {
            'acquisition': acquisition_name,
            'seed': seed,
            'rounds': round_results
        }

    def run_comparison(self,
                      X, y, formulas: List[str],
                      groups: Optional[np.ndarray] = None) -> pd.DataFrame:
        print(f"\n{'='*80}")
        print("ACTIVE LEARNING SIMULATION")
        print(f"{'='*80}")
        print(f"Configuration:")
        print(f"  Initial labeled: {self.config.n_initial}")
        print(f"  Samples per round: {self.config.k_per_round}")
        print(f"  Number of rounds: {self.config.n_rounds}")
        print(f"  Random seeds: {len(self.config.random_seeds)}")
        print(f"  Test split: {'Group-based' if self.config.use_group_split else 'Random'}")

                                       
        strategies = [
            (AcquisitionFunction.uncertainty_sampling, "Uncertainty"),
            (AcquisitionFunction.target_oriented_uncertainty, "Target-Oriented"),
            (AcquisitionFunction.random_sampling, "Random (Baseline)")
        ]

        all_results = []

        for acq_fn, acq_name in strategies:
            print(f"\n{'-'*80}")
            print(f"Running: {acq_name}")
            print(f"{'-'*80}")

            for seed_idx, seed in enumerate(self.config.random_seeds):
                print(f"  Seed {seed_idx+1}/{len(self.config.random_seeds)}: {seed}", end='\r')

                trial_result = self.run_single_trial(
                    X, y, formulas, groups,
                    acquisition_fn=acq_fn,
                    acquisition_name=acq_name,
                    seed=seed
                )

                                           
                for round_metrics in trial_result['rounds']:
                    row = {
                        'acquisition': acq_name,
                        'seed': seed,
                        **round_metrics
                    }
                    all_results.append(row)

            print()                           

        results_df = pd.DataFrame(all_results)
        self.results = results_df

                       
        self._print_summary(results_df)

        return results_df

    def _print_summary(self, results_df: pd.DataFrame):
        print(f"\n{'='*80}")
        print("RESULTS SUMMARY")
        print(f"{'='*80}")

                                        
        summary = results_df.groupby(['acquisition', 'round']).agg({
            'mae': ['mean', 'std'],
            'rmse': ['mean', 'std'],
            'r2': ['mean', 'std']
        }).round(4)

        print("\nMAE (Mean ± Std):")
        print(summary['mae'])

                              
        print(f"\n{'-'*80}")
        print("Improvement from Round 0 to Round 1:")
        print(f"{'-'*80}")

        for acq in results_df['acquisition'].unique():
            subset = results_df[results_df['acquisition'] == acq]

            mae_r0 = subset[subset['round'] == 0]['mae'].mean()
            mae_r1 = subset[subset['round'] == 1]['mae'].mean()
            improvement = (mae_r0 - mae_r1) / mae_r0 * 100

            print(f"{acq:25s}: {mae_r0:.4f} → {mae_r1:.4f} "
                  f"({improvement:+.1f}% improvement)")

                          
        self._statistical_test(results_df)

    def _statistical_test(self, results_df: pd.DataFrame):
        from scipy.stats import ttest_rel

        print(f"\n{'-'*80}")
        print("Statistical Significance (Paired t-test)")
        print(f"{'-'*80}")

                                     
        r1_data = results_df[results_df['round'] == 1]

        strategies = r1_data['acquisition'].unique()
        baseline = "Random (Baseline)"

        if baseline not in strategies:
            print("Baseline not found")
            return

        baseline_mae = r1_data[r1_data['acquisition'] == baseline]['mae'].values

        for strategy in strategies:
            if strategy == baseline:
                continue

            strategy_mae = r1_data[r1_data['acquisition'] == strategy]['mae'].values

                                        
            t_stat, p_value = ttest_rel(baseline_mae, strategy_mae)

            sig_marker = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else "ns"))

            print(f"{strategy} vs {baseline}:")
            print(f"  t = {t_stat:.3f}, p = {p_value:.4f} {sig_marker}")

    def plot_results(self, save_path: str):
        if self.results is None or len(self.results) == 0:
            print("No results to plot. Run simulation first.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

                                  
        ax = axes[0]

        for acq in self.results['acquisition'].unique():
            subset = self.results[self.results['acquisition'] == acq]

                                  
            learning_curve = subset.groupby('round')['mae'].agg(['mean', 'std'])

            rounds = learning_curve.index
            means = learning_curve['mean'].values
            stds = learning_curve['std'].values

                                  
            ax.plot(rounds, means, 'o-', linewidth=2.5, markersize=8,
                   label=acq, alpha=0.8)
            ax.fill_between(rounds, means - stds, means + stds,
                           alpha=0.2)

        ax.set_xlabel('Active Learning Round', fontsize=13, fontweight='bold')
        ax.set_ylabel('MAE on Test Set (eV)', fontsize=13, fontweight='bold')
        ax.set_title('Learning Curves: Active Learning vs Random',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(sorted(self.results['round'].unique()))

                                                 
        ax = axes[1]

        r1_summary = self.results[self.results['round'] == 1].groupby('acquisition')['mae'].agg(['mean', 'std'])
        r1_summary = r1_summary.sort_values('mean')

        colors = ['#E15759' if 'Random' in idx else '#4E79A7' for idx in r1_summary.index]

        bars = ax.bar(range(len(r1_summary)), r1_summary['mean'],
                     yerr=r1_summary['std'], capsize=5,
                     color=colors, edgecolor='black', linewidth=1.5,
                     alpha=0.8)

        ax.set_xticks(range(len(r1_summary)))
        ax.set_xticklabels(r1_summary.index, rotation=15, ha='right')
        ax.set_ylabel('MAE on Test Set (eV)', fontsize=13, fontweight='bold')
        ax.set_title(f'Round 1 Performance Comparison\n'
                    f'(n₀={self.config.n_initial}, K={self.config.k_per_round})',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

                          
        for i, (bar, val, std) in enumerate(zip(bars, r1_summary['mean'], r1_summary['std'])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.005,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Results plot saved: {save_path}")
        plt.close()

    def export_summary(self, save_path: str):
        if self.results is None or len(self.results) == 0:
            print("No results to export. Run simulation first.")
            return

        summary = self.results.groupby(['acquisition', 'round']).agg({
            'mae': ['mean', 'std'],
            'rmse': ['mean', 'std'],
            'r2': ['mean', 'std'],
            'n_labeled': 'mean'
        }).reset_index()

                              
        summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]

        summary.to_csv(save_path, index=False)
        print(f"✓ Summary exported: {save_path}")


                                            
                    
                                            

def run_active_learning_simulation(
    X, y, formulas: List[str],
    groups: Optional[np.ndarray] = None,
    n_initial: int = 200,
    k_per_round: int = 50,
    n_rounds: int = 1,
    n_seeds: int = 10,
    save_dir: str = './outputs'
) -> pd.DataFrame:
    import os
    os.makedirs(save_dir, exist_ok=True)

    config = ActiveLearningConfig(
        n_initial=n_initial,
        k_per_round=k_per_round,
        n_rounds=n_rounds,
        random_seeds=list(range(n_seeds)),
        use_group_split=(groups is not None),
        target_value=1.34                             
    )

    simulator = ActiveLearningSimulator(config)
    results = simulator.run_comparison(X, y, formulas, groups)

                    
    plot_path = os.path.join(save_dir, 'active_learning_results.png')
    simulator.plot_results(plot_path)

                    
    summary_path = os.path.join(save_dir, 'active_learning_summary.csv')
    simulator.export_summary(summary_path)

    return results


if __name__ == "__main__":
    print("Active Learning Simulation module loaded.")
    print("Use run_active_learning_simulation() or ActiveLearningSimulator class.")
