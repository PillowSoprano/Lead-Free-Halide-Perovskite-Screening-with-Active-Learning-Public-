
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, Tuple, List
import os


class MultiRoundALSimulator:

    def __init__(self,
                 model_class,
                 model_params: dict,
                 n_initial: int = 200,
                 k_per_round: int = 50,
                 n_rounds: int = 6,
                 random_state: int = None):
        self.model_class = model_class
        self.model_params = model_params
        self.n_initial = n_initial
        self.k_per_round = k_per_round
        self.n_rounds = n_rounds
        self.random_state = random_state

    def uncertainty_acquisition(self, mu: np.ndarray, sigma: np.ndarray, k: int) -> np.ndarray:
        return np.argsort(sigma)[-k:]

    def target_oriented_acquisition(self, mu: np.ndarray, sigma: np.ndarray,
                                    k: int, target: float = 1.34, beta: float = 0.5) -> np.ndarray:
        score = sigma - beta * np.abs(mu - target)
        return np.argsort(score)[-k:]

    def random_acquisition(self, mu: np.ndarray, sigma: np.ndarray, k: int) -> np.ndarray:
        n = len(mu)
        return np.random.choice(n, size=k, replace=False)

    def run_single_trial(self,
                        X_pool: np.ndarray,
                        y_pool: np.ndarray,
                        X_test: np.ndarray,
                        y_test: np.ndarray,
                        acquisition_fn: str = 'target_oriented',
                        verbose: bool = False) -> Dict:
        n_pool = len(X_pool)
        all_pool_indices = np.arange(n_pool)

                                        
        L_indices = np.random.choice(all_pool_indices, size=self.n_initial, replace=False)
        U_indices = np.setdiff1d(all_pool_indices, L_indices)

        labeled_sizes = [self.n_initial]
        test_maes = []
        test_rmses = []

        for r in range(self.n_rounds + 1):                                    
                                        
            model = self.model_class(**self.model_params)
            model.fit(X_pool[L_indices], y_pool[L_indices])

                                        
            y_pred_test, _, _ = model.predict_with_uncertainty(X_test)
            mae = np.mean(np.abs(y_pred_test - y_test))
            rmse = np.sqrt(np.mean((y_pred_test - y_test)**2))

            test_maes.append(mae)
            test_rmses.append(rmse)

            if verbose:
                print(f"  Round {r}: Labeled={len(L_indices)}, Test MAE={mae:.4f}")

                                 
            if r == self.n_rounds:
                break

                                              
            if len(U_indices) < self.k_per_round:
                if verbose:
                    print(f"  Warning: Only {len(U_indices)} unlabeled samples left, stopping early.")
                break

                                       
            mu_U, sigma_U, _ = model.predict_with_uncertainty(X_pool[U_indices])

                                                      
            if acquisition_fn == 'uncertainty':
                selected_local = self.uncertainty_acquisition(mu_U, sigma_U, self.k_per_round)
            elif acquisition_fn == 'target_oriented':
                selected_local = self.target_oriented_acquisition(mu_U, sigma_U, self.k_per_round)
            elif acquisition_fn == 'random':
                selected_local = self.random_acquisition(mu_U, sigma_U, self.k_per_round)
            else:
                raise ValueError(f"Unknown acquisition function: {acquisition_fn}")

                                                          
            selected_global = U_indices[selected_local]

                                            
            L_indices = np.concatenate([L_indices, selected_global])
            U_indices = np.setdiff1d(U_indices, selected_global)

            labeled_sizes.append(len(L_indices))

        return {
            'labeled_sizes': labeled_sizes,
            'test_maes': test_maes,
            'test_rmses': test_rmses
        }

    def run_multiple_seeds(self,
                          X: np.ndarray,
                          y: np.ndarray,
                          groups: np.ndarray,
                          n_seeds: int = 10,
                          acquisition_fns: List[str] = None) -> Dict:
        if acquisition_fns is None:
            acquisition_fns = ['target_oriented', 'uncertainty', 'random']

        print("="*80)
        print(f"MULTI-ROUND ACTIVE LEARNING SIMULATION (R={self.n_rounds} rounds)")
        print("="*80)
        print(f"Initial labeled: {self.n_initial}")
        print(f"Per-round query: {self.k_per_round}")
        print(f"Total rounds:    {self.n_rounds}")
        print(f"Random seeds:    {n_seeds}")
        print(f"Acquisition fns: {acquisition_fns}")
        print("="*80)

        results = {fn: [] for fn in acquisition_fns}

        for seed in range(n_seeds):
            print(f"\n[Seed {seed+1}/{n_seeds}]")
            np.random.seed(seed if self.random_state is None else self.random_state + seed)

                                                                
            unique_groups = np.unique(groups)
            n_test_groups = max(1, len(unique_groups) // 5)                          
            test_groups = np.random.choice(unique_groups, size=n_test_groups, replace=False)

            test_mask = np.isin(groups, test_groups)
            pool_mask = ~test_mask

            X_pool = X[pool_mask]
            y_pool = y[pool_mask]
            X_test = X[test_mask]
            y_test = y[test_mask]

            print(f"  Pool size: {len(X_pool)}, Test size: {len(X_test)}")

                                           
            for fn in acquisition_fns:
                trial_result = self.run_single_trial(
                    X_pool, y_pool, X_test, y_test,
                    acquisition_fn=fn,
                    verbose=False
                )
                results[fn].append(trial_result)

        return results

    def aggregate_results(self, results: Dict) -> pd.DataFrame:
        rows = []

        for fn, trials in results.items():
            n_rounds_actual = len(trials[0]['test_maes'])

            for r in range(n_rounds_actual):
                maes = [trial['test_maes'][r] for trial in trials]
                rmses = [trial['test_rmses'][r] for trial in trials]
                labeled_size = trials[0]['labeled_sizes'][r]

                rows.append({
                    'acquisition_fn': fn,
                    'round': r,
                    'labeled_size': labeled_size,
                    'mean_mae': np.mean(maes),
                    'std_mae': np.std(maes),
                    'mean_rmse': np.mean(rmses),
                    'std_rmse': np.std(rmses)
                })

        return pd.DataFrame(rows)

    def plot_learning_curves(self, df_agg: pd.DataFrame, save_path: str = None):
        fig, ax = plt.subplots(figsize=(10, 6))

        acquisition_fns = df_agg['acquisition_fn'].unique()
        colors = {'target_oriented': 'coral', 'uncertainty': 'steelblue', 'random': 'gray'}
        labels = {
            'target_oriented': 'Target-Oriented',
            'uncertainty': 'Uncertainty Sampling',
            'random': 'Random (Baseline)'
        }

        for fn in acquisition_fns:
            df_fn = df_agg[df_agg['acquisition_fn'] == fn]

            x = df_fn['labeled_size'].values
            y_mean = df_fn['mean_mae'].values
            y_std = df_fn['std_mae'].values

            color = colors.get(fn, 'black')
            label = labels.get(fn, fn)

                             
            ax.plot(x, y_mean, marker='o', linewidth=2.5, color=color, label=label)

                                    
            ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2, color=color)

        ax.set_xlabel('Labeled Set Size', fontsize=13)
        ax.set_ylabel('Test MAE (eV)', fontsize=13)
        ax.set_title('Multi-Round Active Learning Curves', fontsize=14, weight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nSaved learning curves to {save_path}")

        return fig

    def statistical_analysis(self, results: Dict, final_round: int = None) -> pd.DataFrame:
        acquisition_fns = list(results.keys())
        n_seeds = len(results[acquisition_fns[0]])

        if final_round is None:
            final_round = len(results[acquisition_fns[0]][0]['test_maes']) - 1

                                      
        mae_dict = {}
        for fn in acquisition_fns:
            mae_dict[fn] = [trial['test_maes'][final_round] for trial in results[fn]]

                          
        comparisons = []
        for i, fn1 in enumerate(acquisition_fns):
            for fn2 in acquisition_fns[i+1:]:
                mae1 = np.array(mae_dict[fn1])
                mae2 = np.array(mae_dict[fn2])

                               
                t_stat, p_value = stats.ttest_rel(mae1, mae2)

                                       
                mean1 = np.mean(mae1)
                mean2 = np.mean(mae2)
                improvement_pct = (mean1 - mean2) / mean1 * 100

                comparisons.append({
                    'fn1': fn1,
                    'fn2': fn2,
                    'mean1': mean1,
                    'mean2': mean2,
                    'improvement_pct': improvement_pct,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': '***' if p_value < 0.01 else '**' if p_value < 0.05 else '*' if p_value < 0.10 else ''
                })

        return pd.DataFrame(comparisons)


def run_multiround_al_experiment(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_class,
    model_params: dict,
    n_initial: int = 200,
    k_per_round: int = 50,
    n_rounds: int = 6,
    n_seeds: int = 10,
    save_dir: str = './outputs'
) -> Dict:
    os.makedirs(save_dir, exist_ok=True)

                          
    simulator = MultiRoundALSimulator(
        model_class=model_class,
        model_params=model_params,
        n_initial=n_initial,
        k_per_round=k_per_round,
        n_rounds=n_rounds,
        random_state=42
    )

                     
    results = simulator.run_multiple_seeds(
        X, y, groups,
        n_seeds=n_seeds,
        acquisition_fns=['target_oriented', 'uncertainty', 'random']
    )

                       
    df_agg = simulator.aggregate_results(results)

                          
    simulator.plot_learning_curves(
        df_agg,
        save_path=os.path.join(save_dir, 'multiround_al_curves.png')
    )

                          
    df_stats = simulator.statistical_analysis(results)

    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE (Final Round)")
    print("="*80)
    print(df_stats.to_string(index=False))

                  
    df_agg.to_csv(os.path.join(save_dir, 'multiround_al_aggregated.csv'), index=False)
    df_stats.to_csv(os.path.join(save_dir, 'multiround_al_statistics.csv'), index=False)

                  
    summary_path = os.path.join(save_dir, 'multiround_al_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("MULTI-ROUND ACTIVE LEARNING SUMMARY\n")
        f.write("="*80 + "\n\n")

        f.write("Experiment Setup:\n")
        f.write(f"  Initial labeled: {n_initial}\n")
        f.write(f"  Per-round query: {k_per_round}\n")
        f.write(f"  Total rounds:    {n_rounds}\n")
        f.write(f"  Random seeds:    {n_seeds}\n\n")

        f.write("Final Round Performance (Round {}):\n".format(n_rounds))
        f.write("-"*40 + "\n")
        final_round_data = df_agg[df_agg['round'] == n_rounds]
        for _, row in final_round_data.iterrows():
            f.write(f"  {row['acquisition_fn']:20s}: {row['mean_mae']:.4f} ± {row['std_mae']:.4f} eV\n")

        f.write("\nPairwise Comparisons:\n")
        f.write("-"*40 + "\n")
        f.write(df_stats.to_string(index=False) + "\n\n")

        f.write("Interpretation:\n")
        f.write("-"*40 + "\n")
        significant_comparisons = df_stats[df_stats['p_value'] < 0.05]
        if len(significant_comparisons) > 0:
            f.write("Significant differences found at final round!\n")
            f.write("Multi-round curves demonstrate AL value better than single-round p-value.\n")
        else:
            f.write("No significant difference at final round.\n")
            f.write("Consider: (1) More rounds, (2) Different acquisition strategies, (3) Task inherently easy.\n")

    print(f"\nSaved summary to {summary_path}")
    print("\n" + "="*80)
    print("MULTI-ROUND AL EXPERIMENT COMPLETE!")
    print("="*80)

    return {
        'results_raw': results,
        'aggregated': df_agg,
        'statistics': df_stats
    }


               
if __name__ == "__main__":
    print("\nThis is a module for multi-round active learning simulation.")
    print("Import and use run_multiround_al_experiment() from your main pipeline.")
    print("\nExample:")
    print("""
    from active_learning_multiround import run_multiround_al_experiment

    results = run_multiround_al_experiment(
        X=X, y=y, groups=b_site_groups,
        model_class=ImprovedEnsemblePredictor,
        model_params={'n_models': 15, 'random_state': 42},
        n_initial=200,
        k_per_round=50,
        n_rounds=6,
        n_seeds=10,
        save_dir='./outputs'
    )
    """)
