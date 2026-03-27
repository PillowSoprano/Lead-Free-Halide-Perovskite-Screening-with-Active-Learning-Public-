
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16, 'axes.labelsize': 18, 'axes.titlesize': 18,
                     'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14})
import seaborn as sns
from sklearn.linear_model import LinearRegression
from typing import Dict, Tuple, List
import os


class CalibrationSensitivityAnalyzer:

    def __init__(self, x_pbe: np.ndarray, y_exp: np.ndarray):
        self.x = x_pbe.reshape(-1, 1)
        self.y = y_exp
        self.n = len(y_exp)

    def loocv_analysis(self) -> Dict:
        slopes = np.zeros(self.n)
        intercepts = np.zeros(self.n)
        predictions = np.zeros(self.n)

        for i in range(self.n):
                                  
            mask = np.ones(self.n, dtype=bool)
            mask[i] = False

                                         
            reg = LinearRegression().fit(self.x[mask], self.y[mask])
            slopes[i] = reg.coef_[0]
            intercepts[i] = reg.intercept_

                                    
            predictions[i] = reg.predict(self.x[i:i+1])[0]

                          
        errors = predictions - self.y
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))

        return {
            'slopes': slopes,
            'intercepts': intercepts,
            'predictions': predictions,
            'mae': mae,
            'rmse': rmse,
            'slope_mean': slopes.mean(),
            'slope_std': slopes.std(),
            'intercept_mean': intercepts.mean(),
            'intercept_std': intercepts.std()
        }

    def bootstrap_ci(self, n_bootstrap: int = 10000, ci: float = 0.95, random_state: int = 0) -> Dict:
                                                                  
        rng = np.random.default_rng(random_state)

        slopes = np.zeros(n_bootstrap)
        intercepts = np.zeros(n_bootstrap)

        for b in range(n_bootstrap):
                                                 
            idx = rng.integers(0, self.n, self.n)

                                     
            reg = LinearRegression().fit(self.x[idx], self.y[idx])
            slopes[b] = reg.coef_[0]
            intercepts[b] = reg.intercept_

                                      
        alpha = 1 - ci
        lower = alpha / 2
        upper = 1 - alpha / 2

        slope_ci = np.quantile(slopes, [lower, upper])
        intercept_ci = np.quantile(intercepts, [lower, upper])

        return {
            'slopes': slopes,
            'intercepts': intercepts,
            'slope_ci': slope_ci,
            'intercept_ci': intercept_ci,
            'slope_mean': slopes.mean(),
            'intercept_mean': intercepts.mean()
        }

    def plot_loocv_bootstrap(self, loocv_results: Dict, bootstrap_results: Dict,
                             save_path: str = None):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

                     
        ax = axes[0, 0]
        ax.hist(loocv_results['slopes'], bins=10, edgecolor='black', alpha=0.7)
        ax.axvline(loocv_results['slope_mean'], color='red', linestyle='--',
                   label=f"Mean: {loocv_results['slope_mean']:.3f}")
        ax.set_xlabel('Slope (LOOCV)', fontsize=18)
        ax.set_ylabel('Frequency', fontsize=18)
        ax.set_title('(a) LOOCV Slope Distribution', fontsize=18, weight='bold')
        ax.legend()
        ax.grid(alpha=0.3)


        ax = axes[0, 1]
        ax.hist(loocv_results['intercepts'], bins=10, edgecolor='black', alpha=0.7)
        ax.axvline(loocv_results['intercept_mean'], color='red', linestyle='--',
                   label=f"Mean: {loocv_results['intercept_mean']:.3f}")
        ax.set_xlabel('Intercept (LOOCV)', fontsize=18)
        ax.set_ylabel('Frequency', fontsize=18)
        ax.set_title('(b) LOOCV Intercept Distribution', fontsize=18, weight='bold')
        ax.legend()
        ax.grid(alpha=0.3)


        ax = axes[1, 0]
        ax.hist(bootstrap_results['slopes'], bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(bootstrap_results['slope_ci'][0], color='red', linestyle='--',
                   label=f"95% CI: [{bootstrap_results['slope_ci'][0]:.3f}, {bootstrap_results['slope_ci'][1]:.3f}]")
        ax.axvline(bootstrap_results['slope_ci'][1], color='red', linestyle='--')
        ax.set_xlabel('Slope (Bootstrap)', fontsize=18)
        ax.set_ylabel('Frequency', fontsize=18)
        ax.set_title('(c) Bootstrap Slope Distribution', fontsize=18, weight='bold')
        ax.legend()
        ax.grid(alpha=0.3)


        ax = axes[1, 1]
        ax.hist(bootstrap_results['intercepts'], bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(bootstrap_results['intercept_ci'][0], color='red', linestyle='--',
                   label=f"95% CI: [{bootstrap_results['intercept_ci'][0]:.3f}, {bootstrap_results['intercept_ci'][1]:.3f}]")
        ax.axvline(bootstrap_results['intercept_ci'][1], color='red', linestyle='--')
        ax.set_xlabel('Intercept (Bootstrap)', fontsize=18)
        ax.set_ylabel('Frequency', fontsize=18)
        ax.set_title('(d) Bootstrap Intercept Distribution', fontsize=18, weight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved LOOCV/Bootstrap plot to {save_path}")

        return fig


class CandidateSetSensitivity:

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

                                        
        self.gap_center = 1.34
        self.gap_halfwidth = 0.25
        self.ehull_max = 0.05
        self.sigma_min = 0.05
        self.sigma_max = 0.50

                                                         
        self.heavy_d = {'Au', 'Pd', 'Pt', 'Hg', 'Ir', 'Rh', 'Os', 'Re', 'W', 'Ta'}

    def apply_calibration(self, strategy: str, a_global: float, b_global: float,
                         a_sn: float = None, b_sn: float = None,
                         a_ge: float = None, b_ge: float = None) -> np.ndarray:
        mu = self.df['mu_pbe_pred'].values

        if strategy == 'S0':
                            
            return mu

        elif strategy == 'S1':
                                
            return a_global * mu + b_global

        elif strategy == 'S2':
                                    
            gap_calibrated = a_global * mu + b_global           

                                                
            if 'formula' in self.df.columns:
                for i, formula in enumerate(self.df['formula']):
                    if 'Sn' in formula and a_sn is not None:
                        gap_calibrated[i] = a_sn * mu[i] + b_sn
                    elif 'Ge' in formula and a_ge is not None:
                        gap_calibrated[i] = a_ge * mu[i] + b_ge

            return gap_calibrated

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def select_candidates(self, gap_used: np.ndarray) -> np.ndarray:
        ok_gap = ((gap_used >= self.gap_center - self.gap_halfwidth) &
                  (gap_used <= self.gap_center + self.gap_halfwidth))
        ok_ehull = self.df['e_hull'].values <= self.ehull_max
        ok_sigma = ((self.df['sigma'].values >= self.sigma_min) &
                    (self.df['sigma'].values <= self.sigma_max))

        return ok_gap & ok_ehull & ok_sigma

    def jaccard_overlap(self, mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        intersection = np.sum(mask_a & mask_b)
        union = np.sum(mask_a | mask_b)
        return intersection / union if union > 0 else np.nan

    def count_heavy_d(self, mask: np.ndarray) -> Tuple[int, float]:
        if 'formula' not in self.df.columns:
            return 0, np.nan

        candidates = self.df[mask]
        heavy_d_count = 0

        for formula in candidates['formula']:
                                                                     
            if any(elem in formula for elem in self.heavy_d):
                heavy_d_count += 1

        total = len(candidates)
        fraction = heavy_d_count / total if total > 0 else np.nan

        return heavy_d_count, fraction

    def compare_strategies(self, a_global: float, b_global: float,
                          a_sn: float = None, b_sn: float = None,
                          a_ge: float = None, b_ge: float = None) -> pd.DataFrame:
        results = []

        strategies = ['S0', 'S1', 'S2']
        masks = {}

        for strategy in strategies:
            gap_used = self.apply_calibration(strategy, a_global, b_global,
                                              a_sn, b_sn, a_ge, b_ge)
            mask = self.select_candidates(gap_used)
            masks[strategy] = mask

            n_candidates = np.sum(mask)
            heavy_d_count, heavy_d_frac = self.count_heavy_d(mask)

            results.append({
                'Strategy': strategy,
                'N_candidates': n_candidates,
                'Heavy_d_count': heavy_d_count,
                'Heavy_d_fraction': heavy_d_frac
            })

                                    
        jaccard_s0_s1 = self.jaccard_overlap(masks['S0'], masks['S1'])
        jaccard_s0_s2 = self.jaccard_overlap(masks['S0'], masks['S2'])
        jaccard_s1_s2 = self.jaccard_overlap(masks['S1'], masks['S2'])

        df_results = pd.DataFrame(results)

                                                    
        print(f"\nJaccard Overlaps:")
        print(f"  S0 vs S1: {jaccard_s0_s1:.3f}")
        print(f"  S0 vs S2: {jaccard_s0_s2:.3f}")
        print(f"  S1 vs S2: {jaccard_s1_s2:.3f}")

        return df_results, {
            'J_S0_S1': jaccard_s0_s1,
            'J_S0_S2': jaccard_s0_s2,
            'J_S1_S2': jaccard_s1_s2
        }


def run_full_sensitivity_analysis(
    x_pbe_calib: np.ndarray,
    y_exp_calib: np.ndarray,
    df_candidates: pd.DataFrame,
    a_global: float,
    b_global: float,
    a_sn: float = None,
    b_sn: float = None,
    a_ge: float = None,
    b_ge: float = None,
    random_state: int = 42,
    save_dir: str = './outputs'
):
    os.makedirs(save_dir, exist_ok=True)

    print("="*80)
    print("CALIBRATION SENSITIVITY ANALYSIS")
    print("="*80)

                       
    print("\n[1/3] Running LOOCV analysis...")
    analyzer = CalibrationSensitivityAnalyzer(x_pbe_calib, y_exp_calib)
    loocv_results = analyzer.loocv_analysis()

    print(f"\nLOOCV Results:")
    print(f"  Slope:     {loocv_results['slope_mean']:.4f} ± {loocv_results['slope_std']:.4f}")
    print(f"  Intercept: {loocv_results['intercept_mean']:.4f} ± {loocv_results['intercept_std']:.4f}")
    print(f"  MAE:       {loocv_results['mae']:.4f} eV")
    print(f"  RMSE:      {loocv_results['rmse']:.4f} eV")

                     
    print("\n[2/3] Running Bootstrap (10000 iterations)...")
    bootstrap_results = analyzer.bootstrap_ci(n_bootstrap=10000, random_state=random_state)

    print(f"\nBootstrap Results (95% CI):")
    print(f"  Slope:     {bootstrap_results['slope_mean']:.4f} "
          f"[{bootstrap_results['slope_ci'][0]:.4f}, {bootstrap_results['slope_ci'][1]:.4f}]")
    print(f"  Intercept: {bootstrap_results['intercept_mean']:.4f} "
          f"[{bootstrap_results['intercept_ci'][0]:.4f}, {bootstrap_results['intercept_ci'][1]:.4f}]")

                           
    print("\n[3/3] Plotting distributions...")
    analyzer.plot_loocv_bootstrap(
        loocv_results,
        bootstrap_results,
        save_path=os.path.join(save_dir, 'calibration_robustness.png')
    )

                                  
    print("\n" + "="*80)
    print("CANDIDATE SET SENSITIVITY")
    print("="*80)

    sensitivity = CandidateSetSensitivity(df_candidates)
    df_comparison, jaccard_dict = sensitivity.compare_strategies(
        a_global, b_global, a_sn, b_sn, a_ge, b_ge
    )

    print("\nCandidate Set Comparison:")
    print(df_comparison.to_string(index=False))

                  
    df_comparison.to_csv(os.path.join(save_dir, 'candidate_set_sensitivity.csv'), index=False)

                  
    summary_path = os.path.join(save_dir, 'calibration_sensitivity_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("CALIBRATION SENSITIVITY ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")

        f.write("1. LOOCV Analysis (n=15)\n")
        f.write("-"*40 + "\n")
        f.write(f"Slope:     {loocv_results['slope_mean']:.4f} ± {loocv_results['slope_std']:.4f}\n")
        f.write(f"Intercept: {loocv_results['intercept_mean']:.4f} ± {loocv_results['intercept_std']:.4f}\n")
        f.write(f"MAE:       {loocv_results['mae']:.4f} eV\n")
        f.write(f"RMSE:      {loocv_results['rmse']:.4f} eV\n\n")

        f.write("2. Bootstrap CI (B=10000, 95% CI)\n")
        f.write("-"*40 + "\n")
        f.write(f"Slope:     {bootstrap_results['slope_mean']:.4f} "
                f"[{bootstrap_results['slope_ci'][0]:.4f}, {bootstrap_results['slope_ci'][1]:.4f}]\n")
        f.write(f"Intercept: {bootstrap_results['intercept_mean']:.4f} "
                f"[{bootstrap_results['intercept_ci'][0]:.4f}, {bootstrap_results['intercept_ci'][1]:.4f}]\n\n")

        f.write("3. Candidate Set Sensitivity\n")
        f.write("-"*40 + "\n")
        f.write(df_comparison.to_string(index=False) + "\n\n")

        f.write("Jaccard Overlaps:\n")
        f.write(f"  S0 vs S1: {jaccard_dict['J_S0_S1']:.3f}\n")
        f.write(f"  S0 vs S2: {jaccard_dict['J_S0_S2']:.3f}\n")
        f.write(f"  S1 vs S2: {jaccard_dict['J_S1_S2']:.3f}\n")

    print(f"\nSaved summary to {summary_path}")
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)

    return {
        'loocv': loocv_results,
        'bootstrap': bootstrap_results,
        'candidate_sensitivity': df_comparison,
        'jaccard': jaccard_dict
    }


               
if __name__ == "__main__":
    print("\nThis is a module for calibration sensitivity analysis.")
    print("Import and use run_full_sensitivity_analysis() from your main pipeline.")
    print("\nExample:")
    print("""
    from calibration_sensitivity import run_full_sensitivity_analysis

    results = run_full_sensitivity_analysis(
        x_pbe_calib=x_pbe,
        y_exp_calib=y_exp,
        df_candidates=df_full,
        a_global=0.89,
        b_global=0.59,
        a_sn=0.93,
        b_sn=0.52,
        save_dir='./outputs'
    )
    """)
