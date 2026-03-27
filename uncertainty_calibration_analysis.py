import warnings
import os
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, norm
from scipy import interpolate
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({'font.size': 16, 'axes.labelsize': 18, 'axes.titlesize': 18,
                     'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14})

warnings.filterwarnings("ignore")
                                           

class UncertaintyCalibrator:

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray,
                 sigma: np.ndarray, name: str = "Model"):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.sigma = np.array(sigma)
        self.name = name

                        
        self.errors = np.abs(self.y_pred - self.y_true)
        self.residuals = self.y_pred - self.y_true

                         
        self.results = {}

    def compute_error_uncertainty_correlation(self) -> Dict:
                                                          
        spearman_rho, spearman_p = spearmanr(self.sigma, self.errors)

                                      
        pearson_r, pearson_p = pearsonr(self.sigma, self.errors)

        results = {
            'spearman_rho': spearman_rho,
            'spearman_p': spearman_p,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'interpretation': self._interpret_correlation(spearman_rho)
        }

        self.results['correlation'] = results
        return results

    @staticmethod
    def _interpret_correlation(rho: float) -> str:
        if rho > 0.7:
            return "Strong positive - Uncertainty is highly informative"
        elif rho > 0.5:
            return "Moderate positive - Uncertainty is somewhat informative"
        elif rho > 0.3:
            return "Weak positive - Uncertainty has limited value"
        else:
            return "Very weak - Uncertainty may not be useful"

    def compute_calibration_curve(self, n_bins: int = 10) -> Dict:
                             
        sorted_indices = np.argsort(self.sigma)
        sigma_sorted = self.sigma[sorted_indices]
        errors_sorted = self.errors[sorted_indices]

                     
        bin_size = len(sigma_sorted) // n_bins
        bin_means_sigma = []
        bin_means_error = []
        bin_rmv = []                      
        bin_rmse = []
        bin_counts = []

        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(sigma_sorted)

            bin_sigma = sigma_sorted[start_idx:end_idx]
            bin_errors = errors_sorted[start_idx:end_idx]

            bin_means_sigma.append(bin_sigma.mean())
            bin_means_error.append(bin_errors.mean())
            bin_rmv.append(np.sqrt((bin_sigma ** 2).mean()))
            bin_rmse.append(np.sqrt((bin_errors ** 2).mean()))
            bin_counts.append(len(bin_sigma))

                                                              
        ece = np.mean(np.abs(np.array(bin_means_sigma) - np.array(bin_means_error)))

                                             
        rmsce = np.sqrt(np.mean((np.array(bin_means_sigma) - np.array(bin_means_error)) ** 2))

        results = {
            'bin_means_sigma': np.array(bin_means_sigma),
            'bin_means_error': np.array(bin_means_error),
            'bin_rmv': np.array(bin_rmv),
            'bin_rmse': np.array(bin_rmse),
            'bin_counts': np.array(bin_counts),
            'n_bins': n_bins,
            'ece': ece,                              
            'rmsce': rmsce,                          
            'interpretation': self._interpret_calibration(ece)
        }

        self.results['calibration_curve'] = results
        return results

    @staticmethod
    def _interpret_calibration(ece: float) -> str:
        if ece < 0.05:
            return "Excellent calibration"
        elif ece < 0.1:
            return "Good calibration"
        elif ece < 0.15:
            return "Moderate calibration"
        else:
            return "Poor calibration - σ does not match true error"

    def compute_coverage_and_sharpness(self,
                                        confidence_levels: list = [0.68, 0.90, 0.95]) -> Dict:
        results = {}

        for conf_level in confidence_levels:
                                                   
            z = norm.ppf((1 + conf_level) / 2)

                               
            lower = self.y_pred - z * self.sigma
            upper = self.y_pred + z * self.sigma

                                                      
            in_interval = (self.y_true >= lower) & (self.y_true <= upper)
            coverage = in_interval.mean()

                                            
            interval_width = upper - lower
            sharpness = interval_width.mean()

                                                           
            miscalibration = abs(coverage - conf_level)

            results[f'{int(conf_level*100)}%'] = {
                'z_score': z,
                'coverage': coverage,
                'expected_coverage': conf_level,
                'sharpness': sharpness,
                'miscalibration': miscalibration,
                'n_in_interval': int(in_interval.sum()),
                'n_total': len(self.y_true)
            }

        self.results['coverage'] = results
        return results

    def compute_nll(self) -> Dict:
                                           
        sigma_safe = np.maximum(self.sigma, 1e-6)

                                
        z_scores = (self.y_true - self.y_pred) / sigma_safe

                             
        nll_per_sample = 0.5 * np.log(2 * np.pi) + np.log(sigma_safe) + 0.5 * z_scores**2

                     
        nll = nll_per_sample.mean()

                                   
        mse_component = (0.5 * z_scores**2).mean()
        sigma_component = np.log(sigma_safe).mean()

        results = {
            'nll': nll,
            'nll_per_sample': nll_per_sample,
            'mse_component': mse_component,
            'sigma_component': sigma_component,
            'interpretation': self._interpret_nll(nll)
        }

        self.results['nll'] = results
        return results

    @staticmethod
    def _interpret_nll(nll: float) -> str:
        if nll < 0.5:
            return "Excellent - Very low NLL"
        elif nll < 1.0:
            return "Good - Low NLL"
        elif nll < 2.0:
            return "Moderate - NLL acceptable"
        else:
            return "Poor - High NLL indicates bad calibration"

    def compute_crps(self) -> Dict:
        sigma_safe = np.maximum(self.sigma, 1e-6)
        z = (self.y_true - self.y_pred) / sigma_safe

                                     
        phi_z = norm.cdf(z)
        pdf_z = norm.pdf(z)

                                   
        crps_per_sample = sigma_safe * (
            z * (2 * phi_z - 1) + 2 * pdf_z - 1 / np.sqrt(np.pi)
        )

        crps = crps_per_sample.mean()

        results = {
            'crps': crps,
            'crps_per_sample': crps_per_sample,
            'interpretation': self._interpret_crps(crps)
        }

        self.results['crps'] = results
        return results

    @staticmethod
    def _interpret_crps(crps: float) -> str:
        if crps < 0.2:
            return "Excellent predictive performance"
        elif crps < 0.4:
            return "Good predictive performance"
        elif crps < 0.6:
            return "Moderate predictive performance"
        else:
            return "Poor predictive performance"

    def run_full_analysis(self) -> Dict:
        print(f"\n{'='*80}")
        print(f"Uncertainty Calibration Analysis: {self.name}")
        print(f"{'='*80}")
        print(f"Dataset size: {len(self.y_true)}")
        print(f"Mean uncertainty: {self.sigma.mean():.4f} eV")
        print(f"Mean absolute error: {self.errors.mean():.4f} eV")

                        
        print(f"\n{'-'*80}")
        print("1. Error-Uncertainty Correlation")
        print(f"{'-'*80}")
        corr_results = self.compute_error_uncertainty_correlation()
        print(f"Spearman ρ = {corr_results['spearman_rho']:.4f} "
              f"(p = {corr_results['spearman_p']:.4e})")
        print(f"Pearson r  = {corr_results['pearson_r']:.4f} "
              f"(p = {corr_results['pearson_p']:.4e})")
        print(f"Interpretation: {corr_results['interpretation']}")

                              
        print(f"\n{'-'*80}")
        print("2. Calibration Curve (Binned Analysis)")
        print(f"{'-'*80}")
        calib_results = self.compute_calibration_curve(n_bins=10)
        print(f"Expected Calibration Error (ECE) = {calib_results['ece']:.4f} eV")
        print(f"RMSE Calibration Error = {calib_results['rmsce']:.4f} eV")
        print(f"Interpretation: {calib_results['interpretation']}")

                     
        print(f"\n{'-'*80}")
        print("3. Prediction Interval Coverage")
        print(f"{'-'*80}")
        coverage_results = self.compute_coverage_and_sharpness(
            confidence_levels=[0.68, 0.90, 0.95]
        )
        for level, stats in coverage_results.items():
            print(f"{level} interval (±{stats['z_score']:.2f}σ):")
            print(f"  Coverage: {stats['coverage']:.3f} "
                  f"(expected: {stats['expected_coverage']:.3f})")
            print(f"  Sharpness: {stats['sharpness']:.4f} eV")
            print(f"  Miscalibration: {stats['miscalibration']:.4f}")

                
        print(f"\n{'-'*80}")
        print("4. Negative Log-Likelihood (NLL)")
        print(f"{'-'*80}")
        nll_results = self.compute_nll()
        print(f"NLL = {nll_results['nll']:.4f}")
        print(f"  MSE component: {nll_results['mse_component']:.4f}")
        print(f"  Sigma component: {nll_results['sigma_component']:.4f}")
        print(f"Interpretation: {nll_results['interpretation']}")

                 
        print(f"\n{'-'*80}")
        print("5. Continuous Ranked Probability Score (CRPS)")
        print(f"{'-'*80}")
        crps_results = self.compute_crps()
        print(f"CRPS = {crps_results['crps']:.4f} eV")
        print(f"Interpretation: {crps_results['interpretation']}")

        print(f"\n{'='*80}\n")

        return self.results

    def plot_calibration_diagnostics(self, save_path: str):
        # Publication-quality font settings
        import matplotlib
        matplotlib.rcParams.update({
            'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 18,
            'xtick.labelsize': 14, 'ytick.labelsize': 14,
            'legend.fontsize': 14, 'figure.titlesize': 18,
        })
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

                                                       
        ax = axes[0, 0]

                      
        ax.scatter(self.sigma, self.errors, alpha=0.4, s=80,
                  c='#4E79A7', edgecolors='none')

                        
        z = np.polyfit(self.sigma, self.errors, 1)
        p = np.poly1d(z)
        x_line = np.linspace(self.sigma.min(), self.sigma.max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2.5,
               label=f'Linear fit (slope={z[0]:.2f})')

                                          
        max_val = max(self.sigma.max(), self.errors.max())
        ax.plot([0, max_val], [0, max_val], 'k:', linewidth=2,
               alpha=0.5, label='Ideal (error = σ)')

                    
        corr = self.results['correlation']
        textstr = f"Spearman ρ = {corr['spearman_rho']:.3f}\n"
        textstr += f"Pearson r = {corr['pearson_r']:.3f}"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
               fontsize=14, verticalalignment='top', bbox=props)

        ax.set_xlabel('Predicted Uncertainty (σ, eV)', fontsize=18, fontweight='bold')
        ax.set_ylabel('Absolute Error (eV)', fontsize=18, fontweight='bold')
        ax.set_title('(a) Error vs Uncertainty Correlation', fontsize=18, fontweight='bold')
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3)

                                            
        ax = axes[0, 1]

        calib = self.results['calibration_curve']

                                     
        ax.scatter(calib['bin_means_sigma'], calib['bin_means_error'],
                  s=calib['bin_counts']*3, alpha=0.6, c='#E15759',
                  edgecolors='black', linewidth=1.5, label='Binned data')

                                  
        max_val = max(calib['bin_means_sigma'].max(),
                     calib['bin_means_error'].max())
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2.5,
               label='Perfect calibration')

                             
        textstr = f"ECE = {calib['ece']:.4f} eV\n"
        textstr += f"RMSCE = {calib['rmsce']:.4f} eV"
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
               fontsize=14, verticalalignment='top', bbox=props)

        ax.set_xlabel('Mean Predicted σ in Bin (eV)', fontsize=18, fontweight='bold')
        ax.set_ylabel('Mean Absolute Error in Bin (eV)', fontsize=18, fontweight='bold')
        ax.set_title(f'(b) Calibration Curve ({calib["n_bins"]} bins)',
                    fontsize=18, fontweight='bold')
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3)

                                               
        ax = axes[1, 0]

        coverage_data = self.results['coverage']

                      
        conf_levels = []
        observed_coverage = []
        sharpness_vals = []

        for level_str, stats in coverage_data.items():
            conf_levels.append(stats['expected_coverage'])
            observed_coverage.append(stats['coverage'])
            sharpness_vals.append(stats['sharpness'])

        conf_levels = np.array(conf_levels)
        observed_coverage = np.array(observed_coverage)

                                   
        ax.scatter(conf_levels * 100, observed_coverage * 100,
                  s=200, c='#F28E2B', edgecolors='black', linewidth=2,
                  zorder=3, label='Observed coverage')

                                  
        ax.plot([0, 100], [0, 100], 'k--', linewidth=2.5,
               label='Perfect calibration', zorder=2)

                               
        ax.fill_between([0, 100], [0, 100], [5, 105],
                       alpha=0.1, color='green', label='±5% tolerance')
        ax.fill_between([0, 100], [-5, 95], [0, 100],
                       alpha=0.1, color='green')

                         
        for i, (exp, obs) in enumerate(zip(conf_levels * 100, observed_coverage * 100)):
            ax.annotate(f'{int(exp)}%',
                       xy=(exp, obs), xytext=(5, 5),
                       textcoords='offset points', fontsize=14)

        ax.set_xlabel('Expected Coverage (%)', fontsize=18, fontweight='bold')
        ax.set_ylabel('Observed Coverage (%)', fontsize=18, fontweight='bold')
        ax.set_title('(c) Prediction Interval Coverage', fontsize=18, fontweight='bold')
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3)

                                                            
        ax = axes[1, 1]

                          
        z_scores = self.residuals / np.maximum(self.sigma, 1e-6)

                   
        n, bins, patches = ax.hist(z_scores, bins=30, density=True,
                                   alpha=0.6, color='#59A14F',
                                   edgecolor='black', linewidth=1.2,
                                   label='Observed')

                                    
        x = np.linspace(-4, 4, 1000)
        ax.plot(x, norm.pdf(x), 'r-', linewidth=2.5,
               label='N(0,1) (ideal)')

                    
        mean_z = z_scores.mean()
        std_z = z_scores.std()
        textstr = f"Mean = {mean_z:.3f}\n"
        textstr += f"Std = {std_z:.3f}\n"
        textstr += f"(Ideal: 0, 1)"
        props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
        ax.text(0.65, 0.95, textstr, transform=ax.transAxes,
               fontsize=14, verticalalignment='top', bbox=props)

        ax.set_xlabel('Standardized Residuals (z-score)', fontsize=18, fontweight='bold')
        ax.set_ylabel('Density', fontsize=18, fontweight='bold')
        ax.set_title('(d) Residual Distribution Check', fontsize=18, fontweight='bold')
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Calibration diagnostics saved: {save_path}")
        plt.close()


                                            
                                
                                            

def analyze_model_uncertainty(model, X_test, y_test, save_dir='./outputs'):
    os.makedirs(save_dir, exist_ok=True)

                                      
    y_pred, sigma, _ = model.predict_with_uncertainty(X_test)

                           
    calibrator = UncertaintyCalibrator(
        y_true=y_test,
        y_pred=y_pred,
        sigma=sigma,
        name="Ensemble Model"
    )

                       
    results = calibrator.run_full_analysis()

                    
    plot_path = os.path.join(save_dir, 'uncertainty_calibration_diagnostics.png')
    calibrator.plot_calibration_diagnostics(plot_path)

                         
    summary_data = {
        'Metric': [],
        'Value': [],
        'Interpretation': []
    }

                 
    summary_data['Metric'].append('Spearman ρ')
    summary_data['Value'].append(f"{results['correlation']['spearman_rho']:.4f}")
    summary_data['Interpretation'].append(results['correlation']['interpretation'])

                 
    summary_data['Metric'].append('ECE')
    summary_data['Value'].append(f"{results['calibration_curve']['ece']:.4f} eV")
    summary_data['Interpretation'].append(results['calibration_curve']['interpretation'])

                    
    cov_95 = results['coverage']['95%']
    summary_data['Metric'].append('95% Coverage')
    summary_data['Value'].append(f"{cov_95['coverage']:.3f}")
    summary_data['Interpretation'].append(
        f"Expected: {cov_95['expected_coverage']:.3f}, "
        f"Miscal: {cov_95['miscalibration']:.3f}"
    )

         
    summary_data['Metric'].append('NLL')
    summary_data['Value'].append(f"{results['nll']['nll']:.4f}")
    summary_data['Interpretation'].append(results['nll']['interpretation'])

          
    summary_data['Metric'].append('CRPS')
    summary_data['Value'].append(f"{results['crps']['crps']:.4f} eV")
    summary_data['Interpretation'].append(results['crps']['interpretation'])

    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(save_dir, 'uncertainty_calibration_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Calibration summary saved: {summary_path}")

    return results


                                            
                    
                                            

def example_usage():

    print("="*80)
    print("UNCERTAINTY CALIBRATION ANALYSIS - EXAMPLE")
    print("="*80)

                                             
    np.random.seed(42)
    n_samples = 200

    y_true = np.random.randn(n_samples) * 0.5 + 1.5
    sigma_true = np.abs(np.random.randn(n_samples) * 0.2 + 0.3)

                                             
    y_pred = y_true + np.random.randn(n_samples) * sigma_true * 0.8
    sigma_pred = sigma_true + np.random.randn(n_samples) * 0.05

             
    calibrator = UncertaintyCalibrator(y_true, y_pred, sigma_pred,
                                       name="Synthetic Example")
    results = calibrator.run_full_analysis()

          
    os.makedirs('./outputs', exist_ok=True)
    calibrator.plot_calibration_diagnostics('./outputs/example_calibration.png')

    print("\n✓ Example analysis complete!")


if __name__ == "__main__":
    example_usage()
