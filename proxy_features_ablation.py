import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymatgen.core import Composition, Element
from typing import Dict, List, Tuple
import os


class ProxyFeatureExtractor:

                      
    HALOGENS = {'F', 'Cl', 'Br', 'I'}

    @staticmethod
    def extract_proxy_features(comp: Composition, formula: str) -> Dict[str, float]:
        elements = comp.elements
        fractions = [comp.get_atomic_fraction(el) for el in elements]

                                                
        cations = []
        cation_fracs = []
        anions = []
        anion_fracs = []

        for el, frac in zip(elements, fractions):
            if el.symbol in ProxyFeatureExtractor.HALOGENS:
                anions.append(el)
                anion_fracs.append(frac)
            else:
                cations.append(el)
                cation_fracs.append(frac)

                             
        features = {
            't_proxy': np.nan,
            'mu_proxy': np.nan,
            'delta_chi': np.nan
        }

                                                
        if len(cations) == 0 or len(anions) == 0:
            return features

                                                            
        try:
            cation_radii = []
            for cat in cations:
                try:
                                                                     
                    if hasattr(cat, 'average_ionic_radius'):
                        r = cat.average_ionic_radius
                    else:
                                                   
                        r = cat.atomic_radius
                    if r and r > 0:
                        cation_radii.append(r)
                except:
                    pass

            anion_radii = []
            for an in anions:
                try:
                    if hasattr(an, 'average_ionic_radius'):
                        r = an.average_ionic_radius
                    else:
                        r = an.atomic_radius
                    if r and r > 0:
                        anion_radii.append(r)
                except:
                    pass

                                           
            if len(cation_radii) > 0 and len(anion_radii) > 0:
                r_cation_max = max(cation_radii)
                r_cation_min = min(cation_radii)
                r_anion_mean = np.mean(anion_radii)

                                        
                t_proxy = (r_cation_max + r_anion_mean) / (np.sqrt(2) * (r_cation_min + r_anion_mean))
                features['t_proxy'] = t_proxy

                                         
                mu_proxy = r_cation_min / r_anion_mean
                features['mu_proxy'] = mu_proxy

        except Exception as e:
            pass

                                      
        try:
            cation_chi = [cat.X for cat in cations if cat.X is not None]
            anion_chi = [an.X for an in anions if an.X is not None]

            if len(cation_chi) > 0 and len(anion_chi) > 0:
                chi_cation_mean = np.mean(cation_chi)
                chi_anion_mean = np.mean(anion_chi)
                features['delta_chi'] = abs(chi_cation_mean - chi_anion_mean)

        except Exception as e:
            pass

        return features

    @staticmethod
    def add_proxy_features_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        df_new = df.copy()

        proxy_features = []
        for formula in df['formula']:
            try:
                comp = Composition(formula)
                feats = ProxyFeatureExtractor.extract_proxy_features(comp, formula)
            except:
                feats = {
                    't_proxy': np.nan,
                    'mu_proxy': np.nan,
                    'delta_chi': np.nan
                }
            proxy_features.append(feats)

                          
        df_proxy = pd.DataFrame(proxy_features)
        df_new = pd.concat([df_new, df_proxy], axis=1)

                                                   
        for col in ['t_proxy', 'mu_proxy', 'delta_chi']:
            if col in df_new.columns:
                median_val = df_new[col].median()
                df_new[col].fillna(median_val, inplace=True)

        return df_new


class ProxyFeatureAblation:

    def __init__(self, heavy_d_elements: set = None):
        if heavy_d_elements is None:
            self.heavy_d = {'Au', 'Pd', 'Pt', 'Hg', 'Ir', 'Rh', 'Os', 'Re', 'W', 'Ta'}
        else:
            self.heavy_d = heavy_d_elements

    def contains_heavy_d(self, formula: str) -> bool:
        return any(elem in formula for elem in self.heavy_d)

    def analyze_candidate_sets(self,
                               df_baseline: pd.DataFrame,
                               df_proxy: pd.DataFrame,
                               mask_baseline: np.ndarray,
                               mask_proxy: np.ndarray) -> Dict:
                             
        cand_baseline = df_baseline[mask_baseline]
        n_baseline = len(cand_baseline)
        heavy_d_baseline = sum(self.contains_heavy_d(f) for f in cand_baseline['formula'])
        frac_baseline = heavy_d_baseline / n_baseline if n_baseline > 0 else 0

                           
        cand_proxy = df_proxy[mask_proxy]
        n_proxy = len(cand_proxy)
        heavy_d_proxy = sum(self.contains_heavy_d(f) for f in cand_proxy['formula'])
        frac_proxy = heavy_d_proxy / n_proxy if n_proxy > 0 else 0

        return {
            'baseline': {
                'n_candidates': n_baseline,
                'heavy_d_count': heavy_d_baseline,
                'heavy_d_fraction': frac_baseline
            },
            'proxy': {
                'n_candidates': n_proxy,
                'heavy_d_count': heavy_d_proxy,
                'heavy_d_fraction': frac_proxy
            },
            'reduction': {
                'delta_fraction': frac_baseline - frac_proxy,
                'relative_reduction': ((frac_baseline - frac_proxy) / frac_baseline * 100)
                                      if frac_baseline > 0 else 0
            }
        }

    def plot_comparison(self,
                       df_baseline: pd.DataFrame,
                       df_proxy: pd.DataFrame,
                       mask_baseline: np.ndarray,
                       mask_proxy: np.ndarray,
                       save_path: str = None):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                                   
        ax = axes[0]
        comparison = self.analyze_candidate_sets(df_baseline, df_proxy,
                                                 mask_baseline, mask_proxy)

        categories = ['Baseline\n(29 features)', '+Proxy\n(32 features)']
        fractions = [
            comparison['baseline']['heavy_d_fraction'],
            comparison['proxy']['heavy_d_fraction']
        ]

        bars = ax.bar(categories, fractions, color=['steelblue', 'coral'],
                      edgecolor='black', alpha=0.8)
        ax.set_ylabel('Heavy d Fraction', fontsize=12)
        ax.set_title('(a) Heavy d False Positive Rate', fontsize=13, weight='bold')
        ax.set_ylim(0, max(fractions) * 1.2)
        ax.grid(axis='y', alpha=0.3)

                          
        for bar, frac in zip(bars, fractions):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{frac:.2%}',
                   ha='center', va='bottom', fontsize=11, weight='bold')

                                           
        ax = axes[1]

                                                                   
        cand_baseline = df_baseline[mask_baseline]
        cand_proxy = df_proxy[mask_proxy]

        heavy_d_mask_baseline = cand_baseline['formula'].apply(self.contains_heavy_d)
        heavy_d_mask_proxy = cand_proxy['formula'].apply(self.contains_heavy_d)

        sigma_baseline_heavy = cand_baseline.loc[heavy_d_mask_baseline, 'sigma'].values
        sigma_proxy_heavy = cand_proxy.loc[heavy_d_mask_proxy, 'sigma'].values

        bins = np.linspace(0.05, 0.50, 20)
        ax.hist(sigma_baseline_heavy, bins=bins, alpha=0.6, label='Baseline',
               color='steelblue', edgecolor='black')
        ax.hist(sigma_proxy_heavy, bins=bins, alpha=0.6, label='+Proxy',
               color='coral', edgecolor='black')

        ax.set_xlabel('Uncertainty (eV)', fontsize=12)
        ax.set_ylabel('Count (Heavy d candidates)', fontsize=12)
        ax.set_title('(b) Uncertainty Distribution for Heavy d Materials', fontsize=13, weight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved proxy features comparison plot to {save_path}")

        return fig


def run_proxy_ablation_experiment(
    df_with_features: pd.DataFrame,
    X_baseline: np.ndarray,
    y: np.ndarray,
    model_class,
    model_params: dict,
    mask_candidates_baseline: np.ndarray,
    a_global: float,
    b_global: float,
    gap_center: float = 1.34,
    gap_halfwidth: float = 0.25,
    ehull_max: float = 0.05,
    sigma_max: float = 0.50,
    save_dir: str = './outputs'
) -> Dict:
    os.makedirs(save_dir, exist_ok=True)

    print("="*80)
    print("PROXY FEATURES ABLATION EXPERIMENT")
    print("="*80)

                                
    print("\n[1/4] Adding proxy features...")
    df_proxy = ProxyFeatureExtractor.add_proxy_features_to_dataframe(df_with_features)

                                
    proxy_cols = ['t_proxy', 'mu_proxy', 'delta_chi']
    X_proxy = np.hstack([X_baseline, df_proxy[proxy_cols].values])

    print(f"  Baseline features: {X_baseline.shape[1]}")
    print(f"  +Proxy features:   {X_proxy.shape[1]}")

                           
    print("\n[2/4] Retraining model with +proxy features...")
    model_proxy = model_class(**model_params)
    model_proxy.fit(X_proxy, y)

                                  
    mu_proxy, sigma_proxy, _ = model_proxy.predict_with_uncertainty(X_proxy)

                                                         
    print("\n[2.5/4] Applying calibration (same as baseline)...")
    predicted_gap_exp_proxy = a_global * mu_proxy + b_global

                      
    df_proxy['mu_pbe_pred'] = mu_proxy
    df_proxy['predicted_gap_exp'] = predicted_gap_exp_proxy
    df_proxy['sigma'] = sigma_proxy

    print(f"  Calibration: Exp = {a_global:.4f} × PBE + {b_global:.4f}")

                                                                    
    print("\n[3/4] Re-screening candidates...")

                                    
    ehull_col = 'e_hull' if 'e_hull' in df_proxy.columns else 'e_above_hull'

                                                                              
                                                                     
    mask_proxy = (
        (df_proxy['predicted_gap_exp'].values >= gap_center - gap_halfwidth) &
        (df_proxy['predicted_gap_exp'].values <= gap_center + gap_halfwidth) &
        (df_proxy[ehull_col].values <= ehull_max) &
        (sigma_proxy <= sigma_max)                                       
    )

    print(f"  Baseline candidates: {np.sum(mask_candidates_baseline)}")
    print(f"  +Proxy candidates:   {np.sum(mask_proxy)}")

                     
    print("\n[4/4] Analyzing results...")
    ablation = ProxyFeatureAblation()

    comparison = ablation.analyze_candidate_sets(
        df_with_features, df_proxy,
        mask_candidates_baseline, mask_proxy
    )

    print("\nHeavy d Analysis:")
    print(f"  Baseline: {comparison['baseline']['heavy_d_count']}/{comparison['baseline']['n_candidates']} "
          f"({comparison['baseline']['heavy_d_fraction']:.2%})")
    print(f"  +Proxy:   {comparison['proxy']['heavy_d_count']}/{comparison['proxy']['n_candidates']} "
          f"({comparison['proxy']['heavy_d_fraction']:.2%})")
    print(f"  Reduction: {comparison['reduction']['relative_reduction']:.1f}%")

          
    ablation.plot_comparison(
        df_with_features, df_proxy,
        mask_candidates_baseline, mask_proxy,
        save_path=os.path.join(save_dir, 'proxy_features_ablation.png')
    )

                  
    summary_path = os.path.join(save_dir, 'proxy_ablation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("PROXY FEATURES ABLATION SUMMARY\n")
        f.write("="*80 + "\n\n")

        f.write("Features:\n")
        f.write(f"  Baseline: {X_baseline.shape[1]} features (compositional only)\n")
        f.write(f"  +Proxy:   {X_proxy.shape[1]} features (+3 structural proxies)\n\n")

        f.write("Proxy Features Added:\n")
        f.write("  1. t_proxy:    Pseudo tolerance factor\n")
        f.write("  2. mu_proxy:   Pseudo octahedral factor\n")
        f.write("  3. delta_chi:  Cation-anion electronegativity difference\n\n")

        f.write("Heavy d False Positive Analysis:\n")
        f.write(f"  Baseline: {comparison['baseline']['heavy_d_fraction']:.2%}\n")
        f.write(f"  +Proxy:   {comparison['proxy']['heavy_d_fraction']:.2%}\n")
        f.write(f"  Reduction: {comparison['reduction']['relative_reduction']:.1f}%\n\n")

        f.write("Interpretation:\n")
        if comparison['reduction']['relative_reduction'] > 10:
            f.write("  Proxy features provide modest reduction in heavy d false positives.\n")
            f.write("  However, problem likely persists due to electronic structure effects.\n")
        else:
            f.write("  Proxy features have limited impact on heavy d false positives.\n")
            f.write("  This supports the hypothesis that the blind spot is primarily electronic.\n")

    print(f"\nSaved summary to {summary_path}")
    print("\n" + "="*80)
    print("ABLATION EXPERIMENT COMPLETE!")
    print("="*80)

    return {
        'comparison': comparison,
        'df_proxy': df_proxy,
        'X_proxy': X_proxy,
        'mask_proxy': mask_proxy
    }


               
if __name__ == "__main__":
    print("\nThis is a module for proxy features ablation study.")
    print("Import and use run_proxy_ablation_experiment() from your main pipeline.")
    print("\nExample:")
    print("""
    from proxy_features_ablation import run_proxy_ablation_experiment

    results = run_proxy_ablation_experiment(
        df_with_features=df,
        X_baseline=X,
        y=y,
        model_class=ImprovedEnsemblePredictor,
        model_params={'n_models': 15, 'random_state': 42},
        mask_candidates_baseline=candidate_mask,
        save_dir='./outputs'
    )
    """)
