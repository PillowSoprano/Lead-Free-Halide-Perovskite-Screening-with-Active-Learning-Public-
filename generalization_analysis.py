import warnings
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold, GroupShuffleSplit, LeaveOneGroupOut,
    KFold
)
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pymatgen.core import Composition, Element

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

plt.rcParams.update({'font.size': 16, 'axes.labelsize': 18, 'axes.titlesize': 18,
                     'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14})

warnings.filterwarnings("ignore")


class ChemicalGroupExtractor:

                                 
    A_SITE = {"Cs", "Rb", "K", "Na", "Ba", "Sr", "Ca", "Li"}
    B_SITE = {
        "Bi", "Sb", "Cu", "Ag", "In", "Ga", "Sn", "Ge",
        "Fe", "Mn", "Co", "Ni", "Ti", "V", "Cr", "Zr", "Nb",
        "Pd", "Pt", "Au", "Zn", "Cd", "Hg", "Tl", "Y", "La"
    }
    X_SITE_HALIDE = {"Cl", "Br", "I"}

                       
    ALKALI_METALS = {"Li", "Na", "K", "Rb", "Cs"}
    ALKALINE_EARTH = {"Be", "Mg", "Ca", "Sr", "Ba"}
    TRANSITION_METALS = {
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd"
    }
    POST_TRANSITION = {"Al", "Ga", "In", "Tl", "Sn", "Pb", "Bi"}

    @classmethod
    def get_b_site_element(cls, formula: str) -> str:
        comp = Composition(formula)
        comp_dict = comp.as_dict()

                                  
        b_elements = [elem for elem in comp_dict.keys() if elem in cls.B_SITE]

        if not b_elements:
            return "Unknown"

        if len(b_elements) == 1:
            return b_elements[0]
        else:
                                                           
            return "-".join(sorted(b_elements))

    @classmethod
    def get_chemical_family(cls, formula: str) -> str:
        comp = Composition(formula)
        elements = {str(e) for e in comp.elements}

                      
        b_elements = elements.intersection(cls.B_SITE)

        if not b_elements:
            return "Unknown"

                              
        if b_elements.intersection(cls.POST_TRANSITION):
            if "Sn" in b_elements or "Ge" in b_elements:
                return "Group-14 (Sn/Ge)"
            elif "Bi" in b_elements or "Sb" in b_elements:
                return "Group-15 (Bi/Sb)"
            else:
                return "Post-transition"
        elif b_elements.intersection(cls.TRANSITION_METALS):
            return "Transition-metal"
        elif len(b_elements) > 1:
            return "Double-perovskite"
        else:
            return "Other"

    @classmethod
    def get_halide_type(cls, formula: str) -> str:
        comp = Composition(formula)
        comp_dict = comp.as_dict()

        halides = {elem: count for elem, count in comp_dict.items()
                  if elem in cls.X_SITE_HALIDE}

        if not halides:
            return "Unknown"

                                     
        primary_halide = max(halides, key=halides.get)
        return primary_halide

    @classmethod
    def cluster_by_composition(cls, formulas: List[str],
                               feature_df: pd.DataFrame,
                               n_clusters: int = 5) -> np.ndarray:
                                                
        from sklearn.preprocessing import StandardScaler

        X = feature_df.values
        X_scaled = StandardScaler().fit_transform(X)

                           
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        cluster_labels = kmeans.fit_predict(X_scaled)

        return cluster_labels


                                            
                          
                                            

class GeneralizationEvaluator:

    def __init__(self, model_class, model_params: Dict = None):
        self.model_class = model_class
        self.model_params = model_params or {}
        self.results = {}

    def evaluate_random_split(self, X, y, formulas: List[str],
                              n_splits: int = 5,
                              test_size: float = 0.2) -> Dict:
        print(f"\n{'='*80}")
        print("1. Random Split (Baseline)")
        print(f"{'='*80}")

        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        fold_scores = []
        for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

                         
            model = self.model_class(**self.model_params)
            model.fit(X_train, y_train)

                     
            y_pred = model.predict(X_test)

                     
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            fold_scores.append({'r2': r2, 'rmse': rmse, 'mae': mae})

                   
        results = {
            'split_type': 'Random',
            'n_splits': n_splits,
            'mean_r2': np.mean([s['r2'] for s in fold_scores]),
            'std_r2': np.std([s['r2'] for s in fold_scores]),
            'mean_rmse': np.mean([s['rmse'] for s in fold_scores]),
            'std_rmse': np.std([s['rmse'] for s in fold_scores]),
            'mean_mae': np.mean([s['mae'] for s in fold_scores]),
            'std_mae': np.std([s['mae'] for s in fold_scores]),
            'fold_scores': fold_scores
        }

        print(f"Mean R² = {results['mean_r2']:.4f} ± {results['std_r2']:.4f}")
        print(f"Mean RMSE = {results['mean_rmse']:.4f} ± {results['std_rmse']:.4f} eV")
        print(f"Mean MAE = {results['mean_mae']:.4f} ± {results['std_mae']:.4f} eV")

        self.results['random'] = results
        return results

    def evaluate_group_by_element(self, X, y, formulas: List[str],
                                   n_splits: int = 5) -> Dict:
        print(f"\n{'='*80}")
        print("2. Group Split by B-site Element")
        print(f"{'='*80}")

                                              
        groups = np.array([
            ChemicalGroupExtractor.get_b_site_element(f) for f in formulas
        ])

                                 
        unique_groups, counts = np.unique(groups, return_counts=True)
        print(f"\nFound {len(unique_groups)} unique B-site groups:")
        for group, count in sorted(zip(unique_groups, counts),
                                   key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {group}: {count} materials")

                      
        gkf = GroupKFold(n_splits=min(n_splits, len(unique_groups)))

        fold_scores = []
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            test_groups = np.unique(groups[test_idx])
            print(f"\nFold {fold+1}: Test groups = {', '.join(test_groups[:5])}"
                  f"{' ...' if len(test_groups) > 5 else ''}")

                         
            model = self.model_class(**self.model_params)
            model.fit(X_train, y_train)

                     
            y_pred = model.predict(X_test)

                     
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            fold_scores.append({
                'r2': r2, 'rmse': rmse, 'mae': mae,
                'test_groups': test_groups
            })

                   
        results = {
            'split_type': 'Group-by-B-element',
            'n_splits': len(fold_scores),
            'n_groups': len(unique_groups),
            'mean_r2': np.mean([s['r2'] for s in fold_scores]),
            'std_r2': np.std([s['r2'] for s in fold_scores]),
            'mean_rmse': np.mean([s['rmse'] for s in fold_scores]),
            'std_rmse': np.std([s['rmse'] for s in fold_scores]),
            'mean_mae': np.mean([s['mae'] for s in fold_scores]),
            'std_mae': np.std([s['mae'] for s in fold_scores]),
            'fold_scores': fold_scores,
            'groups': groups,
            'unique_groups': unique_groups
        }

        print(f"\nOverall Performance:")
        print(f"Mean R² = {results['mean_r2']:.4f} ± {results['std_r2']:.4f}")
        print(f"Mean RMSE = {results['mean_rmse']:.4f} ± {results['std_rmse']:.4f} eV")
        print(f"Mean MAE = {results['mean_mae']:.4f} ± {results['std_mae']:.4f} eV")

        self.results['group_element'] = results
        return results

    def evaluate_leave_one_element_out(self, X, y, formulas: List[str],
                                       min_samples: int = 10) -> Dict:
        print(f"\n{'='*80}")
        print("3. Leave-One-Element-Out (LOEO)")
        print(f"{'='*80}")

                                              
        groups = np.array([
            ChemicalGroupExtractor.get_b_site_element(f) for f in formulas
        ])

                                           
        unique_groups, counts = np.unique(groups, return_counts=True)
        valid_groups = unique_groups[counts >= min_samples]

        print(f"\nTesting {len(valid_groups)} elements with ≥{min_samples} samples")

        element_scores = []
        for group in valid_groups:
                                     
            test_mask = groups == group
            train_mask = ~test_mask

            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]

            print(f"\n  Testing on {group}: {test_mask.sum()} samples")

                         
            model = self.model_class(**self.model_params)
            model.fit(X_train, y_train)

                     
            y_pred = model.predict(X_test)

                     
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            print(f"    R² = {r2:.4f}, RMSE = {rmse:.4f} eV, MAE = {mae:.4f} eV")

            element_scores.append({
                'element': group,
                'n_test': test_mask.sum(),
                'r2': r2,
                'rmse': rmse,
                'mae': mae
            })

                   
        results = {
            'split_type': 'Leave-One-Element-Out',
            'n_elements_tested': len(element_scores),
            'mean_r2': np.mean([s['r2'] for s in element_scores]),
            'std_r2': np.std([s['r2'] for s in element_scores]),
            'mean_rmse': np.mean([s['rmse'] for s in element_scores]),
            'std_rmse': np.std([s['rmse'] for s in element_scores]),
            'mean_mae': np.mean([s['mae'] for s in element_scores]),
            'std_mae': np.std([s['mae'] for s in element_scores]),
            'element_scores': element_scores,
            'groups': groups
        }

        print(f"\nOverall Performance:")
        print(f"Mean R² = {results['mean_r2']:.4f} ± {results['std_r2']:.4f}")
        print(f"Mean RMSE = {results['mean_rmse']:.4f} ± {results['std_rmse']:.4f} eV")
        print(f"Mean MAE = {results['mean_mae']:.4f} ± {results['std_mae']:.4f} eV")

                         
        sorted_elements = sorted(element_scores, key=lambda x: x['r2'])
        print(f"\nWorst extrapolation: {sorted_elements[0]['element']} "
              f"(R² = {sorted_elements[0]['r2']:.4f})")
        print(f"Best extrapolation: {sorted_elements[-1]['element']} "
              f"(R² = {sorted_elements[-1]['r2']:.4f})")

        self.results['loeo'] = results
        return results

    def evaluate_cluster_split(self, X, y, formulas: List[str],
                               feature_df: pd.DataFrame,
                               n_clusters: int = 5) -> Dict:
        print(f"\n{'='*80}")
        print(f"4. Cluster-Based Split ({n_clusters} clusters)")
        print(f"{'='*80}")

                           
        cluster_labels = ChemicalGroupExtractor.cluster_by_composition(
            formulas, feature_df, n_clusters=n_clusters
        )

                                   
        unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
        print(f"\nCluster sizes:")
        for cluster, count in zip(unique_clusters, counts):
            print(f"  Cluster {cluster}: {count} materials")

                               
        fold_scores = []
        for cluster_id in unique_clusters:
            test_mask = cluster_labels == cluster_id
            train_mask = ~test_mask

            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]

            print(f"\n  Testing cluster {cluster_id}: {test_mask.sum()} samples")

                         
            model = self.model_class(**self.model_params)
            model.fit(X_train, y_train)

                     
            y_pred = model.predict(X_test)

                     
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            print(f"    R² = {r2:.4f}, RMSE = {rmse:.4f} eV")

            fold_scores.append({
                'cluster': cluster_id,
                'n_test': test_mask.sum(),
                'r2': r2,
                'rmse': rmse,
                'mae': mae
            })

                   
        results = {
            'split_type': f'Cluster ({n_clusters})',
            'n_clusters': n_clusters,
            'mean_r2': np.mean([s['r2'] for s in fold_scores]),
            'std_r2': np.std([s['r2'] for s in fold_scores]),
            'mean_rmse': np.mean([s['rmse'] for s in fold_scores]),
            'std_rmse': np.std([s['rmse'] for s in fold_scores]),
            'mean_mae': np.mean([s['mae'] for s in fold_scores]),
            'std_mae': np.std([s['mae'] for s in fold_scores]),
            'fold_scores': fold_scores,
            'cluster_labels': cluster_labels
        }

        print(f"\nOverall Performance:")
        print(f"Mean R² = {results['mean_r2']:.4f} ± {results['std_r2']:.4f}")
        print(f"Mean RMSE = {results['mean_rmse']:.4f} ± {results['std_rmse']:.4f} eV")

        self.results['cluster'] = results
        return results

    def run_all_evaluations(self, X, y, formulas: List[str],
                           feature_df: pd.DataFrame) -> Dict:
        print(f"\n{'='*80}")
        print("GENERALIZATION ANALYSIS: COMPREHENSIVE EVALUATION")
        print(f"{'='*80}")
        print(f"Total samples: {len(X)}")
        print(f"Unique formulas: {len(set(formulas))}")

                             
        self.evaluate_random_split(X, y, formulas, n_splits=5)
        self.evaluate_group_by_element(X, y, formulas, n_splits=5)
        self.evaluate_leave_one_element_out(X, y, formulas, min_samples=10)
        self.evaluate_cluster_split(X, y, formulas, feature_df, n_clusters=5)

        return self.results

    def plot_comparison(self, save_path: str):
        if not self.results:
            print("No results to plot. Run evaluations first.")
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

                      
        split_names = []
        r2_means = []
        r2_stds = []
        rmse_means = []
        rmse_stds = []
        mae_means = []
        mae_stds = []

        for key in ['random', 'group_element', 'loeo', 'cluster']:
            if key in self.results:
                res = self.results[key]
                split_names.append(res['split_type'])
                r2_means.append(res['mean_r2'])
                r2_stds.append(res['std_r2'])
                rmse_means.append(res['mean_rmse'])
                rmse_stds.append(res['std_rmse'])
                mae_means.append(res['mean_mae'])
                mae_stds.append(res['std_mae'])

        x_pos = np.arange(len(split_names))
        colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2']

        # R² subplot
        ax = axes[0]
        r2_plot = [max(v, -1.0) for v in r2_means]  # clip bars at -1 for display
        r2_err_plot = [min(s, abs(v + 1.0)) if v < -1.0 else s
                       for v, s in zip(r2_means, r2_stds)]
        bars = ax.bar(x_pos, r2_plot, yerr=r2_err_plot, capsize=5,
                     color=colors[:len(split_names)],
                     edgecolor='black', linewidth=1.5, alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(split_names, rotation=15, ha='right', fontsize=14)
        ax.set_ylabel('R² Score', fontsize=18, fontweight='bold')
        ax.set_title('Model Performance: R²\n(Higher is Better)',
                    fontsize=18, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(-1.0, 1.0)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='-')

        for i, (bar, val, std) in enumerate(zip(bars, r2_means, r2_stds)):
            display_y = max(val, -1.0)
            if val < -1.0:
                # annotate true value below the clipped bar
                ax.text(bar.get_x() + bar.get_width()/2., -0.95,
                       f'{val:.2f}*', ha='center', va='bottom',
                       fontsize=12, color='red', fontweight='bold')
            else:
                ax.text(bar.get_x() + bar.get_width()/2.,
                       display_y + min(std, 1.0 - display_y - 0.05) + 0.02,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=14)
        ax.text(0.98, 0.02, '* clipped; true value shown',
               transform=ax.transAxes, fontsize=12, ha='right',
               va='bottom', color='red', style='italic')

        # RMSE subplot
        ax = axes[1]
        bars = ax.bar(x_pos, rmse_means, yerr=rmse_stds, capsize=5,
                     color=colors[:len(split_names)],
                     edgecolor='black', linewidth=1.5, alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(split_names, rotation=15, ha='right', fontsize=14)
        ax.set_ylabel('RMSE (eV)', fontsize=18, fontweight='bold')
        ax.set_title('Model Performance: RMSE\n(Lower is Better)',
                    fontsize=18, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        rmse_max = max(v + s for v, s in zip(rmse_means, rmse_stds))
        ax.set_ylim(0, min(rmse_max * 1.25, 1.5))

        for i, (bar, val, std) in enumerate(zip(bars, rmse_means, rmse_stds)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=14)

        # MAE subplot
        ax = axes[2]
        bars = ax.bar(x_pos, mae_means, yerr=mae_stds, capsize=5,
                     color=colors[:len(split_names)],
                     edgecolor='black', linewidth=1.5, alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(split_names, rotation=15, ha='right', fontsize=14)
        ax.set_ylabel('MAE (eV)', fontsize=18, fontweight='bold')
        ax.set_title('Model Performance: MAE\n(Lower is Better)',
                    fontsize=18, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        mae_max = max(v + s for v, s in zip(mae_means, mae_stds))
        ax.set_ylim(0, min(mae_max * 1.25, 1.2))

        for i, (bar, val, std) in enumerate(zip(bars, mae_means, mae_stds)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=14)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Comparison plot saved: {save_path}")
        plt.close()

    def export_summary(self, save_path: str):
        if not self.results:
            print("No results to export. Run evaluations first.")
            return

        summary_data = []
        for key, res in self.results.items():
            summary_data.append({
                'Split_Type': res['split_type'],
                'Mean_R2': res['mean_r2'],
                'Std_R2': res['std_r2'],
                'Mean_RMSE': res['mean_rmse'],
                'Std_RMSE': res['std_rmse'],
                'Mean_MAE': res['mean_mae'],
                'Std_MAE': res['std_mae']
            })

        df = pd.DataFrame(summary_data)
        df.to_csv(save_path, index=False)
        print(f"✓ Summary exported: {save_path}")


                                            
                    
                                            

def analyze_generalization(model_class, model_params: Dict,
                           X, y, formulas: List[str], feature_df: pd.DataFrame,
                           save_dir: str = './outputs'):
    import os
    os.makedirs(save_dir, exist_ok=True)

    evaluator = GeneralizationEvaluator(model_class, model_params)
    results = evaluator.run_all_evaluations(X, y, formulas, feature_df)

                    
    plot_path = os.path.join(save_dir, 'generalization_comparison.png')
    evaluator.plot_comparison(plot_path)

                    
    summary_path = os.path.join(save_dir, 'generalization_summary.csv')
    evaluator.export_summary(summary_path)

    return results


if __name__ == "__main__":
    print("Generalization Analysis module loaded.")
    print("Use analyze_generalization() or GeneralizationEvaluator class.")
