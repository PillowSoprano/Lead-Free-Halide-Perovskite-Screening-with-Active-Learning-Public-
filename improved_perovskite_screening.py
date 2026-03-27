import warnings
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import yaml
from dotenv import load_dotenv

load_dotenv()

import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.utils import resample

                   
from mp_api.client import MPRester
from pymatgen.core import Composition, Element

                  
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

               
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({'font.size': 16, 'axes.labelsize': 18, 'axes.titlesize': 18,
                     'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14})

warnings.filterwarnings("ignore")

                                            
                          
                                            

@dataclass
class Config:
         
    api_key: str = os.getenv('MATERIALS_PROJECT_API_KEY', '')
    
                    
    min_elements: int = 3
    max_elements: int = 5
    bandgap_min: float = 0.3
    bandgap_max: float = 4.0
    
                      
    n_ensemble_models: int = 15                     
    test_size: float = 0.2
    random_state: int = 42
    
                        
    target_bandgap: float = 1.34                             
    bandgap_tolerance: float = 0.25                   
    max_e_above_hull: float = 0.05
    max_uncertainty: float = 0.5                                  
    min_uncertainty: float = 0.05                               
    
                     
    active_learning_batch_size: int = 15
    
            
    save_dir: str = './outputs'


                                            
                              
                                            

class EnhancedFeatureExtractor:
    
    @staticmethod
    def get_compositional_features(comp: Composition) -> Dict[str, float]:
        features = {}
        
                                      
        atomic_numbers = []
        atomic_masses = []
        electronegativities = []
        atomic_radii = []
        ionization_energies = []
        electron_affinities = []
        
                               
        is_alkali = []
        is_alkaline_earth = []
        is_transition = []
        is_halogen = []
        is_chalcogen = []
        
        fractions = []
        
        for element in comp.elements:
            fraction = comp.get_atomic_fraction(element)
            fractions.append(fraction)
            elem_obj = Element(element)
            
                              
            atomic_numbers.append(elem_obj.Z)
            atomic_masses.append(elem_obj.atomic_mass)
            
                               
            en = elem_obj.X if elem_obj.X else 0
            electronegativities.append(en)
            
                    
            radius = elem_obj.atomic_radius if elem_obj.atomic_radius else 1.5
            atomic_radii.append(float(radius))
            
                               
            ie = elem_obj.ionization_energy if elem_obj.ionization_energy else 5.0
            ionization_energies.append(float(ie))
            
                                             
            ea = elem_obj.electron_affinity if hasattr(elem_obj, 'electron_affinity') else 0
            electron_affinities.append(float(ea) if ea else 0)
            
                                   
            Z = elem_obj.Z
            group = elem_obj.group
            
            is_alkali.append(1 if group == 1 else 0)
            is_alkaline_earth.append(1 if group == 2 else 0)
            is_transition.append(1 if (21 <= Z <= 30) or (39 <= Z <= 48) or (72 <= Z <= 80) else 0)
            is_halogen.append(1 if group == 17 else 0)
            is_chalcogen.append(1 if group == 16 else 0)
        
                                                        
        features['Z_mean'] = np.average(atomic_numbers, weights=fractions)
        features['Z_std'] = np.sqrt(np.average((np.array(atomic_numbers) - features['Z_mean'])**2, weights=fractions))
        features['Z_range'] = max(atomic_numbers) - min(atomic_numbers)
        
        features['EN_mean'] = np.average(electronegativities, weights=fractions)
        features['EN_std'] = np.sqrt(np.average((np.array(electronegativities) - features['EN_mean'])**2, weights=fractions))
        features['EN_range'] = max(electronegativities) - min(electronegativities)
        features['EN_max'] = max(electronegativities)
        features['EN_min'] = min(electronegativities)
        
        features['Radius_mean'] = np.average(atomic_radii, weights=fractions)
        features['Radius_std'] = np.sqrt(np.average((np.array(atomic_radii) - features['Radius_mean'])**2, weights=fractions))
        features['Radius_range'] = max(atomic_radii) - min(atomic_radii)
        
        features['Mass_mean'] = np.average(atomic_masses, weights=fractions)
        features['Mass_std'] = np.sqrt(np.average((np.array(atomic_masses) - features['Mass_mean'])**2, weights=fractions))
        
        features['IE_mean'] = np.average(ionization_energies, weights=fractions)
        features['IE_std'] = np.sqrt(np.average((np.array(ionization_energies) - features['IE_mean'])**2, weights=fractions))
        
                                
        features['N_elements'] = len(comp.elements)
        features['n_alkali'] = sum(is_alkali)
        features['n_alkaline_earth'] = sum(is_alkaline_earth)
        features['n_transition'] = sum(is_transition)
        features['n_halogen'] = sum(is_halogen)
        features['n_chalcogen'] = sum(is_chalcogen)
        
                
        n_metals = sum([1 for e in comp.elements if Element(e).is_metal])
        features['metal_ratio'] = n_metals / len(comp.elements)
        
                                                                  
        if len(electronegativities) >= 2:
            features['EN_diff_max'] = max(electronegativities) - min(electronegativities)
        else:
            features['EN_diff_max'] = 0
        
        return features
    
    @staticmethod
    def get_perovskite_descriptors(formula: str, comp: Composition) -> Dict[str, float]:
        descriptors = {}
        
        comp_dict = comp.as_dict()
        
                           
        A_site = {"Cs", "Rb", "K", "Na", "Ba", "Sr", "Ca", "Li"}
        B_site = {
            "Bi", "Sb", "Cu", "Ag", "In", "Ga", "Sn", "Ge",
            "Fe", "Mn", "Co", "Ni", "Ti", "V", "Cr", "Zr", "Nb",
            "Pd", "Pt", "Au", "Zn", "Cd", "Hg", "Tl", "Y", "La"
        }
        X_site = {"Cl", "Br", "I", "F", "O", "S", "Se"}
        
                                  
        total_A = sum([count for elem, count in comp_dict.items() if elem in A_site])
        total_B = sum([count for elem, count in comp_dict.items() if elem in B_site])
        total_X = sum([count for elem, count in comp_dict.items() if elem in X_site])
        
        descriptors['A_site_count'] = total_A
        descriptors['B_site_count'] = total_B
        descriptors['X_site_count'] = total_X
        
                               
        if total_B > 0:
            descriptors['A_to_B_ratio'] = total_A / total_B
            descriptors['X_to_B_ratio'] = total_X / total_B
        else:
            descriptors['A_to_B_ratio'] = 0
            descriptors['X_to_B_ratio'] = 0
        
                                                   
                                                 
                                                                  
        A_elements = [elem for elem in comp.elements if str(elem) in A_site]
        B_elements = [elem for elem in comp.elements if str(elem) in B_site]
        X_elements = [elem for elem in comp.elements if str(elem) in X_site]
        
        if A_elements and B_elements and X_elements:
            r_A = np.mean([Element(e).atomic_radius for e in A_elements if Element(e).atomic_radius])
            r_B = np.mean([Element(e).atomic_radius for e in B_elements if Element(e).atomic_radius])
            r_X = np.mean([Element(e).atomic_radius for e in X_elements if Element(e).atomic_radius])
            
            if r_B and r_X:
                descriptors['tolerance_factor_approx'] = (r_A + r_X) / (np.sqrt(2) * (r_B + r_X))
            else:
                descriptors['tolerance_factor_approx'] = 1.0
        else:
            descriptors['tolerance_factor_approx'] = 1.0
        
        return descriptors
    
    @classmethod
    def extract_all_features(cls, comp: Composition, formula: str) -> pd.Series:
        comp_features = cls.get_compositional_features(comp)
        perov_features = cls.get_perovskite_descriptors(formula, comp)
        
        all_features = {**comp_features, **perov_features}
        return pd.Series(all_features)


                                            
                               
                                            

class PerovskiteValidator:
    
                             
    A_SITE = {"Cs", "Rb", "K", "Na", "Ba", "Sr", "Ca", "Li"}
    B_SITE = {
        "Bi", "Sb", "Cu", "Ag", "In", "Ga", "Sn", "Ge",
        "Fe", "Mn", "Co", "Ni", "Ti", "V", "Cr", "Zr", "Nb",
        "Pd", "Pt", "Au", "Zn", "Cd", "Hg", "Tl", "Y", "La"
    }
    X_SITE_HALIDE = {"Cl", "Br", "I"}                     
    X_SITE_OXIDE = {"O"}
    X_SITE_CHALCOGEN = {"S", "Se"}
    
                                                                 
    VALID_PATTERNS = [
        (0.8, 1.3, 2.5, 3.8),                    
        (1.5, 2.5, 3.5, 4.8),                           
        (1.2, 1.9, 4.0, 5.2),                            
        (1.7, 2.4, 5.2, 6.8),                           
    ]
    
    PEROVSKITE_SPACE_GROUPS = [
        221, 225, 166, 194, 62, 71, 167, 136, 99,
        140, 139, 164, 15, 63, 14, 148, 12, 74
    ]
    
    @classmethod
    def is_valid_halide_perovskite(cls, formula: str, comp_dict: Dict, 
                                   space_group: Optional[int] = None) -> bool:
        elements_in_formula = set(comp_dict.keys())
        
                                             
        n_A = len(elements_in_formula.intersection(cls.A_SITE))
        n_B = len(elements_in_formula.intersection(cls.B_SITE))
        n_X_halide = len(elements_in_formula.intersection(cls.X_SITE_HALIDE))
        
        if n_A < 1 or n_B < 1 or n_X_halide < 1:
            return False
        
                                    
        if len(elements_in_formula) < 3 or len(elements_in_formula) > 5:
            return False
        
                                                                     
                                                                       
        oxide_chalcogen = elements_in_formula.intersection(
            cls.X_SITE_OXIDE.union(cls.X_SITE_CHALCOGEN)
        )
        
        if oxide_chalcogen:
                                                       
            total_halide = sum([count for elem, count in comp_dict.items() 
                               if elem in cls.X_SITE_HALIDE])
            total_oxide = sum([count for elem, count in comp_dict.items() 
                              if elem in oxide_chalcogen])
            
            if total_oxide / (total_halide + total_oxide + 1e-8) > 0.1:
                return False                            
        
                                     
        total_A = sum([count for elem, count in comp_dict.items() if elem in cls.A_SITE])
        total_B = sum([count for elem, count in comp_dict.items() if elem in cls.B_SITE])
        total_X = sum([count for elem, count in comp_dict.items() if elem in cls.X_SITE_HALIDE])
        
        if total_A == 0 or total_B == 0 or total_X == 0:
            return False
        
        ratio_A_to_B = total_A / total_B
        ratio_X_to_B = total_X / total_B
        
                                      
        for a_min, a_max, x_min, x_max in cls.VALID_PATTERNS:
            if (a_min <= ratio_A_to_B <= a_max) and (x_min <= ratio_X_to_B <= x_max):
                return True
        
                                                  
        if space_group and space_group in cls.PEROVSKITE_SPACE_GROUPS:
            return True
        
        return False
    
    @staticmethod
    def is_physically_reasonable(formula: str, predicted_gap: float, 
                                 uncertainty: float) -> Tuple[bool, str]:
        comp = Composition(formula)
        elements = {str(e) for e in comp.elements}
        
                                                                 
        if 'F' in elements and ('Li' in elements or 'Cu' in elements):
            return False, "LiF/CuF are wide-gap insulators (>3 eV)"
        
                                                      
        if 'ClO' in formula or 'SO' in formula or 'PO' in formula:
            return False, "Contains polyatomic anions (not simple halides)"
        
                                                
        if uncertainty > 0.5:
            return False, f"Uncertainty too high ({uncertainty:.3f} eV)"
        
                                                       
        if predicted_gap < 0.1 or predicted_gap > 4.0:
            return False, f"Predicted gap outside physical range ({predicted_gap:.2f} eV)"
        
        return True, "Physically reasonable"


                                            
                            
                                            

class StratifiedPBECalibrator:
    
    def __init__(self):
        # Calibration dataset: 20 inorganic lead-free halide perovskites
        # PBE gaps from Materials Project (lowest-Ehull entry per formula).
        # Experimental gaps from peer-reviewed optical measurements (UV-vis /
        # diffuse reflectance / PL onset).  All values verified against primary
        # literature; previous erroneous entries corrected (see commit notes).
        # MA/FA organic-inorganic hybrids excluded: MP formula search returns
        # incorrect mp-ids for these compounds.
        # References stored in calibration_refs dict below.
        self.calibration_data = {
            'formula': [
                # Sn-based ABX3 (n=2)
                'CsSnI3', 'CsSnBr3',
                # Ge-based ABX3 (n=3)
                'CsGeI3', 'CsGeBr3', 'CsGeCl3',
                # Bi-based A3B2X9 (n=2)
                'Cs3Bi2Br9', 'Cs3Bi2Cl9',
                # Sb-based A3B2X9 (n=3)
                'Cs3Sb2I9', 'Cs3Sb2Br9', 'Cs3Sb2Cl9',
                # Double perovskites A2B'B''X6 (n=5)
                'Cs2AgBiBr6', 'Cs2AgBiCl6', 'Cs2AgInCl6',
                'Cs2AgSbCl6', 'Cs2AgSbBr6',
                # Vacancy-ordered A2BX6 (n=5)
                'Cs2SnI6', 'Cs2SnBr6', 'Cs2SnCl6',
                'Cs2TiI6', 'Cs2PdBr6',
            ],
            'pbe_gap': [
                # Sn
                0.450, 0.968,
                # Ge
                0.991, 0.786, 2.154,
                # Bi
                2.600, 3.230,
                # Sb
                1.883, 1.984, 2.395,
                # Double
                1.355, 1.866, 1.342, 1.662, 1.192,
                # Vacancy
                0.294, 1.439, 2.590, 0.864, 0.756,
            ],
            'exp_gap': [
                # Sn  — Chung 2012 Nature; Stoumpos 2013 IC
                1.31, 1.75,
                # Ge  — Stoumpos 2015 IC (all three)
                1.63, 2.32, 3.67,
                # Bi  — Hoye 2016 CEJ (both)
                2.60, 3.30,
                # Sb  — Saparov 2015 CM; Vargas 2017 JPCL (×2)
                2.05, 2.67, 3.56,
                # Double — Slavney 2016 JACS (×2); Volonakis 2017 JPCL;
                #          Babu 2021 JPCL; Deng 2020 AFM
                1.95, 2.77, 3.23, 2.54, 1.64,
                # Vacancy — Saparov 2016 CM; Kaltzoglou 2016 JPCC (×2);
                #           Baranwal 2018 ChemSusChem; Babu 2020 JPCC
                1.26, 1.82, 3.90, 1.02, 1.60,
            ],
            'material_type': [
                'Sn', 'Sn',
                'Ge', 'Ge', 'Ge',
                'Bi', 'Bi',
                'Sb', 'Sb', 'Sb',
                'double', 'double', 'double', 'double', 'double',
                'vacancy', 'vacancy', 'vacancy', 'vacancy', 'vacancy',
            ]
        }
        # Full references for SI table
        self.calibration_refs = {
            'CsSnI3':    'Chung et al. Nature 485, 486 (2012)',
            'CsSnBr3':   'Stoumpos et al. Inorg. Chem. 52, 9019 (2013)',
            'CsGeI3':    'Stoumpos et al. Inorg. Chem. 54, 2757 (2015)',
            'CsGeBr3':   'Stoumpos et al. Inorg. Chem. 54, 2757 (2015)',
            'CsGeCl3':   'Stoumpos et al. Inorg. Chem. 54, 2757 (2015)',
            'Cs3Bi2Br9': 'Hoye et al. Chem. Eur. J. 22, 2605 (2016)',
            'Cs3Bi2Cl9': 'Hoye et al. Chem. Eur. J. 22, 2605 (2016)',
            'Cs3Sb2I9':  'Saparov et al. Chem. Mater. 27, 5622 (2015)',
            'Cs3Sb2Br9': 'Vargas et al. J. Phys. Chem. Lett. 8, 1412 (2017)',
            'Cs3Sb2Cl9': 'Vargas et al. J. Phys. Chem. Lett. 8, 1412 (2017)',
            'Cs2AgBiBr6':'Slavney et al. J. Am. Chem. Soc. 138, 2138 (2016)',
            'Cs2AgBiCl6':'Slavney et al. J. Am. Chem. Soc. 138, 2138 (2016)',
            'Cs2AgInCl6':'Volonakis et al. J. Phys. Chem. Lett. 8, 772 (2017)',
            'Cs2AgSbCl6':'Babu et al. J. Phys. Chem. Lett. 12, 4571 (2021)',
            'Cs2AgSbBr6':'Deng et al. Adv. Funct. Mater. 30, 2002131 (2020)',
            'Cs2SnI6':   'Saparov et al. Chem. Mater. 28, 2315 (2016)',
            'Cs2SnBr6':  'Kaltzoglou et al. J. Phys. Chem. C 120, 11777 (2016)',
            'Cs2SnCl6':  'Kaltzoglou et al. J. Phys. Chem. C 120, 11777 (2016)',
            'Cs2TiI6':   'Baranwal et al. ChemSusChem 11, 3794 (2018)',
            'Cs2PdBr6':  'Babu et al. J. Phys. Chem. C 124, 10580 (2020)',
        }
        
        self.global_model = None
        self.stratified_models = {}
        self.bootstrap_slopes = []
        self.bootstrap_intercepts = []
    
    def fit(self, n_bootstrap: int = 1000):
        pbe = np.array(self.calibration_data['pbe_gap'])
        exp = np.array(self.calibration_data['exp_gap'])
        types = self.calibration_data['material_type']
        
                      
        slope, intercept, r_value, p_value, std_err = linregress(pbe, exp)
        self.global_model = {'slope': slope, 'intercept': intercept, 
                            'r2': r_value**2, 'p_value': p_value}
        
        print(f"\n{'='*80}")
        print("PBE → Experimental Bandgap Calibration")
        print(f"{'='*80}")
        print(f"\nGlobal Model (n={len(pbe)}):")
        print(f"  Exp_gap = {slope:.3f} × PBE_gap + {intercept:.3f}")
        print(f"  R² = {r_value**2:.3f}, p-value = {p_value:.4e}")
        
                                        
        print(f"\nBootstrap Analysis ({n_bootstrap} iterations)...")
        for i in range(n_bootstrap):
            pbe_boot, exp_boot = resample(pbe, exp)
            slope_boot, intercept_boot, _, _, _ = linregress(pbe_boot, exp_boot)
            self.bootstrap_slopes.append(slope_boot)
            self.bootstrap_intercepts.append(intercept_boot)
        
        slope_ci = np.percentile(self.bootstrap_slopes, [2.5, 97.5])
        intercept_ci = np.percentile(self.bootstrap_intercepts, [2.5, 97.5])
        
        print(f"  95% CI for slope: [{slope_ci[0]:.3f}, {slope_ci[1]:.3f}]")
        print(f"  95% CI for intercept: [{intercept_ci[0]:.3f}, {intercept_ci[1]:.3f}]")
        
                           
        print(f"\nStratified Models:")
        df_cal = pd.DataFrame(self.calibration_data)
        
        for mat_type in df_cal['material_type'].unique():
            subset = df_cal[df_cal['material_type'] == mat_type]
            
            if len(subset) >= 3:                          
                pbe_sub = subset['pbe_gap'].values
                exp_sub = subset['exp_gap'].values
                
                slope_sub, intercept_sub, r_sub, _, _ = linregress(pbe_sub, exp_sub)
                self.stratified_models[mat_type] = {
                    'slope': slope_sub,
                    'intercept': intercept_sub,
                    'r2': r_sub**2,
                    'n': len(subset)
                }
                
                print(f"  {mat_type:10s} (n={len(subset):2d}): "
                      f"Exp = {slope_sub:.3f}×PBE + {intercept_sub:.3f}, "
                      f"R²={r_sub**2:.3f}")
    
    def calibrate(self, pbe_bandgaps: np.ndarray, 
                  material_types: Optional[List[str]] = None) -> np.ndarray:
        if self.global_model is None:
            self.fit()
        
        pbe_array = np.array(pbe_bandgaps)
        
        if material_types is None:
                              
            return (self.global_model['slope'] * pbe_array + 
                    self.global_model['intercept'])
        else:
                                                   
            calibrated = np.zeros_like(pbe_array)
            
            for i, (pbe, mat_type) in enumerate(zip(pbe_array, material_types)):
                if mat_type in self.stratified_models:
                    model = self.stratified_models[mat_type]
                    calibrated[i] = model['slope'] * pbe + model['intercept']
                else:
                                               
                    calibrated[i] = (self.global_model['slope'] * pbe + 
                                    self.global_model['intercept'])
            
            return calibrated
    
    def plot_calibration(self, save_path: str):
        if self.global_model is None:
            self.fit()
        
        pbe = np.array(self.calibration_data['pbe_gap'])
        exp = np.array(self.calibration_data['exp_gap'])
        types = self.calibration_data['material_type']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
                                     
        type_colors = {
            'Sn': '#E15759',
            'Ge': '#4E79A7',
            'double': '#59A14F',
            'vacancy': '#F28E2B'
        }
        
        for mat_type in set(types):
            mask = np.array(types) == mat_type
            ax1.scatter(pbe[mask], exp[mask], 
                       s=150, alpha=0.8, 
                       c=type_colors.get(mat_type, '#999999'),
                       edgecolors='black', linewidth=2,
                       label=mat_type)
        
                  
        x_line = np.linspace(pbe.min(), pbe.max(), 100)
        y_line = self.global_model['slope'] * x_line + self.global_model['intercept']
        ax1.plot(x_line, y_line, 'b--', linewidth=2.5, 
                label=f"Fit: y={self.global_model['slope']:.2f}x+{self.global_model['intercept']:.2f}")
        
                                     
        y_upper = np.percentile(
            [s * x_line + i for s, i in zip(self.bootstrap_slopes, self.bootstrap_intercepts)],
            97.5, axis=0
        )
        y_lower = np.percentile(
            [s * x_line + i for s, i in zip(self.bootstrap_slopes, self.bootstrap_intercepts)],
            2.5, axis=0
        )
        ax1.fill_between(x_line, y_lower, y_upper, alpha=0.2, color='blue', 
                        label='95% CI')
        
                    
        ax1.plot([0, 4], [0, 4], 'k:', linewidth=2, alpha=0.5, label='Ideal (1:1)')
        
        ax1.set_xlabel('PBE Band Gap (eV)', fontsize=18, fontweight='bold')
        ax1.set_ylabel('Experimental Band Gap (eV)', fontsize=18, fontweight='bold')
        ax1.set_title(f'(a) PBE → Experimental Calibration (n={len(pbe)})\n'
                     f'Slope = {self.global_model["slope"]:.3f}, '
                     f'Intercept = {self.global_model["intercept"]:.3f} eV, '
                     f'R² = {self.global_model["r2"]:.3f}',
                     fontsize=18, fontweight='bold')
        ax1.legend(fontsize=14)
        ax1.tick_params(axis='both', labelsize=14)
        ax1.grid(True, alpha=0.3)
        
                          
        residuals = exp - (self.global_model['slope'] * pbe + self.global_model['intercept'])
        ax2.scatter(pbe, residuals, s=120, alpha=0.6, c='#E15759', edgecolors='black')
        ax2.axhline(0, color='black', linestyle='--', linewidth=2)
        ax2.axhline(residuals.std(), color='red', linestyle=':', alpha=0.5, 
                   label=f'±1σ = ±{residuals.std():.3f} eV')
        ax2.axhline(-residuals.std(), color='red', linestyle=':', alpha=0.5)
        
        ax2.set_xlabel('PBE Band Gap (eV)', fontsize=18, fontweight='bold')
        ax2.set_ylabel('Residual (Exp − Predicted, eV)', fontsize=18, fontweight='bold')
        ax2.set_title('(b) Calibration Residuals', fontsize=18, fontweight='bold')
        ax2.legend(fontsize=14)
        ax2.tick_params(axis='both', labelsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Calibration plot saved: {save_path}")
        plt.close()


                                            
                         
                                            

class ImprovedEnsemblePredictor:
    
    def __init__(self, n_models: int = 15, random_state: int = 42):
        self.n_models = n_models
        self.random_state = random_state
        self.models = []
        self.scaler = StandardScaler()
        self.feature_names = None
        self.cv_scores = None
        
    def _create_models(self):
        models = []
        
        for i in range(self.n_models):
            if i % 3 == 0:
                                                       
                model = RandomForestRegressor(
                    n_estimators=500,
                    max_depth=25,
                    min_samples_split=3,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    random_state=self.random_state + i,
                    n_jobs=-1,
                    bootstrap=True
                )
            elif i % 3 == 1:
                                   
                model = GradientBoostingRegressor(
                    n_estimators=400,
                    max_depth=7,
                    learning_rate=0.05,
                    subsample=0.8,
                    min_samples_split=3,
                    min_samples_leaf=2,
                    random_state=self.random_state + i
                )
            else:
                                                            
                model = RandomForestRegressor(
                    n_estimators=400,
                    max_depth=20,
                    min_samples_split=4,
                    min_samples_leaf=3,
                    max_features='log2',
                    random_state=self.random_state + i,
                    n_jobs=-1,
                    bootstrap=True
                )
            
            models.append(model)
        
        return models
    
    def fit(self, X, y, feature_names=None):
        self.feature_names = feature_names
        
                        
        X_scaled = self.scaler.fit_transform(X)
        
                       
        self.models = self._create_models()
        
                                 
                                              
        n_splits = min(5, max(2, len(X)))

        print(f"\nTraining ensemble ({self.n_models} models)...")
        print(f"Performing {n_splits}-fold cross-validation...")

        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
                          
        cv_scores_list = []
        for i, model in enumerate(self.models):
                                              
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_boot = X_scaled[indices]
            y_boot = y[indices]
            
                       
            model.fit(X_boot, y_boot)
            
                                                       
            cv_score = cross_val_score(model, X_scaled, y, cv=kfold, 
                                      scoring='r2', n_jobs=-1).mean()
            cv_scores_list.append(cv_score)
            
            print(f"  Model {i+1:2d}/{self.n_models}: "
                  f"CV R² = {cv_score:.4f}", end='\r')
        
        print()                           
        
        self.cv_scores = np.array(cv_scores_list)
        print(f"\nCross-validation R² scores:")
        print(f"  Mean: {self.cv_scores.mean():.4f}")
        print(f"  Std:  {self.cv_scores.std():.4f}")
        print(f"  Min:  {self.cv_scores.min():.4f}")
        print(f"  Max:  {self.cv_scores.max():.4f}")
        
    def predict_with_uncertainty(self, X):
        X_scaled = self.scaler.transform(X)
        
        all_preds = np.array([model.predict(X_scaled) for model in self.models])
        
        mean_preds = all_preds.mean(axis=0)
        std_preds = all_preds.std(axis=0)
        
        return mean_preds, std_preds, all_preds
    
    def predict(self, X):
        mean_preds, _, _ = self.predict_with_uncertainty(X)
        return mean_preds
    
    def get_feature_importance(self):
        rf_models = [m for m in self.models if isinstance(m, RandomForestRegressor)]
        
        if not rf_models:
            return None
        
        importances = np.array([m.feature_importances_ for m in rf_models])
        mean_importance = importances.mean(axis=0)
        std_importance = importances.std(axis=0)
        
        if self.feature_names is not None:
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': mean_importance,
                'std': std_importance
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            return mean_importance


                                            
                               
                                            

def physics_aware_active_learning(
    df: pd.DataFrame,
    model: ImprovedEnsemblePredictor,
    feature_df: pd.DataFrame,
    config: Config
) -> pd.DataFrame:
    
    print(f"\n{'='*80}")
    print("Physics-Aware Active Learning Recommendations")
    print(f"{'='*80}")
    
                              
    X_all = feature_df.values
    mean_preds, std_preds, _ = model.predict_with_uncertainty(X_all)
    
                                
    mask = np.ones(len(df), dtype=bool)
    
    print(f"\nInitial candidates: {len(df)}")
    
                                       
    uncertainty_mask = (std_preds >= config.min_uncertainty) &\
                       (std_preds <= config.max_uncertainty)
    mask &= uncertainty_mask
    print(f"After uncertainty filter ({config.min_uncertainty:.2f}-{config.max_uncertainty:.2f} eV): "
          f"{mask.sum()}")
    
                             
    gap_mask = (mean_preds >= config.target_bandgap - config.bandgap_tolerance) &\
               (mean_preds <= config.target_bandgap + config.bandgap_tolerance)
    mask &= gap_mask
    print(f"After bandgap filter ({config.target_bandgap-config.bandgap_tolerance:.2f}-"
          f"{config.target_bandgap+config.bandgap_tolerance:.2f} eV): {mask.sum()}")
    
                         
    stability_mask = df['e_above_hull'] <= config.max_e_above_hull
    mask &= stability_mask
    print(f"After stability filter (E_hull < {config.max_e_above_hull:.3f} eV/atom): {mask.sum()}")
    
                                       
    physical_validity = []
    for idx in range(len(df)):
        if mask[idx]:
            formula = df.iloc[idx]['formula']
            pred_gap = mean_preds[idx]
            uncertainty = std_preds[idx]
            
            is_valid, reason = PerovskiteValidator.is_physically_reasonable(
                formula, pred_gap, uncertainty
            )
            physical_validity.append(is_valid)
        else:
            physical_validity.append(False)
    
    physical_mask = np.array(physical_validity)
    mask &= physical_mask
    print(f"After physical validation: {mask.sum()}")
    
    if mask.sum() == 0:
        print("\n⚠️  Warning: No materials passed all filters!")
        print("Consider relaxing constraints.")
        return pd.DataFrame()
    
                                 
    candidates = df[mask].copy()
    candidates['predicted_gap_exp'] = mean_preds[mask]
    candidates['uncertainty'] = std_preds[mask]
    
                             
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    
                                                                              
    uncert_score = scaler.fit_transform(
        std_preds[mask].reshape(-1, 1)
    ).flatten()
    
                                                             
    gap_score = scaler.fit_transform(
        -np.abs(mean_preds[mask] - config.target_bandgap).reshape(-1, 1)
    ).flatten()
    
                                                           
    stability_score = scaler.fit_transform(
        -candidates['e_above_hull'].values.reshape(-1, 1)
    ).flatten()
    
                               
    total_score = (0.4 * uncert_score +                                         
                   0.4 * gap_score +                                                
                   0.2 * stability_score)                                             
    
    candidates['priority_score'] = total_score
    
                      
    recommendations = candidates.sort_values('priority_score', ascending=False)
    
                       
    print(f"\n{'='*80}")
    print("Recommendation Statistics")
    print(f"{'='*80}")
    print(f"Total recommended materials: {len(recommendations)}")
    print(f"\nPredicted bandgap range: {recommendations['predicted_gap_exp'].min():.3f} - "
          f"{recommendations['predicted_gap_exp'].max():.3f} eV")
    print(f"Mean uncertainty: {recommendations['uncertainty'].mean():.3f} eV")
    print(f"Stability range (E_hull): {recommendations['e_above_hull'].min():.4f} - "
          f"{recommendations['e_above_hull'].max():.4f} eV/atom")
    
                         
    print(f"\n{'='*80}")
    print(f"Top {min(config.active_learning_batch_size, len(recommendations))} "
          f"Recommended Materials for DFT Validation")
    print(f"{'='*80}")
    
    display_cols = ['formula', 'material_id', 'predicted_gap_exp', 
                   'uncertainty', 'e_above_hull', 'priority_score']
    print(recommendations[display_cols].head(config.active_learning_batch_size).to_string(index=False))
    
    return recommendations


                                            
                        
                                            

class ImprovedVisualizer:
    
    @staticmethod
    def plot_model_performance(y_true, y_pred, y_std, save_path: str, 
                               material_ids=None):
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
                                            
        ax1.errorbar(y_true, y_pred, yerr=y_std, 
                     fmt='o', alpha=0.6, ecolor='lightgray', 
                     elinewidth=1.5, capsize=2, markersize=8,
                     color='#4E79A7', markeredgecolor='black', markeredgewidth=0.5)
        
                                 
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 
                'r--', linewidth=2.5, label='Ideal (1:1)', zorder=10)
        
                           
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
                         
        textstr = f'R² = {r2:.3f}\nRMSE = {rmse:.3f} eV\nMAE = {mae:.3f} eV'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes,
                fontsize=14, verticalalignment='top', bbox=props)

        ax1.set_xlabel('DFT Band Gap (eV)', fontsize=18, fontweight='bold')
        ax1.set_ylabel('Predicted Band Gap (eV)', fontsize=18, fontweight='bold')
        ax1.set_title('Model Performance: Predicted vs Actual\n(with epistemic uncertainty)',
                     fontsize=18, fontweight='bold')
        ax1.legend(fontsize=14)
        ax1.grid(True, alpha=0.3)
        
                              
        residuals = y_pred - y_true
        ax2.scatter(y_true, residuals, alpha=0.6, s=80,
                   c='#E15759', edgecolors='black', linewidth=0.5)
        ax2.axhline(0, color='black', linestyle='--', linewidth=2)
        ax2.axhline(residuals.std(), color='red', linestyle=':', alpha=0.5,
                   label=f'±1σ = ±{residuals.std():.3f} eV')
        ax2.axhline(-residuals.std(), color='red', linestyle=':', alpha=0.5)
        
        ax2.set_xlabel('DFT Band Gap (eV)', fontsize=18, fontweight='bold')
        ax2.set_ylabel('Residual (Predicted - Actual, eV)', fontsize=18, fontweight='bold')
        ax2.set_title('Residual Analysis', fontsize=18, fontweight='bold')
        ax2.legend(fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Model performance plot saved: {save_path}")
        plt.close()
    
    @staticmethod
    def plot_uncertainty_analysis(predicted_gaps, uncertainties,
                                  stabilities, save_path: str,
                                  target_gap: float = 1.34,
                                  bandgap_tolerance: float = 0.25,
                                  min_uncertainty: float = 0.05,
                                  max_uncertainty: float = 0.50):

        fig, ax = plt.subplots(figsize=(12, 8))

        scatter = ax.scatter(predicted_gaps, uncertainties,
                           c=stabilities, cmap='RdYlGn_r',
                           s=80, alpha=0.5, edgecolors='none')

        cbar = plt.colorbar(scatter, ax=ax, label='Energy Above Hull (eV/atom)')

        gap_min, gap_max = target_gap - bandgap_tolerance, target_gap + bandgap_tolerance

        ax.axvline(target_gap, color='red', linestyle='--',
                  linewidth=2, alpha=0.5, label=f'Target Gap ({target_gap} eV)')
        ax.axvspan(gap_min, gap_max, alpha=0.1, color='green',
                  label=f'Target Gap Range (±{bandgap_tolerance} eV)')
        ax.axhspan(min_uncertainty, max_uncertainty, alpha=0.1, color='blue',
                  label=f'AL Uncertainty Window ({min_uncertainty}–{max_uncertainty} eV)')

        ax.text(target_gap, (min_uncertainty + max_uncertainty) / 2,
               'High-Priority\nCandidates',
               fontsize=16, fontweight='bold',
               ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

        ax.set_xlabel('ML-Predicted Band Gap (eV)',
                     fontsize=18, fontweight='bold')
        ax.set_ylabel('Prediction Uncertainty — Ensemble Std (eV)',
                     fontsize=18, fontweight='bold')
        ax.set_title('Figure 3: Uncertainty vs ML-Predicted Gap for 1,117 Compounds',
                    fontsize=18, fontweight='bold')
        ax.tick_params(axis='both', labelsize=14)
        ax.legend(loc='upper right', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Uncertainty analysis plot saved: {save_path}")
        plt.close()
    
    @staticmethod
    def plot_feature_importance(importance_df, save_path: str, top_n: int = 15):
        
        top_features = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
        
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_features['importance'].values[::-1], 
               xerr=top_features['std'].values[::-1],
               color=colors[::-1], edgecolor='black', linewidth=1.5,
               capsize=5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['feature'].values[::-1], fontsize=14)
        ax.set_xlabel('Feature Importance', fontsize=18, fontweight='bold')
        ax.set_title(f'Top {top_n} Feature Importances\n(with ensemble variability)',
                    fontsize=18, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Feature importance plot saved: {save_path}")
        plt.close()


                                            
               
                                            

def main():

    config = Config()

                                                 
    os.makedirs(config.save_dir, exist_ok=True)

    print("="*80)
    print("IMPROVED LEAD-FREE HALIDE PEROVSKITE SCREENING")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Target bandgap: {config.target_bandgap} ± {config.bandgap_tolerance} eV")
    print(f"  Max stability: {config.max_e_above_hull} eV/atom")
    print(f"  Uncertainty range: {config.min_uncertainty} - {config.max_uncertainty} eV")
    print(f"  Ensemble size: {config.n_ensemble_models} models")
    
                                                
                              
                                                
    
    print(f"\n{'='*80}")
    print("Step 1: Data Acquisition from Materials Project")
    print(f"{'='*80}")
    
    data = []
    
    try:
        with MPRester(config.api_key) as mpr:
            docs = mpr.materials.summary.search(
                exclude_elements=["Pb"],
                num_elements=(config.min_elements, config.max_elements),
                band_gap=(config.bandgap_min, config.bandgap_max),
                fields=[
                    "material_id",
                    "formula_pretty",
                    "band_gap",
                    "energy_above_hull",
                    "composition",
                    "elements",
                    "symmetry"
                ]
            )
        
        print(f"API returned {len(docs)} materials")
        print("Applying physics-based filters...")
        
        for doc in docs:
            comp_dict = doc.composition.as_dict()
            space_group = doc.symmetry.number if hasattr(doc, 'symmetry') and doc.symmetry else None
            
                                           
            if PerovskiteValidator.is_valid_halide_perovskite(
                doc.formula_pretty, comp_dict, space_group
            ):
                data.append({
                    'material_id': doc.material_id,
                    'formula': doc.formula_pretty,
                    'band_gap': doc.band_gap,
                    'e_above_hull': doc.energy_above_hull,
                    'composition_obj': doc.composition,
                    'space_group': space_group
                })
        
        df = pd.DataFrame(data)
        print(f"✓ Filtered to {len(df)} valid halide perovskite materials")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    if df.empty or len(df) < 30:
        print(f"❌ Insufficient data ({len(df)} materials)")
        return
    
                                                
                                 
                                                
    
    print(f"\n{'='*80}")
    print("Step 2: Feature Engineering")
    print(f"{'='*80}")
    
    extractor = EnhancedFeatureExtractor()
    
    feature_list = []
    for idx, row in df.iterrows():
        features = extractor.extract_all_features(
            row['composition_obj'], 
            row['formula']
        )
        feature_list.append(features)
    
    feature_df = pd.DataFrame(feature_list).fillna(0)
    
    print(f"✓ Extracted {len(feature_df.columns)} features")
    print(f"  Features: {', '.join(list(feature_df.columns)[:5])}...")
    
                                                
                            
                                                
    
    print(f"\n{'='*80}")
    print("Step 3: Ensemble Model Training")
    print(f"{'='*80}")
    
    X = feature_df.values
    y = df['band_gap'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state
    )
    
    print(f"Training set: {len(X_train)} | Test set: {len(X_test)}")
    
                    
    model = ImprovedEnsemblePredictor(
        n_models=config.n_ensemble_models,
        random_state=config.random_state
    )
    model.fit(X_train, y_train, feature_names=feature_df.columns.tolist())
    
              
    y_pred, y_std, _ = model.predict_with_uncertainty(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\n{'='*80}")
    print("Model Performance on Test Set")
    print(f"{'='*80}")
    print(f"  R² = {r2:.4f}")
    print(f"  RMSE = {rmse:.4f} eV")
    print(f"  MAE = {mae:.4f} eV")
    print(f"  Mean Uncertainty = {y_std.mean():.3f} eV")
    print(f"  Median Uncertainty = {np.median(y_std):.3f} eV")

                                                
                                                
                                                

    print(f"\n{'='*80}")
    print("Step 3.5: Uncertainty Calibration and Validation")
    print(f"{'='*80}")

                                           
    from uncertainty_calibration_analysis import analyze_model_uncertainty

                                 
    uncertainty_results = analyze_model_uncertainty(
        model=model,
        X_test=X_test,
        y_test=y_test,
        save_dir=config.save_dir
    )

                                                
                                       
                                                

    print(f"\n{'='*80}")
    print("Step 3.6: Generalization Analysis (Rigorous Splitting)")
    print(f"{'='*80}")

                                  
    from generalization_analysis import analyze_generalization

                                                 
    generalization_results = analyze_generalization(
        model_class=ImprovedEnsemblePredictor,
        model_params={'n_models': 5, 'random_state': config.random_state},                              
        X=X,
        y=y,
        formulas=df['formula'].tolist(),
        feature_df=feature_df,
        save_dir=config.save_dir
    )

                                                
                             
                                                
    
    print(f"\n{'='*80}")
    print("Step 4: PBE → Experimental Bandgap Calibration")
    print(f"{'='*80}")
    
    calibrator = StratifiedPBECalibrator()
    calibrator.fit(n_bootstrap=1000)
    
                       
    df['predicted_gap_pbe'] = model.predict(X)
    df['predicted_gap_exp'] = calibrator.calibrate(df['predicted_gap_pbe'])
    _, df['prediction_uncertainty'], _ = model.predict_with_uncertainty(X)
    
                                                
                               
                                                
    
    print(f"\n{'='*80}")
    print("Step 5: Virtual Screening for Photovoltaic Candidates")
    print(f"{'='*80}")
    
                              
    candidates = df[
        (df['predicted_gap_exp'] >= config.target_bandgap - config.bandgap_tolerance) &
        (df['predicted_gap_exp'] <= config.target_bandgap + config.bandgap_tolerance) &
        (df['e_above_hull'] <= config.max_e_above_hull) &
        (df['prediction_uncertainty'] <= config.max_uncertainty)
    ].copy()
    
                                    
    valid_candidates = []
    for idx, row in candidates.iterrows():
        is_valid, reason = PerovskiteValidator.is_physically_reasonable(
            row['formula'], 
            row['predicted_gap_exp'],
            row['prediction_uncertainty']
        )
        if is_valid:
            valid_candidates.append(idx)
    
    candidates = candidates.loc[valid_candidates]
    
                                 
    candidates['gap_diff'] = abs(candidates['predicted_gap_exp'] - config.target_bandgap)
    top_candidates = candidates.sort_values('gap_diff').head(20)
    
    print(f"\n✓ Found {len(candidates)} valid candidates")
    print(f"\nTop 20 Candidates:")
    display_cols = ['formula', 'material_id', 'predicted_gap_exp', 
                   'band_gap', 'e_above_hull', 'prediction_uncertainty']
    print(top_candidates[display_cols].to_string(index=False))
    
          
    candidates_path = os.path.join(config.save_dir, 'improved_candidates.csv')
    candidates.to_csv(candidates_path, index=False)
    print(f"\n✓ Saved candidates: {candidates_path}")
    
                                                
                                             
                                                
    
    print(f"\n{'='*80}")
    print("Step 6: Active Learning Recommendations")
    print(f"{'='*80}")
    
    recommendations = physics_aware_active_learning(
        df, model, feature_df, config
    )
    
    if not recommendations.empty:
        rec_path = os.path.join(config.save_dir, 'active_learning_recommendations_improved.csv')
        recommendations.to_csv(rec_path, index=False)
        print(f"\n✓ Saved recommendations: {rec_path}")

                                                
                                          
                                                

    print(f"\n{'='*80}")
    print("Step 6.5: Active Learning Simulation (Retrospective)")
    print(f"{'='*80}")

                                 
    from active_learning_simulation import run_active_learning_simulation
    from generalization_analysis import ChemicalGroupExtractor

                                               
    b_site_groups = np.array([
        ChemicalGroupExtractor.get_b_site_element(f) for f in df['formula'].tolist()
    ])

                                                                     
    al_results = run_active_learning_simulation(
        X=X,
        y=y,
        formulas=df['formula'].tolist(),
        groups=b_site_groups,
        n_initial=200,                                  
        k_per_round=50,                                
        n_rounds=1,                                     
        n_seeds=10,                                                
        save_dir=config.save_dir
    )

                                                
                                                          
                                                

    print(f"\n{'='*80}")
    print("Step 6.6: Calibration Robustness Analysis")
    print(f"{'='*80}")

                                           
    from calibration_sensitivity import run_full_sensitivity_analysis

                              
    x_pbe_calib = np.array(calibrator.calibration_data['pbe_gap'])
    y_exp_calib = np.array(calibrator.calibration_data['exp_gap'])

                                           
    a_global = calibrator.global_model['slope']
    b_global = calibrator.global_model['intercept']

                                                              
    a_sn = calibrator.stratified_models.get('Sn', {}).get('slope', None)
    b_sn = calibrator.stratified_models.get('Sn', {}).get('intercept', None)
    a_ge = calibrator.stratified_models.get('Ge', {}).get('slope', None)
    b_ge = calibrator.stratified_models.get('Ge', {}).get('intercept', None)

    print(f"\nRunning LOOCV + Bootstrap analysis (n={len(x_pbe_calib)} calibration points)...")
    print(f"Global calibration: y = {a_global:.4f}x + {b_global:.4f}")

                                                                             
    df_for_calib = df.copy()
    df_for_calib['mu_pbe_pred'] = df['predicted_gap_pbe']                             
    df_for_calib['sigma'] = df['prediction_uncertainty']                          
    df_for_calib['e_hull'] = df['e_above_hull']                                   

                                             
    calibration_results = run_full_sensitivity_analysis(
        x_pbe_calib=x_pbe_calib,
        y_exp_calib=y_exp_calib,
        df_candidates=df_for_calib,
        a_global=a_global,
        b_global=b_global,
        a_sn=a_sn,
        b_sn=b_sn,
        a_ge=a_ge,
        b_ge=b_ge,
        random_state=config.random_state,                                 
        save_dir=os.path.join(config.save_dir, 'calibration_robustness')
    )

    print(f"\n✓ Calibration robustness analysis complete!")
    print(f"  Output: {config.save_dir}/calibration_robustness/")

                                                
                                       
                                                

    print(f"\n{'='*80}")
    print("Step 6.7: Proxy Features Ablation Analysis")
    print(f"{'='*80}")

                                  
    from proxy_features_ablation import run_proxy_ablation_experiment

                                                                                          
                                                                             
    gap_center = config.target_bandgap
    gap_halfwidth = config.bandgap_tolerance
    ehull_max = config.max_e_above_hull
    sigma_max = config.max_uncertainty

    mask_baseline = (
        (df['predicted_gap_exp'] >= gap_center - gap_halfwidth) &
        (df['predicted_gap_exp'] <= gap_center + gap_halfwidth) &
        (df['e_above_hull'] <= ehull_max) &
        (df['prediction_uncertainty'] <= sigma_max)                                     
    )

    print(f"\nBaseline candidates (29 features): {np.sum(mask_baseline)}")

                                     
    feature_names = model.feature_names if hasattr(model, 'feature_names') else\
                    [f'feature_{i}' for i in range(X.shape[1])]
    feature_df = pd.DataFrame(X, columns=feature_names)
    df_for_ablation = pd.concat([df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)

                                                                        
    df_for_ablation['e_hull'] = df_for_ablation['e_above_hull']                         
    df_for_ablation['sigma'] = df_for_ablation['prediction_uncertainty']                         

    print(f"\nRunning proxy features ablation experiment...")
    print(f"  This will retrain the model with +3 proxy features...")

                                                                                          
    ablation_results = run_proxy_ablation_experiment(
        df_with_features=df_for_ablation,
        X_baseline=X,
        y=y,
        model_class=ImprovedEnsemblePredictor,
        model_params={
            'n_models': config.n_ensemble_models,
            'random_state': config.random_state
        },
        mask_candidates_baseline=mask_baseline,
        a_global=a_global,
        b_global=b_global,
        gap_center=gap_center,
        gap_halfwidth=gap_halfwidth,
        ehull_max=ehull_max,
        sigma_max=sigma_max,
        save_dir=os.path.join(config.save_dir, 'proxy_ablation')
    )

    print(f"\n✓ Proxy features ablation complete!")
    print(f"  Output: {config.save_dir}/proxy_ablation/")

                                                
                                                      
                                                

    print(f"\n{'='*80}")
    print("Step 6.8: Multi-Round Active Learning Analysis")
    print(f"{'='*80}")

                                  
    from active_learning_multiround import run_multiround_al_experiment

    print(f"\nRunning multi-round AL simulation (6 rounds, 10 seeds)...")
    print(f"  This extends Step 6.5 to show learning curves across rounds...")

                                       
    multiround_al_results = run_multiround_al_experiment(
        X=X,
        y=y,
        groups=b_site_groups,
        model_class=ImprovedEnsemblePredictor,
        model_params={
            'n_models': config.n_ensemble_models,
            'random_state': config.random_state
        },
        n_initial=200,
        k_per_round=50,
        n_rounds=6,                                   
        n_seeds=10,
        save_dir=os.path.join(config.save_dir, 'multiround_al')
    )

    print(f"\n✓ Multi-round AL simulation complete!")
    print(f"  Output: {config.save_dir}/multiround_al/")

                                                
                           
                                                

    print(f"\n{'='*80}")
    print("Step 7: Generating Visualizations")
    print(f"{'='*80}")
    
    viz = ImprovedVisualizer()
    
                       
    viz.plot_model_performance(
        y_test, y_pred, y_std,
        os.path.join(config.save_dir, 'improved_model_performance.png')
    )
    
                     
    calibrator.plot_calibration(
        os.path.join(config.save_dir, 'improved_pbe_calibration.png')
    )
    
                          
    viz.plot_uncertainty_analysis(
        df['predicted_gap_exp'].values,
        df['prediction_uncertainty'].values,
        df['e_above_hull'].values,
        os.path.join(config.save_dir, 'improved_uncertainty_analysis.png'),
        target_gap=config.target_bandgap,
        bandgap_tolerance=config.bandgap_tolerance,
        min_uncertainty=config.min_uncertainty,
        max_uncertainty=config.max_uncertainty
    )
    
                        
    importance_df = model.get_feature_importance()
    if importance_df is not None:
        viz.plot_feature_importance(
            importance_df,
            os.path.join(config.save_dir, 'improved_feature_importance.png'),
            top_n=15
        )
    
                                                
                   
                                                
    
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Dataset: {len(df)} halide perovskite materials")
    print(f"Model Performance: R² = {r2:.3f}, RMSE = {rmse:.3f} eV")
    print(f"Uncertainty Calibration:")
    print(f"  Spearman ρ = {uncertainty_results['correlation']['spearman_rho']:.3f} "
          f"({uncertainty_results['correlation']['interpretation']})")
    print(f"  95% Coverage = {uncertainty_results['coverage']['95%']['coverage']:.3f} "
          f"(expected: 0.95)")
    print(f"  CRPS = {uncertainty_results['crps']['crps']:.3f} eV")
    print(f"Generalization (vs Random baseline):")
    if 'random' in generalization_results and 'loeo' in generalization_results:
        random_r2 = generalization_results['random']['mean_r2']
        loeo_r2 = generalization_results['loeo']['mean_r2']
        r2_drop = (random_r2 - loeo_r2) / random_r2 * 100
        print(f"  Random Split R² = {random_r2:.3f}")
        print(f"  Leave-One-Element-Out R² = {loeo_r2:.3f} (drop: {r2_drop:.1f}%)")
    print(f"PBE Calibration: Exp = {calibrator.global_model['slope']:.2f}×PBE + "
          f"{calibrator.global_model['intercept']:.2f}")
                                                
    loocv_slope_min = calibration_results['loocv']['slopes'].min()
    loocv_slope_max = calibration_results['loocv']['slopes'].max()
    print(f"  LOOCV Robustness: Slope range [{loocv_slope_min:.3f}, "
          f"{loocv_slope_max:.3f}] (mean ± std: "
          f"{calibration_results['loocv']['slope_mean']:.3f} ± "
          f"{calibration_results['loocv']['slope_std']:.3f})")
    print(f"  Bootstrap 95% CI: Slope [{calibration_results['bootstrap']['slope_ci'][0]:.3f}, "
          f"{calibration_results['bootstrap']['slope_ci'][1]:.3f}]")
    print(f"Candidates: {len(candidates)} materials (gap {config.target_bandgap}±{config.bandgap_tolerance} eV)")
    if ablation_results:
                                                                               
        comparison = ablation_results.get('comparison', {})
        heavy_d_baseline = comparison.get('baseline', {}).get('heavy_d_count', 0)
        heavy_d_proxy = comparison.get('proxy', {}).get('heavy_d_count', 0)
        n_baseline = comparison.get('baseline', {}).get('n_candidates', 1)
        n_proxy = comparison.get('proxy', {}).get('n_candidates', 1)
        frac_baseline = heavy_d_baseline / max(n_baseline, 1) * 100
        frac_proxy = heavy_d_proxy / max(n_proxy, 1) * 100

        if heavy_d_baseline > 0 or heavy_d_proxy > 0:
            delta = (heavy_d_proxy - heavy_d_baseline) / max(heavy_d_baseline, 1) * 100
            if delta <= 0:
                print(f"  Proxy Features: Heavy-d reduction = {-delta:.1f}% "
                      f"({heavy_d_baseline} → {heavy_d_proxy}, "
                      f"fraction: {frac_baseline:.1f}% → {frac_proxy:.1f}%)")
            else:
                print(f"  Proxy Features: Heavy-d increase = {delta:.1f}% "
                      f"({heavy_d_baseline} → {heavy_d_proxy}, "
                      f"fraction: {frac_baseline:.1f}% → {frac_proxy:.1f}%)")
    print(f"Active Learning: {len(recommendations)} high-priority materials for DFT")
    if multiround_al_results:
                                                            
        df_agg = multiround_al_results['aggregated']
        final_round = df_agg['round'].max()
        final_round_data = df_agg[df_agg['round'] == final_round]

        uncertainty_row = final_round_data[final_round_data['acquisition_fn'] == 'uncertainty']
        random_row = final_round_data[final_round_data['acquisition_fn'] == 'random']

        if len(uncertainty_row) > 0 and len(random_row) > 0:
            final_mae_unc = uncertainty_row['mean_mae'].values[0]
            final_mae_random = random_row['mean_mae'].values[0]
            improvement = (final_mae_random - final_mae_unc) / final_mae_random * 100
            print(f"  Multi-Round AL: {improvement:.1f}% improvement over random (Round {final_round})")
    
    print(f"\n{'='*80}")
    print("RECOMMENDED NEXT STEPS")
    print(f"{'='*80}")
    print(f"1. Perform HSE06 calculations on top 5 candidates:")
    for i, (idx, row) in enumerate(top_candidates.head(5).iterrows(), 1):
        print(f"   {i}. {row['formula']} (predicted: {row['predicted_gap_exp']:.2f} eV)")

    print(f"\n2. Prioritize active learning recommendations for DFT validation")
    print(f"   (Materials with high uncertainty in sparse chemical space)")

    print(f"\n3. Consider experimental synthesis for most promising candidates")
    print(f"   (Low E_hull, optimal bandgap, high model confidence)")

    print(f"\n4. Review robustness analysis outputs for paper revision:")
    print(f"   - Calibration robustness: {config.save_dir}/calibration_robustness/")
    print(f"   - Proxy features ablation: {config.save_dir}/proxy_ablation/")
    print(f"   - Multi-round AL curves: {config.save_dir}/multiround_al/")
    print(f"   See ROBUSTNESS_ANALYSIS_GUIDE.md for interpretation")

    print(f"\n{'='*80}")
    print("✓ Pipeline Complete with Robustness Analyses!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
