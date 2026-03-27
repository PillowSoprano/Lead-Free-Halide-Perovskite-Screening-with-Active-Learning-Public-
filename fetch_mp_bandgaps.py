"""
Fetch PBE band gaps from Materials Project for calibration compounds.
Run this after confirming the candidate formula list.
"""

import os
from dotenv import load_dotenv
import pandas as pd
from mp_api.client import MPRester

load_dotenv()
API_KEY = os.getenv("MATERIALS_PROJECT_API_KEY")

# ── Candidate formulas (expand this list after literature search) ──────────
# Format: (formula, material_type, experimental_gap_eV, reference)
# PBE gap will be fetched from MP; if multiple entries exist, lowest-energy one is used.

CALIBRATION_CANDIDATES = [
    # ── Sn-based (ABX3) ──────────────────────────────────────────────────
    # NOTE: CsSnI3 corrected from 1.27→1.31 (Chung 2012 Nature)
    # NOTE: MASnI3 corrected from 1.17→1.30 (Noel 2014 EES)
    # NOTE: FASnI3 corrected from 1.48→1.41 (Koh 2015 JACS)
    # NOTE: RbSnI3 corrected from 1.32→1.38 (Babu 2019 JPCC)
    ("CsSnI3",   "Sn", 1.31, "Chung et al. Nature 485, 486 (2012)"),
    ("CsSnBr3",  "Sn", 1.75, "Stoumpos et al. Inorg.Chem. 52, 9019 (2013)"),
    ("CsSnCl3",  "Sn", 2.90, "Stoumpos et al. Inorg.Chem. 52, 9019 (2013)"),
    ("MASnI3",   "Sn", 1.30, "Noel et al. Energy Environ.Sci. 7, 3061 (2014)"),
    ("MASnBr3",  "Sn", 2.15, "Hao et al. Nat.Photon. 8, 489 (2014)"),
    ("MASnCl3",  "Sn", 3.05, "Stoumpos et al. Inorg.Chem. 52, 9019 (2013)"),
    ("FASnI3",   "Sn", 1.41, "Koh et al. J.Am.Chem.Soc. 137, 2494 (2015)"),
    ("RbSnI3",   "Sn", 1.38, "Babu et al. J.Phys.Chem.C 123, 4009 (2019)"),
    ("CsSnI2Br", "Sn", 1.37, "Sabba et al. J.Phys.Chem.C 119, 1763 (2015)"),
    ("CsSnIBr2", "Sn", 1.65, "Sabba et al. J.Phys.Chem.C 119, 1763 (2015)"),

    # ── Ge-based (ABX3) ──────────────────────────────────────────────────
    # NOTE: MAGeI3 corrected from 1.55→1.90 (Krishnamoorthy 2015 JMCA)
    ("CsGeI3",   "Ge", 1.63, "Stoumpos et al. Inorg.Chem. 54, 2757 (2015)"),
    ("CsGeBr3",  "Ge", 2.32, "Stoumpos et al. Inorg.Chem. 54, 2757 (2015)"),
    ("CsGeCl3",  "Ge", 3.67, "Stoumpos et al. Inorg.Chem. 54, 2757 (2015)"),
    ("MAGeI3",   "Ge", 1.90, "Krishnamoorthy et al. J.Mater.Chem.A 3, 23829 (2015)"),
    ("MAGeBr3",  "Ge", 2.64, "Krishnamoorthy et al. J.Mater.Chem.A 3, 23829 (2015)"),

    # ── Bi-based (A3B2X9) ────────────────────────────────────────────────
    ("Cs3Bi2I9",  "Bi", 2.03, "Lehner et al. Angew.Chem. 54, 8546 (2015)"),
    ("Cs3Bi2Br9", "Bi", 2.60, "Hoye et al. Chem.Eur.J. 22, 2605 (2016)"),
    ("Cs3Bi2Cl9", "Bi", 3.30, "Hoye et al. Chem.Eur.J. 22, 2605 (2016)"),
    ("MA3Bi2I9",  "Bi", 2.10, "Lyu et al. Nano Res. 9, 692 (2016)"),
    ("MA3Bi2Br9", "Bi", 2.75, "Johansson et al. Chem.Mater. 31, 6706 (2019)"),
    ("FA3Bi2I9",  "Bi", 2.08, "Johansson et al. Chem.Mater. 31, 6706 (2019)"),
    ("Rb3Bi2I9",  "Bi", 2.19, "Johansson et al. Chem.Mater. 31, 6706 (2019)"),

    # ── Sb-based (A3B2X9) ────────────────────────────────────────────────
    ("Cs3Sb2I9",  "Sb", 2.05, "Saparov et al. Chem.Mater. 27, 5622 (2015)"),
    ("Cs3Sb2Br9", "Sb", 2.67, "Vargas et al. J.Phys.Chem.Lett. 8, 1412 (2017)"),
    ("Cs3Sb2Cl9", "Sb", 3.56, "Vargas et al. J.Phys.Chem.Lett. 8, 1412 (2017)"),
    ("MA3Sb2I9",  "Sb", 2.14, "Saparov et al. Chem.Mater. 27, 5622 (2015)"),
    ("MA3Sb2Br9", "Sb", 2.74, "Johansson et al. Chem.Mater. 31, 6706 (2019)"),
    ("FA3Sb2I9",  "Sb", 2.11, "Johansson et al. Chem.Mater. 31, 6706 (2019)"),

    # ── Double perovskites (A2B'B''X6) ───────────────────────────────────
    # NOTE: Cs2AgBiBr6 corrected from 2.19→1.95 (indirect optical gap, Slavney 2016)
    # NOTE: Cs2AgSbCl6 corrected from 2.94→2.54 (Babu 2021 JPCL)
    ("Cs2AgBiBr6",  "double", 1.95, "Slavney et al. J.Am.Chem.Soc. 138, 2138 (2016) [indirect]"),
    ("Cs2AgBiCl6",  "double", 2.77, "Slavney et al. J.Am.Chem.Soc. 138, 2138 (2016)"),
    ("Cs2AgInCl6",  "double", 3.23, "Volonakis et al. J.Phys.Chem.Lett. 8, 772 (2017)"),
    ("Cs2AgSbCl6",  "double", 2.54, "Babu et al. J.Phys.Chem.Lett. 12, 4571 (2021)"),
    ("Cs2AgSbBr6",  "double", 1.64, "Deng et al. Adv.Funct.Mater. 30, 2002131 (2020)"),
    ("Cs2NaBiI6",   "double", 1.67, "Zhou et al. Angew.Chem. 58, 15213 (2019)"),
    ("Cs2NaBiCl6",  "double", 3.07, "Greul et al. J.Mater.Chem.A 5, 19972 (2017)"),
    ("Cs2AgTlBr6",  "double", 0.95, "Slavney et al. Angew.Chem. 57, 12765 (2018)"),

    # ── Vacancy-ordered (A2BX6) ───────────────────────────────────────────
    # NOTE: Cs2SnI6 corrected from 1.62→1.26 (Saparov 2016 CM)
    # NOTE: Cs2TiI6 corrected from 2.02→1.02 (Baranwal 2018) — likely original typo
    ("Cs2SnI6",  "vacancy", 1.26, "Saparov et al. Chem.Mater. 28, 2315 (2016)"),
    ("Cs2SnBr6", "vacancy", 1.82, "Kaltzoglou et al. J.Phys.Chem.C 120, 11777 (2016)"),
    ("Cs2SnCl6", "vacancy", 3.90, "Kaltzoglou et al. J.Phys.Chem.C 120, 11777 (2016)"),
    ("Cs2TiI6",  "vacancy", 1.02, "Baranwal et al. ChemSusChem 11, 3794 (2018)"),
    ("Cs2TiBr6", "vacancy", 1.78, "Euvrard et al. J.Mater.Chem.A 7, 24948 (2019)"),
    ("Cs2PdBr6", "vacancy", 1.60, "Babu et al. J.Phys.Chem.C 124, 10580 (2020)"),
]

def fetch_pbe_gaps(candidates):
    """Query Materials Project for PBE band gaps."""
    results = []

    with MPRester(API_KEY) as mpr:
        for formula, mat_type, exp_gap, ref in candidates:
            print(f"Querying: {formula} ...", end=" ")
            try:
                docs = mpr.materials.summary.search(
                    formula=formula,
                    fields=["material_id", "formula_pretty",
                            "band_gap", "energy_above_hull",
                            "is_stable"]
                )

                if not docs:
                    print(f"❌ Not found in MP")
                    results.append({
                        "formula": formula,
                        "material_type": mat_type,
                        "pbe_gap": None,
                        "exp_gap": exp_gap,
                        "reference": ref,
                        "mp_id": None,
                        "e_above_hull": None,
                        "note": "Not found in MP"
                    })
                    continue

                # Pick lowest energy-above-hull entry with band_gap > 0
                valid = [d for d in docs if d.band_gap is not None and d.band_gap > 0]
                if not valid:
                    valid = docs  # fall back to all

                best = min(valid, key=lambda d: d.energy_above_hull if d.energy_above_hull is not None else 999)

                print(f"✓  mp_id={best.material_id}  "
                      f"PBE={best.band_gap:.3f} eV  "
                      f"Ehull={best.energy_above_hull:.4f} eV/atom")

                results.append({
                    "formula": formula,
                    "material_type": mat_type,
                    "pbe_gap": best.band_gap,
                    "exp_gap": exp_gap,
                    "reference": ref,
                    "mp_id": best.material_id,
                    "e_above_hull": best.energy_above_hull,
                    "note": "OK" if exp_gap is not None else "exp_gap missing"
                })

            except Exception as e:
                print(f"⚠️  Error: {e}")
                results.append({
                    "formula": formula,
                    "material_type": mat_type,
                    "pbe_gap": None,
                    "exp_gap": exp_gap,
                    "reference": ref,
                    "mp_id": None,
                    "e_above_hull": None,
                    "note": f"Error: {e}"
                })

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("=" * 60)
    print("Fetching PBE band gaps from Materials Project")
    print("=" * 60)

    df = fetch_pbe_gaps(CALIBRATION_CANDIDATES)

    out_path = "./outputs/calibration_candidates_mp.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✓ Saved: {out_path}")

    # Summary
    complete = df[df["pbe_gap"].notna() & df["exp_gap"].notna()]
    print(f"\nComplete pairs (PBE + Exp both available): {len(complete)}")
    print(f"Missing PBE gap (not in MP):               "
          f"{df['pbe_gap'].isna().sum()}")
    print(f"Missing exp gap (TBD from literature):     "
          f"{df['exp_gap'].isna().sum()}")

    print("\nComplete pairs so far:")
    print(complete[["formula", "material_type", "pbe_gap",
                    "exp_gap", "reference"]].to_string(index=False))
