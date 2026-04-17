# C4 Photosynthesis Parameter Discrepancies: Primary Sources vs. vC2021 vs. Woodford 2025

**Date:** 2026-04-17
**Scope:** Setaria viridis NADP-ME parameterization used in `SetariaViridis_vC2021` library entry.

## Summary

The Helios C4 model follows the **vC2021 simple-Arrhenius re-fit convention**: activation energies for Vpmax, gm, and Kp are vC2021's own re-fits of primary-source data (Boyd 2015, Ubierna 2017), not the primary authors' directly reported values. Woodford et al. (2025) — co-authored by von Caemmerer — introduces inconsistencies with vC2021, including one citation error and one sign error inherited from the vC2021 spreadsheet. The primary sources agree among themselves; the two vC publications disagree with each other on several numerical values that should be equivalent.

## Discrepancy table

| Parameter | Primary source (direct) | vC2021 paper Table 1 | vC2021 spreadsheet | Woodford 2025 paper Table 1 | Woodford 2025 spreadsheet | Current Helios |
|---|---|---|---|---|---|---|
| Ea(Vpmax) | Boyd 2015: 1.01 ± 0.07 (peaked; ΔS=94.8 J/mol/K, Hd=73.3 kJ/mol) | **50.1** (re-fit of Boyd) | 50.1 | **94.8**¹ | 50.1 | 50.1 ✓ |
| Ea(Vcmax) | Boyd 2015: 78.0 ± 4.1 | 78.0 | 78.0 | 78.0 | 78.0 | 78.0 ✓ |
| Ea(Kp) | Boyd 2015: 36.3 ± 2.4; DiMario & Cousins 2019: 36.3 | **38.3** (re-fit) | 38.3 | **36.3** | 38.3 | 38.3 ✓ |
| Ea(Kc) | Boyd 2015: 64.2 ± 4.5 | 64.2 | 64.2 | 64.2 | 64.2 | 64.2 ✓ |
| Ea(Ko) | Boyd 2015: 10.5 ± 4.8 | 10.5 | 10.5 | 10.5 | 10.5 | 10.5 ✓ |
| Ea(gm) | Ubierna 2017 (Setaria): 63.8 (peaked); (Z. mays): 40.6 | **49.8** (re-fit of Setaria, 10–35 °C) | 49.8 | **40.6**² | 49.8 (hard-coded) | 49.8 ✓ |
| Ea(γ\*) | Boyd 2015: Ea(S_c/o) = −31.1, so Ea(γ\*) = +31.1 | −31.1³ | **−31.1** (error) | **+31.1** | **−31.1** (error persists) | +31.1 ✓ (fixed on this branch) |
| Ea(Rd) | Farquhar 1980: 66.4 | 66.4 | 66.4 | 66.4 | 66.4 | 66.4 ✓ |
| fcyc | — | 0.3 | 0.3 | **0.45** | **0.45** | 0.3 → **0.45** |
| H_Jcyc | — | 2 (implicit) | 2 (implicit) | **3.4** | **3.4** | implicit 2 → **3.4** |
| z (at default fcyc) | — | 0.964 | 0.964 | **1.45** | **1.45** | 0.964 → **1.45** |
| Jmax(T) form | June 2004 / Yamori 2010: Gaussian | peaked Arrhenius re-fit | Gaussian (Topt=43, Ω=26) | Gaussian (Topt=43, Ω=26) | Gaussian | peaked Arrhenius ≈ |

¹ **Woodford 2025 paper, Table 1: Ea(Vpmax) = 94.8 is a citation error.** 94.8 is the entropy parameter ΔS (units: J·mol⁻¹·K⁻¹) from Boyd 2015's peaked function, not an activation energy in kJ/mol. Their own fitting spreadsheet uses 50.1 — contradicting the paper text.

² **Woodford 2025 paper, Table 1: Ea(gm) = 40.6 is Ubierna's Zea mays value, not Setaria.** Ubierna 2017 reports Ea(gm)_Setaria = 63.8 (peaked) or 49.8 (re-fit, per vC2021). The spreadsheet uses 49.8 hard-coded inside the formula — contradicting the paper text.

³ **vC2021 Table 1 is correct on γ\*** (positive); only the shipping spreadsheet has the sign wrong. Boyd 2015 Table I reports Ea(S_c/o) = −31.1 (specificity falls with T). Because γ\* = 0.5/S_c/o, Ea(γ\*) = +31.1. The spreadsheet copies Boyd's S_c/o row into the γ\* row without the required reciprocal sign flip. Woodford 2025 paper silently corrects this — but the new Woodford fitting spreadsheet propagates the same error.

## Pattern analysis

**vC2021 paper vs. vC2021 spreadsheet agree** (except γ\*, a transcription error caught by the paper). Both use simple-Arrhenius re-fits of peaked/partial primary-source data: Vpmax 50.1, Kp 38.3, gm 49.8.

**Woodford 2025 paper vs. Woodford 2025 spreadsheet disagree on every activation energy.** The spreadsheet silently uses vC2021's re-fit values (50.1, 38.3, 49.8) and even carries the γ\* sign error forward. The paper Table 1 substitutes different numbers — some citation errors (Vpmax 94.8 misreads ΔS; gm 40.6 is Zea mays), some primary-source values (Kp 36.3 from Boyd 2015 direct). The paper never acknowledges that its Table 1 values do not match its own fitting tool.

**Woodford's structural contributions are independently correct.** The explicit H_Jcyc parameter, the updated z formula `z = (H_J·(1−fcyc) + H_Jcyc·fcyc)/(h·(1−fcyc))`, and fcyc = 0.45 (derived from ~0.41–0.58 theoretical range given NDH dominance per Ermakova 2024) are real improvements. Both the paper and spreadsheet agree on these.

## Implications for Helios

1. **Keep vC2021 re-fit activation energies** (Vpmax 50.1, Kp 38.3, gm 49.8). These are internally self-consistent for a simple-Arrhenius framework applied to Setaria. Adopting Woodford's paper-Table-1 numbers would introduce citation errors (Vpmax ΔS misread as Ea), wrong-species values (gm from Zea mays), and cherry-picking (Kp from Boyd direct while everything else stays vC2021 re-fit). Clarify comments to cite the re-fit source (vC2021) explicitly rather than the underlying primary sources.

2. **Update γ\* activation energy sign to +31.1** — already done on this branch. Both Woodford spreadsheets perpetuate the vC2021 spreadsheet transcription error; the physically correct sign (and Woodford paper Table 1) is positive.

3. **Adopt Woodford's structural updates:**
   - Add `H_J = 3` and `H_Jcyc = 3.4` as explicit `C4ModelCoefficients` fields.
   - Replace the hard-coded `(3 − fcyc)/(h·(1−fcyc))` z expression with `(H_J·(1−fcyc) + H_Jcyc·fcyc)/(h·(1−fcyc))`.
   - Update `fcyc` default 0.3 → 0.45 in both the struct and the `SetariaViridis_vC2021` library entry.
   - Regenerate C4 regression test anchors (both the 14-point spreadsheet test at 25 °C and the T-response test at 25/35 °C are affected).

4. **Defer Jmax Gaussian form.** Structural change to `PhotosyntheticTemperatureResponseParameters`, out of scope for this branch. The current peaked-Arrhenius approximation matches the Gaussian at 25 °C and 43 °C by construction and only diverges materially outside ~15–50 °C.

## Sources

- Boyd RA, Gandin A, Cousins AB (2015) *Plant Physiol* 169:1850. DOI 10.1104/pp.15.00586.
- Ubierna N, Gandin A, Boyd RA, Cousins AB (2017) *New Phytol* 214:66. DOI 10.1111/nph.14359.
- DiMario RJ, Cousins AB (2019) *J Exp Bot* 70:995.
- von Caemmerer S (2021) *J Exp Bot* 72:6003. DOI 10.1093/jxb/erab266.
- von Caemmerer S (2021) Setaria spreadsheet `C4__model_setaria__11-06-2021.xlsm`.
- Woodford R, Ermakova M, Furbank RT, von Caemmerer S (2025) *bioRxiv* 10.1101/2025.06.03.657559.
- Woodford et al. (2025) fitting tool `S1 - C4 Gas Exchange Fitting Tool_template.xlsm`.
