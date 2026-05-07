/** \file "FluspectB.h" Fluspect-B leaf fluorescence model (Vilfan et al. 2016) for Helios RadiationModel SIF plugin.
 *
 * This module ports the fluspect_B_CX.m routine from SCOPE v2.0
 * (https://github.com/Christiaanvandertol/SCOPE) to C++. Given leaf
 * biochemistry (PROSPECT-PRO + CX inputs) and Optipar specific-absorption
 * coefficients, it computes:
 *   (a) PROSPECT-PRO-CX leaf reflectance and transmittance over 400-2400 nm
 *   (b) Forward and backward fluorescence excitation-emission matrices
 *       M_f[wlf, wle] and M_b[wlf, wle] (units: dimensionless fluorescence
 *       efficiency per excitation-emission wavelength pair). The M matrices,
 *       when multiplied by incoming excitation flux and the fluorescence
 *       quantum yield Phi_F, give the per-leaf fluorescence emission at each
 *       emission wavelength.
 *
 * Inputs and outputs closely follow the MATLAB reference; see SCOPE's
 * src/RTMs/fluspect_B_CX.m. Optipar specific-absorption coefficients are
 * loaded from plugins/radiation/spectral_data/fluspect_B_optipar.xml.
 */

#ifndef HELIOS_FLUSPECT_B_H
#define HELIOS_FLUSPECT_B_H

#include <cstddef>
#include <string>
#include <vector>

namespace helios {

    //! Fluspect-B leaf biochemistry inputs (PROSPECT-PRO + CX pigments + fluorescence parameters).
    struct FluspectBiochemistry {
        float Cab = 40.f;    //!< chlorophyll a+b content, ug/cm^2
        float Cca = 10.f;    //!< carotenoid content, ug/cm^2
        float Cw = 0.009f;   //!< equivalent water thickness, cm
        float Cdm = 0.012f;  //!< dry matter, g/cm^2 (PROSPECT-D)
        float Cs = 0.f;      //!< brown pigments (senescence), unitless
        float Cant = 1.f;    //!< anthocyanin content, ug/cm^2
        float Cp = 0.f;      //!< protein content, g/cm^2 (PROSPECT-PRO)
        float Cbc = 0.f;     //!< carbon-based constituents, g/cm^2 (PROSPECT-PRO)
        float N = 1.5f;      //!< mesophyll structure parameter, unitless
        float V2Z = 0.f;     //!< violaxanthin->zeaxanthin conversion state, [0, 1]
        float fqe = 0.01f;   //!< intrinsic fluorescence quantum efficiency
    };

    //! Optipar specific-absorption coefficient set consumed by Fluspect-B.
    //!
    //! All arrays are defined on the same monotonic wavelength grid
    //! `wavelengths_nm`, typically 400-2400 nm at 1 nm spacing (the default
    //! shipped by fluspect_B_optipar.xml).
    struct FluspectOptipar {
        std::vector<float> wavelengths_nm;
        std::vector<float> nr;     //!< refractive index
        std::vector<float> Kab;    //!< specific absorption coefficient, chlorophyll a+b
        std::vector<float> Kca;    //!< specific absorption coefficient, carotenoids
        std::vector<float> KcaV;   //!< violaxanthin-dominant Kca (V2Z = 0 end)
        std::vector<float> KcaZ;   //!< zeaxanthin-dominant Kca (V2Z = 1 end)
        std::vector<float> Kdm;    //!< dry matter
        std::vector<float> Kw;     //!< water
        std::vector<float> Ks;     //!< brown pigments
        std::vector<float> Kant;   //!< anthocyanin
        std::vector<float> Kp;     //!< protein (PROSPECT-PRO)
        std::vector<float> Kcbc;   //!< carbon-based constituents (PROSPECT-PRO)
        std::vector<float> phi;    //!< PSII normalised fluorescence emission spectrum (unitless)
    };

    //! Output of Fluspect-B computation for one leaf biochemistry.
    struct FluspectKernel {
        //! PROSPECT wavelength grid (nm). Same size as `refl`/`tran`.
        std::vector<float> wavelengths_nm;

        //! Leaf directional-hemispherical reflectance spectrum (same grid as wavelengths_nm).
        std::vector<float> refl;
        //! Leaf directional-hemispherical transmittance spectrum.
        std::vector<float> tran;

        //! Excitation wavelength grid (nm). Subset of PROSPECT grid at user-chosen step.
        std::vector<float> wle;
        //! Emission wavelength grid (nm). Fixed at 640-848 nm at 4 nm step (Fluspect native).
        std::vector<float> wlf;

        //! Forward fluorescence excitation-emission matrix. Indexed Mf[wlf_idx][wle_idx].
        //! Each entry is the fluorescence emission per unit excitation flux, integrated
        //! over the leaf, emitted from the face receiving the excitation.
        std::vector<std::vector<float>> Mf;
        //! Backward fluorescence excitation-emission matrix. Same shape as Mf. Emitted
        //! from the face opposite the one receiving the excitation.
        std::vector<std::vector<float>> Mb;
    };

    //! Load Fluspect-B Optipar coefficients from a Helios spectral-library XML asset.
    /**
     * Reads globaldata_vec2 entries named fluspect_optipar_<field> (nr, Kab, Kca, KcaV,
     * KcaZ, Kdm, Kw, Ks, Kant, Kp, Kcbc, phi) from Context global data. If any entries
     * are missing from Context global data but present in the named XML file on disk,
     * loads them first via Context::loadXML.
     *
     * \param[in] xml_path Absolute path to fluspect_B_optipar.xml (resolved by caller).
     * \param[out] optipar Destination coefficient set.
     * \throws helios_runtime_error on any missing coefficient or grid-mismatch.
     */
    void loadFluspectOptipar(const std::string &xml_path, FluspectOptipar &optipar);

    //! Compute Fluspect-B leaf optics + fluorescence matrices for a single leaf biochemistry.
    /**
     * Implements the doubling-method algorithm from fluspect_B_CX.m (SCOPE v2.0). The
     * excitation grid is 400-750 nm at `excitation_step_nm` spacing (default 5 nm to match
     * SCOPE's int=5 internal resolution). The emission grid is 640-848 nm at 4 nm spacing
     * (Fluspect native; corresponds to wlf = 640:4:850 with MATLAB's inclusive/exclusive
     * endpoint convention yielding 848 as the last sample).
     *
     * Numerical agreement with MATLAB fluspect_B_CX is to within ~1e-6 per Mf/Mb element
     * given identical Optipar coefficients.
     *
     * \param[in] biochem Leaf biochemistry inputs.
     * \param[in] optipar Optipar coefficients loaded via loadFluspectOptipar.
     * \param[in] excitation_step_nm Excitation wavelength grid spacing in nm. Must be > 0.
     *                               The grid is 400 nm to 750 nm inclusive.
     * \return Filled FluspectKernel. If `biochem.fqe <= 0`, Mf/Mb are left empty (caller
     *         should treat as no fluorescence). Callers that want a quantum-yield-agnostic
     *         kernel can pass `fqe = 1` and multiply by their own Phi_F externally — this
     *         is how RadiationModel uses the kernel so that leaves sharing biochemistry
     *         but differing in J/Jmax can share a cached kernel.
     */
    FluspectKernel computeFluspectKernel(const FluspectBiochemistry &biochem,
                                         const FluspectOptipar &optipar,
                                         float excitation_step_nm = 5.f);

} // namespace helios

#endif // HELIOS_FLUSPECT_B_H
