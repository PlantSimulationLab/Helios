/** \file "FluspectB.h" Fluspect-B/CX leaf fluorescence model for the Helios RadiationModel SIF pipeline.
 *
 * Implements the leaf radiative transfer and chlorophyll-fluorescence model described in
 *   - Vilfan, N., van der Tol, C., Muller, O., Rascher, U., Verhoef, W. (2016) Fluspect-B: A model for leaf
 *     fluorescence, reflectance and transmittance spectra. Remote Sensing of Environment 186:596-615;
 *   - Vilfan, N., Van der Tol, C., Yang, P., Wyber, R., Malenovsky, Z., Robinson, S.A., Verhoef, W. (2018)
 *     Extending Fluspect to simulate xanthophyll driven leaf reflectance dynamics. Remote Sensing of
 *     Environment 211:345-356 (violaxanthin/zeaxanthin carotenoid absorption mixing);
 *   - Jacquemoud, S., Baret, F. (1990) PROSPECT: A model of leaf optical properties spectra. Remote Sensing of
 *     Environment 34:75-91 (plate model and N-layer Stokes solution),
 * with the protein and carbon-based-constituent absorption terms of PROSPECT-PRO (Feret et al. 2021).
 * Given leaf biochemistry and Optipar specific-absorption coefficients, it computes:
 *   (a) leaf directional-hemispherical reflectance and transmittance over the Optipar grid (400-2400 nm)
 *   (b) Forward and backward fluorescence excitation-emission matrices
 *       M_f[wlf, wle] and M_b[wlf, wle] (units: dimensionless fluorescence
 *       efficiency per excitation-emission wavelength pair). The M matrices,
 *       when multiplied by incoming excitation flux and the fluorescence
 *       quantum yield Phi_F, give the per-leaf fluorescence emission at each
 *       emission wavelength.
 *
 * Optipar specific-absorption coefficients are loaded from
 * plugins/radiation/spectral_data/fluspect_B_optipar.xml.
 *
 * Copyright (C) 2016-2026 Brian Bailey
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
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
        //! over the leaf, emitted from the face opposite the one receiving the excitation
        //! (Vilfan et al. 2016: "forward" = detected from the leaf side turned away from the light).
        std::vector<std::vector<float>> Mf;
        //! Backward fluorescence excitation-emission matrix. Same shape as Mf. Emitted
        //! from the face receiving the excitation (toward the light source).
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
     * Implements the doubling-method algorithm of Vilfan et al. (2016), Sec. 2.4 and Appendices
     * A-C. The excitation grid is 400-750 nm at `excitation_step_nm` spacing (default 5 nm). The
     * emission grid is 640-848 nm at 4 nm spacing. Each kernel column includes the excitation
     * grid spacing as a quadrature weight, so kernel values scale with `excitation_step_nm`.
     *
     * Agreement with the reference kernels in plugins/radiation/tests/reference/ is to within
     * ~1e-6 relative per Mf/Mb element given identical Optipar coefficients.
     *
     * \param[in] biochem Leaf biochemistry inputs.
     * \param[in] optipar Optipar coefficients loaded via loadFluspectOptipar.
     * \param[in] excitation_step_nm Excitation wavelength grid spacing in nm. Must be > 0.
     *                               The grid is 400 nm to 750 nm inclusive.
     * \return Filled FluspectKernel. If `biochem.fqe <= 0`, Mf/Mb are left empty (caller
     *         should treat as no fluorescence). Callers that want a quantum-yield-agnostic
     *         kernel can pass `fqe = 1` and multiply by their own Phi_F externally — this
     *         is how RadiationModel uses the kernel so that leaves sharing biochemistry
     *         but differing in electron_transport_ratio can share a cached kernel.
     */
    FluspectKernel computeFluspectKernel(const FluspectBiochemistry &biochem,
                                         const FluspectOptipar &optipar,
                                         float excitation_step_nm = 5.f);

} // namespace helios

#endif // HELIOS_FLUSPECT_B_H
