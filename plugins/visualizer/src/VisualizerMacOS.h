/** \file "VisualizerMacOS.h" macOS application settings for the Visualizer plug-in.

    Copyright (C) 2016-2026 Brian Bailey

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    SPDX-License-Identifier: LGPL-2.1-or-later

*/

#ifndef HELIOS_VISUALIZER_MACOS_H
#define HELIOS_VISUALIZER_MACOS_H

#ifdef __APPLE__

namespace helios {
namespace visualizer_macos {

    //! Activation policy of the process's Cocoa application, which decides whether it has a Dock icon
    enum class ActivationPolicy {
        //! An ordinary application, with a Dock icon and a menu bar
        Regular,
        //! No Dock icon, but may show windows and take focus
        Accessory,
        //! No Dock icon, no menu bar, and never takes focus: a background process
        Prohibited
    };

    //! Set the activation policy of the process's Cocoa application
    /**
     * GLFW makes the process a Regular application, with a Dock icon, when it creates its first window, so a headless Visualizer, whose only window is hidden, would otherwise show an icon for as long
     * as the process lives. Must be called after GLFW has created a window, when the application exists.
     * \param[in] policy Activation policy to set.
     */
    void setActivationPolicy(ActivationPolicy policy);

    //! Activation policy of the process's Cocoa application
    [[nodiscard]] ActivationPolicy getActivationPolicy();

} // namespace visualizer_macos
} // namespace helios

#endif

#endif
