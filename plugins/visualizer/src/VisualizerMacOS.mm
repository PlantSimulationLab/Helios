/** \file "VisualizerMacOS.mm" macOS application settings for the Visualizer plug-in.

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

#include "VisualizerMacOS.h"

#import <AppKit/AppKit.h>

#include "global.h"

namespace helios {
namespace visualizer_macos {

    void setActivationPolicy(ActivationPolicy policy) {
        if (NSApp == nil) {
            helios_runtime_error("ERROR (Visualizer): The Cocoa application does not exist yet, so its activation policy cannot be set. It is created when GLFW creates its first window.");
        }
        NSApplicationActivationPolicy ns_policy = NSApplicationActivationPolicyRegular;
        if (policy == ActivationPolicy::Accessory) {
            ns_policy = NSApplicationActivationPolicyAccessory;
        } else if (policy == ActivationPolicy::Prohibited) {
            ns_policy = NSApplicationActivationPolicyProhibited;
        }
        const BOOL accepted = [NSApp setActivationPolicy:ns_policy];
        if ([NSApp activationPolicy] != ns_policy) {
            helios_runtime_error("ERROR (Visualizer): macOS did not change the activation policy of the Cocoa application, which decides whether the process has a Dock icon, from " +
                                 std::to_string(long([NSApp activationPolicy])) + " to " + std::to_string(long(ns_policy)) + " (call returned " + std::to_string(int(accepted)) + ").");
        }
    }

    ActivationPolicy getActivationPolicy() {
        if (NSApp == nil) {
            helios_runtime_error("ERROR (Visualizer): The Cocoa application does not exist yet, so it has no activation policy. It is created when GLFW creates its first window.");
        }
        switch ([NSApp activationPolicy]) {
            case NSApplicationActivationPolicyAccessory:
                return ActivationPolicy::Accessory;
            case NSApplicationActivationPolicyProhibited:
                return ActivationPolicy::Prohibited;
            default:
                return ActivationPolicy::Regular;
        }
    }

} // namespace visualizer_macos
} // namespace helios
