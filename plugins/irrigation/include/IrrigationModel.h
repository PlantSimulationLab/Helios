/** \file "IrrigationModel.h" Primary header file for irrigation plug-in.

    Copyright (C) 2016-2025 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef IRRIGATIONMODEL
#define IRRIGATIONMODEL

#include "Context.h"

namespace helios {

//! Node in irrigation network
struct IrrigationNode {
    double x; //!< x coordinate
    double y; //!< y coordinate
    double pressure; //!< calculated pressure
    bool fixed; //!< if true, pressure is fixed
};

//! Pipe in irrigation network
struct IrrigationPipe {
    int n1; //!< index of first node
    int n2; //!< index of second node
    double length; //!< pipe length
    double diameter; //!< pipe diameter
    double kminor; //!< minor loss coefficient
};

//! Irrigation model class
class IrrigationModel {
public:
    //! Constructor
    /**
     * \param[in] context Pointer to the Helios context
     */
    explicit IrrigationModel(Context *context);

    //! Read irrigation network from DXF file
    int readDXF(const std::string &filename);

    //! Solve for pressure distribution
    int solve();

    //! Write network with pressures to DXF file
    int writeDXF(const std::string &filename) const;

    //! Self-test routine
    int selfTest();

private:
    Context *context; //!< Helios context
    std::vector<IrrigationNode> nodes; //!< nodes in network
    std::vector<IrrigationPipe> pipes; //!< pipes in network

    int parseLineEntity(const std::map<int, std::string> &entity);
    int buildNetwork();
    std::vector<double> solveLinear(std::vector<std::vector<double>> A, std::vector<double> b) const;
};

}

#endif