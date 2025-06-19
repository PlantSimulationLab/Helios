/** \file "IrrigationModel.cpp" Primary source file for irrigation plug-in.

    Copyright (C) 2016-2025 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "IrrigationModel.h"

using namespace helios;

IrrigationModel::IrrigationModel(Context *context) { this->context = context; }

int IrrigationModel::parseLineEntity(const std::map<int, std::string> &entity) {
    if (entity.count(10) && entity.count(20) && entity.count(11) && entity.count(21)) {
        IrrigationPipe p;
        double x1 = std::stod(entity.at(10));
        double y1 = std::stod(entity.at(20));
        double x2 = std::stod(entity.at(11));
        double y2 = std::stod(entity.at(21));

        std::map<std::pair<double,double>, int> node_index;
        for (size_t i=0;i<nodes.size();++i) {
            node_index[{nodes[i].x,nodes[i].y}] = static_cast<int>(i);
        }
        int n1, n2;
        auto it = node_index.find({x1,y1});
        if (it==node_index.end()) {
            IrrigationNode node; node.x=x1; node.y=y1; node.pressure=0.0; node.fixed=false;
            nodes.push_back(node); n1 = static_cast<int>(nodes.size()-1);
        } else n1 = it->second;
        node_index[{nodes[n1].x,nodes[n1].y}] = n1;
        it = node_index.find({x2,y2});
        if (it==node_index.end()) {
            IrrigationNode node; node.x=x2; node.y=y2; node.pressure=0.0; node.fixed=false;
            nodes.push_back(node); n2 = static_cast<int>(nodes.size()-1);
        } else n2 = it->second;
        node_index[{nodes[n2].x,nodes[n2].y}] = n2;

        p.n1 = n1; p.n2 = n2;
        p.length = std::hypot(x2 - x1, y2 - y1);
        p.diameter = 0.05; //m
        p.kminor = 1.0;
        pipes.push_back(p);
        return 0;
    }
    return 1;
}

int IrrigationModel::readDXF(const std::string &filename) {
    std::ifstream in(filename.c_str());
    if (!in.is_open()) return 1;
    std::vector<std::string> lines; std::string line;
    while (std::getline(in,line)) lines.push_back(line);
    in.close();

    std::map<int, std::string> entity;
    for (size_t i=0;i+1<lines.size();i+=2) {
        std::string code = lines[i];
        std::string value = lines[i+1];
        if (code == "0") {
            if (!entity.empty()) {
                if (entity[0] == "LINE") parseLineEntity(entity);
                entity.clear();
            }
            entity[0] = value;
        } else {
            int c = std::stoi(code);
            entity[c] = value;
        }
    }
    if (!entity.empty() && entity[0]=="LINE") parseLineEntity(entity);

    if (!nodes.empty()) nodes[0].fixed = true, nodes[0].pressure=50.0; // supply node

    return 0;
}

std::vector<double> IrrigationModel::solveLinear(std::vector<std::vector<double>> A, std::vector<double> b) const {
    const int n = static_cast<int>(A.size());
    for (int i=0;i<n;i++) {
        int pivot=i;
        for (int j=i+1;j<n;j++) if (std::fabs(A[j][i])>std::fabs(A[pivot][i])) pivot=j;
        std::swap(A[i],A[pivot]); std::swap(b[i],b[pivot]);
        double diag=A[i][i];
        if (std::fabs(diag)<1e-12) continue;
        for (int j=i;j<n;j++) A[i][j]/=diag; b[i]/=diag;
        for (int k=0;k<n;k++) {
            if (k==i) continue;
            double f=A[k][i];
            for (int j=i;j<n;j++) A[k][j]-=f*A[i][j];
            b[k]-=f*b[i];
        }
    }
    return b;
}

int IrrigationModel::solve() {
    size_t n = nodes.size();
    std::vector<int> mapIndex(n,-1);
    int unknowns=0;
    for (size_t i=0;i<n;i++) if (!nodes[i].fixed) mapIndex[i]=unknowns++;
    std::vector<std::vector<double>> A(unknowns, std::vector<double>(unknowns,0.0));
    std::vector<double> b(unknowns,0.0);

    for (const auto &p : pipes) {
        double R = p.length; //simple resistance
        int i=p.n1; int j=p.n2;
        int ai=mapIndex[i]; int aj=mapIndex[j];
        if (ai>=0) {
            A[ai][ai]+=1.0/R;
            if (aj>=0) A[ai][aj]-=1.0/R; else b[ai]+=nodes[j].pressure/R;
        }
        if (aj>=0) {
            A[aj][aj]+=1.0/R;
            if (ai>=0) A[aj][ai]-=1.0/R; else b[aj]+=nodes[i].pressure/R;
        }
    }

    std::vector<double> x = solveLinear(A,b);
    for (size_t i=0;i<n;i++) if (mapIndex[i]>=0) nodes[i].pressure=x[mapIndex[i]];
    return 0;
}

int IrrigationModel::writeDXF(const std::string &filename) const {
    std::ofstream out(filename.c_str());
    if (!out.is_open()) return 1;
    out << "0\nSECTION\n2\nENTITIES\n";
    for (const auto &p : pipes) {
        const auto &n1 = nodes[p.n1];
        const auto &n2 = nodes[p.n2];
        out << "0\nLINE\n8\n0\n10\n" << n1.x << "\n20\n" << n1.y << "\n11\n" << n2.x << "\n21\n" << n2.y << "\n";
    }
    for (const auto &node : nodes) {
        out << "0\nTEXT\n8\n0\n10\n" << node.x << "\n20\n" << node.y << "\n40\n0.2\n1\n" << node.pressure << "\n";
    }
    out << "0\nENDSEC\n0\nEOF\n";
    out.close();
    return 0;
}

int IrrigationModel::selfTest() {
    std::cout << "Running irrigation plug-in self-test..." << std::endl;
    readDXF("../plugins/irrigation/doc/simple_network.dxf");
    solve();
    writeDXF("../plugins/irrigation/doc/simple_network_out.dxf");
    for (size_t i=0;i<nodes.size();++i) {
        std::cout << "Node " << i << " pressure: " << nodes[i].pressure << std::endl;
    }
    return 0;
}