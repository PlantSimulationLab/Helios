/** \file "WeberPennTree.h" Primary header file for Weber-Penn tree architecture model.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include <random>
#include "Context.h"

struct WeberPennTreeParameters {

    // Name of tree
    std::string label;

    // General tree shape ID
    int Shape;

    // Fractional branchless area at tree base
    float BaseSize, BaseSizeV;

    // Number of splits at tree base
    uint BaseSplits, BaseSplitsV;

    // Fractional height of split (if BaseSplits>0)
    float BaseSplitSize, BaseSplitSizeV;

    // Size and scaling of tree
    float Scale, ScaleV, ZScale, ZScaleV;

    // Radius/length ratio, reduction
    float Ratio, RatioPower;

    // Sinusoidal cross-section variation
    int Lobes;
    float LobeDepth;

    // Exponential expansion at base of tree
    float Flare;

    // Levels of recursion
    int Levels;

    // Splits
    std::vector<float> nSegSplits;
    std::vector<float> nSplitAngle;
    std::vector<float> nSplitAngleV;

    // Curvature resolution and angles
    std::vector<int> nCurveRes;
    std::vector<float> nCurve;
    std::vector<float> nCurveV;
    std::vector<float> nCurveBack;

    // Relative length, cross sectional scaling
    std::vector<float> nLength;
    std::vector<float> nLengthV;
    std::vector<float> nTaper;

    // Spiraling angle, # branches
    std::vector<float> nDownAngle;
    std::vector<float> nDownAngleV;
    std::vector<float> nRotate;
    std::vector<float> nRotateV;
    std::vector<int> nBranches;

    // PNG image file for leaf mask
    std::string LeafFile;

    // Number of leaves per parent, shape ID
    int Leaves, LeafShape;

    // Leaf length, relative x-scale
    float LeafScale, LeafScaleX;

    // Color of trunk and branches
    helios::RGBcolor WoodColor;

    // image file for wood texture map
    std::string WoodFile;

    // leaf angle cumulative distribution function (elevation)
    std::vector<float> leafAngleCDF;
};

class WeberPennTree {
public:
    //! Weber-Penn Tree constructor
    /**  \param[in] context Pointer to the Helios context
     */
    WeberPennTree(helios::Context *context);

    //! Unit testing routine
    static int selfTest(int argc = 0, char **argv = nullptr);

    //! Load tree library from an XML file
    /**
     * \param[in] filename XML file with path relative to build directory
     * \param[in] silent Disable output messages
     */
    void loadXML(const char *filename, bool silent = false);

    //! Construct a Weber-Penn tree using a tree already in the library
    /**
     * \param[in] treename Name of a tree loaded in the library (\sa WeberPennTree::loadXML).
     * \param[in] origin (x,y,z) location to place tree.  Note that this position is the location of the center of the trunk base.
     * \return Identifier for tree.
     */
    uint buildTree(const char *treename, helios::vec3 origin);

    //! Construct a Weber-Penn tree using a tree already in the library
    /**
     * \param[in] treename Name of a tree loaded in the library (\sa WeberPennTree::loadXML).
     * \param[in] origin (x,y,z) location to place tree.  Note that this position is the location of the center of the trunk base.
     * \param[in] scale Scaling factor to apply to entire tree. Default scale is 1 or 100%, and scale<1 makes tree smaller, scale>1 makes tree bigger.
     * \return Identifier for tree.
     */
    uint buildTree(const char *treename, helios::vec3 origin, float scale);

    //! Get the unique universal identifiers (UUIDs) for the primitives that make up the tree trunk
    /**
     * \param[in] TreeID Identifier of tree.
     */
    std::vector<uint> getTrunkUUIDs(const uint TreeID);

    //! Get the unique universal identifiers (UUIDs) for the primitives that make up the tree branches
    /**
     * \param[in] TreeID Identifier of tree.
     */
    std::vector<uint> getBranchUUIDs(const uint TreeID);

    //! Get the unique universal identifiers (UUIDs) for the primitives that make up the tree leaves
    /**
     * \param[in] TreeID Identifier of tree.
     */
    std::vector<uint> getLeafUUIDs(const uint TreeID);

    //! Get the unique universal identifiers (UUIDs) for all primitives that make up the tree
    /**
     * \param[in] TreeID Identifier of tree.
     */
    std::vector<uint> getAllUUIDs(const uint TreeID);

    //! Only create branch primitives up to a certain recursion level (leaves are still created for all levels).
    /**
     * Very small branches may add a lot of unnecessary triangles. This function limits the number of recursive branch levels to generate primitives. The default is to generate primitives for recursion levels 0-2.
     * \param[in] level Branch recursion levels for which primitives should be generated. For example, level=1 would generate primitives for the trunk (level 0), and the first branching level (level 1).
     */
    void setBranchRecursionLevel(const uint level);

    //! Set the radial triangle subdivisions for trunks.
    /**
     * For example, if trunk_segs = 3 the trunk would be a triangular prism, if trunk_segs=4 the trunk would be a rectangular prism, if trunk_segs=5 the trunk cross-section would be a pentagon, and so on. Note that trunk_segs must be greater than 2.
     * \param[in] trunk_segs Number of radial triangle subdivisions for the trunk.
     */
    void setTrunkSegmentResolution(const uint trunk_segs);

    //! Set the radial triangle subdivisions for branches.
    /**
     * For example, if branch_segs = 3 the branches would be a triangular prism, if branch_segs=4 the branches would be a rectangular prism, if branch_segs=5 the branch cross-section would be a pentagon, and so on. Note that branch_segs must be
     * greater than 2.
     * \param[in] branch_segs Number of radial triangle subdivisions for branches.
     */
    void setBranchSegmentResolution(const uint branch_segs);

    //! Set the number of sub-patch divisions for leaves.
    /**
     * \param[in] leaf_segs Number of leaf sub-patches in the (x,y) directions.
     */
    void setLeafSubdivisions(const helios::int2 leaf_segs);

    //! Get the architectural parameters for a tree in the currently loaded library
    /**
     * \param[in] treename Name of a tree in the library.
     * \return Set of tree parameters.
     */
    WeberPennTreeParameters getTreeParameters(const char *treename);

    //! Set the architectural parameters for a tree in the currently loaded library
    /**
     * \param[in] treename Name of a tree in the library.
     * \param[in] parameters Set of tree parameters.
     */
    void setTreeParameters(const char *treename, const WeberPennTreeParameters parameters);

    //! Seed the random number generator. This can be useful for generating repeatable trees, say, within a loop.
    /**
     * \param[in] seed Random number seed
     */
    void seedRandomGenerator(const uint seed);

    //! Add optional output primitive data values to the Context
    /**
     * \param[in] label Name of primitive data.
     */
    void optionalOutputPrimitiveData(const char *label);

    //! Disable all print messages to the screen except for fatal error messages
    void disableMessages();

    //! Enable all print messages to the screen
    void enableMessages();

private:
    helios::Context *context;

    //! UUIDs for trunk primitives
    std::vector<std::vector<uint>> UUID_trunk;

    //! UUIDs for branch primitives
    std::vector<std::vector<uint>> UUID_branch;

    //! UUIDs for leaf primitives
    std::vector<std::vector<uint>> UUID_leaf;

    std::map<std::string, WeberPennTreeParameters> trees_library;

    //! Spawn a branch, which recurses until leaves
    /**
     * \param[in] parameters Tree geometric parameters
     * \param[in] n Recursive level of branch (n=0 is the trunk, up to maximum n=Levels-1)
     * \param[in] seg_start Segment index along branch (=0 if nSegSplits=0)
     * \param[in] base_position (x,y,z) coordinate of branch base
     * \param[in] parent_normal (nx,ny,nz) unit vector pointing in the direction of the parent branch segment
     * \param[in] child_rotation Spherical coordinates (azimuth, elevation, radius) of the child branch segment relative to the parent branch segment
     * \param[in] length_parent Length of the parent branch in meters
     * \param[in] radius_parent Radius in meters of the parent branch segment
     * \param[in] offset_child Length in meters of the child along the parent's branch
     * \param[in] origin (x,y,z) coordinate of the tree base, used to compute the absolute position of the branch
     * \param[in] scale Tree scaling factor
     * \param[in] leaf_template UUID of the leaf primitive prototype that is copied to form leaves
     * \param[in] base_size Base size computed (with the configured variance) for this specific tree instance
     * \param[in] base_splits Number of base splits computed (with the configured variance) for this specific tree instance
     */
    void recursiveBranch(WeberPennTreeParameters parameters, uint n, uint seg_start, helios::vec3 base_position, helios::vec3 parent_normal, helios::SphericalCoord child_rotation, float length_parent, float radius_parent, float offset_child,
                         helios::vec3 origin, float scale, const uint leaf_template, float base_size, uint base_splits);

    float getVariation(float V);

    std::minstd_rand0 generator;

    float ShapeRatio(uint shape, float ratio);

    uint branchLevels;

    uint trunk_segs;

    uint branch_segs;

    helios::int2 leaf_segs;

    //! Names of additional primitive data to add to the Context
    std::vector<std::string> output_prim_data;

    //! Flag denoting whether messages should be printed to screen
    bool printmessages;
};
