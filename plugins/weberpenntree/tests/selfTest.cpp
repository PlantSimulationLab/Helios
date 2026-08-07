#include "WeberPennTree.h"

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest.h>
#include "doctest_utils.h"

using namespace helios;

DOCTEST_TEST_CASE("WeberPennTree Constructor") {
    Context context;
    DOCTEST_CHECK_NOTHROW(WeberPennTree weberpenntree(&context));
}

DOCTEST_TEST_CASE("WeberPennTree Build Default Library Trees") {
    float spacing = 5;
    std::vector<std::string> trees = {"Almond", "Apple", "Avocado", "Lemon", "Olive", "Orange", "Peach", "Pistachio", "Walnut"};

    for (int i = 0; i < trees.size(); i++) {
        Context context;
        WeberPennTree weberpenntree(&context);
        weberpenntree.disableMessages();

        DOCTEST_CHECK_NOTHROW(weberpenntree.setBranchRecursionLevel(1));
        DOCTEST_CHECK_NOTHROW(weberpenntree.setLeafSubdivisions(make_int2(3, 3)));

        vec3 origin(i * spacing, 0, 0);
        DOCTEST_CHECK_NOTHROW(weberpenntree.buildTree(trees.at(i).c_str(), origin, 0.75f));

        // Verify that primitives were created
        std::vector<uint> all_UUIDs = context.getAllUUIDs();
        DOCTEST_CHECK(!all_UUIDs.empty());
    }
}

DOCTEST_TEST_CASE("WeberPennTree Individual Tree Types") {

    DOCTEST_SUBCASE("Almond Tree") {
        Context context;
        WeberPennTree weberpenntree(&context);
        DOCTEST_CHECK_NOTHROW(weberpenntree.buildTree("Almond", make_vec3(0, 0, 0)));
        DOCTEST_CHECK(!context.getAllUUIDs().empty());
    }

    DOCTEST_SUBCASE("Apple Tree") {
        Context context;
        WeberPennTree weberpenntree(&context);
        DOCTEST_CHECK_NOTHROW(weberpenntree.buildTree("Apple", make_vec3(0, 0, 0)));
        DOCTEST_CHECK(!context.getAllUUIDs().empty());
    }

    DOCTEST_SUBCASE("Avocado Tree") {
        Context context;
        WeberPennTree weberpenntree(&context);
        DOCTEST_CHECK_NOTHROW(weberpenntree.buildTree("Avocado", make_vec3(0, 0, 0)));
        DOCTEST_CHECK(!context.getAllUUIDs().empty());
    }

    DOCTEST_SUBCASE("Lemon Tree") {
        Context context;
        WeberPennTree weberpenntree(&context);
        DOCTEST_CHECK_NOTHROW(weberpenntree.buildTree("Lemon", make_vec3(0, 0, 0)));
        DOCTEST_CHECK(!context.getAllUUIDs().empty());
    }

    DOCTEST_SUBCASE("Olive Tree") {
        Context context;
        WeberPennTree weberpenntree(&context);
        DOCTEST_CHECK_NOTHROW(weberpenntree.buildTree("Olive", make_vec3(0, 0, 0)));
        DOCTEST_CHECK(!context.getAllUUIDs().empty());
    }

    DOCTEST_SUBCASE("Orange Tree") {
        Context context;
        WeberPennTree weberpenntree(&context);
        DOCTEST_CHECK_NOTHROW(weberpenntree.buildTree("Orange", make_vec3(0, 0, 0)));
        DOCTEST_CHECK(!context.getAllUUIDs().empty());
    }

    DOCTEST_SUBCASE("Peach Tree") {
        Context context;
        WeberPennTree weberpenntree(&context);
        DOCTEST_CHECK_NOTHROW(weberpenntree.buildTree("Peach", make_vec3(0, 0, 0)));
        DOCTEST_CHECK(!context.getAllUUIDs().empty());
    }

    DOCTEST_SUBCASE("Pistachio Tree") {
        Context context;
        WeberPennTree weberpenntree(&context);
        DOCTEST_CHECK_NOTHROW(weberpenntree.buildTree("Pistachio", make_vec3(0, 0, 0)));
        DOCTEST_CHECK(!context.getAllUUIDs().empty());
    }

    DOCTEST_SUBCASE("Walnut Tree") {
        Context context;
        WeberPennTree weberpenntree(&context);
        DOCTEST_CHECK_NOTHROW(weberpenntree.buildTree("Walnut", make_vec3(0, 0, 0)));
        DOCTEST_CHECK(!context.getAllUUIDs().empty());
    }
}

DOCTEST_TEST_CASE("WeberPennTree Recursion Level") {

    Context context;
    WeberPennTree weberpenntree(&context);

    // Test setting different recursion levels
    DOCTEST_CHECK_NOTHROW(weberpenntree.setBranchRecursionLevel(0));
    DOCTEST_CHECK_NOTHROW(weberpenntree.setBranchRecursionLevel(1));
    DOCTEST_CHECK_NOTHROW(weberpenntree.setBranchRecursionLevel(2));
}

DOCTEST_TEST_CASE("WeberPennTree Parameter Validation") {

    // These checks deliberately build no geometry: an invalid parameter set must be rejected by
    // setTreeParameters before it can reach buildTree. Previously setTreeParameters performed no
    // validation at all, so a degenerate parameter set produced a silently empty or malformed tree
    // instead of an error.
    Context context;
    WeberPennTree weberpenntree(&context);
    weberpenntree.disableMessages();

    WeberPennTreeParameters valid_parameters = weberpenntree.getTreeParameters("Lemon");

    // The unmodified library parameters must remain acceptable.
    DOCTEST_CHECK_NOTHROW(weberpenntree.setTreeParameters("Lemon", valid_parameters));

    // A zero nCurveRes is a divisor in the trunk and branch segment-length calculations. It used to
    // produce a tree with zero primitives and no diagnostic.
    {
        WeberPennTreeParameters parameters = valid_parameters;
        parameters.nCurveRes.at(0) = 0;
        DOCTEST_CHECK_THROWS_AS(weberpenntree.setTreeParameters("Lemon", parameters), std::runtime_error);
    }

    // nCurveRes must also be positive for every level that is actually recursed into, not just the trunk.
    {
        WeberPennTreeParameters parameters = valid_parameters;
        parameters.nCurveRes.at(1) = 0;
        DOCTEST_CHECK_THROWS_AS(weberpenntree.setTreeParameters("Lemon", parameters), std::runtime_error);
    }

    // Levels indexes the fixed-size-4 parameter arrays. Levels==4 made the leaf path index
    // nDownAngle.at(4), which silently produced a tree with no leaves at all.
    {
        WeberPennTreeParameters parameters = valid_parameters;
        parameters.Levels = 4;
        DOCTEST_CHECK_THROWS_AS(weberpenntree.setTreeParameters("Lemon", parameters), std::runtime_error);
    }

    // Levels beyond the array bounds would throw a bare std::out_of_range from deep inside the recursion.
    {
        WeberPennTreeParameters parameters = valid_parameters;
        parameters.Levels = 5;
        DOCTEST_CHECK_THROWS_AS(weberpenntree.setTreeParameters("Lemon", parameters), std::runtime_error);
    }

    // A tree with no levels has no branches or leaves to generate.
    {
        WeberPennTreeParameters parameters = valid_parameters;
        parameters.Levels = 0;
        DOCTEST_CHECK_THROWS_AS(weberpenntree.setTreeParameters("Lemon", parameters), std::runtime_error);
    }

    // A parameter array shorter than the number of levels cannot be indexed safely.
    {
        WeberPennTreeParameters parameters = valid_parameters;
        parameters.nDownAngle.resize(2);
        DOCTEST_CHECK_THROWS_AS(weberpenntree.setTreeParameters("Lemon", parameters), std::runtime_error);
    }

    // Negative or zero overall scale collapses the tree to a point.
    {
        WeberPennTreeParameters parameters = valid_parameters;
        parameters.Scale = 0;
        DOCTEST_CHECK_THROWS_AS(weberpenntree.setTreeParameters("Lemon", parameters), std::runtime_error);
    }

    // A rejected parameter set must not be left behind in the library.
    WeberPennTreeParameters parameters_after = weberpenntree.getTreeParameters("Lemon");
    DOCTEST_CHECK(parameters_after.Levels == valid_parameters.Levels);
    DOCTEST_CHECK(parameters_after.nCurveRes.at(0) == valid_parameters.nCurveRes.at(0));
    DOCTEST_CHECK(parameters_after.Scale == valid_parameters.Scale);
}

DOCTEST_TEST_CASE("WeberPennTree Leaf Subdivisions") {

    Context context;
    WeberPennTree weberpenntree(&context);

    // Test setting different leaf subdivisions
    DOCTEST_CHECK_NOTHROW(weberpenntree.setLeafSubdivisions(make_int2(1, 1)));
    DOCTEST_CHECK_NOTHROW(weberpenntree.setLeafSubdivisions(make_int2(3, 3)));
    DOCTEST_CHECK_NOTHROW(weberpenntree.setLeafSubdivisions(make_int2(5, 5)));
}

DOCTEST_TEST_CASE("WeberPennTree Scaled Trees") {

    Context context;
    WeberPennTree weberpenntree(&context);

    // Test building trees with different scales
    DOCTEST_CHECK_NOTHROW(weberpenntree.buildTree("Apple", make_vec3(0, 0, 0), 0.5f));
    DOCTEST_CHECK_NOTHROW(weberpenntree.buildTree("Apple", make_vec3(5, 0, 0), 1.0f));
    DOCTEST_CHECK_NOTHROW(weberpenntree.buildTree("Apple", make_vec3(10, 0, 0), 1.5f));

    DOCTEST_CHECK(!context.getAllUUIDs().empty());
}

DOCTEST_TEST_CASE("WeberPennTree UUID Getters After Primitive Deletion") {

    Context context;
    WeberPennTree weberpenntree(&context);
    weberpenntree.disableMessages();

    weberpenntree.setBranchRecursionLevel(1);
    weberpenntree.setLeafSubdivisions(make_int2(1, 1));

    uint TreeID = weberpenntree.buildTree("Lemon", make_vec3(0, 0, 0));

    std::vector<uint> leaf_UUIDs = weberpenntree.getLeafUUIDs(TreeID);
    DOCTEST_CHECK(!leaf_UUIDs.empty());

    // Delete all leaf primitives directly through the Context, which leaves the
    // UUIDs cached inside the plug-in pointing at primitives that no longer exist.
    DOCTEST_CHECK_NOTHROW(context.deletePrimitive(leaf_UUIDs));

    // The getters must not report UUIDs of deleted primitives.
    std::vector<uint> leaf_UUIDs_after = weberpenntree.getLeafUUIDs(TreeID);
    DOCTEST_CHECK(leaf_UUIDs_after.empty());

    // Trunk and branch primitives were not deleted, so those getters should be unaffected.
    std::vector<uint> trunk_UUIDs_after = weberpenntree.getTrunkUUIDs(TreeID);
    DOCTEST_CHECK(!trunk_UUIDs_after.empty());
    DOCTEST_CHECK(context.doesPrimitiveExist(trunk_UUIDs_after));

    std::vector<uint> branch_UUIDs_after = weberpenntree.getBranchUUIDs(TreeID);
    DOCTEST_CHECK(context.doesPrimitiveExist(branch_UUIDs_after));

    // getAllUUIDs is what most users hand off to other plug-ins, so it must contain
    // only live UUIDs. Passing a dangling UUID to the Context throws.
    std::vector<uint> all_UUIDs_after = weberpenntree.getAllUUIDs(TreeID);
    DOCTEST_CHECK(!all_UUIDs_after.empty());
    DOCTEST_CHECK(context.doesPrimitiveExist(all_UUIDs_after));
    for (uint UUID: all_UUIDs_after) {
        DOCTEST_CHECK_NOTHROW(std::ignore = context.getPrimitiveArea(UUID));
    }

    // Deleting the remainder of the tree should leave every getter empty rather than
    // returning stale UUIDs.
    DOCTEST_CHECK_NOTHROW(context.deletePrimitive(all_UUIDs_after));
    DOCTEST_CHECK(weberpenntree.getTrunkUUIDs(TreeID).empty());
    DOCTEST_CHECK(weberpenntree.getBranchUUIDs(TreeID).empty());
    DOCTEST_CHECK(weberpenntree.getAllUUIDs(TreeID).empty());

    // Every getter must reject an out-of-range tree ID with a helios_runtime_error rather than
    // letting a std::vector::at() escape from inside the plug-in. getAllUUIDs is the case that
    // matters here: it indexes UUID_trunk and UUID_branch but historically bounds-checked only
    // UUID_leaf, so a tree ID valid for one container but not the others slipped past the guard.
    uint invalid_TreeID = TreeID + 1;
    DOCTEST_CHECK_THROWS_AS(std::ignore = weberpenntree.getTrunkUUIDs(invalid_TreeID), std::runtime_error);
    DOCTEST_CHECK_THROWS_AS(std::ignore = weberpenntree.getBranchUUIDs(invalid_TreeID), std::runtime_error);
    DOCTEST_CHECK_THROWS_AS(std::ignore = weberpenntree.getLeafUUIDs(invalid_TreeID), std::runtime_error);
    DOCTEST_CHECK_THROWS_AS(std::ignore = weberpenntree.getAllUUIDs(invalid_TreeID), std::runtime_error);
}

DOCTEST_TEST_CASE("WeberPennTree Single Branch Per Trunk Segment") {

    // Regression test for a division by zero in the trunk's branch-offset calculation. The offset of
    // each first-level branch along the trunk was computed with a float(s+0.5)/float(stems_per_segment-1)
    // term, which is a division by zero whenever exactly one branch is placed per trunk segment
    // (nBranches[1] close to nCurveRes[0]). The resulting infinity propagated into recursiveBranch as
    // offset_child, where it made the position ratio -inf; the ratio was then clamped to 0, so every
    // branch was built at the length corresponding to the very base of the tree instead of its true
    // height. No NaN ever reached the geometry, which is why the failure was invisible.
    //
    // The observable is the mean surface area of the first-level branch primitives, which is
    // proportional to branch length. Shape 1 (spherical) is used because it maximizes the effect:
    // ShapeRatio is 0.2 at ratio 0 but averages about 0.735 across the crown, so clamping every
    // branch to ratio 0 shortens them by a factor of roughly 3.7. Lemon's own shape (7, tend flame)
    // only differs by a factor of 1.5, which is too small a margin to test against reliably.
    //
    // Branch z-extent is deliberately NOT used as the observable: the attachment points along the
    // trunk are computed correctly even with the bug, so the extent barely moves and masks the error.
    //
    // The comparison is against the same tree built with two branches per trunk segment, which
    // avoids the degenerate divisor entirely while keeping the shape function identical. Expressing
    // the check as a ratio between two builds cancels the tree geometry and the random variation,
    // leaving only the effect of the bug.
    auto build_and_measure_branch_area = [](uint branches_per_segment, float &area_out, uint &branch_count_out) {
        Context context;
        WeberPennTree weberpenntree(&context);
        weberpenntree.disableMessages();

        // Only the trunk and the first branch level are needed, which keeps these trees cheap to build.
        weberpenntree.setBranchRecursionLevel(1);
        weberpenntree.setLeafSubdivisions(make_int2(1, 1));

        WeberPennTreeParameters parameters = weberpenntree.getTreeParameters("Lemon");
        parameters.Shape = 1; // spherical, which maximizes the length error caused by the bug
        parameters.BaseSplits = 0; // take the non-splitting trunk path, which is where the bug lives
        parameters.BaseSplitsV = 0;
        parameters.nCurveRes.at(0) = 5;
        parameters.nBranches.at(1) = 5 * branches_per_segment;
        parameters.nLengthV.at(1) = 0; // remove length variation so the comparison is deterministic
        parameters.nCurveV.at(0) = 0;
        parameters.Levels = 2;
        weberpenntree.setTreeParameters("Lemon", parameters);

        uint TreeID = weberpenntree.buildTree("Lemon", make_vec3(0, 0, 0));

        std::vector<uint> branch_UUIDs = weberpenntree.getBranchUUIDs(TreeID);
        branch_count_out = branch_UUIDs.size();

        area_out = 0.f;
        for (uint UUID: branch_UUIDs) {
            for (const vec3 &vertex: context.getPrimitiveVertices(UUID)) {
                DOCTEST_REQUIRE(std::isfinite(vertex.x));
                DOCTEST_REQUIRE(std::isfinite(vertex.y));
                DOCTEST_REQUIRE(std::isfinite(vertex.z));
            }
            area_out += context.getPrimitiveArea(UUID);
        }
    };

    // One branch per trunk segment: stems/nCurveRes == 1, which is the degenerate divisor.
    float area_one_per_segment = 0.f;
    uint count_one_per_segment = 0;
    build_and_measure_branch_area(1, area_one_per_segment, count_one_per_segment);

    // Two branches per trunk segment: the divisor is 1, so the offset calculation is well defined.
    float area_two_per_segment = 0.f;
    uint count_two_per_segment = 0;
    build_and_measure_branch_area(2, area_two_per_segment, count_two_per_segment);

    DOCTEST_REQUIRE(count_one_per_segment > 0);
    DOCTEST_REQUIRE(count_two_per_segment > 0);

    // Normalize by the number of branch primitives so the two builds are directly comparable.
    float area_per_branch_one = area_one_per_segment / float(count_one_per_segment);
    float area_per_branch_two = area_two_per_segment / float(count_two_per_segment);

    // Both builds sample the same shape function over the same trunk, so their mean branch areas must
    // agree closely. With the division by zero, the one-per-segment build has every branch clamped to
    // ShapeRatio(1,0)=0.2 while the other averages about 0.735, so the ratio below collapses to
    // roughly 0.3. The 0.7 threshold sits well clear of both outcomes.
    DOCTEST_CHECK(area_per_branch_one > 0.7f * area_per_branch_two);
}

DOCTEST_TEST_CASE("WeberPennTree Leaf Angle Distribution") {

    // Build a tree whose leaf inclination distribution is prescribed through the <LeafAngleDist> tag,
    // then measure the realized distribution from the leaf normals and check it reproduces the input.
    //
    // The prescribed distribution is bimodal (neither spherical nor uniform) and is deliberately
    // confined to inclinations below 90 degrees. The sampled inclination is applied as a rotation of
    // the leaf about the y-axis, so an inclination of theta and one of pi-theta yield leaf normals
    // with the same |n.z| and cannot be told apart by the measurement below. Keeping all of the
    // probability mass below 90 degrees makes the measured angle equal to the sampled angle, so the
    // measured histogram can be compared bin-for-bin against the input.
    const uint Nbins = 12;
    const float dTheta = float(M_PI) / float(Nbins); // 15 degrees

    // Probability densities (units of 1/rad) for each of the 12 bins spanning 0 to pi.
    const std::vector<float> leaf_angle_PDF = {0.2809f, 1.1234f, 0.3370f, 0.2247f, 0.6741f, 1.1796f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    // The plug-in requires sum(g_L*dTheta) == 1 to within 0.001, otherwise it discards the
    // distribution and silently reverts to its default leaf placement.
    float PDF_integral = 0.f;
    for (float density: leaf_angle_PDF) {
        PDF_integral += density * dTheta;
    }
    DOCTEST_REQUIRE(PDF_integral == doctest::Approx(1.f).epsilon(0.001));

    // Write a temporary tree library containing a single tree that uses this distribution.
    std::string xml_path = "plugins/weberpenntree/xml/.tmp_leafangle_test.xml";
    {
        std::ofstream xml_file(xml_path);
        DOCTEST_REQUIRE(xml_file.good());
        xml_file << "<helios>\n";
        xml_file << "<WeberPennTree label=\"LeafAngleTest\">\n";
        xml_file << "  <Shape> 1 </Shape>\n";
        xml_file << "  <BaseSize> 0.4 </BaseSize>\n";
        xml_file << "  <BaseSplits> 3 </BaseSplits>\n";
        xml_file << "  <BaseSplitSize> 0.1 </BaseSplitSize>\n";
        xml_file << "  <Scale> 6 </Scale>\n";
        xml_file << "  <ScaleV> 0 </ScaleV>\n";
        xml_file << "  <ZScale> 1 </ZScale>\n";
        xml_file << "  <ZScaleV> 0 </ZScaleV>\n";
        xml_file << "  <Ratio> 0.02 </Ratio>\n";
        xml_file << "  <RatioPower> 0.5 </RatioPower>\n";
        xml_file << "  <Lobes> 7 </Lobes>\n";
        xml_file << "  <LobeDepth> 0.1 </LobeDepth>\n";
        xml_file << "  <Flare> 0.8 </Flare>\n";
        xml_file << "  <Levels> 3 </Levels>\n";
        xml_file << "  <nSegSplits> 1 0 0 0 </nSegSplits>\n";
        xml_file << "  <nSplitAngle> 10 30 0 0 </nSplitAngle>\n";
        xml_file << "  <nSplitAngleV> 0 0 0 0 </nSplitAngleV>\n";
        xml_file << "  <nCurveRes> 5 5 5 0 </nCurveRes>\n";
        xml_file << "  <nCurve> 30 40 -40 0 </nCurve>\n";
        xml_file << "  <nCurveV> 10 10 10 0 </nCurveV>\n";
        xml_file << "  <nCurveBack> 0 0 0 0 </nCurveBack>\n";
        xml_file << "  <nLength> 1 0.4 0.3 0 </nLength>\n";
        xml_file << "  <nLengthV> 0 0.05 0.05 0 </nLengthV>\n";
        xml_file << "  <nTaper> 1 1 1 0 </nTaper>\n";
        xml_file << "  <nDownAngle> 60 60 60 100 </nDownAngle>\n";
        xml_file << "  <nDownAngleV> 0 0 0 0 </nDownAngleV>\n";
        xml_file << "  <nRotate> 140 140 180 140 </nRotate>\n";
        xml_file << "  <nRotateV> 5 5 40 20 </nRotateV>\n";
        xml_file << "  <nBranches> 0 40 30 0 </nBranches>\n";
        xml_file << "  <Leaves> 10 </Leaves>\n";
        xml_file << "  <LeafFile> plugins/weberpenntree/leaves/AvocadoLeaf.png </LeafFile>\n";
        xml_file << "  <LeafScale> 0.2 </LeafScale>\n";
        xml_file << "  <LeafScaleX> 0.2 </LeafScaleX>\n";
        xml_file << "  <WoodFile> plugins/weberpenntree/wood/wood.jpg </WoodFile>\n";
        xml_file << "  <LeafAngleDist>";
        for (float density: leaf_angle_PDF) {
            xml_file << " " << density;
        }
        xml_file << " </LeafAngleDist>\n";
        xml_file << "</WeberPennTree>\n";
        xml_file << "</helios>\n";
    } // ofstream closed here so the file is complete before it is read back

    Context context;
    // The leaf angle is drawn using the Context's random number generator (not the plug-in's), so
    // seeding the Context is what makes this test reproducible.
    context.seedRandomGenerator(9137);

    WeberPennTree weberpenntree(&context);
    weberpenntree.disableMessages();

    // One patch per leaf, so that each leaf contributes exactly one sample to the histogram.
    weberpenntree.setLeafSubdivisions(make_int2(1, 1));

    DOCTEST_CHECK_NOTHROW(weberpenntree.loadXML(xml_path.c_str(), true));

    uint TreeID = 0;
    DOCTEST_CHECK_NOTHROW(TreeID = weberpenntree.buildTree("LeafAngleTest", make_vec3(0, 0, 0)));

    std::vector<uint> leaf_UUIDs = weberpenntree.getLeafUUIDs(TreeID);

    std::remove(xml_path.c_str());

    // A large sample is needed to keep the sampling noise well below the comparison tolerance.
    DOCTEST_REQUIRE(leaf_UUIDs.size() > 1000);

    // Bin the leaf inclination angles. The inclination of a leaf is the angle of its normal from
    // vertical, folded into the upper hemisphere since the normal sign carries no angle information.
    std::vector<float> measured_distribution(Nbins, 0.f);
    for (uint UUID: leaf_UUIDs) {
        vec3 normal = context.getPrimitiveNormal(UUID);
        float theta = acos_safe(fabsf(normal.z));
        uint bin = uint(floorf(theta / dTheta));
        if (bin >= Nbins) { // guards theta exactly equal to pi/2
            bin = Nbins - 1;
        }
        measured_distribution.at(bin) += 1.f;
    }
    for (float &bin_fraction: measured_distribution) {
        bin_fraction /= float(leaf_UUIDs.size());
    }

    // The prescribed distribution has no probability mass at or above 90 degrees, so no leaf may be
    // binned there. Note that a distribution failing the plug-in's normalization check is discarded
    // at load time and leaf placement silently falls back to the default branch-relative
    // orientation; that fallback is caught by the bin-by-bin comparison below.
    for (uint i = Nbins / 2; i < Nbins; i++) {
        DOCTEST_REQUIRE(measured_distribution.at(i) == 0.f);
    }

    // Compare the measured bin probabilities against the prescribed ones. The tolerance is set
    // comfortably above the binomial sampling noise (about 0.044 at three standard deviations for
    // the sample size required above) while remaining far smaller than the bin probabilities
    // themselves, which range up to 0.31.
    const float probability_tolerance = 0.05f;
    for (uint i = 0; i < Nbins; i++) {
        float expected_probability = leaf_angle_PDF.at(i) * dTheta;
        DOCTEST_CHECK(fabsf(measured_distribution.at(i) - expected_probability) < probability_tolerance);
    }
}

DOCTEST_TEST_CASE("WeberPennTree Leaf Angle Distribution Normalization Error") {

    // A leaf angle distribution that does not integrate to 1 cannot be sampled correctly, so it must
    // be reported as an error rather than being discarded in favor of the default leaf placement.
    // The distribution below integrates to 1.5.
    std::string xml_path = "plugins/weberpenntree/xml/.tmp_leafangle_badnorm_test.xml";
    {
        std::ofstream xml_file(xml_path);
        DOCTEST_REQUIRE(xml_file.good());
        xml_file << "<helios>\n";
        xml_file << "<WeberPennTree label=\"LeafAngleBadNorm\">\n";
        xml_file << "  <Shape> 1 </Shape>\n";
        xml_file << "  <BaseSize> 0.4 </BaseSize>\n";
        xml_file << "  <BaseSplits> 3 </BaseSplits>\n";
        xml_file << "  <BaseSplitSize> 0.1 </BaseSplitSize>\n";
        xml_file << "  <Scale> 6 </Scale>\n";
        xml_file << "  <ScaleV> 0 </ScaleV>\n";
        xml_file << "  <ZScale> 1 </ZScale>\n";
        xml_file << "  <ZScaleV> 0 </ZScaleV>\n";
        xml_file << "  <Ratio> 0.02 </Ratio>\n";
        xml_file << "  <RatioPower> 0.5 </RatioPower>\n";
        xml_file << "  <Lobes> 7 </Lobes>\n";
        xml_file << "  <LobeDepth> 0.1 </LobeDepth>\n";
        xml_file << "  <Flare> 0.8 </Flare>\n";
        xml_file << "  <Levels> 3 </Levels>\n";
        xml_file << "  <nSegSplits> 1 0 0 0 </nSegSplits>\n";
        xml_file << "  <nSplitAngle> 10 30 0 0 </nSplitAngle>\n";
        xml_file << "  <nSplitAngleV> 0 0 0 0 </nSplitAngleV>\n";
        xml_file << "  <nCurveRes> 5 5 5 0 </nCurveRes>\n";
        xml_file << "  <nCurve> 30 40 -40 0 </nCurve>\n";
        xml_file << "  <nCurveV> 10 10 10 0 </nCurveV>\n";
        xml_file << "  <nCurveBack> 0 0 0 0 </nCurveBack>\n";
        xml_file << "  <nLength> 1 0.4 0.3 0 </nLength>\n";
        xml_file << "  <nLengthV> 0 0.05 0.05 0 </nLengthV>\n";
        xml_file << "  <nTaper> 1 1 1 0 </nTaper>\n";
        xml_file << "  <nDownAngle> 60 60 60 100 </nDownAngle>\n";
        xml_file << "  <nDownAngleV> 0 0 0 0 </nDownAngleV>\n";
        xml_file << "  <nRotate> 140 140 180 140 </nRotate>\n";
        xml_file << "  <nRotateV> 5 5 40 20 </nRotateV>\n";
        xml_file << "  <nBranches> 0 40 30 0 </nBranches>\n";
        xml_file << "  <Leaves> 10 </Leaves>\n";
        xml_file << "  <LeafFile> plugins/weberpenntree/leaves/AvocadoLeaf.png </LeafFile>\n";
        xml_file << "  <LeafScale> 0.2 </LeafScale>\n";
        xml_file << "  <LeafScaleX> 0.2 </LeafScaleX>\n";
        xml_file << "  <WoodFile> plugins/weberpenntree/wood/wood.jpg </WoodFile>\n";
        xml_file << "  <LeafAngleDist> 0.4214 1.6851 0.5055 0.3371 1.0112 1.7694 0 0 0 0 0 0 </LeafAngleDist>\n";
        xml_file << "</WeberPennTree>\n";
        xml_file << "</helios>\n";
    }

    Context context;
    WeberPennTree weberpenntree(&context);
    weberpenntree.disableMessages();

    // Loading must fail rather than quietly reverting to the default leaf placement, and it must not
    // write to stdout/stderr while doing so.
    std::string captured_output;
    std::string captured_error;
    {
        capture_cout cout_capture;
        capture_cerr cerr_capture;
        DOCTEST_CHECK_THROWS(weberpenntree.loadXML(xml_path.c_str(), true));
        captured_output = cout_capture.get_captured_output();
        captured_error = cerr_capture.get_captured_output();
    }

    std::remove(xml_path.c_str());

    DOCTEST_CHECK(captured_output.empty());
    DOCTEST_CHECK(captured_error.empty());

    // The rejected tree must not be left behind in the library in a partially-initialized state.
    DOCTEST_CHECK_THROWS(std::ignore = weberpenntree.getTreeParameters("LeafAngleBadNorm"));
}

int WeberPennTree::selfTest(int argc, char **argv) {
    return helios::runDoctestWithValidation(argc, argv);
}
