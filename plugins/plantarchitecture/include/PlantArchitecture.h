/** \file "PlantArchitecture.h" Primary header file for plant architecture plug-in.

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

#ifndef PLANT_ARCHITECTURE
#define PLANT_ARCHITECTURE

#include <functional>
#include <utility>
#include "Context.h"
#include "Hungarian.h"

// Constants
constexpr float C_molecular_wt = 12.01; // g C mol^-1

// forward declarations of classes/structs
class PlantArchitecture;
class CollisionDetection;
struct Shoot;
struct Phytomer;
struct PlantSpatialStateVisitor;
struct PlantPoseState;
struct PoseGroupReference;
struct PoseRigidTransform;

//! Function that builds a fruit prototype and returns its object ID
/**
 * Arguments are the Context, the number of subdivisions, and the organ build parameters of the plant bearing the fruit: the entries of the map passed to
 * \ref PlantArchitecture::buildPlantInstanceFromLibrary() that the plant model declares as shaping its flowers and fruit (see \ref PlantArchBuildParamsValidation). The map is empty for a
 * plant not built from the library. A prototype function reads the keys it understands and ignores the rest.
 * \note A custom prototype function assigned to a library model receives only that model's declared organ parameters. It cannot be given keys of its own, because
 * \ref PlantArchitecture::buildPlantInstanceFromLibrary() rejects any name the model does not declare.
 */
using FruitPrototypeFunction = uint (*)(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters);

//! Function that builds a flower prototype and returns its object ID
/**
 * Arguments are the Context, the number of subdivisions, whether the flower is open, and the build parameters of the plant bearing the flower. See \ref FruitPrototypeFunction.
 */
using FlowerPrototypeFunction = uint (*)(helios::Context *context_ptr, uint subdivisions, bool flower_is_open, const std::map<std::string, float> &build_parameters);

//! Random architecture model parameter value of type float.
struct RandomParameter_float {
public:
    //! Constructor initializing to a constant default value of 0.
    /**
     * In order to make this a randomly varying parameter, the initialize() method must be called to set the random number generator.
     */
    explicit RandomParameter_float() {
        constval = 0.f;
        distribution = "constant";
        generator = nullptr;
        sampled = false;
    }

    //! Constructor initializing to a constant value.
    /**
     * In order to make this a randomly varying parameter, the initialize() method must be called to set the random number generator.
     */
    explicit RandomParameter_float(float val) {
        constval = val;
        distribution = "constant";
        generator = nullptr;
        sampled = false;
    }

    //! Constructor initializing the random number generator.
    /**
     * \param[in] rand_generator Pointer to a random number generator. Note: it is recommended to use the random number generator from the Context, which can be retrieved using the getContextRandomGenerator() method.
     */
    explicit RandomParameter_float(std::minstd_rand0 *rand_generator) {
        constval = 0.f;
        distribution = "constant";
        generator = rand_generator;
        sampled = false;
    }

    void initialize(float a_val, std::minstd_rand0 *rand_generator) {
        constval = a_val;
        distribution = "constant";
        generator = rand_generator;
        sampled = false;
    }

    void initialize(std::minstd_rand0 *rand_generator) {
        constval = 1.f;
        distribution = "constant";
        generator = rand_generator;
        sampled = false;
    }

    //! Draw subsequent values of this parameter from a different random number generator
    /**
     * The distribution and its parameters are left as they are; only the source of randomness changes. Any value
     * already drawn is discarded, so that the next call to val() samples from the new generator rather than
     * returning a value that came from the old one.
     * \param[in] rand_generator Pointer to the random number generator to draw from.
     */
    void setRandomGenerator(std::minstd_rand0 *rand_generator) {
        generator = rand_generator;
        sampled = false;
    }

    RandomParameter_float &operator=(float a) {
        this->distribution = "constant";
        this->constval = a;
        this->sampled = false;
        return *this;
    }

    //! Sample the parameter from a uniform distribution
    /**
     * \param[in] minval Minimum value of the distribution.
     * \param[in] maxval Maximum value of the distribution.
     */
    void uniformDistribution(float minval, float maxval) {
        if (minval > maxval) {
            throw(std::runtime_error("ERROR (PlantArchitecture): RandomParameter_float::uniformDistribution() - minval must be less than or equal to maxval."));
        }
        distribution = "uniform";
        distribution_parameters = {minval, maxval};
        sampled = false;
    }

    //! Sample the parameter from a normal distribution
    /**
     * \param[in] mean Mean of the distribution.
     * \param[in] std_dev Standard deviation of the distribution.
     */
    void normalDistribution(float mean, float std_dev) {
        distribution = "normal";
        distribution_parameters = {mean, std_dev};
        sampled = false;
    }

    //! Sample the parameter from a Weibull distribution
    /**
     * \param[in] shape Shape parameter of the distribution.
     * \param[in] scale Scale parameter of the distribution.
     */
    void weibullDistribution(float shape, float scale) {
        distribution = "weibull";
        distribution_parameters = {shape, scale};
        sampled = false;
    }

    //! Value of the parameter, sampling it on first use.
    /** Const because reading a parameter is not a mutation of the model: the sample is a lazily filled cache,
     * so the cache fields are mutable and resample() is const. This lets code that legitimately holds a
     * ShootParameters by const reference -- Shoot::sampleVegetativeBudBreak(), for one -- read its own
     * parameters instead of reaching through the plant instance for the shared per-type snapshot.
     */
    float val() const {
        if (!sampled) {
            constval = resample();
        }
        return constval;
    }

    float resample() const {
        sampled = true;
        if (distribution != "constant") {
            if (generator == nullptr) {
                throw(std::runtime_error("ERROR (PlantArchitecture): Random parameter was not properly initialized with random number generator."));
            }
            if (distribution == "uniform") {
                helios::uniform_real_distribution<float> unif_distribution;
                constval = distribution_parameters.at(0) + unif_distribution(*generator) * (distribution_parameters.at(1) - distribution_parameters.at(0));
            } else if (distribution == "normal") {
                helios::normal_distribution<float> norm_distribution(distribution_parameters.at(0), distribution_parameters.at(1));
                constval = norm_distribution(*generator);
            } else if (distribution == "weibull") {
                helios::weibull_distribution<float> wbull_distribution(distribution_parameters.at(0), distribution_parameters.at(1));
                constval = wbull_distribution(*generator);
            }
        }
        return constval;
    }

    std::string distribution;
    std::vector<float> distribution_parameters;

private:
    //! Lazily filled sample cache. Mutable so that val() can be const -- see the note there.
    mutable bool sampled;
    mutable float constval;


    std::minstd_rand0 *generator;
};

//! Random architecture model parameter value of type int.
struct RandomParameter_int {
public:
    explicit RandomParameter_int() {
        constval = 1;
        distribution = "constant";
        generator = nullptr;
        sampled = false;
    }

    void initialize(int a_val, std::minstd_rand0 *rand_generator) {
        constval = a_val;
        distribution = "constant";
        generator = rand_generator;
        sampled = false;
    }

    void initialize(std::minstd_rand0 *rand_generator) {
        constval = 1;
        distribution = "constant";
        generator = rand_generator;
        sampled = false;
    }

    RandomParameter_int &operator=(int a) {
        this->distribution = "constant";
        this->constval = a;
        this->sampled = false;
        return *this;
    }

    //! Sample the parameter from a uniform distribution
    /**
     * \param[in] minval Minimum value of the distribution.
     * \param[in] maxval Maximum value of the distribution.
     */
    void uniformDistribution(int minval, int maxval) {
        if (minval > maxval) {
            throw(std::runtime_error("ERROR (PlantArchitecture): RandomParameter_int::uniformDistribution() - minval must be less than or equal to maxval."));
        }
        distribution = "uniform";
        distribution_parameters = {minval, maxval};
        sampled = false;
    }

    //! Sample the parameter from a set of discrete values, each chosen with equal probability
    /**
     * \param[in] values Vector of discrete values that may be chosen.
     */
    void discreteValues(const std::vector<int> &values) {
        distribution = "discretevalues";
        distribution_parameters = values;
        sampled = false;
    }

    //! Get the value of the parameter, sampling from its distribution if it has not already been sampled
    /**
     * \return Current value of the parameter.
     */
    int val() {
        if (!sampled) {
            constval = resample();
        }
        return constval;
    }

    int resample() {
        sampled = true;
        if (distribution != "constant") {
            if (generator == nullptr) {
                throw(std::runtime_error("ERROR (PlantArchitecture): Random parameter was not properly initialized with random number generator."));
            }
            if (distribution == "uniform") {
                helios::uniform_int_distribution<> unif_distribution(distribution_parameters.at(0), distribution_parameters.at(1));
                constval = unif_distribution(*generator);
            } else if (distribution == "discretevalues") {
                helios::uniform_int_distribution<> unif_distribution(0, distribution_parameters.size() - 1);
                constval = distribution_parameters.at(unif_distribution(*generator));
            }
        }
        return constval;
    }

    std::string distribution;
    std::vector<int> distribution_parameters;

private:
    bool sampled;
    int constval;
    std::minstd_rand0 *generator;
};

//! Probability distribution of a build parameter of a library plant model
/**
 * Given to \ref PlantArchitecture::buildPlantInstanceFromLibrary() or \ref PlantArchitecture::buildPlantCanopyFromLibrary() in place of a fixed value, so that plants built in one call, or in separate
 * calls with the same distributions, differ from one another, for example in their number of tillers. A parameter that shapes the plant, such as "tiller_count" or "trunk_height", is sampled once for
 * each plant. A parameter that shapes its flowers and fruit, such as "berry_diameter", is sampled once for each flower and fruit prototype, and plants given the same distribution share the same set of
 * prototypes. See \ref PlantArchBuildParamDistributions.
 *
 * The whole range of a distribution must lie within the valid range of its parameter, and a parameter that counts something takes only \ref constant() or \ref discrete() whole-number values. Both are
 * checked before any plant is built. A constant is checked, as a value passed without a distribution is, when the plant is built.
 */
class BuildParameterDistribution {
public:
    //! Kind of distribution
    enum class Type {
        //! A single fixed value
        CONSTANT,
        //! Uniform over a closed interval
        UNIFORM,
        //! Normal, truncated to an interval
        NORMAL,
        //! A set of values, each with a probability
        DISCRETE
    };

    //! A single fixed value, the same as passing the value itself
    /**
     * \param[in] value Value of the parameter. It is checked against the parameter's valid range when the plant is built.
     */
    [[nodiscard]] static BuildParameterDistribution constant(float value);

    //! Uniform distribution over a closed interval
    /**
     * \param[in] min_value Lower end of the interval.
     * \param[in] max_value Upper end of the interval. Must not be less than min_value.
     */
    [[nodiscard]] static BuildParameterDistribution uniform(float min_value, float max_value);

    //! Normal distribution truncated to an interval
    /**
     * Values outside the interval are redrawn. The interval is required because every build parameter has a bounded valid range that the distribution must lie within.
     * \param[in] mean Mean of the untruncated distribution.
     * \param[in] standard_deviation Standard deviation of the untruncated distribution. Must be positive.
     * \param[in] min_value Lower end of the interval.
     * \param[in] max_value Upper end of the interval. Must be greater than min_value, and the interval must hold at least 0.1% of the untruncated distribution.
     */
    [[nodiscard]] static BuildParameterDistribution normal(float mean, float standard_deviation, float min_value, float max_value);

    //! Discrete distribution over a set of values
    /**
     * \param[in] values Values the parameter may take. Must not be empty.
     * \param[in] weights [optional] Relative probability of each value, which need not sum to 1. Must be non-negative, with a positive sum, and as many as there are values. If not given, every value is
     * equally likely.
     */
    [[nodiscard]] static BuildParameterDistribution discrete(const std::vector<float> &values, const std::vector<float> &weights = {});

    //! Kind of distribution
    [[nodiscard]] Type getType() const {
        return type;
    }

    //! Parameters of the distribution
    /**
     * \return {value} for \ref Type::CONSTANT, {min, max} for \ref Type::UNIFORM, {mean, standard deviation, min, max} for \ref Type::NORMAL, and the values for \ref Type::DISCRETE.
     */
    [[nodiscard]] const std::vector<float> &getParameters() const {
        return parameters;
    }

    //! Weights of the values of a \ref Type::DISCRETE distribution, or an empty vector if the values are equally likely or the distribution is not discrete
    [[nodiscard]] const std::vector<float> &getWeights() const {
        return weights;
    }

    //! Smallest value the distribution can take
    [[nodiscard]] float getMinimum() const;

    //! Largest value the distribution can take
    [[nodiscard]] float getMaximum() const;

    //! Draw a value from the distribution. A constant is returned without drawing from the generator.
    /**
     * \param[in] generator Random number generator to draw from.
     * \return Sampled value.
     */
    [[nodiscard]] float sample(std::minstd_rand0 &generator) const;

    bool operator==(const BuildParameterDistribution &other) const {
        return type == other.type && parameters == other.parameters && weights == other.weights;
    }

    bool operator!=(const BuildParameterDistribution &other) const {
        return !(*this == other);
    }

    //! Ordering by type and then parameters, so that a map of distributions can be used as a key
    bool operator<(const BuildParameterDistribution &other) const {
        if (type != other.type) {
            return type < other.type;
        }
        if (parameters != other.parameters) {
            return parameters < other.parameters;
        }
        return weights < other.weights;
    }

private:
    BuildParameterDistribution(Type type, std::vector<float> parameters, std::vector<float> weights) : type(type), parameters(std::move(parameters)), weights(std::move(weights)) {
    }

    Type type;
    std::vector<float> parameters;
    std::vector<float> weights;
};

//! Name, valid range and kind of a build parameter declared by a library plant model
/**
 * Each declaration is the single statement of a parameter's range: the code that reads the parameter checks the value it is given against it, and \ref PlantArchitecture::buildPlantInstanceFromLibrary()
 * checks the whole range of a \ref BuildParameterDistribution against it before any plant is built.
 */
struct BuildParameterDeclaration {
    //! Name of the parameter, the key passed to \ref PlantArchitecture::buildPlantInstanceFromLibrary()
    const char *name;
    //! Smallest valid value
    float min_value;
    //! Largest valid value
    float max_value;
    //! Whether the parameter counts something and must be a whole number
    bool is_integer;
    //! Description of the parameter, quoted in error messages
    const char *description;
};

struct AxisRotation {
public:
    AxisRotation() {
        pitch = 0;
        yaw = 0;
        roll = 0;
        azimuth = 0;
        peduncle_axis = helios::make_vec3(0, 0, 1);
        gravity_fraction = 0;
        contact_tilt_axis = helios::make_vec3(0, 0, 1);
        contact_tilt_angle = 0;
    }

    AxisRotation(float a_pitch, float a_yaw, float a_roll) {
        pitch = a_pitch;
        yaw = a_yaw;
        roll = a_roll;
        azimuth = 0;
        peduncle_axis = helios::make_vec3(0, 0, 1);
        gravity_fraction = 0;
        contact_tilt_axis = helios::make_vec3(0, 0, 1);
        contact_tilt_angle = 0;
    }

    float pitch;
    float yaw;
    float roll;
    float azimuth; // azimuth rotation for inflorescences (aligns to peduncle orientation)
    helios::vec3 peduncle_axis; // direction vector of peduncle at attachment point (for compound yaw rotation)
    float gravity_fraction; // fraction of its angle from straight down by which a fruit is turned to hang, after all other rotations
    helios::vec3 contact_tilt_axis; // unit axis (world frame) of the rotation about the fruit's attachment point by which it has been tilted out of a branch (see PlantArchitecture::enableFruitBranchCollisionAvoidance())
    float contact_tilt_angle; // angle (radians) of that tilt; 0 when the fruit has never been tilted

    AxisRotation operator+(const AxisRotation &a) const;
    AxisRotation operator-(const AxisRotation &a) const;

    friend std::ostream &operator<<(std::ostream &os, const AxisRotation &rot) {
        return os << "AxisRotation<" << rot.pitch << ", " << rot.yaw << ", " << rot.roll << ">";
    }
};

inline AxisRotation make_AxisRotation(float a_pitch, float a_yaw, float a_roll) {
    return {a_pitch, a_yaw, a_roll};
}

inline AxisRotation AxisRotation::operator+(const AxisRotation &a) const {
    return {a.pitch + pitch, a.yaw + yaw, a.roll + roll};
}

inline AxisRotation AxisRotation::operator-(const AxisRotation &a) const {
    return {a.pitch - pitch, a.yaw - yaw, a.roll - roll};
}

enum BudState { BUD_DORMANT = 0, BUD_ACTIVE = 1, BUD_FLOWER_CLOSED = 2, BUD_FLOWER_OPEN = 3, BUD_FRUITING = 4, BUD_DEAD = 5 };

//! Carbon sub-model of the carbohydrate model: the set of crop-specific mechanisms a plant's carbon economy runs on, chosen by CarbohydrateParameters::submodel
enum CarbonSubmodel {
    //! Dicot tree: a volumetric fruit sink that is set and then dropped for lack of carbon (CarbonFruitParameters), a cambium that lays down pipe-model wood and sapwood rings in competition with
    //! storage, and branches judged on their own carbon balance at the end of each season (CarbonWoodParameters)
    CARBON_SUBMODEL_TREE = 0,
    //! Grass or grain crop: tiller outgrowth gated by carbon supply and tillers shed during stem elongation, a culm reserve that is remobilized to grain, roots allocated by stage, and a grain
    //! sink whose number is set by panicle growth in a critical period around flowering and then filled (CarbonGrassParameters)
    CARBON_SUBMODEL_GRASS = 1
};

//! Parameters of the volumetric fruit sink of the carbohydrate model, held in CarbohydrateParameters::fruit
/**
 * A fruit is a sink whose carbon is set by its mature volume and laid down along a dry-matter curve over its development period. Fruit is set in number and then dropped for lack of carbon
 * within a window after set. Defaults are for almond.
 */
struct CarbonFruitParameters {
    //! Dry mass per unit volume of the mature fruit (g DW m^-3)
    float density = 525000;
    //! Carbon content of fruit dry matter (g C / g DW; Grossman & DeJong 1994a)
    float carbon_percentage = 0.475;
    //! Growth respiration of fruit (mol C respired per mol C incorporated). Peach fruit is 0.18 (DeJong & Goudriaan 1989); the lipid-rich almond kernel raises the whole-fruit cost.
    float growth_respiration_fraction = 0.3f;
    //! Fraction of a fruit's development period (fruit set to maturity) at which half of its dry matter is in place. Fruit carbon is charged along this dry-matter curve rather than with fruit
    //! volume. Almond fruit reaches full size ~12 weeks after bloom and holds ~65% of its final dry weight by ~500 degree-days after bloom (~late April; DeJong-lab ALMOND model), i.e. ~45% of a
    //! ~150-day development period.
    float dry_matter_midpoint = 0.4f;
    //! Width of the logistic fruit dry-matter curve, as a fraction of the fruit development period
    float dry_matter_width = 0.1f;
    //! Fraction of a fruit's carbon that is laid down in a second, later phase with its own logistic curve (kernel fill in a nut). 0 leaves the single curve above. The almond kernel is about a
    //! third of fruit dry weight but, being lipid-rich, close to half of its construction carbon.
    float late_fraction = 0.f;
    //! Fraction of the fruit development period at which half of the late-phase carbon is in place
    float late_midpoint = 0.8f;
    //! Width of the late-phase logistic curve, as a fraction of the fruit development period
    float late_width = 0.06f;
    //! Days after fruit set during which fruit grows at its full potential, funded from reserves. Older fruit is built from current photosynthate only: it grows at its potential times the
    //! plant's fruit supply ratio, the photosynthesis left after maintenance, vegetative growth and reserve-funded fruit, relative to the older fruit's potential growth, averaged over
    //! CarbohydrateParameters::supply_averaging_days. Growth lost to a shortfall is not made up. Flowers and expanding leaves are built from stored carbon and fruit from current photosynthate
    //! (Hoch et al. 2013); in sweet cherry reserves reach fruit until ~14 days after full bloom (Ayala & Lang 2015).
    float reserve_funded_days = 14.f;
    //! Days after fruit set after which fruit no longer drops for lack of carbon. Almond drops most of its excess fruit ~20-60 days after full bloom (Lordan et al. 2021).
    float drop_window_end = 63;
    //! Fruit past its reserve-funded period, up to drop_window_end days after set, drops when the growth it has received since set falls below this fraction of its potential growth to date.
    //! The test is made on that fruit as a whole: the most recently set is dropped first until the carbon the young fruit have received, including what the dropped fruit had received, is
    //! at least this fraction of the potential of the fruit that remain. A brief shortfall after a period of full growth therefore drops nothing; a sustained one drops fruit until the rest can
    //! be fed. Apple fruitlets growing at under ~50-60% of the dominant fruit's rate abscise (Celton et al. 2014).
    float abscission_growth_ratio = 0.55f;
};

//! Parameters of the woody-perennial mechanisms of the carbohydrate model, the cambium and end-of-season branch shedding, held in CarbohydrateParameters::wood
struct CarbonWoodParameters {
    //! Rate (1/day) at which an internode's radius approaches its girth target (the pipe-model target, or the sapwood ring): each sub-step closes a fraction 1 - exp(-radial_growth_rate x dt) of
    //! the gap, so that the cambium lays down a ring over the season rather than at once. This growth is paid from photosynthate: it is multiplied by the cambium supply ratio, the share of the
    //! photosynthate left after maintenance, vegetative growth and fruit that radial growth receives in competition with storage, relative to its potential, averaged over
    //! CarbohydrateParameters::supply_averaging_days. Fruit outcompetes the cambium, and extra carbon does not raise wood growth above its potential (FACE: Koerner et al. 2005). 0 stops radial
    //! growth, which is how a caller confines it to a cambial season.
    float radial_growth_rate = 0.025f;
    //! Sapwood cross-sectional area per unit of leaf area it supplies (m^2 sapwood / m^2 leaf; the Huber value). Together with sapwood_functional_years, sets the ring each internode must lay down every
    //! season to replace the sapwood that becomes heartwood: its potential ring area is sapwood_area_per_leaf_area x (the season's peak live leaf area above it) / sapwood_functional_years. A
    //! mature tree thus keeps adding rings at constant leaf area, which the pipe-model girth target alone does not. Non-positive (the default) disables the ring.
    float sapwood_area_per_leaf_area = 0.f;
    //! Number of years an annual ring stays functional sapwood before it becomes heartwood (see sapwood_area_per_leaf_area)
    float sapwood_functional_years = 0.f;
    //! A branch -- a "proleptic" or "sylleptic" shoot with all of its descendants -- is shed at the end of a season when the net photosynthesis of its leaves over the preceding year is less than this multiple of the carbon it spent: the
    //! maintenance respiration of its wood plus the construction (with growth respiration) of its wood, leaves, flowers and fruit. See PlantArchitecture::shedCarbonStarvedBranches().
    float branch_shedding_carbon_ratio = 1.f;
};

//! Temperature response of plant development under the carbohydrate model, held in CarbohydrateParameters::development
/**
 * With the carbohydrate model enabled, a plant develops in thermal time rather than in calendar days: the phyllochron, the ages of phytomers, leaves and buds, the phenological thresholds
 * (PlantArchitecture::setPlantPhenologicalThresholds()), leaf expansion and maturation, and the reproductive windows of the sub-models all advance by developmental days, while
 * respiration, storage, transport and the other carbon fluxes run on calendar time. A developmental day is the thermal time accumulated at reference_temperature in one day, so every
 * parameter given in days is a duration at the reference temperature, and a plant held at that temperature develops exactly as it would in calendar days. The thermal time of an hour at
 * air temperature T (C) is f(T)/24 C day, where f rises linearly from 0 at base_temperature to (optimum_temperature - base_temperature) at optimum_temperature and falls linearly back to 0
 * at maximum_temperature. The air temperature is supplied with PlantArchitecture::accumulateThermalTime(). The plant's dormancy cycle keeps running in calendar days.
 * Defaults are for almond (growing degree hours; Anderson et al. 1986, Acta Horticulturae 184:87); CarbohydrateParameters::forSubmodel(CARBON_SUBMODEL_GRASS) gives grain-sorghum values.
 */
struct CarbonDevelopmentParameters {
    //! Air temperature below which development stops (C)
    float base_temperature = 4.5f;
    //! Air temperature at which development is fastest (C)
    float optimum_temperature = 25.f;
    //! Air temperature above which development stops (C)
    float maximum_temperature = 36.f;
    //! Air temperature at which one calendar day is one developmental day (C): the temperature at which the plant's day-valued parameters apply as given. Choose it near the
    //! growing-season mean for which those parameters were set.
    float reference_temperature = 20.f;
};

//! Parameters of the grass and grain mechanisms of the carbohydrate model, held in CarbohydrateParameters::grass: tillering, the culm reserve, stage-dependent root allocation and a grain sink
/**
 * A culm -- the main stem or a tiller -- ends in a panicle that appears with its flag leaf, and from then the culm's reproductive clock runs in developmental days. Its stages are the
 * plant's phenological thresholds (PlantArchitecture::setPlantPhenologicalThresholds()) counted from the flag leaf: anthesis is the time to fruit set and physiological maturity the
 * time to fruit maturity. The culm's terminal bud must bear its panicle from the flag leaf, i.e. its shoot type has a fruit prototype and
 * no flower prototype.
 * During the <b>critical period</b>, from flag leaf to the start of effective grain filling (anthesis + grain_fill_lag_days), the panicle structure is built from current photosynthate in
 * competition with the elongating stem; at its end the grain number is fixed from the panicle mass built (grains_per_gram_panicle), so a carbon shortfall in the window costs grain number and
 * cannot be made up later. The grain then fills until physiological maturity at a potential rate set by kernel_potential_carbon, from current photosynthate first and then from the culm and
 * root reserves, so that the reserve contribution to grain is an outcome rather than a constant. After maturity the culm's leaves senesce (post_maturity_senescence_days). Defaults are for
 * grain sorghum (Sorghum bicolor); where a rule is borrowed from maize or wheat the docstring says so.
 */
struct CarbonGrassParameters {

    // -- Phenology (days) -- //
    //! Number of main-stem nodes at which stem elongation begins. From then the elongating stem is the dominant sink: tiller buds no longer break and existing tillers are judged on their own
    //! carbon balance (tiller_survival_carbon_ratio). The sorghum library plant's crown is its first six internodes (Nakamura & Goto 1996), so its seventh node elongates.
    int stem_elongation_node = 7;
    //! Developmental days from anthesis to the start of effective grain filling (the lag phase of endosperm cell division). The critical period for grain number runs from flag leaf to the start
    //! of filling (van Oosterom & Hammer 2008, Field Crops Research 108:259). Anthesis and physiological maturity are the plant's phenological thresholds (see CarbonGrassParameters), and the lag
    //! must end before maturity.
    float grain_fill_lag_days = 5.f;
    //! Developmental days after a culm's physiological maturity over which its remaining leaves senesce and are shed, oldest first, at an even rate. Sorghum canopies die within a few weeks
    //! of maturity in senescent hybrids, while stay-green hybrids keep about twice the green leaf area at maturity (Borrell et al. 2000, Crop Science 40:1037); a longer period is a more
    //! stay-green genotype. Leaves that reach PlantArchitecture::setPlantPhenologicalThresholds()'s max_leaf_lifespan first are shed then.
    float post_maturity_senescence_days = 15.f;
    //! Air temperature below which grain filling stops (C). From anthesis a culm's panicle clock runs on the grain-filling temperatures rather than CarbohydrateParameters::development's: its
    //! developmental days per day are (min(T, grain_fill_optimum_temperature) - grain_fill_base_temperature) over the same expression at the plant's reference temperature. Sorghum grain
    //! filling has a lower base than development before anthesis (5.7 C and 23.5 C; Hammer & Muchow 1994, Field Crops Research 36:221; APSIM-Sorghum), so grain fills faster than the plant
    //! develops in a cool autumn.
    float grain_fill_base_temperature = 5.7f;
    //! Air temperature at and above which grain filling runs at its fastest (C); there is no decline above it (see grain_fill_base_temperature)
    float grain_fill_optimum_temperature = 23.5f;

    // -- Culm -- //
    //! Rate (1/developmental day) at which a culm's internodes close the gap between their radius and the pipe-model radius set by the leaf and inflorescence area above them: each step closes
    //! 1 - exp(-girth_growth_rate x dt) of it. Without the carbohydrate model the gap closes by half each step, so when the panicle appears with the flag leaf -- adding its area above every node
    //! -- the whole culm would thicken within a day or two and its cost would fall on the panicle's critical period. A cereal internode adds girth over the one to three weeks of its elongation.
    float girth_growth_rate = 0.1f;
    //! Largest radius to which surplus photosynthate may thicken an internode of a culm, as a multiple of its pipe-model radius (the radius set by the leaf and inflorescence area above it).
    //! Until a culm flowers its stem is the residual sink, as in APSIM-Sorghum, where the stem takes the dry matter the other organs do not: photosynthate beyond what maintenance, growth and
    //! storage take is laid down as stem, thickening the internodes of the culms that have not yet reached anthesis, with growth respiration. Beyond this limit, and after anthesis, the surplus
    //! is forgone (PlantCarbonBudget::photosynthesis_downregulation). Sorghum stem diameter varies by about a quarter with the assimilate each plant receives (22% thinner at high plant density;
    //! Kebrom, McKinley & Mullet 2017, Biotechnology for Biofuels 10:159). 1 or less disables it.
    float culm_thickening_max = 1.3f;

    // -- Panicle -- //
    //! Structural carbon of a full-size panicle (rachis, branches and glumes, without grain) at the end of the critical period when carbon does not limit (mol C). A grain sorghum panicle holds
    //! ~8 g DW of structure at anthesis: 8 g x 0.45 g C/g / 12 g/mol. Built at a constant potential rate over the critical period from current photosynthate, at its potential times the
    //! reproductive supply ratio; it is not funded from reserves, which cereals are still accumulating at this stage.
    float panicle_potential_carbon = 0.3f;
    //! Carbon content of panicle dry matter (g C / g DW)
    float panicle_carbon_percentage = 0.45f;
    //! Growth respiration of panicle structure (mol C respired per mol C incorporated)
    float panicle_growth_respiration_fraction = 0.25f;
    //! Grains set per gram of panicle dry mass built over the critical period (g^-1). Grain number tracks panicle growth over the window in sorghum (van Oosterom & Hammer 2008) and is
    //! nearly constant per gram of spike at anthesis in wheat (78 g^-1; Fischer 1985). With the default panicle this gives ~2000 grains per culm; the value is a calibration target.
    float grains_per_gram_panicle = 250.f;

    // -- Grain -- //
    //! Potential structural carbon of one grain (mol C). Sorghum kernels are ~25-30 mg at ~0.42 g C/g DW. This is a ceiling that filling approaches at a constant potential rate from the
    //! end of the critical period to physiological maturity: sorghum grain weight responds to late source (Gambín et al. 2008), unlike maize, whose potential kernel weight is set by supply around silking.
    float kernel_potential_carbon = 1.05e-3f;
    //! Growth respiration of grain (mol C respired per mol C incorporated); starch-rich grain
    float grain_growth_respiration_fraction = 0.2f;
    //! Carbohydrate concentration (g C / g DW) to which grain filling may draw down the culm and root reserves. Above it the stem water-soluble carbohydrate is remobilized to the grain;
    //! below it the grain grows only on current photosynthate, so that filling does not starve the culm (CarbohydrateParameters::carbohydrate_growth_threshold, 0.05 for the grass sub-model, is where a
    //! shoot is short of carbon and, after carbohydrate_starvation_days, aborts its panicle), and sits 0.02 below it so that maintenance does not take a culm drained by its grain under the threshold. Wheat stems retain ~5-10% of dry mass as WSC at
    //! maturity after heavy remobilization; grain-sorghum culms hold 47-183 mg glucose/g (0.02-0.07 g C / g DW) at black layer (McBee & Miller 1993).
    float culm_reserve_floor = 0.07f;

    // -- Tillers -- //
    //! Tiller bud outgrowth follows a carbon supply/demand index (Alam et al. 2014, New Phytologist 203:155): the plant's photosynthate supply rate (over the current advanceTime() call) relative to its vegetative growth demand
    //! averaged over CarbohydrateParameters::supply_averaging_days. The probability that a tiller bud breaks is scaled by 0 at or below this index, rising linearly to 1 at
    //! tiller_supply_demand_max; above that the bud breaks at the probability set by its shoot type, which is the genotype's propensity to tiller. Before any demand has been recorded the
    //! factor is 1.
    float tiller_supply_demand_min = 0.5f;
    //! Supply/demand index at and above which carbon does not limit tiller bud outgrowth (see tiller_supply_demand_min)
    float tiller_supply_demand_max = 1.5f;
    //! A tiller is shed while the main stem elongates -- until the tiller's own panicle appears or the main stem reaches anthesis -- when the net photosynthesis of its own leaves since
    //! elongation began, judged once CarbohydrateParameters::supply_averaging_days have elapsed, is less than this multiple of what it spent on maintenance and growth over the same time. Tillers
    //! stop forming and begin to die at the onset of stem elongation, youngest first, when the elongating stem becomes the dominant sink (Xie et al. 2015; Kebrom et al. 2012). Non-positive
    //! disables tiller death.
    float tiller_survival_carbon_ratio = 1.f;
    //! Number of leaves a tiller must have before it is judged on its own carbon balance (tiller_survival_carbon_ratio). A young tiller builds leaves faster than its own few small ones pay
    //! for them and is carried by the rest of the plant; its carbon records restart when it reaches this many leaves, and it is judged on its balance from then, once
    //! CarbohydrateParameters::supply_averaging_days have passed. No measurement for sorghum was found: four is the leaf number at which the leaves of a fully lit tiller of the library plant
    //! begin to pay for its growth.
    int tiller_establishment_leaves = 4;

    // -- Roots -- //
    //! Fraction of the plant's net photosynthate allocated to root growth at emergence. It falls linearly to root_allocation_late as the main stem reaches its flag leaf and is 0 after
    //! anthesis. Belowground allocation peaks early and declines as reproductive growth begins (Pausch & Kuzyakov 2018); CERES-Wheat gives roots at least 35% of assimilate until terminal
    //! spikelet.
    float root_allocation_early = 0.35f;
    //! Fraction of net photosynthate allocated to root growth at flag leaf (see root_allocation_early)
    float root_allocation_late = 0.05f;
};

//! Parameters of the carbohydrate model
/**
 * The carbohydrate model is a shared carbon accounting -- pools in every shoot and in the root system, photosynthate credited from the leaves, maintenance respiration, the construction of
 * stem, root, leaf and flower tissue, phloem transport, finite storage, sugar-starch partitioning and carbon-limited growth -- on which a crop-specific sub-model (\ref submodel) runs the
 * mechanisms that differ between crops: how reproductive sinks are set and filled, what competes for surplus photosynthate, and how shoots are judged. The parameters of those mechanisms are
 * grouped in nested structures (\ref fruit, \ref wood); the parameters here are shared by every sub-model.
 *
 * Defaults are for mature almond (Prunus dulcis), with peach values (Grossman & DeJong 1994a,b) where almond measurements
 * are not available. Every carbohydrate concentration is expressed per gram of stem dry mass, as g C / g DW; 1 mg
 * glucose-equivalent per g DW, the unit of most non-structural carbohydrate (NSC) assays, is 4.0e-4 g C / g DW.
 */
struct CarbohydrateParameters {

    //! Carbon sub-model the plant runs on. CARBON_SUBMODEL_TREE (the default) uses \ref fruit and \ref wood.
    CarbonSubmodel submodel = CARBON_SUBMODEL_TREE;

    // -- Stem -- //
    //! Dry density of mature stem tissue (g DW m^-3; almond wood)
    float stem_density = 675000;
    //! Structural carbon of stem tissue (g C / g DW): total wood carbon (0.457, peach current-year stem; Grossman & DeJong 1994a) less ~0.03 g C / g DW of non-structural carbohydrate
    float stem_structural_carbon_percentage = 0.43f;
    //! Age at which a new internode reaches its mature carbon density (days)
    float maturity_age = 120;
    //! Carbon density of new internode tissue as a fraction of the mature density
    float initial_density_ratio = 0.25;
    //! Ratio of shoot (stem) dry mass to root dry mass. Excavated 13-year-old almond: 5.2-6.6 (Muhammad et al. 2020); 7-year-old peach: ~3.
    float shoot_root_ratio = 5;

    // -- Leaves -- //
    //! Carbon content of leaf dry matter (g C / g DW; Grossman & DeJong 1994a)
    float leaf_carbon_percentage = 0.453;
    //! Specific leaf area (m^2 / g DW). Almond leaf mass per area is 109-124 g/m^2 (Pica et al. 2022).
    float SLA = 0.009f;
    //! Fraction of a leaf's final dry mass per unit area that is in place when its blade expands. The rest is laid down as the leaf thickens over leaf_maturation_days, so leaf carbon is charged
    //! partly with blade area and partly with leaf age. 1 charges the whole leaf with its area.
    float leaf_mass_fraction_at_expansion = 1.f;
    //! Days from a leaf's emergence over which its dry mass per unit area rises from leaf_mass_fraction_at_expansion to its final value (1/SLA)
    float leaf_maturation_days = 60.f;

    // -- Flowers -- //
    //! Carbon cost to produce a flower, including its growth respiration (mol C flower^-1)
    float total_flower_cost = 8.33e-4;

    // -- Respiration -- //
    //! Maintenance respiration of mature wood (trunk and scaffolds) at 20 C (mol C respired / mol structural C / day). Peach trunk: 1.75e-4 (Grossman & DeJong 1994a).
    float r_m_w_20 = 1.75e-4f;
    //! Maintenance respiration at 20 C of wood older than one year but thinner than branch_radius_threshold (mol C / mol C / day). Peach branches: 1.8e-3 (Grossman & DeJong 1994a).
    float r_m_w_branch_20 = 1.8e-3f;
    //! Maintenance respiration at 20 C of wood younger than one year (mol C / mol C / day). Peach current-year stems: 5.2e-3 (Grossman & DeJong 1994a).
    float r_m_w_current_20 = 5.2e-3f;
    //! Internode radius below which wood older than one year respires at r_m_w_branch_20 rather than r_m_w_20 (m)
    float branch_radius_threshold = 0.005f;
    //! Maintenance respiration of roots at 20 C (mol C respired / mol structural C / day). Peach: 1.8e-3 (Grossman & DeJong 1994a).
    float r_m_r_20 = 1.8e-3f;
    //! Q10 of maintenance respiration. Peach stems and leaves: ~2.0 below 30 C (Grossman & DeJong 1994b).
    float respiration_Q10 = 2.0f;
    //! Growth respiration of wood, roots and leaves (mol C respired per mol C incorporated into structure; DeJong & Goudriaan 1989)
    float growth_respiration_fraction = 0.211;

    // -- Storage and transport -- //
    //! Carbohydrate storage capacity of roots per unit dry mass, relative to wood. Peach root-crown xylem holds ~1.5x the carbohydrate of trunk xylem (Da Silva et al. 2014), and the L-PEACH
    //! storage capacities are in the same ratio. Excavated almond roots hold 36% of tree NSC in 13% of its biomass (Esparza et al. 2001).
    float root_storage_ratio = 1.5f;
    //! Rate (1/day) at which phloem transport evens out differences in carbohydrate concentration between shoots and roots. 0 leaves every shoot to live on its own carbon.
    /**
     * Phloem moves carbohydrate across a tree within a day, and measured almond twig, branch and trunk NSC concentrations are nearly identical (Tixier et al. 2020).
     * A rate well above 1/day keeps an actively growing shoot, which is charged for its new leaves every day against a small wood volume, supplied from the
     * rest of the plant; at slower rates such shoots sit persistently below the pruning threshold and die while the plant is well stocked.
     */
    float carbon_mixing_rate = 10.f;
    //! Carbohydrate storage capacity of wood: the largest carbohydrate concentration it holds (g C / g DW). 0.05 is ~125 mg glucose-eq/g DW, within the 110-170 mg/g autumn maximum of almond
    //! shoot wood (Tixier et al. 2020). The root system holds root_storage_ratio times as much per unit dry mass. Carbohydrate the plant cannot store or spend leaves it below ground (root exudation and
    //! respiration) and is reported in PlantCarbonBudget::storage_overflow.
    float storage_capacity = 0.05f;
    //! Rate (1/day) at which storage takes up carbohydrate toward capacity. Storage is a sink that competes with radial growth for the photosynthate left after maintenance, vegetative
    //! growth and fruit: it demands storage_refill_rate x (capacity - pool) per day, the two share the remainder in proportion to their demands, and photosynthate beyond both is forgone
    //! (PlantCarbonBudget::photosynthesis_downregulation). Storage refills in parallel with growth within 2-6 weeks of leaf-out (Klein et al. 2016).
    float storage_refill_rate = 0.03f;

    // -- Sugar-starch partitioning -- //
    //! Intercept of the sugar target (g C / g DW). Tissue holds as soluble sugar sugar_target_intercept + sugar_target_slope x its carbohydrate concentration (at most all of
    //! it) and the rest as starch, so the sugar share falls as carbohydrate rises and some starch remains at low concentrations. 0.0064 g C/g DW is 16 mg glucose/g DW, from a linear fit to monthly
    //! medians of debarked almond twig wood across California orchards, 2017-2019 (Carbohydrate Observatory; Davidson et al. 2021, Sci. Rep. 11:5181, Fig. 5).
    float sugar_target_intercept = 0.0064f;
    //! Slope of the sugar target: soluble sugar gained per unit of carbohydrate (see sugar_target_intercept; Davidson et al. 2021).
    float sugar_target_slope = 0.37f;
    //! Rate (1/day) at which sugar and starch relax toward the sugar target
    float starch_conversion_rate = 0.2f;

    // -- Starvation and growth modulation -- //
    //! Days a shoot's carbohydrate stays below carbohydrate_growth_threshold before its flowers and fruit are aborted, one at a time until it recovers. A shoot whose carbohydrate is exhausted
    //! dies.
    float carbohydrate_starvation_days = 2;
    //! Carbohydrate concentration (g C / g DW) below which a shoot is short of carbon: below it the phyllochron lengthens, internode elongation, leaf expansion and vegetative bud break slow in
    //! proportion, and radial growth stops; a shoot that stays below it for carbohydrate_starvation_days aborts its flowers and fruit (~50 mg glucose/g DW)
    float carbohydrate_growth_threshold = 0.02f;
    //! Days over which the supply ratios of the sinks built from current photosynthate (fruit, cambium) are averaged. Apple fruit drop responds to carbon deficits lasting more than ~4 days,
    //! not 1-2 days (Robinson & Lakso 2011).
    float supply_averaging_days = 5;

    // -- Sub-model mechanisms -- //
    //! Volumetric fruit sink (CARBON_SUBMODEL_TREE)
    CarbonFruitParameters fruit;
    //! Cambium and branch shedding (CARBON_SUBMODEL_TREE)
    CarbonWoodParameters wood;
    //! Tillers, culm reserve, roots and grain (CARBON_SUBMODEL_GRASS)
    CarbonGrassParameters grass;
    //! Temperature response of development (all sub-models)
    CarbonDevelopmentParameters development;

    //! Default parameters of a carbon sub-model
    /**
     * The default-constructed structure is the tree sub-model with almond values. For CARBON_SUBMODEL_GRASS the shared parameters are given grain-sorghum values: a culm of low dry density
     * that matures within weeks, thin leaves, herbaceous maintenance respiration, a culm reserve capacity of ~0.15 g C / g DW (stem water-soluble carbohydrate reaches 30-50% of dry mass in
     * cereals) that fills within days, a growth threshold of 0.05 g C / g DW, and a shoot:root ratio of 4.
     * \param[in] submodel Sub-model to return the defaults of.
     * \return Parameters with \ref submodel set and every shared parameter at that sub-model's default.
     */
    [[nodiscard]] static CarbohydrateParameters forSubmodel(CarbonSubmodel submodel);
};

//! Carbon fluxes of one plant in mol C, summed over the most recent call to PlantArchitecture::advanceTime
struct PlantCarbonBudget {
    //! Net leaf photosynthesis credited to the shoots
    float net_photosynthesis = 0;
    //! Maintenance respiration of the shoots' wood
    float wood_maintenance_respiration = 0;
    //! Maintenance respiration of the roots
    float root_maintenance_respiration = 0;
    //! Structural carbon built into wood
    float wood_growth = 0;
    //! Structural carbon built into roots
    float root_growth = 0;
    //! Structural carbon built into leaves
    float leaf_growth = 0;
    //! Structural carbon built into fruit
    float fruit_growth = 0;
    //! Carbon spent constructing flowers (including their growth respiration)
    float flower_construction = 0;
    //! Growth respiration of wood, roots, leaves and fruit
    float growth_respiration = 0;
    //! Carbohydrate (sugar and starch) lost with branches shed for carbon starvation
    float shed_branch_carbohydrate = 0;
    //! Carbohydrate above the plant's storage capacity, released below ground as root exudation and respiration
    float storage_overflow = 0;
    //! Net photosynthesis forgone because no sink could take it after storage and radial growth (not included in net_photosynthesis)
    float photosynthesis_downregulation = 0;
    //! Number of fruit dropped for lack of carbon (a count, not mol C)
    float fruit_dropped = 0;
    //! Structural carbon built into panicle structure during the critical period (CARBON_SUBMODEL_GRASS)
    float panicle_growth = 0;
    //! Structural carbon built into grain (CARBON_SUBMODEL_GRASS)
    float grain_growth = 0;
    //! The part of grain growth (with its growth respiration) paid from the culm and root reserves rather than from the sub-step's photosynthate (CARBON_SUBMODEL_GRASS). Included in grain_growth
    //! and growth_respiration; reported so that the reserve contribution to grain can be read off.
    float reserve_remobilization = 0;
    //! Number of tillers shed for a negative carbon balance during stem elongation (a count, not mol C; CARBON_SUBMODEL_GRASS). Their carbohydrate is in shed_branch_carbohydrate.
    float tillers_shed = 0;
};

//! Grain sink state of a plant under the carbohydrate model's grass sub-model, summed over its culms, as returned by PlantArchitecture::getPlantGrainStatus
struct PlantGrainStatus {
    //! Number of culms (main stem and tillers) bearing a panicle
    float culms = 0;
    //! Grains set on those culms whose critical period has ended
    float grain_number = 0;
    //! Structural carbon built into grain (mol C)
    float grain_carbon_molC = 0;
    //! Structural carbon the grain could have received so far at its potential filling rate (mol C)
    float grain_potential_carbon_molC = 0;
    //! Structural carbon built into panicle structure (mol C)
    float panicle_carbon_molC = 0;
    //! Carbohydrate (sugar and starch) held in the culms, the stem reserve available for remobilization (mol C)
    float culm_carbohydrate_molC = 0;
};

//! Parameters for the nitrogen model
struct NitrogenParameters {

    // -- Leaf Nitrogen Content (Area Basis) -- //
    //! target leaf nitrogen content per unit area (g N/m²)
    float target_leaf_N_area = 1.5f;
    //! minimum viable leaf nitrogen content per unit area (g N/m²)
    float minimum_leaf_N_area = 0.5f;

    // -- Allocation Parameters -- //
    //! fraction of nitrogen uptake allocated to roots
    float root_allocation_fraction = 0.15f;

    // -- Rate Limiting Parameters -- //
    //! maximum rate at which leaves can accumulate nitrogen per unit area (g N/m²/day)
    float max_N_accumulation_rate = 0.1f;

    // -- Remobilization Parameters -- //
    //! first-order rate (1/day) at which a mature leaf releases its nitrogen above minimum_leaf_N_area while the plant cannot meet its growing organs' demand
    /**
     * Each day of shortfall a mature leaf can give up this fraction of its remobilizable nitrogen, oldest leaves first. Labelling studies in nitrogen-limited plants find mature leaves exporting
     * 5-16 % of their nitrogen to young leaves over ten days (Diaz et al. 2008), and crop models use a few percent per day.
     */
    float leaf_remobilization_rate = 0.03f;
    //! fraction of a leaf's nitrogen above minimum_leaf_N_area resorbed into the plant as it senesces, by the time it is shed at max_leaf_lifespan (0.0-1.0)
    float leaf_remobilization_efficiency = 0.70f;

    // -- Leaf Senescence Parameters -- //
    //! fraction of max_leaf_lifespan, at the end of a leaf's life, over which an unstressed leaf senesces (0.0-1.0)
    /**
     * A senescing leaf returns its resorbable nitrogen to the plant gradually, yellowing as it does, and is shed at max_leaf_lifespan. Has no effect while max_leaf_lifespan keeps its
     * effectively infinite default.
     */
    float leaf_senescence_duration_fraction = 0.25f;
    //! fraction of max_leaf_lifespan by which nitrogen stress brings the onset of leaf senescence forward, at full stress (0.0-1.0)
    /**
     * Scaled by (1 - nitrogen_stress_factor): a plant at its target leaf nitrogen senesces its leaves on schedule, and a nitrogen-starved plant starts senescing them earlier.
     * The sum with leaf_senescence_duration_fraction must not exceed 1.
     */
    float stress_senescence_advance_fraction = 0.30f;

    // -- Fruit Nitrogen Parameters -- //
    //! nitrogen content per unit fruit area (g N/m²)
    float fruit_N_area = 1.0f;
};

//! Add geometry to the Context consisting of a series of Cone objects to form a tube-like shape
/**
 * \param[in] radial_subdivisions Number of subdivisions around the circumference of the tube (must be >= 3).
 * \param[in] vertices (x,y,z) Cartesian coordinates of vertices forming the centerline of the tube.
 * \param[in] radii Radius of the tube at each specified vertex.
 * \param[in] colors Color of the tube at each specified vertex.
 * \param[in] context_ptr Pointer to the Helios context.
 * \return Object ID of the tube, or Phytomer::no_petiole_objID if the tube was too small to build.
 */
uint makePetioleTube(uint radial_subdivisions, const std::vector<helios::vec3> &vertices, const std::vector<float> &radii, const std::vector<helios::RGBcolor> &colors, helios::Context *context_ptr);

struct VegetativeBud {

    // state of the bud
    BudState state = BUD_DORMANT;
    // label of the shoot type that will be produced if the bud breaks into a shoot
    std::string shoot_type_label;
    // ID of the shoot that the bud will produce if it breaks into a shoot
    uint shoot_ID = -1;
    //! Days since the bud became active, which is since it was created or since the dormancy break that woke it
    float time_counter = 0;
    //! Days after becoming active at which the bud breaks into a shoot
    /**
     * Drawn for this bud from ShootParameters::vegetative_bud_break_time of the shoot bearing it, and compared with time_counter.
     */
    float break_time = 0;
    //! Whether break_time has been drawn since the bud last became active
    bool break_time_drawn = false;
};

//! Grain sink of a culm's terminal panicle under the carbohydrate model's grass sub-model, CARBON_SUBMODEL_GRASS
struct GrainSink {
    //! Structural carbon built into the panicle's rachis, branches and glumes during the critical period (mol C)
    float panicle_carbon_molC = 0;
    //! Whether the grain number has been fixed at the end of the critical period
    bool grain_number_set = false;
    //! Number of grains set (0 until the end of the critical period)
    float grain_number = 0;
    //! Structural carbon built into the grain so far (mol C)
    float grain_carbon_molC = 0;
    //! Structural carbon the grain could have received so far at its potential filling rate (mol C); grain_carbon_molC falls behind it when supply is short
    float grain_potential_carbon_molC = 0;
    //! Leaves the culm bore when it reached physiological maturity, from which its post-maturity senescence is counted (-1 before maturity)
    int leaves_at_maturity = -1;
    //! Panicle age (developmental days since the flag leaf) up to which the panicle's and grain's potential growth has been accounted
    float accounted_age = 0;
};

struct FloralBud {

    // state of the bud
    BudState state = BUD_DORMANT;
    // amount of time since the bud flowered (=0 if it has not yet flowered)
    float time_counter = 0;
    // cumulative age since bud break (first transition out of dormant/active state)
    float age = 0;
    //=0 for axillary buds, =1 for terminal buds
    bool isterminal = false;
    // For axillary buds: index of the petiole within the internode that this floral bud originates from
    // For terminal buds: index of the phytomer within the shoot that this floral bud originates from
    uint parent_index = 0;
    // Index of the bud within the petiole that this floral bud originates from
    uint bud_index = 0;
    // Scaling factor fraction of the fruit (if present), ranging from 0 to 1
    float current_fruit_scale_factor = 1;
    float previous_fruit_scale_factor = 0;
    //! Fraction of the mature fruit's structural carbon charged to the carbohydrate model so far (0 = nothing charged yet)
    float fruit_carbon_fraction_charged = 0;
    //! Fraction of the mature fruit's structural carbon the fruit could have laid down so far: its potential, which fruit_carbon_fraction_charged falls behind when growth is carbon-limited
    float fruit_potential_fraction = 0;
    //! Fraction of the mature fruit's structural carbon credited to the fruit when fruit drop is judged: what it has been charged, plus its share of what the young fruit dropped beside it had
    //! been charged (see CarbonFruitParameters::abscission_growth_ratio)
    float fruit_drop_credit_fraction = 0;
    //! Days for this bud's inflorescence to expand from its initial quarter size to full size, overriding the plant-level dd_to_fruit_maturity when positive.
    /**
     * The plant-level threshold is a fruit-ripening period, which is the right clock for a fruit but not for an inflorescence that finishes elongating before the fruit it subtends even sets. A maize tassel
     * is fully expanded at VT, some two months ahead of the ear it shares a plant with, so the two cannot be driven from one number. Left at -1, the bud follows the plant-level value.
     */
    float dd_to_inflorescence_maturity = -1;
    //! Grain sink of this bud's panicle (carbohydrate model, CARBON_SUBMODEL_GRASS); reset when the bud sets fruit
    GrainSink grain;
    //! Days by which this bud's first reproductive transition after it becomes active comes later than the plant-level threshold
    /**
     * Drawn when the bud becomes active. See PlantArchitecture::setPlantPhenologicalSpread().
     */
    float flowering_delay = 0;
    //! Days by which this bud's fruit takes longer than the plant-level dd_to_fruit_maturity to reach full size
    /**
     * Drawn at fruit set. See PlantArchitecture::setPlantPhenologicalSpread().
     */
    float fruit_maturity_delay = 0;


    helios::vec3 base_position;
    AxisRotation base_rotation;
    helios::vec3 bending_axis;

    std::vector<helios::vec3> inflorescence_bases;
    std::vector<AxisRotation> inflorescence_rotation; // pitch, yaw, roll for each flower/fruit
    std::vector<float> inflorescence_base_scales; // individual base scale for each flower/fruit (before growth scaling)
    std::vector<uint> peduncle_objIDs;
    std::vector<uint> inflorescence_objIDs;

    //! Unit axis (world frame) of the rotation about base_position by which the peduncle and its fruit have been swung out of branches (see PlantArchitecture::enableFruitBranchCollisionAvoidance())
    helios::vec3 contact_swing_axis = helios::make_vec3(0, 0, 1);
    //! Angle (radians) of that swing; 0 when the peduncle has never been swung
    float contact_swing_angle = 0;
};

//! The sampled shape of one generic leaf blade: everything needed to rebuild its lattice exactly
/**
 * The blade-shape parameters of \ref LeafPrototype are random parameters, resampled after every blade is built, so the values a particular leaf was built with cannot be read back from the prototype
 * afterwards. A leaf reshaped in place (\ref PlantArchitecture::setPetioleLeafShape(), \ref PlantArchitecture::setPetioleLeafRoll()) starts from these recorded values, so only what is being changed
 * changes. Lengths are in units of the blade length, as in the unit-length prototype.
 */
struct LeafShapeValues {
    //! Ratio of blade width to length (\ref LeafPrototype::leaf_aspect_ratio)
    float aspect_ratio = 1.f;
    //! \ref LeafPrototype::midrib_fold_fraction
    float midrib_fold_fraction = 0.f;
    //! \ref LeafPrototype::longitudinal_curvature
    float longitudinal_curvature = 0.f;
    //! \ref LeafPrototype::longitudinal_curvature_exponent
    float longitudinal_curvature_exponent = 4.f;
    //! \ref LeafPrototype::lateral_curvature
    float lateral_curvature = 0.f;
    //! \ref LeafPrototype::petiole_roll
    float petiole_roll = 0.f;
    //! \ref LeafPrototype::wave_period
    float wave_period = 0.f;
    //! \ref LeafPrototype::wave_amplitude
    float wave_amplitude = 0.f;
    //! \ref LeafPrototype::roll_fraction
    float roll_fraction = 0.f;
    //! Number of lattice subdivisions along the blade
    uint subdivisions_x = 1;
    //! Number of lattice subdivisions across the blade (even)
    uint subdivisions_y = 2;
    //! Translation of the lattice in the prototype frame: \ref LeafPrototype::leaf_offset, plus the petiolule's length when the blade sits beyond one
    helios::vec3 lattice_offset;
};

//! Shape of a leaf blade's midrib, as returned by PlantArchitecture::getLeafMidrib()
/**
 * One entry per station of the blade's lattice along its length (LeafShapeValues::subdivisions_x + 1 stations), from the base of the blade to its tip. See \ref PlantArch_blademidrib.
 */
struct LeafMidrib {
    //! Position of each midrib station in world coordinates
    std::vector<helios::vec3> positions;
    //! Unit normal of the blade's upper side at each station, in world coordinates, perpendicular to the midrib there and ignoring the blade's fold and curvature across the midrib
    std::vector<helios::vec3> normals;
    //! Twist of the blade about its midrib at each station, in radians, right-handed about the midrib's direction from base to tip, measured from the blade bent along the same midrib without twisting
    std::vector<float> rolls;
};

struct LeafPrototype {
public:
    //! Constructor - sets random number generator
    explicit LeafPrototype(std::minstd_rand0 *generator);

    //! Constructor - does not set random number generator
    LeafPrototype() = default;

    //! Copy constructor
    /**
     * Copies every member. Written out rather than defaulted so that copying a LeafPrototype does not raise a deprecation warning for its deprecated buckle members.
     */
    LeafPrototype(const LeafPrototype &a) {
        duplicate(a);
    }

    //! Custom prototype function for creating leaf prototypes
    /**
     * \note This is a function pointer, so it is not carried by a ShootParameters reconstructed from values alone. See \ref ShootParameters::inheritCustomFunctionsFrom().
     */
    uint (*prototype_function)(helios::Context *, LeafPrototype *prototype_parameters, int compound_leaf_index) = nullptr;

    //! OBJ model file to load for the leaf
    /**
     *\note If this is set, the leaf will be loaded from the OBJ file and the other leaf parameters will be ignored.
     */
    std::string OBJ_model_file;

    //! Image texture file for the leaf
    /**
     *\note Key is the index of the compound leaf (=0 is the tip leaf, <0 increases down left side of the leaflet, >0 increases down the right side of the leaflet), value is the texture file.
     */
    std::map<int, std::string> leaf_texture_file;

    //! Sets of image texture files for the leaf, from which the unique prototypes draw so that leaves vary across a plant and can change as they age
    /**
     * Key is the index of the compound leaf, as for \ref LeafPrototype::leaf_texture_file, and a single entry applies to every leaflet. Value is a list of texture variants, each of which is a list of the images a leaf of that variant passes through as it ages, starting from the youngest.
     * Unique prototype k uses variant (k modulo the number of variants), so \ref LeafPrototype::unique_prototypes must be at least the number of variants. Each leaf draws a unique prototype at random and keeps that prototype's variant.
     * Every variant must have the same number of images, and if there is more than one, \ref LeafPrototype::leaf_texture_stage_ages gives when each one begins. Variants may have different outlines, but the images of one variant must match in resolution and transparency (see Context::validateColorTexture()).
     * If this is set, \ref LeafPrototype::leaf_texture_file is ignored. A leaf rebuilt without a unique prototype, as by Phytomer::setPetioleLeafGeometry(), uses the first variant.
     *\note Only available with the default \ref LeafPrototype::prototype_function and without \ref LeafPrototype::OBJ_model_file.
     */
    std::map<int, std::vector<std::vector<std::string>>> leaf_texture_series;

    //! Class map of the leaf texture, from which LeafOptics gives the leaf's tissues (such as a pale midrib) their own spectra
    /**
     * Keyed the same way as \ref LeafPrototype::leaf_texture_file. Value is a single-channel PNG of integer classes with the resolution and transparency of the leaf texture (see
     * helios::Context::setPrimitiveSpectralTexture()), which can be a simple file name as for the texture, and whose class definitions are in an XML file of the same name beside it (see the LeafOptics
     * documentation). Each leaf triangle that covers a class (helios::classMapAreaFractions()) records the class map's path as the primitive data "spectral_class_map", and LeafOptics::run() then gives
     * the leaf a spectral texture whose classes are derived from the leaf's own properties. Without LeafOptics, the class map is not used and the leaf keeps whatever spectra or values it is given.
     *\note Not available with \ref LeafPrototype::leaf_texture_series or \ref LeafPrototype::OBJ_model_file.
     */
    std::map<int, std::string> leaf_class_map_file;

    //! Age of a leaf in days at which it begins each image of its variant in \ref LeafPrototype::leaf_texture_series
    /**
     * Has one value per image, starting at 0 and strictly increasing. Required if the variants have more than one image.
     */
    std::vector<float> leaf_texture_stage_ages;

    //! Which unique prototype is being built, which selects its variant in \ref LeafPrototype::leaf_texture_series. Set by the plug-in immediately before it calls the prototype function.
    uint texture_series_selector = 0;

    //! Check that \ref LeafPrototype::leaf_texture_series can be used by this prototype, and raise an error explaining how to correct it if not
    /**
     * \param[in] shoot_type_label Label of the shoot type the prototype belongs to, for the error message.
     */
    void validateTextureSeries(const std::string &shoot_type_label) const;

    //! Get the images a leaflet passes through as it ages, from \ref LeafPrototype::leaf_texture_series
    /**
     * \param[in] compound_leaf_index Index of the leaflet within the compound leaf.
     * \param[in] prototype_index Which of the unique prototypes the leaflet was built from.
     * \return The variant's images, starting from the youngest.
     */
    [[nodiscard]] const std::vector<std::string> &getTextureSeriesVariant(int compound_leaf_index, uint prototype_index) const;

    //! Get which image of a variant in \ref LeafPrototype::leaf_texture_series a leaf of the given age uses
    /**
     * \param[in] age Age of the leaf in days.
     * \return Index of the image, which is 0 if \ref LeafPrototype::leaf_texture_stage_ages is empty.
     */
    [[nodiscard]] uint getTextureStageAtAge(float age) const;

    //! Ratio of leaf width to leaf length
    RandomParameter_float leaf_aspect_ratio;

    //! Fraction of folding along the leaf midrib. =0 means leaf is flat, =1 means leaf is completely folded in half along midrib with its halves raised; a negative value folds the halves down by as much (=-1: folded shut downward).
    RandomParameter_float midrib_fold_fraction;

    // Parameters for leaf curvature
    //! Leaf curvature factor along the longitudinal/length (x-direction). (+curves upward, -curved downward)
    RandomParameter_float longitudinal_curvature;

    //! How the longitudinal curvature is distributed along the blade, as the exponent of a power law in distance from the leaf base
    /**
     * The deflection at a fraction \c x along the blade is <code>longitudinal_curvature * x^longitudinal_curvature_exponent</code>, so a larger exponent keeps the inner blade straight and concentrates
     * the bending near the tip, while a smaller one spreads it evenly along the leaf. At the default of 4 the blade has reached only 6% of its tip deflection by its own midpoint, which suits a leaf that
     * stays straight and then turns over at the end; a grass blade that arcs continuously along its whole length is better described by an exponent near 2.
     * \note This changes where the curvature acts, not how much there is: the deflection at the tip is <code>longitudinal_curvature</code> whatever the exponent.
     */
    RandomParameter_float longitudinal_curvature_exponent;
    //! Leaf curvature factor along the lateral/width (y-direction). (+curves upward, -curved downward)
    /**
     * The cross-section takes the slope profile of <code>lateral_curvature * (y / leaf_aspect_ratio)^4</code>, laid out along the arc length from the midrib, so the blade bends across its width without
     * stretching: its area and texture outline are unchanged whatever the value.
     */
    RandomParameter_float lateral_curvature;

    //! How far the blade is rolled about its midrib toward its upper (adaxial) side, as a fraction of a closed tube
    /**
     * Each cross-section of the blade is wrapped onto a circular arc about the midrib with half-angle <code>roll_fraction * pi</code>: 0 leaves the blade flat and 1 brings its two edges together in a closed
     * tube. Wrapping a strip onto a cylinder does not stretch it, so the blade keeps its area and texture outline while the area it presents to the light falls - for a strip of uniform width, the projected
     * width is <code>sin(theta) / theta</code> of the flat width for half-angle theta (2/pi at 0.5). This is how grasses such as wheat, maize and rice respond to water stress. Must be between 0 and 1.
     *
     * Rolling is composed with the other blade-shape terms in this order: the midrib fold, lateral curvature and roll together set the cross-section, which is then bent along the blade by the longitudinal
     * curvature, given its waves, and finally deflected under the blade's own weight.
     * \note A blade needs enough lattice cells across its width to form the arc. Leaves rolled on an existing plant with \ref PlantArchitecture::setPetioleLeafRoll() are re-meshed finer when they need it.
     */
    RandomParameter_float roll_fraction;

    //! Creates a rolling at the leaf where the petiole attaches to the leaf blade
    RandomParameter_float petiole_roll;

    // Parameters for leaf wave/wrinkles
    //! Period factor of leaf waves (sets how many waves there are along the leaf length)
    RandomParameter_float wave_period;
    //! Amplitude of leaf waves (sets the height of leaf waves)
    RandomParameter_float wave_amplitude;

    //! How much floppier the blade becomes over its lifetime, as a multiple of its flexibility when it first reaches full size
    /**
     * A blade does not stop bending when it stops growing. Once elongation ceases the leaf loses the active straightening that accompanies growth, while its tissue goes on stiffening and shrinking
     * irreversibly, so the bend accrued during development keeps accumulating with no further change in size. Self-weight accounts for only part of the curvature of a
     * mature blade, so a leaf whose droop follows from its size alone stays too straight, and its oldest leaves - the ones that should be arching over hardest - stay the straightest of all.
     *
     * This is the timescale of the softening: the blade holds its shape for about twice this many days while its tissue matures, then softens over a window of similar length, then stops changing again. Zero disables the effect, which is the default.
     * \note The growth is bounded so that a very old leaf cannot become arbitrarily floppy - see \ref flexibility_aging_max.
     */
    RandomParameter_float flexibility_aging;

    //! Largest multiple of \ref flexibility that ageing can reach, however old the leaf gets
    /**
     * Without a ceiling a long-lived leaf would go on softening indefinitely and eventually hang straight down. Real blades stop changing shape appreciably once their tissue has finished maturing, so the
     * accumulated compliance is capped rather than allowed to grow without limit.
     */
    RandomParameter_float flexibility_aging_max;

    //! How much more compliant the blade is at its tip than at its base
    /**
     * A blade of uniform stiffness bends hardest where the bending moment is largest, which is at the clamped base, leaving the outer half of the leaf nearly straight. Real grass leaves do the reverse: they
     * stay straight near the base and curve most steeply toward the tip, because the midrib carrying the load thins several-fold along the blade and takes the bending stiffness down with it faster than the
     * moment decays. This is the ratio of tip compliance to base compliance; 1 leaves the blade uniform, and values of order ten give the base-straight, tip-curved shape of a grass leaf.
     */
    RandomParameter_float flexibility_taper;

    //! Compliance of the leaf blade in bending, which makes the leaf droop under its own weight as it grows
    /**
     * The blade is treated as a cantilever clamped at its base and loaded by its own weight, so a leaf points along its growing direction when it is small and bends over as it expands, with no separate age
     * term. Larger values give a floppier leaf; zero gives a perfectly rigid one that keeps its prototype shape exactly, which is the behaviour of every species that does not droop.
     * \note Deflection grows with the fifth power of leaf length, so this interacts strongly with <code>leaf_aspect_ratio</code> and with the leaf scale set through \ref PhytomerParameters.
     */
    RandomParameter_float flexibility;

    //! \deprecated Superseded by \ref LeafPrototype::flexibility.
    /**
     * Formerly the fraction along the leaf at which it was bent by a fixed angle, standing in for the leaf drooping under its own weight. The leaf now bends continuously along its whole length, so there is
     * no single station at which it kinks. Retained so that existing code continues to compile: if \ref LeafPrototype::flexibility is left at zero and a buckle angle was set, the pair is converted to the
     * flexibility reproducing the same tip droop on the fully-grown leaf. The mature leaf therefore looks much as it did, but now reaches that shape gradually as it expands rather than being born with it.
     */
    [[deprecated("LeafPrototype::leaf_buckle_length is superseded by LeafPrototype::flexibility, which bends the leaf continuously under its own weight as it grows rather than kinking it at a fixed station. "
                 "The value is converted to an equivalent flexibility for now; set flexibility directly instead.")]] RandomParameter_float leaf_buckle_length;

    //! \deprecated Superseded by \ref LeafPrototype::flexibility.
    /**
     * Formerly the angle, in degrees, through which the leaf was bent at \ref LeafPrototype::leaf_buckle_length. See that parameter for how the value is now interpreted.
     */
    [[deprecated("LeafPrototype::leaf_buckle_angle is superseded by LeafPrototype::flexibility, which bends the leaf continuously under its own weight as it grows rather than kinking it at a fixed station. "
                 "The value is converted to an equivalent flexibility for now; set flexibility directly instead.")]] RandomParameter_float leaf_buckle_angle;

    //! Resolve the bending stiffness to build a leaf with, honouring a deprecated buckle setting when the flexibility was never set
    /**
     * An explicitly-set \ref LeafPrototype::flexibility always wins, so code that has already migrated is unaffected by a stale buckle value left beside it.
     * \return Flexibility to build the leaf with.
     */
    [[nodiscard]] float resolveFlexibility() const;

    //! Amount to shift the leaf
    helios::vec3 leaf_offset;

    //! Leaf subdivision count in each direction
    uint subdivisions = 1;

    //! Number of unique prototypes to generate
    uint unique_prototypes = 1;

    //! Add a petiolule to the base of the leaflet
    /**
     * The petiolule is the stalk joining a leaflet to the rachis of a compound leaf. It is built along the leaflet's midrib axis as a cylinder of length
     * \ref LeafPrototype::petiolule_length, and the blade starts where that cylinder ends. Beyond it the stalk flattens and tapers onto the underside of the
     * blade base, so that it meets the blade plane smoothly rather than ending abruptly against it. Its radius is not a parameter: it is taken from the petiole
     * at the point the leaflet attaches, so the stalk continues the petiole without a step. Its primitives are labelled "petiolule" rather than "leaf" and
     * drawn with the petiole's material; they belong to the leaf object but are not counted in its leaf area.
     */
    bool build_petiolule = false;

    //! Length of the petiolule as a fraction of the leaflet's blade length, keyed by compound-leaf index
    /**
     * Keyed the same way as \ref LeafPrototype::leaf_texture_file, so index 0 is the terminal leaflet, and negative and positive indices are the lateral leaflets
     * on either side, counted from the tip. A single entry applies to every leaflet. Because the petiolule is part of the leaflet prototype it grows with the
     * leaflet, so a fully expanded leaflet of blade length L has a stalk of length L times this value. Only used if \ref LeafPrototype::build_petiolule is true.
     */
    std::map<int, float> petiolule_length = {{0, 0.05f}};

    uint unique_prototype_identifier = 0;

    void duplicate(const LeafPrototype &a) {
        this->leaf_texture_file = a.leaf_texture_file;
        this->leaf_texture_series = a.leaf_texture_series;
        this->leaf_class_map_file = a.leaf_class_map_file;
        this->leaf_texture_stage_ages = a.leaf_texture_stage_ages;
        this->texture_series_selector = a.texture_series_selector;
        this->OBJ_model_file = a.OBJ_model_file;
        this->leaf_aspect_ratio = a.leaf_aspect_ratio;
        this->midrib_fold_fraction = a.midrib_fold_fraction;
        this->longitudinal_curvature = a.longitudinal_curvature;
        this->longitudinal_curvature_exponent = a.longitudinal_curvature_exponent;
        this->lateral_curvature = a.lateral_curvature;
        this->roll_fraction = a.roll_fraction;
        this->petiole_roll = a.petiole_roll;
        this->wave_period = a.wave_period;
        this->wave_amplitude = a.wave_amplitude;
        this->flexibility = a.flexibility;
        this->flexibility_taper = a.flexibility_taper;
        this->flexibility_aging = a.flexibility_aging;
        this->flexibility_aging_max = a.flexibility_aging_max;
        HELIOS_PUSH_IGNORE_DEPRECATED
        this->leaf_buckle_length = a.leaf_buckle_length;
        this->leaf_buckle_angle = a.leaf_buckle_angle;
        HELIOS_POP_IGNORE_DEPRECATED
        this->leaf_offset = a.leaf_offset;
        this->subdivisions = a.subdivisions;
        this->unique_prototypes = a.unique_prototypes;
        this->unique_prototype_identifier = a.unique_prototype_identifier;
        this->build_petiolule = a.build_petiolule;
        this->petiolule_length = a.petiolule_length;
        this->petiolule_radius_fraction = a.petiolule_radius_fraction;
        this->prototype_function = a.prototype_function;
        this->generator = a.generator;
    }

    //! Assignment operator
    LeafPrototype &operator=(const LeafPrototype &a) {
        if (this != &a) {
            this->leaf_texture_file = a.leaf_texture_file;
            this->leaf_texture_series = a.leaf_texture_series;
            this->leaf_class_map_file = a.leaf_class_map_file;
            this->leaf_texture_stage_ages = a.leaf_texture_stage_ages;
            this->texture_series_selector = a.texture_series_selector;
            this->OBJ_model_file = a.OBJ_model_file;
            this->leaf_aspect_ratio = a.leaf_aspect_ratio;
            this->midrib_fold_fraction = a.midrib_fold_fraction;
            this->longitudinal_curvature = a.longitudinal_curvature;
            this->longitudinal_curvature_exponent = a.longitudinal_curvature_exponent;
            this->lateral_curvature = a.lateral_curvature;
            this->roll_fraction = a.roll_fraction;
            this->petiole_roll = a.petiole_roll;
            this->wave_period = a.wave_period;
            this->wave_amplitude = a.wave_amplitude;
            this->flexibility = a.flexibility;
            this->flexibility_taper = a.flexibility_taper;
            this->flexibility_aging = a.flexibility_aging;
            this->flexibility_aging_max = a.flexibility_aging_max;
            HELIOS_PUSH_IGNORE_DEPRECATED
            this->leaf_buckle_length = a.leaf_buckle_length;
            this->leaf_buckle_angle = a.leaf_buckle_angle;
            HELIOS_POP_IGNORE_DEPRECATED
            this->leaf_offset = a.leaf_offset;
            this->subdivisions = a.subdivisions;
            this->unique_prototypes = a.unique_prototypes;
            this->unique_prototype_identifier = a.unique_prototype_identifier;
            this->build_petiolule = a.build_petiolule;
            this->petiolule_length = a.petiolule_length;
            this->petiolule_radius_fraction = a.petiolule_radius_fraction;
            this->prototype_function = a.prototype_function;
            this->generator = a.generator;
            if (this->generator != nullptr) {
                this->sampleIdentifier();
            }
        }
        return *this;
    }

    void sampleIdentifier() {
        assert(generator != nullptr);
        helios::uniform_int_distribution<uint> unif_distribution;
        this->unique_prototype_identifier = unif_distribution(*generator);
    }

    //! Draw the shape of subsequently-built blades from a different random number generator
    /**
     * Redirects every random shape parameter of the prototype, and the prototype itself, at the given generator.
     * Used to build a species' set of unique blade shapes from a private stream keyed on
     * LeafPrototype::unique_prototype_identifier, so that the set does not depend on how much randomness has
     * already been drawn from the Context when the set happens to be built.
     * \param[in] rand_generator Pointer to the random number generator to draw from.
     * \return Generator that was in use before the call, so that a caller can restore it.
     */
    std::minstd_rand0 *setRandomGenerator(std::minstd_rand0 *rand_generator);

    //! Get the random number generator that this prototype draws its shape from
    /**
     * \return Pointer to the random number generator in use, which is null if the prototype was default-constructed.
     */
    [[nodiscard]] std::minstd_rand0 *getRandomGenerator() const {
        return generator;
    }

private:
    std::minstd_rand0 *generator{};

    //! Petiolule radius as a fraction of the leaflet's blade length
    /**
     * Not a user parameter. Phytomer::buildLeafPrototype() sets it from the petiole's radius at the leaflet's attachment point just before the prototype is
     * built, so that the stalk continues the petiole without a step.
     */
    float petiolule_radius_fraction = 0.f;

    //! Shape of the most recent blade built by GenericLeafPrototype() from this prototype
    /**
     * Not a user parameter. The blade-shape parameters are resampled as soon as a blade is built, so the values it was built with are recorded here for Phytomer::buildLeafPrototype() to keep. Only
     * meaningful while \ref LeafPrototype::last_built_shape_valid is true, which it is not for an OBJ-loaded leaf or a custom prototype function.
     */
    LeafShapeValues last_built_shape;
    bool last_built_shape_valid = false;

    friend struct Phytomer;
    friend class PlantArchitecture;
    friend class PlantArchitectureTestHelper;
    friend uint GenericLeafPrototype(helios::Context *context_ptr, LeafPrototype *prototype_parameters, int compound_leaf_index);
};

struct PhytomerParameters {
private:
    struct InternodeParameters {

        //! Angular deviation (in degrees) of this internode’s axis relative to the previous internode along the shoot.  Values other than 0° make the stem zig-zag.
        RandomParameter_float pitch;
        //! Phyllotactic (azimuthal) angle in degrees between the petioles/buds of two successive phytomers.  Typical settings: 180 ° (opposite), 137.5 ° (spiral), 90 ° (decussate).
        RandomParameter_float phyllotactic_angle;
        //! Outside radius (meters) assigned to the internode when it is first created.
        RandomParameter_float radius_initial;
        //! Maximum number of vegetative buds that can form on each petiole; actual bud emergence also depends on vegetative-bud break probability.
        RandomParameter_int max_vegetative_buds_per_petiole;
        //! Maximum number of floral buds (potential flowers/fruit) per petiole; actual emergence also depends on flower-bud break probability.
        RandomParameter_int max_floral_buds_per_petiole;
        //! Diffuse RGB color applied to the internode tube mesh.
        helios::RGBcolor color;
        //! Image texture to map to the internode tube (overrides RGB color).
        std::string image_texture;
        //! Class map of the internode texture, from which the radiation model gives the bark's lighter and darker parts their own spectra
        /**
         * A single-channel PNG of integer classes with the resolution of \ref InternodeParameters::image_texture (which can be a simple file name as for the texture), whose class definitions are in an XML
         * file of the same name beside it (a surface class map; see the radiation plugin documentation, "Spectral Textures"). Each internode triangle records the class map's path as the primitive data
         * "spectral_class_map", and \ref RadiationModel::applySpectralClassMaps() then gives it a spectral texture whose classes scale the bark spectrum named in the definitions. Without that call, the
         * class map is not used. Ignored when the internode has no texture.
         */
        std::string image_class_map_file;
        //! Longitudinal tessellation count of the internode tube.
        uint length_segments;
        //! Number of radial subdivisions around the internode circumference  (4 = square, 5 = pentagon, ≥ 8 ≈ circular).
        uint radial_subdivisions;

        InternodeParameters &operator=(const InternodeParameters &a) {
            if (this != &a) {
                this->pitch = a.pitch;
                this->phyllotactic_angle = a.phyllotactic_angle;
                this->radius_initial = a.radius_initial;
                this->max_vegetative_buds_per_petiole = a.max_vegetative_buds_per_petiole;
                this->max_floral_buds_per_petiole = a.max_floral_buds_per_petiole;
                this->color = a.color;
                this->image_texture = a.image_texture;
                this->image_class_map_file = a.image_class_map_file;
                this->length_segments = a.length_segments;
                this->radial_subdivisions = a.radial_subdivisions;
            }
            return *this;
        }
    };

    struct PetioleParameters {

        //! Number of petioles emerging from a single internode (e.g., 2 for an opposite pattern)
        uint petioles_per_internode;
        //! Angle in degrees of the petiole base axis relative to its parent phytomer axis. Not used for leaves without a petiole (length of 0).
        RandomParameter_float pitch;
        //! Radius in meters of the petiole cross-section; if either radius or length is 0, no petiole geometry is created and leaves attach at the internode tip
        RandomParameter_float radius;
        //! Length in meters of the petiole tube; if either radius or length is 0, no petiole geometry is created and leaves attach at the internode tip
        /**
         * A length of exactly 0 means the leaf has no petiole at all and sits directly on the stem, as the leaf of a grass does. Its blade is then posed in the frame of the stem rather than of a petiole:
         * \ref LeafParameters::pitch is its angle off the internode, and its upper face is turned toward the stem. None of the other petiole parameters are used, and the leaf must be simple (one leaf
         * per petiole). A small positive length gives a short petiole, whose leaf is posed as on any other petiole.
         */
        RandomParameter_float length;
        //! Curvature in degrees per meter applied along the petiole length (positive bends upward, negative downward)
        RandomParameter_float curvature;
        //! Ratio of tip radius to base radius for the petiole (1 = no taper, 0 = pointed tip)
        RandomParameter_float taper;
        //! Diffuse RGB color applied to the petiole mesh
        helios::RGBcolor color;
        //! Number of longitudinal segments used to tessellate the petiole tube
        uint length_segments;
        //! Number of radial subdivisions around the petiole circumference (4 = square, 5 = pentagon, etc.)
        uint radial_subdivisions;

        //! Dimensionless bending compliance of the petiole, which makes the petiole arch toward the ground under the weight of its leaflets as the leaf grows and the petiole ages
        /**
         * The petiole centerline (including the rachis of a compound leaf) is bent as a tapered cantilever clamped at its insertion. Each leaflet's weight, proportional to its mature area times the leaf's
         * current growth fraction, acts at the point where it attaches, and every segment turns downward in its own vertical plane by an amount proportional to the bending moment it carries through the
         * horizontal lever arms of the current bent shape and inversely proportional to the fourth power of its radius. The rotations accumulate from base to tip, so the petiole curves rather than pivoting,
         * and no segment passes hanging straight down. Leaves stay attached and turn with the centerline where they attach.
         *
         * The value is normalized so that a straight, horizontal, untapered petiole whose full-grown leaf weight acts entirely at its tip bends by this many radians at its tip in the small-deflection limit;
         * the result does not depend on the petiole's length. Zero, the default, gives a petiole that keeps the shape it was created with. See \ref PlantArch_petioledroop.
         * \note A petiole whose centerline was set by PlantArchitecture::setPetioleNodePositions(), or that carries a leaf posed by PlantArchitecture::setPetioleLeafGeometry(), is never bent.
         */
        RandomParameter_float flexibility;
        //! Timescale in days over which the petiole's bending compliance grows with age
        /**
         * When positive, the compliance is \ref flexibility multiplied by 1 + age / flexibility_aging, where age is the phytomer's age in days, so it grows linearly and without limit: a petiole goes on
         * lowering after its leaf has stopped growing, bounded only by the geometry of hanging straight down. Zero, the default, disables ageing, so the petiole stops bending once its leaf is full-grown.
         */
        RandomParameter_float flexibility_aging;

        PetioleParameters &operator=(const PetioleParameters &a) {
            if (this != &a) {
                this->petioles_per_internode = a.petioles_per_internode;
                this->pitch = a.pitch;
                this->radius = a.radius;
                this->length = a.length;
                this->curvature = a.curvature;
                this->taper = a.taper;
                this->color = a.color;
                this->length_segments = a.length_segments;
                this->radial_subdivisions = a.radial_subdivisions;
                this->flexibility = a.flexibility;
                this->flexibility_aging = a.flexibility_aging;
            }
            return *this;
        }
    };

    struct LeafParameters {
        //! Number of leaves attached to each petiole; values greater than 1 create a compound leaf
        RandomParameter_int leaves_per_petiole;
        //! Angle in degrees of the leaf axis relative to its parent petiole axis
        /**
         * Positive values turn the leaf toward its upper surface. For a leaf without a petiole (\ref PetioleParameters::length of 0) the pitch is relative to the internode axis, so it must be zero or
         * negative: -20 gives a blade leaving the stem at 20 degrees.
         */
        RandomParameter_float pitch;
        //! Rotation angle in degrees of the leaf about its base within the plane of the lamina
        RandomParameter_float yaw;
        //! Rotation angle in degrees of the leaf about its own midrib axis
        RandomParameter_float roll;
        //! Spacing between adjacent leaflets along the petiole as a fraction of petiole length when leaves_per_petiole>1; the first two leaflets are offset from the tip by half this value
        RandomParameter_float leaflet_offset;
        //! Scale multiplier applied successively to each leaflet along a compound petiole (<1 shrinks, >1 enlarges)
        RandomParameter_float leaflet_scale;
        //! Relative size of the intercalary leaflets of an interruptedly pinnate compound leaf, as a fraction of the major leaflet just distal to them. Zero, the default, gives a simply pinnate leaf whose leaflets shrink monotonically from the tip
        RandomParameter_float intercalary_leaflet_scale;
        //! Overall scaling factor applied to the leaf prototype to set its physical size
        RandomParameter_float prototype_scale;
        //! Prototype definition holding geometric and texture information used to instantiate individual leaves
        LeafPrototype prototype;

        LeafParameters &operator=(const LeafParameters &a) {
            if (this != &a) {
                this->leaves_per_petiole = a.leaves_per_petiole;
                this->pitch = a.pitch;
                this->yaw = a.yaw;
                this->roll = a.roll;
                this->leaflet_offset = a.leaflet_offset;
                this->leaflet_scale = a.leaflet_scale;
                this->intercalary_leaflet_scale = a.intercalary_leaflet_scale;
                this->prototype_scale = a.prototype_scale;
                this->prototype.duplicate(a.prototype);
            }
            return *this;
        }
    };

    struct PeduncleParameters {
        //! Length in meters of the peduncle (inflorescence supporting structure)
        RandomParameter_float length;
        //! Radius in meters of the peduncle
        RandomParameter_float radius;
        //! Angle in degrees of the peduncle axis relative to its parent internode axis
        RandomParameter_float pitch;
        //! Rotation angle in degrees of the peduncle about its own axis
        RandomParameter_float roll;
        //! Angle in degrees by which the peduncle is turned about its parent internode axis, away from the side of the node its leaf is on. At 0 the peduncle leaves the node from the leaf axil; at 180 it is on the opposite side of the node from the leaf.
        RandomParameter_float yaw;
        //! Curvature in degrees per meter along the peduncle (positive bends upward, negative downward)
        RandomParameter_float curvature;
        //! Diffuse RGB color applied to the peduncle mesh
        helios::RGBcolor color;
        //! Number of longitudinal segments used to tessellate the peduncle tube
        uint length_segments;
        //! Number of radial subdivisions around the peduncle circumference (4 = square, 5 = pentagon, etc.)
        uint radial_subdivisions;

        PeduncleParameters &operator=(const PeduncleParameters &a) {
            if (this != &a) {
                this->length = a.length;
                this->radius = a.radius;
                this->pitch = a.pitch;
                this->roll = a.roll;
                this->yaw = a.yaw;
                this->curvature = a.curvature;
                this->color = a.color;
                this->length_segments = a.length_segments;
                this->radial_subdivisions = a.radial_subdivisions;
            }
            return *this;
        }
    };

    struct InflorescenceParameters {

        //! Number of flowers generated on each peduncle
        RandomParameter_int flowers_per_peduncle;
        //! Normalised distance (0‒1) between successive flowers along the peduncle axis
        RandomParameter_float flower_offset;
        //! Angular deviation in degrees of the inflorescence axis relative to its parent peduncle axis
        RandomParameter_float pitch;
        //! Rotation angle in degrees of the inflorescence about its own axis
        RandomParameter_float roll;
        //! Uniform scale factor applied to the flower prototype geometry
        RandomParameter_float flower_prototype_scale;
        //! Pointer to user-supplied function that returns a flower prototype mesh ID (Context*, subdivisions, flower_is_open, build_parameters). See \ref FlowerPrototypeFunction.
        /**
         * \note This is a function pointer, so it is not carried by a ShootParameters reconstructed from values alone. See \ref ShootParameters::inheritCustomFunctionsFrom().
         */
        FlowerPrototypeFunction flower_prototype_function = nullptr;
        //! Uniform scale factor applied to the fruit prototype geometry
        RandomParameter_float fruit_prototype_scale;
        //! Pointer to user-supplied function that returns a fruit prototype mesh ID (Context*, subdivisions, build_parameters). See \ref FruitPrototypeFunction.
        /**
         * \note This is a function pointer, so it is not carried by a ShootParameters reconstructed from values alone. See \ref ShootParameters::inheritCustomFunctionsFrom().
         */
        FruitPrototypeFunction fruit_prototype_function = nullptr;
        //! Class map of the fruit prototype's texture, from which the radiation model gives the fruit's lighter and darker parts their own spectra
        /**
         * A single-channel PNG of integer classes with the resolution of the texture the fruit prototype is mapped with (a simple file name is resolved as for a texture), whose class definitions are in an
         * XML file of the same name beside it (a surface class map; see the radiation plugin documentation, "Spectral Textures"). Each textured triangle of the fruit records the class map's path as the
         * primitive data "spectral_class_map", and \ref RadiationModel::applySpectralClassMaps() then gives it a spectral texture whose classes scale the fruit spectrum named in the definitions. Without
         * that call, the class map is not used. The prototype must use a single texture.
         */
        std::string fruit_class_map_file;
        //! Fraction (0‒1) of gravitational influence used to bend peduncles under fruit load
        RandomParameter_float fruit_gravity_factor_fraction;
        //! Number of distinct prototype meshes to cache for this inflorescence
        uint unique_prototypes;
        //! Days for the inflorescence to expand from its initial quarter size to full size. Zero means it is already full size when it forms, and a negative value defers to the plant-level fruit-maturity threshold, which is the default.
        /**
         * Set this where the inflorescence finishes elongating on a different schedule from the fruit the plant carries, since \ref PlantArchitecture::setPlantPhenologicalThresholds() supplies only one
         * fruit-ripening period for the whole plant. A maize tassel is fully expanded at VT and never grows afterward, roughly two months ahead of the ear's grain fill, so driving it from the plant-level
         * value leaves it undersized for its entire life.
         */
        RandomParameter_float inflorescence_maturity_period;

        InflorescenceParameters &operator=(const InflorescenceParameters &a) {
            if (this != &a) {
                this->flowers_per_peduncle = a.flowers_per_peduncle;
                this->flower_offset = a.flower_offset;
                this->pitch = a.pitch;
                this->roll = a.roll;
                this->flower_prototype_scale = a.flower_prototype_scale;
                this->flower_prototype_function = a.flower_prototype_function;
                this->inflorescence_maturity_period = a.inflorescence_maturity_period;
                this->fruit_prototype_scale = a.fruit_prototype_scale;
                this->fruit_prototype_function = a.fruit_prototype_function;
                this->fruit_class_map_file = a.fruit_class_map_file;
                this->fruit_gravity_factor_fraction = a.fruit_gravity_factor_fraction;
                this->unique_prototypes = a.unique_prototypes;
            }
            return *this;
        }
    };

public:
    /**
     * \brief Parameters defining the characteristics of an internode.
     *
     * This variable encapsulates all parameters related to the physical and structural attributes
     * of an internode, including geometrical properties (e.g., length, radius, pitch, phyllotaxis)
     * and appearance parameters (e.g., color, texture).
     */
    InternodeParameters internode;

    /**
     * \brief Parameters defining the characteristics of the petiole.
     *
     * This variable encapsulates all parameters related to the morphological and structural traits
     * of a petiole, such as its length, radius, curvature, taper, and other geometrical and visual attributes.
     */
    PetioleParameters petiole;

    /**
     * \brief Parameters defining the characteristics of a leaf.
     *
     * This variable encapsulates all parameters related to the structure and geometry
     * of a leaf, such as its attachment configuration, angles, scaling factors, spacing,
     * and prototype definition for appearance and texture.
     */
    LeafParameters leaf;

    /**
     * \brief Parameters defining the characteristics of the peduncle (inflorescence supporting structure).
     *
     * Encapsulates all the geometrical, structural, and visual attributes of the peduncle,
     * including its length, radius, curvature, and appearance details.
     */
    PeduncleParameters peduncle;

    /**
     * \brief Parameters defining the characteristics of the inflorescence.
     *
     * Encapsulates all morphological, structural, and functional attributes associated
     * with the inflorescence, such as the number of flowers per peduncle, spacing,
     * angular orientation, scaling factors for both flowers and fruits,
     * and user-defined prototype functions for flower and fruit mesh generation.
     */
    InflorescenceParameters inflorescence;

    // Custom user-defined function that is called when a phytomer is created
    /**
     * \param[in] phytomer_ptr Pointer to the phytomer to which the function will be applied
     * \param[in] shoot_node_index Index of the phytomer within the shoot starting from 0 at the shoot base
     * \param[in] parent_shoot_node_index Node index of the current shoot along it's parent shoot
     * \param[in] shoot_max_nodes Maximum number of phytomers in the shoot
     * \param[in] plant_age Age of the plant in days
     * \note This is a function pointer, so it is not carried by a ShootParameters reconstructed from values alone. See \ref ShootParameters::inheritCustomFunctionsFrom().
     * \note The function may read the plant, set bud states and meristem flags, change parameters and scale factors, and remove leaves (Phytomer::removeLeaf()). It may call PlantArchitecture::pruneBranch(), which is deferred and takes effect at the end of the current time step, or before the build call that created the phytomer returns. Operations that add or delete plant structure are skipped with a warning or raise an error. A bud set to active while the plant is dormant, including on the time step dormancy ends, is overwritten when the plant breaks dormancy; see \ref PlantArchCallbackEdits.
     */
    void (*phytomer_creation_function)(std::shared_ptr<Phytomer> phytomer_ptr, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) = nullptr;

    // Custom user-defined function that is called for each phytomer on every time step
    /**
     * \param[in] phytomer_ptr Pointer to the phytomer to which the function will be applied
     * \note This is a function pointer, so it is not carried by a ShootParameters reconstructed from values alone. See \ref ShootParameters::inheritCustomFunctionsFrom().
     * \note The function may read the plant, set bud states and meristem flags, change parameters and scale factors, and remove leaves (Phytomer::removeLeaf()). It may call PlantArchitecture::pruneBranch(), which is deferred and takes effect at the end of the current time step, or before the build call that created the phytomer returns. Operations that add or delete plant structure are skipped with a warning or raise an error. A bud set to active while the plant is dormant, including on the time step dormancy ends, is overwritten when the plant breaks dormancy; see \ref PlantArchCallbackEdits.
     */
    void (*phytomer_callback_function)(std::shared_ptr<Phytomer> phytomer_ptr) = nullptr;


    //! Default constructor - does not set random number generator
    PhytomerParameters();

    //! Constructor - sets random number generator
    explicit PhytomerParameters(std::minstd_rand0 *generator);

    //! Draw a new value for every random parameter that varies from one phytomer to the next
    /**
     * The Phytomer constructor calls this on its own copy of the parameters, which is where per-phytomer variation comes from. Assignment and copying never resample, so copying a PhytomerParameters or a
     * ShootParameters consumes no random draws. Parameters with a constant distribution are skipped and consume no draws either.
     */
    void resample();

    friend class PlantArchitecture;
    friend struct Phytomer;
    friend struct Shoot;
};

struct ShootParameters {

    //! Default constructor - does not set random number generator
    ShootParameters();

    //! Constructor - sets random number generator
    explicit ShootParameters(std::minstd_rand0 *generator);

    /**
     * \brief Stores parameters related to a phytomer in the shoot.
     *
     * \note This variable encapsulates properties and behaviors specific to the phytomers of a plant’s shoot system.
     */
    PhytomerParameters phytomer_parameters;

    // ---- Geometric Parameters ---- //

    //! Maximum number of nodes/phytomers along a shoot
    RandomParameter_int max_nodes;
    //! Maximum number of nodes/phytomers that a shoot can produce in a single season (≤ max_nodes)
    RandomParameter_int max_nodes_per_season;
    //! Cross-sectional area of internode in cm² branch area per m² downstream leaf area; set 0 to disable girth scaling
    /** Downstream leaf area is cumulative -- every leaf ever produced above the internode, not reduced when leaves drop or
     * branches are shed -- so this is the pipe model with disused pipes retained. See the girth_area_factor section of the
     * plugin documentation.
     */
    RandomParameter_float girth_area_factor;
    //! Insertion angle (deg) of a shoot of this type that emerges from a bud at the tip of its parent shoot
    /**
     * This is a property of the child shoot, not of the parent it emerges from: when a vegetative bud breaks, the insertion angle of the new shoot is read from the shoot type the bud will become, and the
     * parent shoot contributes only the position of the bud along it (see insertion_angle_decay_rate).
     */
    RandomParameter_float insertion_angle_tip;
    //! Rate (deg/node) at which the insertion angle of a shoot of this type increases with the distance of its bud below the tip of its parent shoot
    /**
     * Like insertion_angle_tip, this is read from the type of the child shoot. The insertion angle is insertion_angle_tip + insertion_angle_decay_rate times the number of nodes between the bud and the
     * parent's tip, capped at 90 degrees.
     */
    RandomParameter_float insertion_angle_decay_rate;
    //! Maximum internode length (m) of a shoot of this type
    /**
     * When a shoot of this type emerges from a bud and growth_requires_dormancy is true, its internode length is reduced from this value by internode_length_decay_rate for each node between the bud and the
     * tip of the parent shoot, down to internode_length_min. All three values are read from the type of the child shoot, not the parent.
     */
    RandomParameter_float internode_length_max;
    //! Minimum internode length (m) of a shoot of this type, the floor for the reduction set by internode_length_decay_rate
    RandomParameter_float internode_length_min;
    //! Rate (m/node) at which the internode length of a shoot of this type decreases with the distance of its bud below the tip of its parent shoot
    /**
     * Applied only when growth_requires_dormancy is true. See internode_length_max.
     */
    RandomParameter_float internode_length_decay_rate;
    //! Roll angle (deg) of the shoot specifying the orientation of the first petiole relative to the parent shoot
    RandomParameter_float base_roll;
    //! Yaw angle (deg) of the shoot relative to the parent shoot
    RandomParameter_float base_yaw;
    //! Gravitropic curvature (deg/m); positive values curve the shoot upward toward vertical
    RandomParameter_float gravitropic_curvature;
    //! Standard deviation (deg/m) of the random curvature that produces wiggle (tortuosity) along the shoot
    /**
     * The shoot is turned by this random curvature times the arc length of each internode segment, so the amount of wiggle does
     * not depend on how many segments an internode is divided into. Values from versions before 1.3.89, which were an angle per
     * internode segment, convert as tortuosity * internode.length_segments / internode_length_max.
     */
    RandomParameter_float tortuosity;
    //! Persistence length (m) over which the random tortuosity perturbation of the shoot curvature decorrelates
    /**
     * Sets the arc-length scale of the Ornstein-Uhlenbeck process applied to the shoot curvature. Larger values give long, smooth
     * meanders; smaller values give rapid, fine-scale wiggle. The noise is applied to curvature (not to the direction), which is
     * what keeps the resulting trajectory smooth: direction noise integrates only once and converges to a nowhere-differentiable
     * Brownian path, whereas curvature noise integrates twice and yields finite curvature everywhere. Values <= 0 fall back to
     * a default of 0.5 m.
     */
    RandomParameter_float tortuosity_persistence_length;
    //! Minimum time (days) between the emergence of successive phytomers along the shoot

    // --- Growth Parameters --- //

    RandomParameter_float phyllochron_min;
    //! Maximum relative elongation rate (m · m⁻¹ · day⁻¹) of the shoot internode; actual rate may be reduced dynamically
    RandomParameter_float elongation_rate_max;
    //! Maximum relative expansion rate of the leaves and petioles on the shoot; negative means expand at the internode elongation rate
    /**
     * Units are meters of leaf expansion per meter of fully-expanded leaf size per day (m · m⁻¹ · day⁻¹), the same
     * form as \ref elongation_rate_max, so a value of 0.1 expands a leaf from nothing to full size in ten days
     * regardless of how large that leaf is.
     *
     * Leaves and internodes need not grow on the same schedule; leaves often finish expanding well before
     * internodes stop elongating. Setting this parameter decouples them. The internode is unaffected and keeps growing at
     * \ref elongation_rate_max.
     *
     * Any negative value -- the default, \ref leaf_expansion_rate_unset -- means the species does not distinguish
     * the two, and leaves expand at the shoot's own \ref elongation_rate_max. This is the behavior of every shoot
     * type that does not set the parameter, including every model in the plant library. Zero is a real rate meaning "leaves never expand", and
     * is therefore distinct from the unset sentinel.
     */
    RandomParameter_float leaf_expansion_rate_max;
    //! Sentinel value of \ref leaf_expansion_rate_max meaning "expand leaves at the internode elongation rate"
    /**
     * The comparison is against zero rather than against this exact value, so any negative rate is treated as
     * unset. A rate is a non-negative quantity, so no negative value can be confused with one a user intended.
     */
    static constexpr float leaf_expansion_rate_unset = -1.f;
    //! Minimum probability that a vegetative bud on a shoot of this type will break and form a new shoot
    /**
     * This is also the probability used everywhere along the shoot when vegetative_bud_break_probability_decay_rate is zero, and when it is negative and growth_requires_dormancy is false. With the
     * defaults (0, with a decay rate of -0.5 and growth_requires_dormancy false) no vegetative bud ever breaks. See the vegetative bud break section of the plug-in documentation.
     */
    RandomParameter_float vegetative_bud_break_probability_min;
    //! Probability that a vegetative bud on a shoot of this type will break at the end of the shoot where the probability is highest
    /**
     * Used only when vegetative_bud_break_probability_decay_rate is non-zero: at the apex for a positive rate, at the base for a negative rate (and growth_requires_dormancy true).
     */
    RandomParameter_float vegetative_bud_break_probability_max;

    //! Rate (1/node) at which the vegetative bud-break probability falls from vegetative_bud_break_probability_max to vegetative_bud_break_probability_min along the shoot
    /**
     * Positive: highest at the apex, falling moving down the shoot. Negative: highest at the base, falling moving up the shoot, but only when growth_requires_dormancy is true; otherwise every bud uses
     * vegetative_bud_break_probability_min. Zero: every bud uses vegetative_bud_break_probability_min.
     *
     * These parameters are read from the shoot bearing the bud. Buds are sampled when a phytomer is created on a shoot whose type has growth_requires_dormancy false, and when dormancy is broken
     * otherwise; a bud that does not break is set dead. Note that PlantArchitecture::makePlantDormant() also kills every bud that is not dormant at the time, including active buds that have not broken yet.
     */
    RandomParameter_float vegetative_bud_break_probability_decay_rate;

    //! Season growth (m) below which no dormant lateral bud on a shoot of this type breaks. When vegetative_bud_break_vigour_saturation is greater than this, the position-dependent probability of a
    //! dormant lateral bud breaking is multiplied by (L - threshold) / (saturation - threshold), limited to [0, 1], where L is the length the shoot grew in the season the bud formed
    //! (Shoot::previous_season_extension_m), so that a weak shoot bears few or no laterals. In almond, proleptic laterals per node rise roughly linearly from none on shoots that grew under ~16 cm to
    //! ~0.23 on shoots that grew ~1 m (Negron et al. 2013, J. Am. Soc. Hortic. Sci. 138:407).
    float vegetative_bud_break_vigour_threshold = 0.f;
    //! Season growth (m) at and above which the position-dependent bud-break probability applies in full (see vegetative_bud_break_vigour_threshold). Not greater than the threshold (the default)
    //! disables the vigour factor.
    float vegetative_bud_break_vigour_saturation = 0.f;
    //! Vigour of a shoot of this type relative to the shoot bearing it. When positive, a shoot of this type that breaks from a dormant lateral bud grows at most lateral_vigour_ratio x (the length its
    //! parent grew in the season the bud formed) per season, converted to a per-season node cap with its own internode length, so that each generation of laterals is weaker than the last (almond
    //! proleptic laterals are about half as long as the shoot bearing them; Negron et al. 2015, Funct. Plant Biol. 42:325). Non-positive (the default) leaves the per-season node cap at the type's value.
    float lateral_vigour_ratio = -1.f;

    //! Share of the cost of the fruit a shoot of this type bears that is charged to the shoot itself when its carbon balance is judged at the season boundary (see
    //! PlantArchitecture::shedCarbonStarvedBranches()); the rest is charged to the shoot bearing it. Almond spurs are only partly carbon-autonomous: their fruit also draws on the
    //! surrounding leaves (Heerema et al. 2008, J. Hortic. Sci. Biotech. 83:274; Saa et al. 2017, Acta Physiol. Plant. 39:107). 1 (the default) charges the whole cost to the bearing shoot.
    float fruit_cost_share = 1.f;
    //! Carbon-dependent survival. When positive, a shoot of this type judged at the season boundary survives with probability 1 / (1 + exp(-(z - carbon_survival_midpoint) /
    //! carbon_survival_scale)), where z = ln(net photosynthesis / (maintenance + growth, with fruit_cost_share of its fruit)), instead of
    //! being shed whenever its photosynthesis falls short of its costs. Non-positive (the default) keeps that deterministic rule.
    float carbon_survival_scale = 0.f;
    //! Value of z at which half the shoots of this type survive (see carbon_survival_scale)
    float carbon_survival_midpoint = 0.f;
    //! Carbon-dependent return bloom. When positive, a shoot of this type flowers in the next season with probability return_bloom_max / (1 + exp(-(z_veg - return_bloom_midpoint) /
    //! return_bloom_scale)), multiplied by return_bloom_bearing_factor if it bore fruit in the season just ended, where z_veg = ln(net photosynthesis / (maintenance + growth other than fruit)). A shoot that does not flower has all its flower buds die when it breaks dormancy. Non-positive (the default) leaves flowering to the flower bud-break probability.
    float return_bloom_scale = 0.f;
    //! Value of z_veg at which flowering is half of return_bloom_max (see return_bloom_scale)
    float return_bloom_midpoint = 0.f;
    //! Flowering probability of a shoot with ample carbon (see return_bloom_scale)
    float return_bloom_max = 1.f;
    //! Factor on the flowering probability of a shoot that bore fruit in the season just ended (see return_bloom_scale). In almond it stands for the suppression of return bloom by the fruit that
    //! carbon does not explain: removing fruit early restores spur survival but not return bloom (Valdebenito et al. 2017, Sci. Hortic. 215:15; Saa et al. 2017).
    float return_bloom_bearing_factor = 1.f;

    //! Maximum number of floral buds at the shoot apex (0 means the shoot forms no terminal floral buds)
    RandomParameter_int max_terminal_floral_buds;
    //! Probability that a phytomer will flower
    RandomParameter_float flower_bud_break_probability;
    //! Probability that a flower will set fruit
    RandomParameter_float fruit_set_probability;
    //! Time (days) after bud creation or dormancy release before a vegetative bud breaks
    RandomParameter_float vegetative_bud_break_time;
    //! Flag indicating whether flower buds require a winter dormancy period to emerge
    bool flowers_require_dormancy;
    //! Flag indicating whether vegetative buds require a winter dormancy period to emerge
    bool growth_requires_dormancy;
    //! Flag indicating determinate shoot growth: true = stop after flowering, false = continue growth
    bool determinate_shoot_growth;

    // ---- Custom Functions ---- //

    /**
     * \brief Defines shoot types for child shoots with their associated probabilities.
     *
     * \param[in] child_shoot_type_labels Vector of labels for child shoot types.
     * \param[in] child_shoot_type_probabilities Vector of probabilities for each child shoot type. Probabilities must sum to 1.
     *
     * \note The sizes of the input vectors must match, and neither input vector can be empty.
     * \note The values set here can be read back with \ref getChildShootTypeLabels() and \ref getChildShootTypeProbabilities().
     */
    void defineChildShootTypes(const std::vector<std::string> &child_shoot_type_labels, const std::vector<float> &child_shoot_type_probabilities);

    //! Get the labels of the child shoot types that may be generated by this shoot type
    /**
     * Returns the labels currently stored in this ShootParameters, which are set by \ref defineChildShootTypes(). An empty vector is a meaningful state rather than an error: a shoot type with no child
     * shoot types defined generates child shoots of its own type.
     *
     * \return Vector of child shoot type labels, parallel to \ref getChildShootTypeProbabilities().
     * \note Once the ShootParameters has been registered with \ref PlantArchitecture::defineShootType() and a plant is built, labels that do not correspond to a defined shoot type are dropped from the
     * stored parameters, so the returned labels are not necessarily those originally passed to \ref defineChildShootTypes().
     */
    [[nodiscard]] std::vector<std::string> getChildShootTypeLabels() const;

    //! Get the probabilities of the child shoot types that may be generated by this shoot type
    /**
     * Returns the probabilities currently stored in this ShootParameters, which are set by \ref defineChildShootTypes(). An empty vector is a meaningful state rather than an error: a shoot type with no
     * child shoot types defined generates child shoots of its own type.
     *
     * \return Vector of child shoot type probabilities, parallel to \ref getChildShootTypeLabels().
     * \note \ref defineChildShootTypes() requires that the probabilities sum to 1, but entries dropped for referring to an undefined shoot type are removed without renormalizing the remainder, so the
     * returned probabilities do not necessarily sum to 1.
     */
    [[nodiscard]] std::vector<float> getChildShootTypeProbabilities() const;

    //! Copy the user-defined callback and prototype function pointers from another ShootParameters
    /**
     * Copies all five function pointers held in the phytomer parameters -- the phytomer creation and callback hooks, and the leaf, flower, and fruit prototype functions -- from \a source into this
     * ShootParameters. All other parameter values are left unchanged, and no random parameters are resampled.
     *
     * This is intended for callers that reconstruct a ShootParameters from plain values, such as a scripting-language binding that flattens the structure to a dictionary, applies user edits, and
     * rebuilds it. Such a reconstruction cannot carry function pointers, so they must be re-attached from the original structure before it is passed to \ref PlantArchitecture::defineShootType() or
     * \ref PlantArchitecture::updateCurrentShootParameters(), both of which replace the stored shoot type entry in its entirety.
     *
     * \param[in] source ShootParameters to copy the function pointers from, typically the structure originally returned by \ref PlantArchitecture::getCurrentShootParameters().
     * \note Pointers that are null in \a source are copied as null, overwriting any existing value. The operation is an unconditional assignment rather than a merge, so repeated calls are idempotent
     * and a deliberately cleared hook stays cleared.
     */
    void inheritCustomFunctionsFrom(const ShootParameters &source);

    //! Assignment operator
    /** Shoot-level random parameters are copied without being resampled, so assignment does not advance the random stream.
     * A ShootParameters holds the DISTRIBUTION for a shoot type; the per-shoot draw is made where a shoot is created, by the
     * Shoot constructor.
     *
     * The nested PhytomerParameters sub-structures and LeafPrototype are different: their assignment
     * operators do resample, because the Phytomer constructor relies on that assignment to re-randomize
     * per-phytomer parameters. LeafPrototype::operator= also calls sampleIdentifier(), which is not
     * distribution sampling but the assignment of a fresh prototype identity for the leaf-prototype cache.
     */
    ShootParameters &operator=(const ShootParameters &a) {
        // Shoot-level parameters are copied by value and are NOT resampled here. Per-shoot variation
        // is produced by the explicit resample() calls in the growth engine at the points where a new
        // shoot is created (see PlantArchitecture.cpp, e.g. base_yaw/insertion_angle_tip when a
        // vegetative bud breaks). Resampling here as well made committing a shoot type through
        // defineShootType()/updateCurrentShootParameters() silently perturb parameters the caller
        // never edited, so an identity round-trip was not a no-op.
        //
        // Note this does NOT apply to the nested PhytomerParameters assignment below: Phytomer's
        // constructor relies on that assignment to re-randomize per-phytomer parameters, so the
        // sub-structure operator= implementations resample deliberately.
        this->phytomer_parameters = a.phytomer_parameters;
        this->max_nodes = a.max_nodes;
        this->max_nodes_per_season = a.max_nodes_per_season;
        this->phyllochron_min = a.phyllochron_min;
        this->elongation_rate_max = a.elongation_rate_max;
        this->leaf_expansion_rate_max = a.leaf_expansion_rate_max;
        this->girth_area_factor = a.girth_area_factor;
        this->vegetative_bud_break_probability_min = a.vegetative_bud_break_probability_min;
        this->vegetative_bud_break_probability_max = a.vegetative_bud_break_probability_max;
        this->flower_bud_break_probability = a.flower_bud_break_probability;
        this->fruit_set_probability = a.fruit_set_probability;
        this->gravitropic_curvature = a.gravitropic_curvature;
        this->tortuosity = a.tortuosity;
        this->tortuosity_persistence_length = a.tortuosity_persistence_length;
        this->vegetative_bud_break_probability_min = a.vegetative_bud_break_probability_min;
        this->vegetative_bud_break_probability_decay_rate = a.vegetative_bud_break_probability_decay_rate;
        this->vegetative_bud_break_vigour_threshold = a.vegetative_bud_break_vigour_threshold;
        this->vegetative_bud_break_vigour_saturation = a.vegetative_bud_break_vigour_saturation;
        this->lateral_vigour_ratio = a.lateral_vigour_ratio;
        this->fruit_cost_share = a.fruit_cost_share;
        this->carbon_survival_scale = a.carbon_survival_scale;
        this->carbon_survival_midpoint = a.carbon_survival_midpoint;
        this->return_bloom_scale = a.return_bloom_scale;
        this->return_bloom_midpoint = a.return_bloom_midpoint;
        this->return_bloom_max = a.return_bloom_max;
        this->return_bloom_bearing_factor = a.return_bloom_bearing_factor;
        this->max_terminal_floral_buds = a.max_terminal_floral_buds;
        this->flower_bud_break_probability = a.flower_bud_break_probability;
        this->fruit_set_probability = a.fruit_set_probability;
        this->vegetative_bud_break_time = a.vegetative_bud_break_time;
        this->insertion_angle_tip = a.insertion_angle_tip;
        this->insertion_angle_decay_rate = a.insertion_angle_decay_rate;
        this->internode_length_max = a.internode_length_max;
        this->internode_length_min = a.internode_length_min;
        this->internode_length_decay_rate = a.internode_length_decay_rate;
        this->base_roll = a.base_roll;
        this->base_yaw = a.base_yaw;
        this->flowers_require_dormancy = a.flowers_require_dormancy;
        this->growth_requires_dormancy = a.growth_requires_dormancy;
        this->child_shoot_type_labels = a.child_shoot_type_labels;
        this->child_shoot_type_probabilities = a.child_shoot_type_probabilities;
        this->determinate_shoot_growth = a.determinate_shoot_growth;
        this->child_shoot_type_labels = a.child_shoot_type_labels;
        this->child_shoot_type_probabilities = a.child_shoot_type_probabilities;
        return *this;
    }

    friend class PlantArchitecture;
    friend struct Shoot;

protected:
    std::vector<std::string> child_shoot_type_labels;
    std::vector<float> child_shoot_type_probabilities;
};

//! Growth state belonging to one individual shoot, as distinct from the shoot type it was built from.
/**
 * ShootParameters describes a shoot *type*: the values shared by every shoot built from it. Two things
 * do not fit that description. The first is per-shoot state that the growth model itself writes as a
 * plant develops. The second arises when a shoot's geometry is measured rather than generated: the
 * values needed to *build* the measured shape and the values that should govern how it grows *next* are
 * then different values.
 *
 * Both live here. Every field carries an "unset" sentinel meaning "defer to the shoot type".
 *
 * \note ShootParameters::girth_area_factor is deliberately NOT overridden here. A reconstruction sets it
 * near zero on the build type precisely to preserve measured radii, and the pipe model only ever
 * increases a radius -- so taking it from the growth type instead would silently thicken measured wood.
 * It stays a property of the shoot the wood belongs to.
 */
struct ShootGrowthState {
    //! Cap on this shoot's total node count for the purpose of further growth; -1 defers to ShootParameters::max_nodes.
    /**
     * Distinct from the shoot type's max_nodes, which is also a build-time capacity check: a shoot built
     * from measured node positions must be allowed to be as long as it was measured to be, without that
     * length becoming the licence for it to keep extending.
     */
    int max_nodes_growth_cap = -1;
    //! Cap on the nodes this shoot may produce in one season; -1 defers to ShootParameters::max_nodes_per_season.
    int max_nodes_per_season_growth_cap = -1;
    //! Label of the shoot type governing this shoot's future growth and the type of the shoots its buds produce; empty defers to Shoot::shoot_type_label.
    std::string growth_type_label;
    //! True if this shoot's existing geometry was prescribed by the caller rather than generated by the growth model.
    bool geometry_is_prescribed = false;
    //! Upper bound (m) on the initial radius of phytomers this shoot grows at its apex; a value of zero or less (the default, -1) means no bound beyond the shoot type's radius_initial.
    /**
     * Set when a shoot emerges, so that the limit that keeps the shoot's first phytomer from being thicker than its bearing axis
     * also applies to every phytomer the shoot grows afterwards. For a lateral that breaks from a bud during advanceTime() it is the
     * bearing phytomer's radius at that phytomer's creation; for an epicormic shoot (PlantArchitecture::addEpicormicShoot()) it is the
     * bearing internode's current radius, since epicormic shoots break from old wood that has thickened since. A shoot that continues
     * its parent's axis (PlantArchitecture::appendShoot()) inherits the parent's bound.
     */
    float internode_radius_cap = -1.f;
};

//! Caller-prescribed geometry for a single internode, used to build a shoot from measured node positions.
/**
 * The ordinary phytomer construction path derives the internode axis by accumulating rotations and then
 * integrates forward by a step length, which can only ever produce an extrapolated shape. When this
 * structure is supplied instead, the two endpoints are taken as given and the axis is read from them, so
 * a shoot can follow a measured path. See PlantArchitecture::addShootFromNodePositions().
 */
struct PrescribedInternode {
    //! Position of the node at the base of the internode.
    helios::vec3 base_node;
    //! Position of the node at the tip of the internode.
    helios::vec3 tip_node;
    //! Radius of the internode at its base node.
    float base_radius;
    //! Radius of the internode at its tip node.
    float tip_radius;
};

struct Phytomer {
public:
    //! Constructor
    /**
     * \param[in] prescribed_internode [optional] Caller-supplied internode endpoints and radii. When null (the default) the internode geometry is generated from the shoot's rotation and curvature parameters
     * in the usual way; when supplied, the given endpoints are used directly and curvature, tortuosity, obstacle avoidance and attraction are not applied to this internode.
     */
    Phytomer(const PhytomerParameters &params, Shoot *parent_shoot, uint phytomer_index, const helios::vec3 &parent_internode_axis, const helios::vec3 &parent_petiole_axis, helios::vec3 internode_base_origin, const AxisRotation &shoot_base_rotation,
             float internode_radius, float internode_length_max, float internode_length_scale_factor_fraction, float leaf_scale_factor_fraction, uint rank, PlantArchitecture *plantarchitecture_ptr, helios::Context *context_ptr,
             const PrescribedInternode *prescribed_internode = nullptr);

    // ---- query info about the phytomer ---- //

    /**
     * \brief Retrieve the positions of nodes associated with an internode.
     *
     * \return A vector of vec3 objects representing the node positions for the current internode.
     */
    [[nodiscard]] std::vector<helios::vec3> getInternodeNodePositions() const;

    /**
     * \brief Retrieve the radii of nodes associated with an internode.
     *
     * \return A vector of floats representing the radii of the nodes for the current internode.
     */
    [[nodiscard]] std::vector<float> getInternodeNodeRadii() const;

    /**
     * \brief Calculates the volume of a phytomer.
     *
     * \param[in] node_number The node index of the phytomer whose volume is to be computed.
     * \return The calculated volume of the phytomer.
     */
    [[nodiscard]] float calculatePhytomerVolume(uint node_number) const;

    /**
     * \brief Retrieves the axis vector of the internode at a given fraction along the internode.
     *
     * \param[in] stem_fraction Fraction along the stem (value between 0.0 and 1.0) for which the axis vector is computed.
     * \return A vec3 representing the axis vector of the internode at the specified internode fraction.
     */
    [[nodiscard]] helios::vec3 getInternodeAxisVector(float stem_fraction) const;

    /**
     * \brief Retrieves the petiole axis vector for a given stem fraction and petiole index.
     *
     * \param[in] stem_fraction Fraction along the stem (value between 0.0 and 1.0) for which the axis vector is computed.
     * \param[in] petiole_index Index of the petiole for which the axis vector is retrieved.
     * \return The axis vector of the specified petiole as a helios::vec3.
     */
    [[nodiscard]] helios::vec3 getPetioleAxisVector(float stem_fraction, uint petiole_index) const;

    //! Get the axis a child structure should be oriented against, substituting a "ghost petiole" when this phytomer has none.
    /**
     * The orientation of anything attached to a phytomer -- a child shoot's first internode, the frame a base rotation's pitch is
     * measured against -- is expressed relative to the petiole axis. A phytomer built from a shoot type with
     * `petiole.petioles_per_internode = 0` has no petiole to supply that axis, which is the normal case for a leafless woody
     * skeleton, and asking for one directly through \ref getPetioleAxisVector() raises an error.
     *
     * This returns the real petiole axis when the phytomer has petioles, and otherwise the axis a petiole would have had: a unit
     * vector perpendicular to the internode, rotated by the phytomer's accumulated phyllotactic angle so that petiole-free phytomers
     * are spaced around the stem the same way petiole-bearing ones are.
     *
     * \param[in] petiole_index Index of the petiole within this phytomer. Ignored when the phytomer has no petioles.
     * \param[in] internode_axis Axis of this phytomer's internode, which the ghost petiole is constructed perpendicular to.
     * \param[in] phytomer_index Index of this phytomer within its shoot, which sets how far the ghost petiole is rotated about the internode.
     * \return Unit vector along the petiole, or along the ghost petiole when there is no petiole.
     */
    [[nodiscard]] helios::vec3 getPetioleAxisVectorOrGhost(uint petiole_index, const helios::vec3 &internode_axis, uint phytomer_index) const;

    /**
     * \brief Retrieves the peduncle axis vector for a given stem fraction, petiole index, and floral bud index.
     *
     * \param[in] stem_fraction Fraction along the peduncle (value between 0.0 and 1.0) for which the axis vector is computed.
     * \param[in] petiole_index Index of the petiole containing the floral bud.
     * \param[in] bud_index Index of the floral bud within the petiole.
     * \return The axis vector of the specified peduncle as a helios::vec3.
     */
    [[nodiscard]] helios::vec3 getPeduncleAxisVector(float stem_fraction, uint petiole_index, uint bud_index) const;

    /**
     * \brief Locates a floral bud of this phytomer within Phytomer::floral_buds.
     *
     * Peduncle geometry and parameters are stored at this location, which is not the bud's (parent_index, bud_index): terminal floral
     * buds are appended to floral_buds after the phytomer's per-petiole entries, one entry per bud, with a parent_index of 0.
     *
     * \param[in] fbud Floral bud held in this phytomer's floral_buds.
     * \return The index of the entry of floral_buds holding the bud (x), and the bud's position within that entry (y).
     */
    [[nodiscard]] helios::uint2 getFloralBudStorageIndex(const FloralBud &fbud) const;

    /**
     * \brief Computes the normalized vector direction along the axis at a given fraction of the stem.
     *
     * \param[in] stem_fraction Fraction along the stem (value between 0.0 and 1.0) for which the axis vector is computed.
     * \param[in] axis_vertices A list of 3D points defining the vertices of the axis.
     * \return A normalized vector direction at the given fraction along the axis.
     */
    [[nodiscard]] static helios::vec3 getAxisVector(float stem_fraction, const std::vector<helios::vec3> &axis_vertices);

    /**
     * \brief Computes the total length of the internode.
     *
     * \return The cumulative length of the internode based on its node positions.
     */
    [[nodiscard]] float getInternodeLength() const;

    /**
     * \brief Retrieves the radius of the internode.
     *
     * \return The radius of the internode.
     */
    [[nodiscard]] float getInternodeRadius() const;

    //! Mean current length of the petioles borne by this phytomer
    /**
     * The value is the petiole's length *right now*, not the mature length it is growing toward: a petiole part-way through its elongation is proportionally shorter, so the value rises as
     * \ref setPetioleScaleFraction(uint, float) advances it toward \ref petiole_length_max. In this it differs from \ref getLeafArea(), which reports the area a leaf is expanding toward. A petiole elongates
     * on its shoot's internode rate rather than on the leaf expansion rate, so it goes on lengthening after the blade it carries has reached full size.
     *
     * The length is measured along the petiole's centerline (its arclength), not as the straight-line distance from its base to its tip. The two differ for a petiole that has been bent by
     * \ref bendPetioleUnderLeafWeight(uint): bending is inextensible, so it shortens the base-to-tip chord while leaving the arclength unchanged, and a drooping petiole therefore reports the same length
     * as a rigid one of the same age. A petiole whose centerline was prescribed with \ref PlantArchitecture::setPetioleNodePositions() reports the arclength of the path that was supplied.
     *
     * Petioles borne at the same node are parallel structures rather than segments in series, so their lengths are not physically additive and this reports their mean rather than their sum (contrast
     * \ref getLeafArea(), which sums because areas do add). Every petiole on a phytomer is built from the same draw of the petiole length parameter, so the mean equals each petiole's individual length
     * unless a caller has prescribed or rescaled them one at a time; use \ref getPetioleLength(uint) const to read one of them.
     *
     * A phytomer built from a shoot type with `petiole.petioles_per_internode = 0` -- the normal case for a leafless woody skeleton -- has no petiole, and so does one whose leaf has been shed by
     * \ref removeLeaf(). Both report zero, which is a statement that no petiole exists rather than a stand-in for an unknown value.
     *
     * \return Mean current arclength of this phytomer's petioles (m), or zero if it has none.
     */
    [[nodiscard]] float getPetioleLength() const;

    //! Current length of one petiole borne by this phytomer
    /**
     * As \ref getPetioleLength(), but for a single petiole rather than the mean over all of them. The value is the current arclength of that petiole's centerline, which bending does not change.
     *
     * \param[in] petiole_index Index of the petiole within this phytomer.
     * \return Current arclength of the petiole (m).
     */
    [[nodiscard]] float getPetioleLength(uint petiole_index) const;

    //! Whether a petiole has a joint in the plant skeleton (see PlantArchitecture::getPlantJoints())
    /**
     * \param[in] petiole_index Index of the petiole within the phytomer.
     * \return True if the petiole has a centerline of non-zero length. False for the petiole of a sessile leaf, and for an index the phytomer does not have.
     */
    [[nodiscard]] bool petioleHasJoint(uint petiole_index) const;

    //! Whether the peduncle of a floral bud has a joint in the plant skeleton (see PlantArchitecture::getPlantJoints())
    /**
     * \param[in] petiole_index Index of the petiole the floral bud belongs to.
     * \param[in] bud_index Index of the floral bud within the petiole.
     * \return True if the peduncle has been built, with a centerline of non-zero length.
     */
    [[nodiscard]] bool peduncleHasJoint(uint petiole_index, uint bud_index) const;

    //! Visit every position and direction stored for one petiole and its leaves. Not intended for use outside the plug-in.
    /**
     * Together with visitFloralBudSpatialState(), visitNodeSpatialState() and the visit functions of \ref Shoot, this is the one place that lists the members holding a plant's geometry in world
     * coordinates. A member added to the phytomer that holds a position or a direction has to be visited here, or re-posing the plant will leave it behind.
     * \param[in] petiole_index Index of the petiole within the phytomer.
     * \param[in] visitor Operation applied to each value.
     */
    void visitPetioleSpatialState(uint petiole_index, PlantSpatialStateVisitor &visitor);

    //! Visit every position and direction stored for one floral bud, its peduncle and its flowers or fruit. Not intended for use outside the plug-in.
    /**
     * \param[in] petiole_index Index of the petiole the floral bud belongs to.
     * \param[in] bud_index Index of the floral bud within the petiole.
     * \param[in] visitor Operation applied to each value.
     */
    void visitFloralBudSpatialState(uint petiole_index, uint bud_index, PlantSpatialStateVisitor &visitor);

    //! Visit every direction stored for the node of the phytomer itself, rather than for one of its organs. Not intended for use outside the plug-in.
    /**
     * \param[in] visitor Operation applied to each value.
     */
    void visitNodeSpatialState(PlantSpatialStateVisitor &visitor);

    //! Let a petiole and its leaves settle under gravity again after they have been turned. Not intended for use outside the plug-in.
    /**
     * Bends the petiole from its rest shape and then lets each blade droop from its own, as advanceTime() does, but without waiting for the leaf to have grown. Does nothing for a species whose petioles and
     * leaves are rigid, or for a petiole or leaf whose geometry was prescribed.
     * \param[in] petiole_index Index of the petiole within the phytomer.
     */
    void resettlePetioleUnderGravity(uint petiole_index);

    /**
     * \brief Retrieves the radius of the internode based on the stem fraction.
     *
     * \param[in] stem_fraction Fraction along the stem (value between 0.0 and 1.0) for which the radius is computed.
     * \return The radius of the internode at the specified stem fraction.
     */
    [[nodiscard]] float getInternodeRadius(float stem_fraction) const;

    /**
     * \brief Calculates the total leaf area for the current phytomer.
     *
     * \return Total leaf area as a float value.
     */
    [[nodiscard]] float getLeafArea() const;

    //! Total one-sided surface area of all flowers/fruit borne on this phytomer
    /**
     * Reported at full size, in the same way as \ref getLeafArea(): an inflorescence part-way through its growth reports the area it is expanding toward rather than its present area.
     *
     * \return Surface area (m^2) of the phytomer's inflorescences
     */
    [[nodiscard]] float getInflorescenceArea() const;

    //! Whether this phytomer's inflorescence is a load that sizes the stem bearing it
    /**
     * True only for a terminal inflorescence on a herbaceous stem -- a grass panicle or tassel, which sits on the shoot apex and continues the stem's own axis, so that the stem must carry it and taper into
     * it. False for a lateral flower cluster, and for anything on a woody stem, whose girth is set by the leaves it supports rather than by a bloom that will abscise. Used by \ref
     * Shoot::refreshDownstreamInflorescenceArea() to decide which inflorescences enter the pipe model; it does not affect what \ref getInflorescenceArea() reports.
     *
     * \return True if this phytomer's inflorescence area should count toward the girth of the stem
     */
    [[nodiscard]] bool inflorescenceLoadsStem() const;

    /**
     * \brief Retrieves the position of the base of a leaf within the phytomer.
     *
     * \param[in] petiole_index Index of the petiole within the phytomer structure.
     * \param[in] leaf_index Index of the leaf within the specified petiole.
     * \return A vec3 indicating the position of the leaf base.
     */
    [[nodiscard]] helios::vec3 getLeafBasePosition(uint petiole_index, uint leaf_index) const;

    /**
     * \brief Checks if the phytomer has a leaf.
     *
     * \return True if the phytomer has at least one leaf, false otherwise.
     */
    [[nodiscard]] bool hasLeaf() const;

    /**
     * \brief Calculates the total leaf area downstream of the current phytomer.
     *
     * \return The total downstream leaf area as a float.
     */
    [[nodiscard]] float calculateDownstreamLeafArea() const;

    // ---- modify the phytomer ---- //

    //! Set the current internode length as a fraction of its maximum length
    /**
     * \param[in] internode_scale_factor_fraction Fraction of the maximum internode length
     * \param[in] update_context_geometry If true, the context geometry will be immediately updated to reflect the new internode length. If false, the geometry will be updated the next time Context geometry information is needed.
     */
    void setInternodeLengthScaleFraction(float internode_scale_factor_fraction, bool update_context_geometry);

    //! Scale the fully-elongated (maximum) internode length as a fraction of its current fully-elongated length
    /**
     * \param[in] scale_factor Fraction by which to scale the fully-elongated internode length
     */
    void scaleInternodeMaxLength(float scale_factor);

    //! Set the fully-elongated (maximum) internode length
    /**
     * \param[in] internode_length_max_new Fully-elongated length of the internode
     */
    void setInternodeMaxLength(float internode_length_max_new);

    //! Scale the fully-elongated maximum length of every petiole on this phytomer, as a fraction of its current fully-elongated length
    /**
     * The petiole counterpart of \ref scaleInternodeMaxLength(float). The petiole's present length is left where it is and only the
     * length it is growing toward changes, so a phytomer creation function can give a leaf born on a young plant a shorter final
     * petiole without moving the petiole that is already there.
     * \param[in] scale_factor Fraction by which to scale the fully-elongated petiole length. Must be positive.
     */
    void scalePetioleMaxLength(float scale_factor);

    //! Set the leaf scale as a fraction of its total fully-elongated scale factor, carrying the petiole to the same fraction. Value is uniformly applied for all leaves/leaflets in the petiole.
    /**
     * The petiole is moved to the same fraction of its own fully-elongated length, which is what makes this the call to use when a leaf and the petiole it sits on are being posed together -- restoring a
     * saved plant, or copying one phytomer's state onto another. The growth model does not use it, because a petiole elongates on its shoot's internode rate rather than on the leaf's (see
     * \ref setPetioleScaleFraction(uint, float)); use \ref setPetioleAndLeafScaleFraction() to advance the two independently.
     * \param petiole_index Index of the petiole to which the leaf belongs
     * \param[in] leaf_scale_factor_fraction Fraction of the total fully-elongated leaf scale factor (i.e., =1 for fully-elongated leaf)
     */
    void setLeafScaleFraction(uint petiole_index, float leaf_scale_factor_fraction);

    //! Set the leaf scale as a fraction of its total fully-elongated scale factor for all petioles in the phytomer. Value is uniformly applied for all leaves/leaflets in the phytomer.
    /**
     * \param[in] leaf_scale_factor_fraction Fraction of the total fully-elongated leaf scale factor (i.e., =1 for fully-elongated leaf)
     */
    void setLeafScaleFraction(float leaf_scale_factor_fraction);

    //! Set the current petiole length as a fraction of its fully-elongated length, leaving the leaves it carries at the size they are.
    /**
     * A petiole is a stem segment rather than part of the blade, and it goes on extending after the blade it carries has finished expanding.
     * PlantArchitecture::advanceTime() therefore drives the petiole with the shoot's Shoot::elongation_rate_instantaneous -- the rate its internodes elongate at -- and the
     * blade with Shoot::leaf_expansion_rate_instantaneous. A shoot type that leaves ShootParameters::leaf_expansion_rate_max unset has one rate for both, and the petiole and blade then advance together.
     *
     * The centerline, the radii and the undeformed rest shape are scaled about the petiole base, and every leaf is re-seated at the arclength fraction along the centerline where it already sat, so the
     * leaves ride out along the petiole as it lengthens without changing size. The bend applied by \ref bendPetioleUnderLeafWeight() is recomputed for the new length.
     *
     * \param[in] petiole_index Index of the petiole within this phytomer.
     * \param[in] petiole_scale_factor_fraction Fraction of the fully-elongated petiole length (i.e., =1 for a fully-elongated petiole)
     */
    void setPetioleScaleFraction(uint petiole_index, float petiole_scale_factor_fraction);

    //! Set the petiole length and the leaf size of one petiole, each as a fraction of its own fully-elongated value.
    /**
     * The two fractions are applied in one pass so that the leaves are scaled, re-seated along the rescaled petiole and bent under their new weight once rather than twice. \ref setLeafScaleFraction(uint,
     * float) and \ref setPetioleScaleFraction() are the special cases in which one of the two fractions is held where it is.
     *
     * \param[in] petiole_index Index of the petiole within this phytomer.
     * \param[in] petiole_scale_factor_fraction Fraction of the fully-elongated petiole length (i.e., =1 for a fully-elongated petiole)
     * \param[in] leaf_scale_factor_fraction Fraction of the total fully-elongated leaf scale factor (i.e., =1 for fully-elongated leaf)
     */
    void setPetioleAndLeafScaleFraction(uint petiole_index, float petiole_scale_factor_fraction, float leaf_scale_factor_fraction);

    //! Prescribe the base position, orientation and size of every leaf on one petiole from measured geometry.
    /**
     * This replaces the placement of all leaves on a petiole with values supplied by the caller, rather than the placement derived from the petiole geometry and the shoot type's leaf parameters. Every leaf on the
     * petiole is set in one call, because a compound leaf's leaflets are not independent: their roll and yaw signs, and the prototype each is a copy of, are all determined by the leaflet's position along the
     * petiole.
     *
     * Each leaf object is rebuilt from its prototype and oriented from the caller's angles in the leaf's rest frame on the petiole, as described for PlantArchitecture::setPetioleLeafGeometry().
     * The prescribed geometry is exempted from the leaf expansion performed by PlantArchitecture::advanceTime(), and prescribed leaves do not droop under self-weight: the blade keeps the prototype's rest shape.
     *
     * \param[in] petiole_index Index of the petiole within this phytomer.
     * \param[in] leaf_bases_new Base position of each leaf in world coordinates, one per leaf on the petiole and ordered as the petiole's existing leaves are.
     * \param[in] leaf_rotations_new Roll, pitch and yaw of each leaf in radians, as intrinsic rotations in the leaf's rest frame on the petiole (see PlantArchitecture::setPetioleLeafGeometry()).
     * \param[in] leaf_sizes_new Fully-elongated size of each leaf in meters. All sizes must be greater than zero.
     * \param[in] leaf_stalk_lengths_new [optional] Length of each leaf's stalk (petiolule) in meters, for a leaf prototype that builds one: the distance from the leaf's base to where its blade begins.
     * One per leaf, all greater than zero; with none given each stalk is the species' share of its blade's length (LeafPrototype::petiolule_length). Either way a stalk is as thick as the petiole is where
     * the leaf's base lies along it.
     * \note Supplying a different number of leaves than the petiole carries raises an error rather than adding or removing leaves; change the count with setPetioleLeafCount() first.
     * \note Rebuilding each leaf object discards any primitive data a caller has attached to it. The object label and material are restored, but other primitive data is not.
     * \param[in] leaf_stalk_bases_new [optional] Where each leaf's stalk leaves the petiole, in world coordinates, one per leaf. A stalk is otherwise built back along its blade's own line from the leaf's base, and a
     * measured stalk seldom lies on that line: it rises from the petiole to a blade held a few millimetres off it. With these the stalk runs straight from the given point to where the blade begins and
     * bends there into the join under the blade, which is unchanged; the blade stays where its base, angles and size put it. A stalk may be no longer than its blade and must reach the blade from behind
     * its base, and it is as thick as the petiole where it leaves it; anything else raises an error. Needs a leaf prototype that builds a stalk.
     */
    void setPetioleLeafGeometry(uint petiole_index, const std::vector<helios::vec3> &leaf_bases_new, const std::vector<AxisRotation> &leaf_rotations_new, const std::vector<float> &leaf_sizes_new,
                                const std::vector<float> &leaf_stalk_lengths_new = {}, const std::vector<helios::vec3> &leaf_stalk_bases_new = {});

    //! Change the number of leaves (leaflets) on a petiole, rebuilding them procedurally.
    /**
     * The existing leaf objects on the petiole are deleted and leaf_count new ones are built from the leaf prototype, sized, oriented and spaced along the petiole exactly as a freshly created phytomer's
     * leaves are, using the phytomer's own leaf parameters. The rebuilt leaves do not droop as they age, and the petiole's expansion fraction is pinned first (see lockPetioleScale()) so that leaf_size_max
     * means the fully-elongated size. Afterwards setPetioleLeafGeometry() expects leaf_count entries.
     * \param[in] petiole_index Index of the petiole within the internode.
     * \param[in] leaf_count New number of leaves on the petiole; must be at least one.
     */
    void setPetioleLeafCount(uint petiole_index, uint leaf_count);

    //! Regenerate a petiole from its own pitch, curvature and length, leaning toward a given direction, and rebuild its leaves on it.
    /**
     * The petiole is generated by the rules the phytomer constructor uses: it leaves the internode tip at pitch_degrees off the internode axis, toward outward_direction projected onto the plane
     * perpendicular to that axis, and turns by curvature degrees per meter of its length about the same axis the pitch is taken about. The resulting path is prescribed with
     * setPetioleNodePositions(), so it is held as given and not bent or rescaled afterwards, and the petiole's leaves are then rebuilt on it procedurally as setPetioleLeafCount() rebuilds them.
     * \param[in] petiole_index Index of the petiole within the internode.
     * \param[in] outward_direction Direction the petiole leans toward, in world coordinates; any vector that is not parallel to the internode axis.
     * \param[in] pitch_degrees Angle between the petiole's base and the internode axis, in degrees.
     * \param[in] curvature Change in the petiole's direction along its length, in degrees per meter; negative bends it down, as the shoot parameter petiole.curvature does.
     * \param[in] length Length of the petiole in meters; must be greater than zero.
     */
    void setPetioleShape(uint petiole_index, const helios::vec3 &outward_direction, float pitch_degrees, float curvature, float length);

    //! Rebuild the leaflets of one petiole with their own count, spacing and sizes.
    /**
     * Sets this phytomer's own leaflet_offset, leaflet_scale and prototype_scale and rebuilds leaf_count leaves on the petiole from them, as setPetioleLeafCount() does. Other phytomers, and the shoot
     * type, are unchanged. A leaflet_offset that would place the outermost leaflets beyond the petiole's base is refused rather than reduced, as the constructor would silently reduce it.
     * \param[in] petiole_index Index of the petiole within the internode.
     * \param[in] leaf_count Number of leaves on the petiole; must be at least one.
     * \param[in] leaflet_offset Spacing between lateral leaflet pairs as a fraction of the petiole length; zero or greater.
     * \param[in] leaflet_scale Size of each lateral pair relative to the pair nearer the tip; greater than zero.
     * \param[in] prototype_scale Fully-elongated size of the terminal leaflet in meters; greater than zero.
     * \param[in] intercalary_leaflet_scale Size of the intercalary leaflets, if any, relative to the primary pair nearer the tip; zero for a leaf with none, in which case every lateral slot is a primary pair.
     * Negative (the default) keeps the phytomer's current value.
     */
    void setPetioleLeaflets(uint petiole_index, uint leaf_count, float leaflet_offset, float leaflet_scale, float prototype_scale, float intercalary_leaflet_scale = -1.f);

    //! Rebuild the leaflets of one petiole turned by their own pitch, yaw and roll.
    /**
     * Sets this phytomer's own leaf.pitch, leaf.yaw and leaf.roll, the angles every leaflet on a petiole is turned by relative to it (the shoot parameters of the same names), and rebuilds the petiole's
     * leaves with them. The values are the phytomer's, so a later rebuild of another of its petioles uses them too; the other petioles are not rebuilt here.
     * \param[in] petiole_index Index of the petiole within the internode.
     * \param[in] pitch_degrees Leaflet pitch in degrees.
     * \param[in] yaw_degrees Leaflet yaw in degrees.
     * \param[in] roll_degrees Leaflet roll in degrees.
     */
    void setPetioleLeafletAngles(uint petiole_index, float pitch_degrees, float yaw_degrees, float roll_degrees);

    //! Reshape the leaflets of one petiole with their own blade shape.
    /**
     * Sets this phytomer's own copy of the leaf prototype's leaf_aspect_ratio, longitudinal_curvature, lateral_curvature and midrib_fold_fraction, and reshapes the petiole's existing leaves to it in place:
     * each leaf keeps its object ID, object data, size, orientation and position, and any other shape term (such as its roll) it already had. The prototype is otherwise the shoot type's: one shape for
     * every phytomer, drawn per unique prototype. The values are the phytomer's, so a later rebuild of another of its petioles uses them too. The shape is recorded by
     * PlantArchitecture::writePlantStructureXML().
     * \param[in] petiole_index Index of the petiole within the internode.
     * \param[in] aspect_ratio Ratio of blade width to length; greater than zero.
     * \param[in] longitudinal_curvature Curvature along the blade's length (+ up, - down).
     * \param[in] lateral_curvature Curvature across the blade's width (+ up, - down).
     * \param[in] midrib_fold_fraction Fold along the midrib, from -1 (halves folded down shut) through 0 (flat) to 1 (halves folded up shut).
     */
    void setPetioleLeafShape(uint petiole_index, float aspect_ratio, float longitudinal_curvature, float lateral_curvature, float midrib_fold_fraction);

    //! Roll the leaflets of one petiole about their midribs
    /**
     * Sets this phytomer's own copy of LeafPrototype::roll_fraction and reshapes the petiole's existing leaves to it in place, as Phytomer::setPetioleLeafShape() does. See
     * PlantArchitecture::setPetioleLeafRoll().
     * \param[in] petiole_index Index of the petiole within the internode.
     * \param[in] roll_fraction Roll as a fraction of a closed tube, from 0 (flat) to 1.
     */
    void setPetioleLeafRoll(uint petiole_index, float roll_fraction);

    //! Shape a leaf currently has, as recorded when it was built or last reshaped
    /**
     * \param[in] petiole_index Index of the petiole within the internode.
     * \param[in] leaf_index Index of the leaf on the petiole.
     * \return The leaf's blade shape. Raises an error for a leaf whose shape was not recorded, which is one not built by GenericLeafPrototype().
     */
    [[nodiscard]] LeafShapeValues getLeafShape(uint petiole_index, uint leaf_index) const;

    //! Rebuild one existing leaf's blade to a new shape inside its existing object
    /**
     * The blade lattice is rebuilt in the leaf's own frame from \p shape and carried through the leaf's transformation matrix, so the leaf keeps its object ID, object data, size, orientation and position;
     * a leaf that droops under its own weight is deflected again from the new rest shape. When the lattice has as many cells across the blade as the new shape needs, only its vertices move and its
     * primitives (and their data) are kept. A roll needs enough cells across the blade to form its arc, and a coarser blade is re-meshed with Context::replacePolymeshObjectPrimitives(): its blade
     * primitives are replaced by finer ones, each taking the primitive data of the old facet whose texture cell contains it.
     * \param[in] petiole_index Index of the petiole within the internode.
     * \param[in] leaf_index Index of the leaf on the petiole.
     * \param[in] shape New blade shape. Its subdivisions_y is the least number of cells wanted across the blade; its subdivisions_x and lattice_offset must be the leaf's own.
     */
    void reshapeLeaf(uint petiole_index, uint leaf_index, const LeafShapeValues &shape);

    //! Whether reshapeLeaf() would keep a leaf's primitives for a new shape, rather than re-mesh its blade
    /**
     * reshapeLeaf() moves the vertices of the existing blade lattice when it has as many cells across the blade as the new shape needs, and otherwise replaces the blade primitives with a finer lattice,
     * which changes their UUIDs. A caller that must keep the UUIDs (for example one that tracks primitives in a renderer) asks first and decides what to do instead of discovering the re-mesh afterwards.
     * \param[in] petiole_index Index of the petiole within the internode.
     * \param[in] leaf_index Index of the leaf on the petiole.
     * \param[in] shape New blade shape, with the same requirements as for reshapeLeaf(); an error is raised for a shape reshapeLeaf() would refuse.
     * \return True if reshapeLeaf() would keep every primitive of the leaf; false if it would re-mesh the blade.
     */
    [[nodiscard]] bool canReshapeLeafInPlace(uint petiole_index, uint leaf_index, const LeafShapeValues &shape) const;

    //! Set whether one leaf bends under its own weight. See PlantArchitecture::setLeafDroop().
    /**
     * \param[in] petiole_index Index of the petiole within the internode.
     * \param[in] leaf_index Index of the leaf on the petiole.
     * \param[in] droops True to bend the blade under its own weight from its rest shape, false to give it its rest shape.
     */
    void setLeafDroop(uint petiole_index, uint leaf_index, bool droops);

    //! Whether one leaf bends under its own weight
    /**
     * \param[in] petiole_index Index of the petiole within the internode.
     * \param[in] leaf_index Index of the leaf on the petiole.
     * \return True if the blade is bent under its own weight from its rest shape.
     */
    [[nodiscard]] bool doesLeafDroop(uint petiole_index, uint leaf_index) const;

    //! Lay one leaf's blade along a prescribed midrib, twisted by a roll at each point. See PlantArchitecture::setLeafMidrib().
    /**
     * \param[in] petiole_index Index of the petiole within the internode.
     * \param[in] leaf_index Index of the leaf on the petiole.
     * \param[in] midrib Points of the midrib in world coordinates, from the base of the blade to its tip.
     * \param[in] rolls Twist of the blade about its midrib at each point, in radians; one per point, or none for an untwisted blade.
     */
    void setLeafMidrib(uint petiole_index, uint leaf_index, const std::vector<helios::vec3> &midrib, const std::vector<float> &rolls);

    //! Lay one leaf's blade along a prescribed midrib, turned to face a given normal at each point. See PlantArchitecture::setLeafMidrib().
    /**
     * \param[in] petiole_index Index of the petiole within the internode.
     * \param[in] leaf_index Index of the leaf on the petiole.
     * \param[in] midrib Points of the midrib in world coordinates, from the base of the blade to its tip.
     * \param[in] normals Normal of the blade's upper side at each point, in world coordinates; one per point.
     */
    void setLeafMidrib(uint petiole_index, uint leaf_index, const std::vector<helios::vec3> &midrib, const std::vector<helios::vec3> &normals);

    //! Current midrib of one leaf's blade. See PlantArchitecture::getLeafMidrib().
    /**
     * \param[in] petiole_index Index of the petiole within the internode.
     * \param[in] leaf_index Index of the leaf on the petiole.
     * \return Midrib stations, normals and rolls of the blade as it now is.
     */
    [[nodiscard]] LeafMidrib getLeafMidrib(uint petiole_index, uint leaf_index) const;

    //! Return one leaf's blade from a prescribed midrib to its own shape. See PlantArchitecture::clearLeafMidrib().
    /**
     * \param[in] petiole_index Index of the petiole within the internode.
     * \param[in] leaf_index Index of the leaf on the petiole.
     */
    void clearLeafMidrib(uint petiole_index, uint leaf_index);

    //! Whether one leaf's blade is laid along a midrib prescribed by setLeafMidrib()
    /**
     * \param[in] petiole_index Index of the petiole within the internode.
     * \param[in] leaf_index Index of the leaf on the petiole.
     * \return True if the blade follows a prescribed midrib.
     */
    [[nodiscard]] bool hasLeafMidrib(uint petiole_index, uint leaf_index) const;

    //! Rebuild the leaflets of one petiole with each turned by its own offset from the leaf's angles.
    /**
     * Every leaflet on a petiole is otherwise turned by the same leaf.pitch, leaf.yaw and leaf.roll, mirrored from one side of the petiole to the other, so the two members of a pair are always mirror images.
     * Scanned compound leaves are not: their pairs' members differ by a median of about 13 degrees in yaw and 12 in pitch. This gives each leaflet its own offset, added to the leaf's angles, and rebuilds
     * the petiole's leaves with them. Yaw is taken in the leaf's mirrored sense, so a positive yaw offset swings a lateral further toward the tip whichever side it is on; on the terminal leaflet, which has
     * no yaw of its own, it turns the leaflet to one side of its petiole's tip (counter-clockwise seen from above for a positive offset). A roll offset on the terminal turns it about its own midrib; the
     * terminal takes no roll from leaf.roll, so the offset is the whole of its roll. The offsets are kept on the phytomer, reapplied by every later rebuild of the petiole with the same number of leaves,
     * cleared by one with a different number, and written to and read from the plant structure XML.
     * \param[in] petiole_index Index of the petiole within the internode.
     * \param[in] pitch_offsets_degrees Pitch offset of each leaflet in degrees, in the petiole's leaf order; one per leaflet.
     * \param[in] yaw_offsets_degrees Yaw offset of each leaflet in degrees; one per leaflet.
     * \param[in] roll_offsets_degrees Roll offset of each leaflet in degrees; one per leaflet.
     */
    void setPetioleLeafletAngleOffsets(uint petiole_index, const std::vector<float> &pitch_offsets_degrees, const std::vector<float> &yaw_offsets_degrees, const std::vector<float> &roll_offsets_degrees);

    //! Delete a petiole's leaves and build leaf_count new ones from this phytomer's leaf parameters (the shared body of setPetioleLeafCount(), setPetioleShape() and setPetioleLeaflets()).
    void rebuildPetioleLeaves(uint petiole_index, uint leaf_count);

    //! Set the fully-elongated (maximum) leaf prototype scale. Value is uniformly applied for all leaves/leaflets in the petiole.
    /**
     * \param petiole_index Index of the petiole to which the leaf belongs
     * \param[in] leaf_prototype_scale Leaf scale factor for fully-elongated leaf
     */
    void setLeafPrototypeScale(uint petiole_index, float leaf_prototype_scale);

    //! Set the fully-elongated (maximum) leaf prototype scale for all petioles in the phytomer. Value is uniformly applied for all leaves/leaflets in the petiole.
    /**
     * \param[in] leaf_prototype_scale Leaf scale factor for fully-elongated leaf
     */
    void setLeafPrototypeScale(float leaf_prototype_scale);

    //! Scales the leaf prototype by the given scale factor. Value is uniformly applied for all leaves/leaflets in the petiole.
    /**
     * This function scales the leaf prototype geometry by the specified factor.
     * \param petiole_index Index of the petiole to which the leaf belongs
     * \param[in] scale_factor Factor by which to scale the leaf prototype. Values less than 0 are clamped to 0.
     */
    void scaleLeafPrototypeScale(uint petiole_index, float scale_factor);

    //! Scales the leaf prototype by the given scale factor for all petioles in the phytomer. Value is uniformly applied for all leaves/leaflets in the petiole.
    /**
     * \param[in] scale_factor Factor by which to scale the leaf prototype. Values less than 0 are clamped to 0.
     */
    void scaleLeafPrototypeScale(float scale_factor);

    //! Scale the size every leaf on this phytomer is expanding toward, leaving the blades where they are.
    /**
     * The leaf counterpart of \ref scaleInternodeMaxLength(float) and \ref scalePetioleMaxLength(float). The blade's
     * present size is left untouched and only the size it is growing toward changes: the expansion fraction moves the
     * other way, so a fully-expanded leaf given a larger target becomes a partly-expanded leaf of the same size and
     * goes on growing on the next call to PlantArchitecture::advanceTime(). This is what hands a leaf built from
     * measured geometry back to the growth model still the size it was measured. It differs from
     * \ref scaleLeafPrototypeScale(float), which rescales the blade itself and leaves the fraction alone.
     *
     * A factor small enough to put the target below the leaf's present size is the one case in which the blade does
     * move: the leaf is taken down to the new target, since no growth step can shrink a leaf that is already past it.
     * The leaflets of a compound leaf are then re-seated along the petiole from \ref LeafParameters::leaflet_offset,
     * which discards a placement prescribed by PlantArchitecture::setPetioleLeafGeometry(); raising the target, the
     * case this function exists for, never re-seats anything.
     *
     * \param[in] scale_factor Factor by which to scale the mature leaf size. Must be positive.
     */
    void scaleLeafSizeMax(float scale_factor);

    //! Scale petiole geometry to match target length and radius
    /**
     * This function rescales an existing petiole's 3D geometry (vertices and radii) to match the specified target dimensions.
     * Used during XML deserialization to restore actual petiole sizes that may differ from parameter-based values.
     * \param petiole_index Index of the petiole to scale
     * \param target_length Target total length of the petiole in meters
     * \param target_base_radius Target radius at the petiole base in meters
     */
    void scalePetioleGeometry(uint petiole_index, float target_length, float target_base_radius);

    //! Prescribe the path of one petiole from measured node positions.
    /**
     * This replaces a petiole's centerline with one supplied by the caller, rather than the path integrated from the shoot type's petiole pitch and curvature. It is the organ-level counterpart of
     * PlantArchitecture::addShootFromNodePositions(), intended for reconstructing a plant from labelled measurements such as a segmented point cloud.
     *
     * The first supplied position is snapped exactly onto the tip of the parent internode, and the remaining positions are translated by the same amount, so the measured shape is preserved exactly and only the
     * absolute position of the petiole changes. This is required rather than cosmetic: Shoot::updateShootNodes() re-seats every petiole base onto its internode tip on every timestep, and it does so by translating
     * the whole phytomer by the discrepancy. A base that merely sits close to the tip therefore produces a small shift that is re-applied on every timestep.
     *
     * The prescribed petiole is exempted from the rescaling performed by PlantArchitecture::advanceTime(), which would otherwise stretch the whole centerline as the leaf expands. The number of supplied nodes is
     * free and need not match the shoot type's PetioleParameters::length_segments; the petiole tube is rebuilt to match.
     *
     * \param[in] petiole_index Index of the petiole within this phytomer.
     * \param[in] petiole_node_positions Positions of the petiole nodes in world coordinates, ordered from the petiole base to its tip. At least two positions are required, no two consecutive positions may be
     * coincident, and the first must lie within a tolerance of the parent internode tip.
     * \param[in] petiole_node_radii Radius of the petiole at each node, one per entry of petiole_node_positions. All radii must be greater than zero.
     * \note Call this before setPetioleLeafGeometry() for the same petiole. The leaf placement reads the petiole axis to orient each leaf, so prescribing the leaves first would orient them from the petiole path
     * this call is about to replace.
     */
    void setPetioleNodePositions(uint petiole_index, const std::vector<helios::vec3> &petiole_node_positions, const std::vector<float> &petiole_node_radii);

    /**
     * \brief Sets the scaling fraction for the inflorescence of a floral bud.
     *
     * \param[in] fbud Reference to the floral bud object whose inflorescence scaling will be updated.
     * \param[in] inflorescence_scale_factor_fraction Fractional value (between 0 and 1) representing the new scale factor for the inflorescence.
     */
    void setInflorescenceScaleFraction(FloralBud &fbud, float inflorescence_scale_factor_fraction) const;

    //! Swing a floral bud's peduncle and everything it carries rigidly about the bud's base position
    /**
     * Rotates the peduncle tube and the flower/fruit objects in the Context, the stored peduncle vertices, the flower/fruit attachment points and their stored peduncle axes, and composes the rotation
     * into FloralBud::contact_swing_axis / contact_swing_angle. Used to move fruit out of branches (see PlantArchitecture::enableFruitBranchCollisionAvoidance()) and to restore such a swing when a
     * plant is read back from XML.
     * \param[in] fbud Floral bud whose peduncle is swung
     * \param[in] axis Unit axis of the rotation, through fbud.base_position
     * \param[in] angle Rotation angle (radians)
     */
    void applyPeduncleSwing(FloralBud &fbud, const helios::vec3 &axis, float angle);

    //! Tilt one flower/fruit of a floral bud rigidly about its own attachment point on the peduncle
    /**
     * Rotates the flower/fruit object in the Context and composes the rotation into the fruit's AxisRotation::contact_tilt_axis / contact_tilt_angle. Used, together with applyPeduncleSwing(), to move
     * fruit out of branches (see PlantArchitecture::enableFruitBranchCollisionAvoidance()) and to restore such a tilt when a plant is read back from XML.
     * \param[in] fbud Floral bud carrying the fruit
     * \param[in] fruit_index Index of the fruit within fbud.inflorescence_objIDs
     * \param[in] axis Unit axis of the rotation, through the fruit's attachment point
     * \param[in] angle Rotation angle (radians)
     */
    void applyFruitContactTilt(FloralBud &fbud, size_t fruit_index, const helios::vec3 &axis, float angle);

    /**
     * \brief Sets the base position of the petiole to the specified position.
     *
     * Updates the position of the petiole, leaf bases, and floral buds, ensuring
     * all associated geometry and objects are translated accordingly.
     *
     * \param[in] base_position New base position for the petiole
     */
    void setPetioleBase(const helios::vec3 &base_position);

    /**
     * \brief Rotates a specified leaf around its base using the given rotation parameters.
     *
     * This function adjusts the leaf's orientation based on the specified petiole and leaf indices
     * and applies the provided axis rotation values.
     *
     * \param[in] petiole_index Index identifying the petiole to which the leaf belongs
     * \param[in] leaf_index Index of the specific leaf within the petiole
     * \param[in] rotation AxisRotation object containing pitch, roll, and yaw values for the rotation
     */
    void rotateLeaf(uint petiole_index, uint leaf_index, const AxisRotation &rotation);

    /**
     * \brief Rotates a petiole and everything anchored to it (its leaves) about the petiole base.
     *
     * Performs a solid-body rotation: the petiole tube and all leaves attached to it rotate together, along with everything the phytomer stores about them, so the leaves remain anchored where they
     * were on the petiole and the turned petiole is what later growth, and a plant structure file, describe. The floral buds of the phytomer and their peduncles and inflorescences are anchored to the
     * internode (not the petiole) and are not affected. The rotation is an increment on the petiole as it stands; to set a petiole's pose as an offset from the way it grew, which can be undone
     * exactly, use PlantArchitecture::setPlantJointPose() with a PLANT_JOINT_PETIOLE joint.
     *
     * \param[in] petiole_index Index identifying the petiole within the phytomer
     * \param[in] rotation AxisRotation, in radians, applied as pitch, then yaw, then roll. A positive pitch tilts the petiole further from the internode and a negative pitch toward it, about the
     *                     axis the petiole was tilted about when it was built. Yaw rotates the petiole around the internode axis (azimuth around the stem). Roll rolls the petiole around its own
     *                     length axis.
     */
    void rotatePetiole(uint petiole_index, const AxisRotation &rotation);

    /**
     * \brief Sets the vegetative bud state for all axillary vegetative buds in the phytomer.
     *
     * \param[in] state The new state to apply to the vegetative buds.
     */
    void setVegetativeBudState(BudState state);

    /**
     * \brief Sets the state of a vegetative bud at the specified petiole and bud indices.
     *
     * \param[in] state The desired vegetative bud state.
     * \param[in] petiole_index Index of the petiole where the bud is located.
     * \param[in] bud_index Index of the bud within the specified petiole.
     */
    void setVegetativeBudState(BudState state, uint petiole_index, uint bud_index);

    /**
     * \brief Sets the vegetative bud's state to the specified state.
     *
     * \param[in] state The desired state to set for the vegetative bud.
     * \param[in,out] vbud Reference to the vegetative bud whose state will be modified.
     */
    static void setVegetativeBudState(BudState state, VegetativeBud &vbud);

    /**
     * \brief Sets the floral bud state for all non-terminal buds.
     *
     * \param[in] state New BudState to apply to all non-terminal buds.
     */
    void setFloralBudState(BudState state);

    /**
     * \brief Sets the state of a specific floral bud.
     *
     * \param[in] state The new state to set for the floral bud.
     * \param[in] petiole_index The index of the petiole containing the target bud.
     * \param[in] bud_index The index of the bud within the specified petiole.
     */
    void setFloralBudState(BudState state, uint petiole_index, uint bud_index);

    /**
     * \brief Sets the state of a floral bud.
     *
     * \param[in] state New state to set for the floral bud.
     * \param[in,out] fbud Reference to the FloralBud object whose state is to be updated.
     */
    void setFloralBudState(BudState state, FloralBud &fbud);

    /**
     * \brief Removes the leaf and its associated properties from the phytomer.
     *
     * This function resets the phytomer's leaf-related attributes, such as petiole
     * radii, vertices, colors, length, and other geometric properties. It also
     * removes associated objects from the context.
     */
    void removeLeaf();

    /**
     * \brief Deletes the phytomer and its associated components.
     *
     * This method handles the cleanup and deletion of all parts associated with
     * the phytomer, including internode, leaves, inflorescence structures, and
     * any child or subsequent phytomers within the shoot.
     * \note Not available inside a phytomer callback (\ref PhytomerParameters::phytomer_creation_function or \ref PhytomerParameters::phytomer_callback_function): a call made from one does nothing, and is reported once, as an aggregated warning, at the end of the enclosing call. See \ref PlantArchCallbackEdits.
     */
    void deletePhytomer();

private:
    //! Cells across the blade that reshapeLeaf() builds a leaf's new shape with
    /**
     * The larger of the leaf's own, the shape's, and what its roll needs, rounded up to even. reshapeLeaf() keeps the leaf's primitives exactly when this equals the leaf's own; shared with
     * canReshapeLeafInPlace() so that the two cannot disagree. Raises an error for a leaf or shape that cannot be reshaped.
     * \param[in] petiole_index Index of the petiole within the internode.
     * \param[in] leaf_index Index of the leaf on the petiole.
     * \param[in] shape_requested New blade shape.
     * \param[in] caller Name of the public method, for error messages.
     */
    [[nodiscard]] uint reshapedLeafSubdivisionsY(uint petiole_index, uint leaf_index, const LeafShapeValues &shape_requested, const char *caller) const;

    //! Rest blade lattice of a leaf in its own object frame, built from its recorded shape, and the number of vertices its mesh has beyond the lattice (none, or a petiolule)
    /**
     * Raises an error for a leaf whose shape was not recorded (one not built by GenericLeafPrototype()) or whose object no longer carries its lattice.
     * \param[in] petiole_index Index of the petiole within the internode.
     * \param[in] leaf_index Index of the leaf on the petiole.
     * \param[in] caller Name of the public method, for error messages.
     * \param[out] appended_count Number of mesh vertices beyond the lattice.
     */
    [[nodiscard]] std::vector<helios::vec3> leafRestLatticeLocal(uint petiole_index, uint leaf_index, const char *caller, size_t &appended_count) const;

    //! Rest positions, in the leaf's own frame, of the petiolule vertices that follow a leaf's blade lattice in its mesh, if any
    /**
     * Taken from the leaf's recorded rest shape where it has one, and otherwise read back from the object, which then has its rest shape.
     */
    [[nodiscard]] std::vector<helios::vec3> leafRestAppendedVerticesLocal(uint petiole_index, uint leaf_index, size_t appended_count) const;

    //! Write a leaf's blade, laid along the midrib recorded in its LeafShapeState, into its object
    void layLeafAlongRecordedMidrib(uint petiole_index, uint leaf_index);

    //! Shared body of the two setLeafMidrib() overloads; exactly one of rolls and normals is used
    void setLeafMidribFromPolyline(uint petiole_index, uint leaf_index, const std::vector<helios::vec3> &midrib, const std::vector<float> *rolls, const std::vector<helios::vec3> *normals);

    //! Remove the leaves and petioles of the phytomer, optionally handing their Context objects to the caller instead of deleting them
    /**
     * \param[out] detached_objIDs If not null, the Context objects of the leaves and petioles are appended here and left in the Context; if null, they are deleted, as by removeLeaf().
     */
    void removeLeaf(std::vector<uint> *detached_objIDs);

    //! Delete the phytomer, the phytomers above it and their descendants, optionally handing their Context objects to the caller instead of deleting them
    /**
     * The plant bookkeeping is identical to deletePhytomer(). In detach mode the removed part of the shoot's internode tube is split off into its own object (see helios::Context::splitTubeObject()).
     * \param[out] detached_objIDs If not null, the removed Context objects are appended here and left in the Context; if null, they are deleted, as by deletePhytomer().
     */
    void deletePhytomer(std::vector<uint> *detached_objIDs);

    //! Pin this petiole's leaf and petiole growth fractions at 1, which is what exempts it from the rescaling in PlantArchitecture::advanceTime().
    /**
     * A leaf's rendered size is leaf_size_max multiplied by current_leaf_scale_factor, so pinning the fraction on a petiole that is still expanding would leave the bookkeeping overstating every leaf on it while
     * the geometry stayed where it was. The fraction being replaced is therefore folded into leaf_size_max first, leaving the product - and so the rendered geometry - unchanged.
     * The petiole's own fraction is pinned the same way, by folding it into \ref petiole_length_max so that the petiole's present length becomes the
     * length it is considered fully elongated at; both fractions have to be pinned, since either one still below 1 would have advanceTime() go on rescaling the measured geometry.
     *
     * Idempotent: a second call finds fractions of 1 and folds nothing.
     *
     * \param[in] petiole_index Index of the petiole within this phytomer.
     */
    void lockPetioleScale(uint petiole_index);

    /**
     * \brief Calculates optimal collision avoidance direction for internode growth.
     *
     * This method performs collision detection analysis to find an optimal growth direction
     * that avoids obstacles. It builds a BVH with target geometry and uses cone-based
     * path optimization to determine the best collision-free direction.
     *
     * \param[in] internode_base_origin Starting position for collision detection cone
     * \param[in] internode_axis Current growth direction before collision avoidance
     * \param collision_detection_active
     */
    helios::vec3 calculateCollisionAvoidanceDirection(const helios::vec3 &internode_base_origin, const helios::vec3 &internode_axis, bool &collision_detection_active) const;

    /**
     * \brief Applies hard obstacle avoidance to internode growth direction.
     *
     * This method checks for solid obstacles in the growth path and calculates rotation
     * needed to avoid them. Uses cone-based detection with a smaller angle than soft
     * collision avoidance for more precise obstacle detection.
     *
     * \param[in] current_position Current position of the growing internode
     * \param[in,out] internode_axis Current growth direction that will be modified if obstacle found
     * \return true if obstacle was found and avoidance was applied, false otherwise
     */
    bool applySolidObstacleAvoidance(const helios::vec3 &current_position, helios::vec3 &internode_axis) const;

    /**
     * \brief Calculates attraction point direction for internode growth.
     *
     * This method uses the CollisionDetection plugin to determine if any attraction points
     * lie within the plant's perception cone. If found, returns the direction toward the
     * closest attraction point for growth guidance.
     *
     * \param[in] internode_base_origin Current position of the growing internode
     * \param[in] internode_axis Current natural growth direction
     * \param[out] attraction_active Set to true if attraction points were found, false otherwise
     * \return Unit vector pointing toward the closest attraction point (if any found)
     */
    helios::vec3 calculateAttractionPointDirection(const helios::vec3 &internode_base_origin, const helios::vec3 &internode_axis, bool &attraction_active) const;

    /**
     * \brief Calculates optimal collision avoidance direction for petiole growth.
     *
     * This method performs collision detection analysis to find an optimal petiole direction
     * that avoids obstacles. It uses the petiole base position as the cone apex and the
     * proposed petiole direction as the central axis to find gaps in the environment.
     *
     * \param[in] petiole_base_origin Starting position for petiole collision detection cone
     * \param[in] proposed_petiole_axis Proposed petiole direction before collision avoidance
     * \param[out] collision_detection_active Flag indicating if collision detection was applied
     * \return Optimal petiole direction toward largest detected gap
     */
    helios::vec3 calculatePetioleCollisionAvoidanceDirection(const helios::vec3 &petiole_base_origin, const helios::vec3 &proposed_petiole_axis, bool &collision_detection_active) const;

    //! Calculate optimal fruit direction to avoid collisions using cone-based gap detection
    /**
     * \param[in] fruit_base_origin Starting position for fruit collision detection cone
     * \param[in] proposed_fruit_axis Proposed fruit direction before collision avoidance
     * \param[out] collision_detection_active Flag indicating if collision detection was applied
     * \return Optimal fruit direction toward largest detected gap
     */
    helios::vec3 calculateFruitCollisionAvoidanceDirection(const helios::vec3 &fruit_base_origin, const helios::vec3 &proposed_fruit_axis, bool &collision_detection_active) const;


public:
    // ---- phytomer data ---- //

    //! Coordinates of internode tube segments. Index is tube segment within internode
    std::vector<std::vector<helios::vec3>> petiole_vertices; // first index is petiole within internode, second index is tube segment within petiole tube
    std::vector<std::vector<helios::vec3>> leaf_bases; // first index is petiole within internode, second index is leaf within petiole
    std::vector<std::vector<std::vector<helios::vec3>>> peduncle_vertices; // first and second index as for floral_buds (see getFloralBudStorageIndex()), third index is tube segment within peduncle
    std::vector<std::vector<std::vector<float>>> peduncle_radii; // first and second index as for floral_buds (see getFloralBudStorageIndex()), third index is tube segment within peduncle
    std::vector<std::vector<float>> peduncle_length; // actual sampled length for each peduncle - indexed as for floral_buds (see getFloralBudStorageIndex())
    std::vector<std::vector<float>> peduncle_radius; // actual sampled radius for each peduncle - indexed as for floral_buds (see getFloralBudStorageIndex())
    std::vector<std::vector<float>> peduncle_pitch; // actual sampled pitch for each peduncle - indexed as for floral_buds (see getFloralBudStorageIndex())
    std::vector<std::vector<float>> peduncle_curvature; // actual sampled curvature for each peduncle - indexed as for floral_buds (see getFloralBudStorageIndex())
    std::vector<std::vector<float>> peduncle_roll; // actual sampled roll for each peduncle - indexed as for floral_buds (see getFloralBudStorageIndex())
    std::vector<std::vector<float>> peduncle_yaw; // actual sampled yaw for each peduncle - indexed as for floral_buds (see getFloralBudStorageIndex())
    float internode_pitch, internode_phyllotactic_angle;
    //! Azimuth of this phytomer around its shoot (radians): the phyllotactic angles of every phytomer up to and including this one, summed.
    /**
     * A phytomer carrying a petiole takes its direction by rotating the previous phytomer's actual petiole by its own
     * internode_phyllotactic_angle, so its azimuth accumulates without this. A phytomer with no petiole has nothing to
     * rotate from, and its ghost direction is built from this stored sum instead.
     */
    float internode_phyllotactic_azimuth = 0;

    std::vector<std::vector<float>> petiole_radii; // first index is petiole within internode, second index is segment within petiole tube
    //! Current arclength of each petiole's centerline. Index is petiole within internode.
    std::vector<float> petiole_length;
    //! Fully-elongated length of each petiole, which its current \ref petiole_length is a fraction of. Index is petiole within internode.
    /**
     * The petiole counterpart of \ref internode_length_max -- the length drawn from PhytomerParameters::PetioleParameters::length when the phytomer was built, which the petiole reaches once
     * \ref current_petiole_scale_factor gets to 1. Recorded rather than read back from the parameter, which the Phytomer constructor resamples once the petiole has been built, so that a later reader would
     * get some other phytomer's draw; PlantArchitecture::readPlantStructureXML() rebuilds a petiole's curvature against this length. A petiole whose centerline was prescribed, or whose scale was otherwise
     * pinned, carries its own measured length here (see lockPetioleScale()).
     */
    std::vector<float> petiole_length_max; // index is petiole within internode
    std::vector<float> petiole_pitch; //!< Pitch angle of each petiole in radians; index is petiole within internode
    //! Whether the leaves of this phytomer have no petiole (PhytomerParameters::PetioleParameters::length of exactly 0) and sit directly on the stem
    /**
     * Such a leaf still has a petiole axis of minimal length, which runs along the leaf's base and gives the side of the stem it grows on, but none of the petiole parameters: \ref Phytomer::petiole_pitch holds the
     * blade's angle off the internode, drawn from PhytomerParameters::LeafParameters::pitch, and the blade is posed in the frame of the stem (see orientLeaf()).
     */
    bool leaf_is_sessile = false;
    std::vector<float> petiole_curvature; // index is petiole within internode
    std::vector<float> petiole_taper; // taper value for each petiole (tip_radius/base_radius ratio)
    std::vector<helios::vec3> petiole_axis_initial; // initial petiole axis (before curvature) for each petiole
    std::vector<helios::vec3> petiole_rotation_axis; // rotation axis for curvature application for each petiole
    std::vector<std::vector<float>> leaf_size_max; // first index is petiole within internode, second index is leaf within petiole
    std::vector<std::vector<AxisRotation>> leaf_rotation; //!< Roll, pitch and yaw applied to each leaf, as angles rather than as the rotations built from them. First index is petiole within internode, second index is leaf within petiole.

    std::vector<helios::RGBcolor> internode_colors; // index is segment within internode tube
    std::vector<helios::RGBcolor> petiole_colors; // index is segment within petiole tube

    //! Object ID of the tube making up each petiole. Index is petiole within internode.
    //! Equal to no_petiole_objID where the petiole has no geometry (suppressed, or too small to build).
    std::vector<uint> petiole_objIDs;

    //! Sentinel marking a petiole that has no Context geometry. Zero is a valid object ID, so
    //! absence cannot be signalled with it.
    static constexpr uint no_petiole_objID = 4294967294;

    //! Object IDs of all petioles on this phytomer that currently have geometry in the Context.
    /**
     * Petioles whose geometry was suppressed, was never built, or has since been deleted are
     * omitted, so the result can be passed directly to Context methods that require every object
     * to exist.
     * \return Object IDs of the existing petiole tubes.
     */
    [[nodiscard]] std::vector<uint> getExistingPetioleObjIDs() const;

    //! Re-deflect one leaf under its own weight for its current size
    /**
     * Deflects the leaf from its prototype's undeformed rest shape rather than from its current geometry, so repeated growth steps cannot accumulate and creep the leaf downward. Does nothing for a species
     * whose leaves are rigid, for a leaf with no cached rest shape, or for a leaf that has not measurably grown since it was last deflected.
     * \param[in] petiole_index Index of the petiole within the internode.
     * \param[in] leaf_index Index of the leaf within the petiole.
     */
    void deformLeafUnderSelfWeight(uint petiole_index, uint leaf_index);

    //! Bend one petiole, together with the leaves it carries, under the weight of its leaflets for the leaf's current size and the petiole's age
    /**
     * The petiole is bent as a tapered cantilever clamped at its insertion by bendPetioleCenterline() (see PhytomerParameters::PetioleParameters::flexibility). The bent shape is always computed from the
     * undeformed rest shape in \ref petiole_rest_offsets rather than from the current shape, so repeated calls do not accumulate. The insertion is clamped, so the petiole keeps leaving the stem at its
     * generated pitch and the droop appears beyond it as curvature along the length. Each leaf stays attached at its arclength along the centerline and is turned
     * by the rotation that carries the centerline's previous tangent at that point onto its new one. Does nothing for a rigid petiole, for a petiole whose centerline was prescribed, for a petiole carrying a
     * prescribed leaf, or when neither the load nor the compliance has changed since the last call.
     * \param[in] petiole_index Index of the petiole within the internode.
     */
    void bendPetioleUnderLeafWeight(uint petiole_index);

    //! Move the leaves of a petiole from one centerline of the petiole to another
    /**
     * Each leaf keeps the fraction of the petiole's length it is attached at, is turned by the rotation that carries the centerline's old direction there onto its new one, and is moved to the new
     * position of that point. The leaf objects are rotated and translated rather than rebuilt, and the stored leaf bases are updated. The petiole's own stored centerline is not changed.
     * \param[in] petiole_index Index of the petiole within the phytomer.
     * \param[in] previous_nodes Centerline the leaves are currently attached along.
     * \param[in] previous_arclength Arclength from the base to each node of previous_nodes.
     * \param[in] new_nodes Centerline to carry the leaves to.
     * \param[in] new_arclength Arclength from the base to each node of new_nodes.
     * \param[in] leaf_arclength_fraction Fraction of the petiole's length at which each leaf is attached, indexed as the petiole's leaves.
     */
    void carryLeavesAlongPetiole(uint petiole_index, const std::vector<helios::vec3> &previous_nodes, const std::vector<float> &previous_arclength, const std::vector<helios::vec3> &new_nodes, const std::vector<float> &new_arclength,
                                 const std::vector<float> &leaf_arclength_fraction);

    //! Record the current centerline of one petiole as its undeformed rest shape
    /**
     * Called wherever a petiole's centerline is generated or replaced wholesale, so that \ref bendPetioleUnderLeafWeight() bends from the new shape. Also marks the petiole as needing to be bent again.
     * \param[in] petiole_index Index of the petiole within the internode.
     */
    void recordPetioleRestShape(uint petiole_index);

    //! Apply the full rotation chain that orients one leaf on its petiole
    /**
     * Rolls, pitches and yaws the leaf, turns it to the azimuth of the petiole it hangs from, and applies the curvature-aware blade-up correction, then records the three angles in \ref leaf_rotation. Called
     * both when a phytomer is first built and when one is rebuilt by PlantArchitecture::readPlantStructureXML(), so that a restored leaf is oriented identically to a grown one.
     * Samples nothing, so the caller owns the random draws and their order.
     * \param[in] objID_leaf Object ID of the leaf to orient. It must already be scaled and still be at the origin.
     * \param[in] petiole_index Index of the petiole within the internode.
     * \param[in] leaf_index Index of the leaf within the petiole.
     * \param[in] leaves_per_petiole Number of leaves on this petiole. One selects the unifoliate roll, more than one the compound roll.
     * \param[in] ind_from_tip Signed offset of this leaf from the middle of the petiole; zero for the tip leaf of a compound leaf and for a unifoliate leaf.
     * \param[in] compound_rotation Rotation about the petiole that separates the leaflets of a compound leaf; zero for a unifoliate leaf.
     * \param[in] petiole_tip_axis Direction of the petiole at its tip, in world coordinates.
     * \param[in] leaf_roll_angle Leaf roll angle (radians), as stored in \ref leaf_rotation rather than as applied.
     * \param[in] leaf_pitch_angle Leaf pitch angle (radians), as stored in \ref leaf_rotation rather than as applied.
     * \param[in] leaf_yaw_angle Leaf yaw angle (radians), as stored in \ref leaf_rotation rather than as applied.
     * \param[in] prescribed Whether the angles were prescribed from measurements. A prescribed leaf does not go through the procedural chain: its angles are intrinsic rotations in the leaf's rest frame on
     * the petiole, the same for every leaf position and node (see PlantArchitecture::setPetioleLeafGeometry()), and it receives no side signs, elevation offsets or blade-up correction.
     * \param[in] terminal_roll_angle Roll (radians) of a compound leaf's terminal leaflet about its own midrib, from its offset in setPetioleLeafletAngleOffsets(). The terminal takes no roll from
     * leaf_roll_angle, so this is the only roll it is given; ignored for every other leaf and for a prescribed one.
     */
    void orientLeaf(uint objID_leaf, uint petiole_index, uint leaf_index, int leaves_per_petiole, float ind_from_tip, float compound_rotation, const helios::vec3 &petiole_tip_axis, float leaf_roll_angle, float leaf_pitch_angle, float leaf_yaw_angle,
                    bool prescribed = false, float terminal_roll_angle = 0.f);

    //! Re-aim one leaf so that its blade faces a given direction, keeping \ref leaf_rotation in step with the geometry
    /**
     * Solves for the roll and pitch that carry this leaf's blade onto \a target_normal and applies the difference as a single rotation about the leaf's own base, so the leaf stays attached to its petiole and
     * keeps the azimuth of the petiole it hangs from. The angles are written back to \ref leaf_rotation, which is what makes the new orientation survive a
     * PlantArchitecture::writePlantStructureXML() / PlantArchitecture::readPlantStructureXML() round trip: the reader replays \ref orientLeaf() from those angles. Rotating the leaf object directly, as
     * Context::setObjectAverageNormal() does, changes the geometry without changing the record and is silently lost on reload.
     *
     * The part of the orientation chain outside roll and pitch -- the petiole's own azimuth and elevation, the compound rotation, and the curvature-aware blade-up correction -- is recovered from the leaf's
     * current transformation matrix rather than recomputed, because the blade-up term depends on the petiole length and leaf size at the moment the leaf was last oriented.
     *
     * \param[in] petiole_index Index of the petiole within the internode.
     * \param[in] leaf_index Index of the leaf within the petiole.
     * \param[in] target_normal Direction the blade should face, in world coordinates. Need not be normalized.
     * \note The leaf must have geometry in the Context. A blade whose facet normals cancel, or a target the roll-pitch pair cannot reach, raises an error rather than leaving the leaf half-turned.
     */
    void setLeafNormal(uint petiole_index, uint leaf_index, const helios::vec3 &target_normal, const helios::vec3 *current_normal = nullptr);

    //! Build one leaflet from this phytomer's leaf prototype
    /**
     * Calls the prototype function, first setting the petiolule radius from the petiole: the mature petiole radius where the leaflet attaches, divided by the
     * leaflet's mature size, both at the distribution means of the shoot type's definition. Petiole girth and leaflet size grow in proportion, so the stalk
     * then continues the petiole without a step at every age; only per-leaf random draws of size can put the two slightly apart.
     * \param[in] leaves_per_petiole Number of leaflets on the petiole.
     * \param[in] ind_from_tip Leaflet position counted from the tip (0 is the terminal leaflet).
     * \param[in] prototype_index Which of the unique prototypes is being built, which selects its variant in LeafPrototype::leaf_texture_series. A leaf that is not one of the unique prototypes uses the default, and so the first variant.
     * \param[in] leaflet_size The size this leaflet will be given, in meters, when it is not the shoot type's (a leaflet rebuilt on one phytomer); with petiole_radius, its petiolule is sized from these. Negative means the shoot type's.
     * \param[in] petiole_radius The radius of the petiole where this leaflet attaches, in meters, when it is not the shoot type's. Negative means the shoot type's.
     * \param[in] petiolule_length The length of this leaflet's own stalk in meters, when it is not the species' share of the blade's length (a leaflet placed from a measurement); needs leaflet_size. Negative means the species'.
     * \return Object ID of the new leaflet.
     */
    uint buildLeafPrototype(int leaves_per_petiole, float ind_from_tip, uint prototype_index = 0, float leaflet_size = -1.f, float petiole_radius = -1.f, float petiolule_length = -1.f);

    //! The thickest stalk (radius, m) a leaf placed by setPetioleLeafGeometry() is given
    /**
     * A placed leaf's stalk is as thick as the petiole is where it leaves it, but no thicker than this share of the leaf's own length: a stalk scales with the leaf it carries, and a small
     * leaflet on a thick rachis otherwise has a stalk nearly the size of its blade.
     *
     * \param[in] leaf_size Length of the leaf's blade, in meters.
     * \return The largest stalk radius, in meters.
     */
    [[nodiscard]] static float placedStalkRadiusMax(float leaf_size) {
        return 0.02f * leaf_size;
    }

    //! The sharpest turn (radians) between a placed leaf's stalk and its blade
    /**
     * A stalk given a base on the petiole by setPetioleLeafGeometry() runs from there to where the blade begins. It may come to the blade from behind, in the blade's own line (no turn), or rise to it
     * from beneath (a quarter turn) and a little past that, for a leaflet held over its petiole; a stalk that would turn further than this runs back along its blade and is refused.
     *
     * \return The largest angle between the stalk's line and the blade's, in radians (120 degrees).
     */
    [[nodiscard]] static float placedStalkTurnMax() {
        return 2.f * PI_F / 3.f;
    }

    //! Move the start of a placed leaf's stalk to where it leaves the petiole
    /**
     * See PlantArchitecture::setPetioleLeafGeometry(). The stalk's first ring is put at stalk_base, across the line from there to where the blade begins, and the ring where the blade begins is mitred
     * between that line and the blade's own; the join under the blade is left as built.
     * \param[in] objID_leaf Object ID of the leaf, already posed and translated.
     * \param[in] petiole_index Index of the petiole the leaf hangs from, whose radius the stalk takes where it leaves it.
     * \param[in] stalk_base Where the stalk leaves the petiole, in world coordinates.
     * \param[in] leaf_size Size of the leaf in meters, which the stalk may not be longer than.
     */
    void placeLeafStalk(uint objID_leaf, uint petiole_index, const helios::vec3 &stalk_base, float leaf_size);

    //! Draw a leaf object's petiolule with the petiole's material
    /**
     * The material of this phytomer's petiole is shared, so the stalk matches the petiole exactly and follows any edit to its material. If the phytomer has no
     * petiole geometry, a material of the name the petiole's would have is created in the petiole colour. A leaf without a petiolule is left unchanged.
     * \param[in] objID_leaf Leaf object whose petiolule primitives are assigned the material.
     */
    void assignPetioluleMaterial(uint objID_leaf);

    //! Label a freshly built leaf object's blade primitives "leaf" and colour its petiolule, as the cached prototypes are.
    /**
     * A generic prototype arrives unlabelled except for a petiolule it may carry, which keeps its own label. A prototype loaded from an OBJ is already labelled by its file groups and is left alone.
     * \param[in] objID_leaf Leaf object to label.
     */
    void labelLeafPrototype(uint objID_leaf);

    //! Rotation about the petiole that separates one leaflet of a compound leaf from the others
    /**
     * Leaflets held away from the petiole tip (a non-zero leaflet offset) are placed on alternating sides of the petiole, while leaflets all attached at the tip are fanned evenly around it. Shared by the
     * Phytomer constructor and PlantArchitecture::readPlantStructureXML() so that a restored compound leaf is arranged like a grown one.
     * \param[in] leaves_per_petiole Number of leaves on the petiole. One returns zero, since a unifoliate leaf is not rotated about its petiole.
     * \param[in] leaf_index Index of the leaf within the petiole.
     * \param[in] leaflet_offset_val Leaflet offset along the petiole, already clamped by clampOffset().
     * \return Rotation about the petiole, in radians.
     */
    static float compoundLeafRotation(int leaves_per_petiole, int leaf_index, float leaflet_offset_val);


    std::vector<std::vector<uint>> leaf_objIDs; // first index is petiole within internode, second index is leaf within petiole tube

    //! Which cached prototype each leaf was copied from, so its undeformed rest shape can be found again. Indexed as <code>leaf_objIDs</code>; -1 for a leaf that does not droop.
    std::vector<std::vector<int>> leaf_prototype_index;

    //! Which image of LeafPrototype::leaf_texture_series the leaves of this phytomer currently use. The largest uint means that none has been applied yet, so that the next update applies one to every leaf.
    uint leaf_texture_stage = std::numeric_limits<uint>::max();

    //! Give the leaves of this phytomer the image of LeafPrototype::leaf_texture_series for their age, if it has changed
    /**
     * All of a phytomer's leaves have the same age, so they change together. Only the color of a leaf changes (see Context::setPrimitiveColorTexture()), so its area and its shape are not affected.
     * Also writes the "leaf_texture_stage" object data of the leaves if it has been enabled with PlantArchitecture::optionalOutputObjectData(), which is 0 for a shoot type without stages.
     */
    void updateLeafTextureStage();

    //! Whether each leaf's pose was prescribed by setPetioleLeafGeometry() rather than generated. Indexed as <code>leaf_objIDs</code>. Recorded by PlantArchitecture::writePlantStructureXML() so that the pose is restored.
    std::vector<std::vector<bool>> leaf_pose_prescribed;

    //! Where the stalk of a leaf placed with its own stalk base leaves the petiole, by the leaf's object ID (see PlantArchitecture::setPetioleLeafGeometry()). Recorded by
    //! PlantArchitecture::writePlantStructureXML() so that the stalk is put back when the plant is read.
    std::map<uint, helios::vec3> leaf_stalk_base_placed;

    //! Each leaflet's own (pitch, yaw, roll) offset from the leaf's angles, in degrees, set by setPetioleLeafletAngleOffsets(). First index is petiole within internode, second leaf within petiole; empty for a
    //! petiole whose leaflets have none.
    std::vector<std::vector<helios::vec3>> leaflet_angle_offsets;

    //! Where a leaf being steered toward a prescribed leaf angle distribution is headed, and how far along it is. Indexed as <code>leaf_objIDs</code>.
    /**
     * Only leaves that have been assigned a target carry one. A leaf reaches its target as it finishes
     * expanding, so that it unfolds into the prescribed orientation rather than snapping to it on the
     * timestep it emerges. See PlantArchitecture::enablePlantLeafAngleDistributionTracking().
     */
    struct LeafAngleSteering {
        //! Whether this leaf has been given a target to steer toward
        bool steered = false;
        //! Direction the blade is to face once the leaf is fully expanded, in world coordinates
        helios::vec3 target_normal;
        //! Leaf scale fraction at the last steering step, so each step covers its share of the remaining turn
        float scale_at_last_step = 0.f;
    };

    //! Steering state of each leaf, parallel to <code>leaf_objIDs</code>. Empty entries are leaves that are not being steered.
    std::vector<std::vector<LeafAngleSteering>> leaf_angle_steering;

    //! Whether each petiole's centerline was prescribed by setPetioleNodePositions() rather than generated. Indexed by petiole within the internode. Recorded by PlantArchitecture::writePlantStructureXML().
    std::vector<bool> petiole_path_prescribed;

    //! Leaf scale at which each leaf was last deflected, used to skip redeforming a leaf that has not measurably grown. Indexed as <code>leaf_objIDs</code>.
    std::vector<std::vector<float>> leaf_last_deformed_scale;

    //! Shape of a leaf recorded on the phytomer itself, rather than in the shared prototype cache
    struct LeafShapeState {
        //! Shape the blade was built with or last reshaped to
        LeafShapeValues shape;
        //! Whether the leaf was reshaped from the shape it was built with (recorded by PlantArchitecture::writePlantStructureXML())
        bool reshaped = false;
        //! Whether the blade droops under its own weight, deflected from rest_vertices
        bool deformable = false;
        //! Undeformed blade lattice in the leaf's own frame, used in place of the prototype cache's when the leaf droops
        std::vector<helios::vec3> rest_vertices;
        //! Undeformed petiolule vertices that follow the lattice in the mesh, if any
        std::vector<helios::vec3> rest_appended_vertices;
        //! Position, in the frame buildLeafBladeLattice() builds the blade in, of the origin of the leaf object's own frame (see PlantArchitecture::LeafRestGeometry::lattice_frame_offset)
        helios::vec3 lattice_frame_offset;
        //! Whether the blade is laid along a midrib prescribed by setLeafMidrib()
        bool has_midrib = false;
        //! Prescribed midrib stations in the leaf's own frame, one per lattice column. Held in the leaf's frame so that any rigid motion of the leaf object carries them.
        std::vector<helios::vec3> midrib_stations;
        //! Twist of the blade at each prescribed midrib station, in radians, in the leaf's own frame
        std::vector<float> midrib_rolls;
        //! Factor by which the blade's rest lattice is scaled, in the leaf's own frame, for its midrib to have the prescribed length
        float midrib_length_scale = 1.f;
        //! Whether the blade drooped before its midrib was prescribed, which clearLeafMidrib() restores
        bool deformable_before_midrib = false;
    };

    //! Shapes of leaves that are not, or no longer, simply copies of a cached prototype: those built fresh by the prototype function, and those reshaped in place. Keyed by leaf object ID.
    std::map<uint, LeafShapeState> leaf_shape_state;

    //! Shape of the most recent blade built by buildLeafPrototype(), if it was built by GenericLeafPrototype()
    LeafShapeValues last_built_leaf_shape;
    bool last_built_leaf_shape_valid = false;

    //! Record on the phytomer the shape of a leaf just built by buildLeafPrototype() rather than copied from the prototype cache
    /**
     * Must be called before the leaf is scaled, rotated or translated: the offset of the leaf's own frame from its blade lattice is read from its transformation matrix, which is then a translation alone.
     */
    void recordBuiltLeafShape(uint objID_leaf);

    //! Position, in the frame of buildLeafBladeLattice(), of the origin of a leaf object's own frame, looked up as getLeafShape() looks up the shape
    [[nodiscard]] helios::vec3 leafLatticeFrameOffset(uint petiole_index, uint leaf_index) const;

    //! Bending flexibility this phytomer's leaves were built with
    /**
     * Resolved once when the phytomer is created and held, rather than being read from the prototype on each timestep. The prototype's value is a random parameter that resamples when it is read, so
     * consulting it every timestep would give one leaf a different stiffness at every step and make it flap rather than settle.
     */
    float leaf_flexibility = 0.f;

    //! Bending compliance of this phytomer's petioles
    /**
     * Resolved once from PhytomerParameters::PetioleParameters::flexibility when the phytomer is created and held for its life, for the same reason as \ref leaf_flexibility.
     */
    float petiole_flexibility = 0.f;

    //! Undeformed rest centerline of each petiole, as offsets of its nodes from the petiole base. Indexed as <code>petiole_vertices</code>.
    /**
     * Scaled with the petiole as its leaf grows and rotated with it by \ref rotatePetiole(); the base itself is not stored, so translation of the petiole by stem growth is carried automatically.
     * \ref bendPetioleUnderLeafWeight() bends this shape, never the current one.
     */
    std::vector<std::vector<helios::vec3>> petiole_rest_offsets;

    //! State each petiole was last bent for: compliance times leaf growth fraction, reference leaflet load, and leaflet count. Indexed by petiole within the internode; x < 0 marks a petiole that must be bent again.
    std::vector<helios::vec3> petiole_bend_state;

    PhytomerParameters phytomer_parameters;

    uint rank;
    //! .x = index of phytomer along shoot, .y = current number of phytomers on parent shoot, .z = maximum number of phytomers on parent shoot
    helios::int3 shoot_index;

    uint plantID;
    uint parent_shoot_ID;
    Shoot *parent_shoot_ptr;

    //! Time since the phytomer was created
    float age = 0;
    bool isdormant = false;

    float current_internode_scale_factor = 1;
    //! How far each petiole's leaves have expanded toward their fully-grown size, as a fraction. Index is petiole within internode.
    std::vector<float> current_leaf_scale_factor;
    //! How far each petiole has elongated toward \ref petiole_length_max, as a fraction. Index is petiole within internode.
    /**
     * Advances independently of \ref current_leaf_scale_factor, because a petiole elongates on its shoot's internode rate while the blade it carries expands on the leaf rate (see
     * \ref setPetioleScaleFraction()). For a shoot type that does not distinguish the two rates -- the default for every species in the plant library -- the two fractions take the same value at every step.
     */
    std::vector<float> current_petiole_scale_factor; // index is petiole within internode

    //! Structural carbon of this phytomer's internode already charged to the carbohydrate model (mol C)
    float internode_carbon_charged_molC = 0;
    //! Net photosynthesis of this phytomer's leaves accumulated by PlantArchitecture::accumulateHourlyLeafPhotosynthesis() since the last time step (mol C)
    float cumulative_net_photosynthesis_molC = 0;
    //! Structural carbon of this phytomer's leaves already charged to the carbohydrate model (mol C)
    float leaf_carbon_charged_molC = 0;

    //! Leaf area borne at or above this phytomer, including on child shoots. Accumulated incrementally by \ref Shoot::propagateDownstreamLeafArea() as leaves are added, removed or rescaled.
    float downstream_leaf_area = 0;

    //! Inflorescence area borne at or above this phytomer, including on child shoots. Refreshed once per shoot per time step by \ref Shoot::refreshDownstreamInflorescenceArea().
    float downstream_inflorescence_area = 0;

    //! Leaf area currently borne at or above this phytomer, including on child shoots (m^2). Unlike downstream_leaf_area it drops when leaves are shed. Refreshed once per time step by \ref
    //! Shoot::refreshLiveDownstreamLeafArea() when the carbohydrate model's sapwood ring is enabled (CarbonWoodParameters::sapwood_area_per_leaf_area).
    float live_downstream_leaf_area = 0;
    //! Highest live_downstream_leaf_area reached during the current cambial season (m^2)
    float ring_season_peak_leaf_area = 0;
    //! Mean internode cross-sectional area when the current cambial season began for this internode (m^2)
    float ring_season_start_area = 0;
    //! Cambial season (PlantInstance::cambium_season_index) that ring_season_peak_leaf_area and ring_season_start_area belong to; -1 before the first
    int ring_season_index = -1;

    std::vector<std::vector<VegetativeBud>> axillary_vegetative_buds; // first index is petiole within internode, second index is bud within petiole
    std::vector<std::vector<FloralBud>> floral_buds; // first index is petiole within internode, second index is bud within petiole

    float internode_radius_initial;
    //! Fully-elongated length of this phytomer's internode, which its current length is a fraction of.
    float internode_length_max;

    std::vector<float> internode_curvature_perturbations; //!< Stochastic curvature perturbation values for each internode segment (for exact XML reconstruction)
    std::vector<float> internode_yaw_perturbations; //!< Stochastic yaw perturbation values for each internode segment (for exact XML reconstruction)
    //! Angle in degrees that each internode segment was turned about the vertical back toward the heading its shoot set out on (for exact XML reconstruction)
    /**
     * The angle is saved rather than recomputed on reading because it is ill-conditioned: on a nearly vertical shoot the heading can lie almost opposite the initial heading, where the direction of the turn is decided by rounding.
     */
    std::vector<float> internode_azimuthal_restoring_angles;
    //! Angle in degrees that each internode segment was turned by the gravitropic response (for exact XML reconstruction)
    /**
     * The angle is saved rather than recomputed on reading because it is discontinuous: its inclination factor doubles where the shoot axis passes below horizontal, so for an axis close to horizontal, rounding decides which side of the jump the segment lands on.
     */
    std::vector<float> internode_gravitropic_angles;
    //! Arc length in meters that each segment of this internode was curved over when it was built, or zero if no curvature was applied to it.
    /**
     * This is the maximum internode length the phytomer was created with divided by its number of segments. It is kept separately because \ref internode_length_max can be rescaled after the internode has been built (see \ref scaleInternodeMaxLength()), and readPlantStructureXML() needs the original value to replay the curvature.
     */
    float internode_curvature_arc_length_step = 0;

    bool build_context_geometry_petiole = true;
    bool build_context_geometry_peduncle = true;

protected:
    helios::vec3 inflorescence_bending_axis;

    //! Position of the phytomer's node -- the tip of its internode -- when its organs were last seated on it
    /**
     * setPetioleBase() moves the phytomer's organs by how far the node has moved since. A phytomer with petioles has this in the base of its first petiole; one without petioles, which can still carry
     * floral buds, has nowhere else to keep it.
     */
    helios::vec3 seated_node_position;

    helios::Context *context_ptr;

    PlantArchitecture *plantarchitecture_ptr;

    void updateInflorescence(FloralBud &fbud);

    //! Radius to build the peduncle with, matched to the culm tip it attaches to
    /**
     * The peduncle joins the culm at a single point, so a peduncle radius that disagrees with the internode radius there renders as an abrupt step. This returns a radius that continues the stem's taper
     * across the junction, falling back to the configured \ref PeduncleParameters::radius when the phytomer has no internode geometry to match.
     *
     * \return Radius (meters) of the peduncle at its base
     */
    [[nodiscard]] float getReconciledPeduncleRadius();

    //! Colors of the peduncle tube nodes, blended from the culm color at its base where the peduncle continues the culm
    /**
     * A peduncle that continues the stem, as \ref getReconciledPeduncleRadius() matches in radius, also has to match it in color, or the junction shows as a band. Its color therefore runs from the
     * internode color at its base to \ref PeduncleParameters::color at its tip, so a peduncle colored to match the inflorescence it carries meets both the culm and the inflorescence without a step. A
     * lateral peduncle is uniformly \ref PeduncleParameters::color.
     *
     * Each tube segment is drawn in the color of its lower node, so the blend is spread over all but the last node: the first segment carries the culm color and the last the peduncle color.
     * \param[in] node_count Number of nodes along the peduncle tube.
     * \return Color of each node, from base to tip
     */
    [[nodiscard]] std::vector<helios::RGBcolor> getPeduncleNodeColors(size_t node_count) const;

    //! Turn a peduncle's direction by its curvature over one segment of its tube
    /**
     * A positive curvature turns the peduncle toward pointing straight up, and a negative one toward hanging straight down, about the horizontal axis across it, without carrying it past that vertical. A
     * peduncle lying along the vertical has no horizontal axis across it: one already pointing the way it is turned toward stays there, and one pointing the opposite way is left unturned, since there is no
     * direction for it to bend in.
     * \param[in] peduncle_axis Unit direction of the peduncle at the start of the segment.
     * \param[in] curvature_degrees_per_meter Peduncle curvature (degrees per meter), as \ref PeduncleParameters::curvature.
     * \param[in] segment_length Length of the segment (meters).
     * \return Unit direction of the peduncle at the end of the segment
     */
    [[nodiscard]] static helios::vec3 bendPeduncleAxis(const helios::vec3 &peduncle_axis, float curvature_degrees_per_meter, float segment_length);

    //! Whether this phytomer's peduncle should take its radius from the culm rather than from its own parameter
    /**
     * True for a terminal inflorescence on a herbaceous stem -- a grass panicle or tassel, which continues the stem's own axis and must match it across the junction. False for a lateral peduncle or one on a
     * woody stem, whose configured radius describes a genuinely slender flower stalk that should not be widened to match the branch bearing it.
     *
     * \return True if the peduncle radius should be derived from the culm-tip radius
     */
    [[nodiscard]] bool peduncleShouldMatchCulm() const;

    //! Re-match this phytomer's peduncle geometry to the current culm-tip radius
    /**
     * Peduncle geometry is built once, when the floral bud appears and the culm is at its thinnest, but the culm keeps thickening afterward. This re-applies \ref getReconciledPeduncleRadius() so the
     * junction stays continuous as the plant grows.
     */
    void updatePeduncleRadii();

    /**
     * \brief Create and position a single flower or fruit with specified rotation parameters.
     *
     * This function creates flower/fruit geometry and applies transformations. It is called by updateInflorescence()
     * during normal growth with randomly sampled parameters, and by XML restoration with saved deterministic parameters.
     *
     * \param[in,out] fbud FloralBud to add the flower/fruit to
     * \param[in] fruit_base Position on peduncle where flower/fruit attaches
     * \param[in] peduncle_axis Direction vector of peduncle at attachment point
     * \param[in] pitch Pitch rotation angle (radians) - rotates about Y-axis
     * \param[in] roll Roll rotation angle (radians) - rotates about X-axis
     * \param[in] azimuth Azimuth rotation angle (radians) - rotates about Z-axis to align with peduncle
     * \param[in] yaw_compound Compound yaw rotation angle (radians) - rotates about peduncle axis
     * \param[in] scale_factor Scale factor to apply to the flower/fruit prototype
     * \param[in] is_open_flower For flowers: true=open, false=closed. Ignored for fruit.
     * \param[in] gravity_fraction Fraction (0-1) of its angle from straight down by which the flower/fruit is finally turned toward hanging, about the horizontal line through its base. 0 for flowers.
     */
    void createInflorescenceGeometry(FloralBud &fbud, const helios::vec3 &fruit_base, const helios::vec3 &peduncle_axis, float pitch, float roll, float azimuth, float yaw_compound, float scale_factor, bool is_open_flower, float gravity_fraction);

    /**
     * Calculate the structural carbon (mol C) of the phytomer's current leaf area: leaf dry mass (area / SLA) times its carbon content. Growth respiration is not included.
     * \return Structural carbon of the phytomer's leaves (mol C).
     */
    [[nodiscard]] float calculatePhytomerConstructionCosts() const;

    /**
     * Calculate the structural carbon (mol C) of the phytomer's internode: its volume times the wood carbon density, which rises from initial_density_ratio of the mature value at emergence
     * to the mature value at maturity_age.
     * \param[in] node_index Index of this phytomer within its shoot.
     * \return Structural carbon of the internode (mol C).
     */
    [[nodiscard]] float calculateInternodeStructuralCarbon(uint node_index) const;

    /**
     * Calculate the total carbon cost (mol C) of constructing the flowers for a given FloralBud object.
     * Iterates over all flower object IDs stored in the FloralBud instance and accumulates the total construction cost.
     *
     * \param[in] fbud References a FloralBud object that contains inflorescence object IDs.
     * \return The total flower carbon construction cost (mol C).
     */
    [[nodiscard]] float calculateFlowerConstructionCosts(const FloralBud &fbud) const;

    /**
     * Calculate the structural carbon (mol C) of a floral bud's fruit at full size. The fruit geometry is scaled uniformly, so the mature volume is the current volume divided by the cube of the
     * scale factor, and dry mass follows volume at CarbonFruitParameters::density.
     * \param[in] fbud Floral bud carrying the fruit.
     * \return Structural carbon of the mature fruit (mol C), 0 if the bud has no fruit geometry. Growth respiration is not included.
     */
    [[nodiscard]] float calculateMatureFruitCarbon(const FloralBud &fbud) const;

    /**
     * Fraction of a fruiting bud's mature dry matter in place at its current stage of development (CARBON_SUBMODEL_TREE): the CarbonFruitParameters dry-matter curve evaluated at the fraction of
     * the fruit's development period (fruit set to maturity) that has elapsed. A plant whose fruit maturity threshold is disabled never develops its fruit, so the fraction stays 0.
     * \param[in] fbud Floral bud carrying the fruit.
     * \return Fraction of the mature fruit carbon in place, from 0 at fruit set to 1 at maturity; 0 for a bud that is not fruiting.
     */
    [[nodiscard]] float calculateFruitDryMatterFraction(const FloralBud &fbud) const;

    friend struct Shoot;
    friend class PlantArchitecture;
};

struct Shoot {

    //! Constructor
    Shoot(uint plant_ID, int shoot_ID, int parent_shoot_ID, uint parent_node, uint parent_petiole_index, uint rank, const helios::vec3 &shoot_base_position, const AxisRotation &shoot_base_rotation, uint current_node_number,
          float internode_length_shoot_initial, ShootParameters &shoot_params, std::string shoot_type_label, PlantArchitecture *plant_architecture_ptr);

    //! Constructs and appends shoot phytomers to the shoot structure.
    /**
     * \param[in] internode_radius Initial radius of the phytomer internode.
     * \param[in] internode_length Length of the internode for each phytomer.
     * \param[in] internode_length_scale_factor_fraction Fraction to scale the internode length.
     * \param[in] leaf_scale_factor_fraction Fraction to scale the leaf size.
     * \param[in] radius_taper Degree of tapering applied to reduce the internode radius along the shoot.
     * \note Not available inside a phytomer callback (\ref PhytomerParameters::phytomer_creation_function or \ref PhytomerParameters::phytomer_callback_function): a call made from one does nothing, and is reported once, as an aggregated warning, at the end of the enclosing call. See \ref PlantArchCallbackEdits.
     */
    void buildShootPhytomers(float internode_radius, float internode_length, float internode_length_scale_factor_fraction, float leaf_scale_factor_fraction, float radius_taper);

    //! Append a phytomer at the shoot apex
    /**
     * \param[in] internode_radius Initial radius of the internode
     * \param[in] internode_length_max Maximum length of the internode
     * \param[in] internode_length_scale_factor_fraction Fraction of the total fully-elongated internode length
     * \param[in] leaf_scale_factor_fraction Fraction of the total fully-elongated leaf scale factor (i.e., =1 for fully-elongated leaf)
     * \return Number of phytomers in the shoot after the new phytomer is appended
     * \note Cannot be used inside a phytomer callback (\ref PhytomerParameters::phytomer_creation_function or \ref PhytomerParameters::phytomer_callback_function): a call made from one raises an error. See \ref PlantArchCallbackEdits.
     */
    int appendPhytomer(float internode_radius, float internode_length_max, float internode_length_scale_factor_fraction, float leaf_scale_factor_fraction, const PhytomerParameters &phytomer_parameters);

    //! Append a phytomer at the shoot apex whose internode geometry is given rather than generated
    /**
     * The phytomer is built at the prescribed endpoints and is fully elongated on creation, so it is not
     * subsequently re-shaped by curvature or elongation. Everything else about the phytomer -- buds,
     * petioles, leaves and the shared internode tube object -- is created exactly as for appendPhytomer().
     * \param[in] prescribed_internode Endpoints and radii of the internode to build.
     * \param[in] phytomer_parameters Parameters of the phytomer to be added.
     * \return Number of phytomers in the shoot after the new phytomer is appended
     * \note Cannot be used inside a phytomer callback (\ref PhytomerParameters::phytomer_creation_function or \ref PhytomerParameters::phytomer_callback_function): a call made from one raises an error. See \ref PlantArchCallbackEdits.
     */
    int appendPhytomerFromNodePositions(const PrescribedInternode &prescribed_internode, const PhytomerParameters &phytomer_parameters);

    //! Shared implementation of appendPhytomer() and appendPhytomerFromNodePositions()
    /**
     * The two differ only in whether the internode geometry is generated or supplied; determining the parent
     * axes beforehand and initializing buds, leaf area and object data afterwards is identical for both.
     * \param[in] prescribed_internode Endpoints and radii to build the internode from, or null to generate it.
     * \return Number of phytomers in the shoot after the new phytomer is appended
     * \note Cannot be used inside a phytomer callback (\ref PhytomerParameters::phytomer_creation_function or \ref PhytomerParameters::phytomer_callback_function): a call made from one raises an error. See \ref PlantArchCallbackEdits.
     */
    int appendPhytomerInternal(float internode_radius, float internode_length_max, float internode_length_scale_factor_fraction, float leaf_scale_factor_fraction, const PhytomerParameters &phytomer_parameters,
                               const PrescribedInternode *prescribed_internode);

    //! Randomly sample the type of a child shoot based on the probabilities defined in the shoot parameters
    /**
     * \return Label of the randomly selected child shoot type.
     */
    [[nodiscard]] std::string sampleChildShootType() const;

    //! Get the node cap governing further apical growth of this shoot
    /**
     * Returns ShootGrowthState::max_nodes_growth_cap when it has been set, and otherwise the shoot
     * type's ShootParameters::max_nodes. This is the value the growth model must consult; the shoot
     * type's max_nodes on its own also serves as a build-time capacity check, so reading it directly
     * conflates "how large may this shoot be built" with "how much further may it grow".
     *
     * \return Maximum number of nodes this shoot may grow to.
     */
    [[nodiscard]] int getGrowthNodeCap();

    //! Get the node cap governing further apical growth of this shoot within the current season
    /**
     * \return Maximum number of nodes this shoot may produce in one season.
     */
    [[nodiscard]] int getGrowthNodeCapPerSeason();

    //! Get the label of the shoot type governing this shoot's future growth
    /**
     * Returns ShootGrowthState::growth_type_label when it has been set, and otherwise this shoot's own
     * type. A shoot built from measured geometry uses the two differently: its own type supplies the
     * parameters its existing wood was built with, while this one supplies the parameters for anything
     * it does next, including the type of the shoots its buds produce.
     *
     * \return Label of the shoot type governing future growth.
     */
    [[nodiscard]] const std::string &getGrowthTypeLabel() const;

    //! Randomly sample whether a vegetative bud should break into a new shoot
    /**
     * \param[in] node_index Index of the node along the shoot
     * \return True if the vegetative bud should break into a new shoot
     */
    [[nodiscard]] bool sampleVegetativeBudBreak(uint node_index);

    [[nodiscard]] bool sampleVegetativeBudBreak_carb(uint node_index);

    //! Factor on the probability that a dormant lateral bud breaks, from this shoot's growth in the season the bud formed
    /**
     * \return The vigour factor in [0, 1] (see ShootParameters::vegetative_bud_break_vigour_threshold), or a negative value if the vigour factor is not enabled for this shoot
     */
    [[nodiscard]] float vigourBudBreakFactor() const;

    //! Randomly sample whether the shoot should produce an epicormic shoot (water sprout) over timestep
    /**
     * \param[in] dt Time step in days
     * \param[out] epicormic_positions_fraction Vector of fractions of the shoot's length where epicormic shoots will be produced
     * \return Number of epicormic shoots to be produced; position of the epicormic shoot as a fraction of the shoot's length
     */
    uint sampleEpicormicShoot(float dt, std::vector<float> &epicormic_positions_fraction) const;

    /**
     * \brief Terminates the apical bud of the shoot.
     * This function marks the apical meristem of the shoot as no longer alive.
     */
    void terminateApicalBud();

    /**
     * \brief Terminates all axillary vegetative buds within the shoot.
     * This function iterates through all phytomers of the current shoot and marks the axillary vegetative buds as no longer alive.
     */
    void terminateAxillaryVegetativeBuds();

    /**
     * \brief Adds terminal floral buds to the shoot.
     * This function creates and initializes a specified number of terminal floral buds at the apex of the shoot.
     * Each bud is assigned its position, rotation, and bending axis, based on the shoot's parameters.
     * \note The number of floral buds is determined by the "max_terminal_floral_buds" parameter in the shoot's configuration,
     * and the parameter is resampled after creating the buds.
     */
    void addTerminalFloralBud();

    /**
     * \brief Calculates the total volume of the shoot's internode tubes.
     * This function iterates through all phytomers of the shoot and sums the volume
     * of the internode tube objects if they exist.
     * \return The total volume of the shoot's internode tubes.
     */
    [[nodiscard]] float calculateShootInternodeVolume() const;

    /**
     * \brief Calculates the total length of the shoot.
     * This function iterates through all phytomers in the shoot and sums their internode lengths.
     * \return Total length of the shoot.
     */
    [[nodiscard]] float calculateShootLength() const;

    /**
     * \brief Determines the shoot axis vector based on a specified fraction along the shoot's length.
     * \param[in] shoot_fraction A float value representing the fraction along the shoot's length (0 to 1).
     * \return A vec3 representing the direction vector of the shoot axis for the computed position.
     * \note The shoot_fraction parameter is clamped within the range of 0 to 1, with 0 corresponding to the base and 1 to the tip of the shoot. The function uses the closest phytomer's internode axis vector for calculation.
     */
    [[nodiscard]] helios::vec3 getShootAxisVector(float shoot_fraction) const;

    //! Visit the positions of one phytomer's internode nodes. Not intended for use outside the plug-in (see Phytomer::visitPetioleSpatialState()).
    /**
     * \param[in] node_index Index of the phytomer along the shoot.
     * \param[in] visitor Operation applied to each value.
     */
    void visitInternodeSpatialState(uint node_index, PlantSpatialStateVisitor &visitor);

    //! Visit what the shoot stores about how it sits on its parent, which turns with the parent node rather than with the shoot. Not intended for use outside the plug-in.
    /**
     * \param[in] visitor Operation applied to each value.
     */
    void visitAttachmentSpatialState(PlantSpatialStateVisitor &visitor);

    //! Visit the positions and directions stored for the shoot as a whole, which turn with the shoot. Not intended for use outside the plug-in.
    /**
     * \param[in] visitor Operation applied to each value.
     */
    void visitShootSpatialState(PlantSpatialStateVisitor &visitor);

    //! Reports whether this shoot has been pruned away entirely
    /**
     * A shoot pruned at node 0 keeps its slot in the plant's shoot tree so that shoot IDs remain
     * stable, but it has no phytomers, no internode geometry, and is no longer listed as a child of
     * its parent. Such a shoot contributes nothing to the plant and cannot be queried for geometry.
     * \return True if the shoot has no phytomers because it was pruned.
     * \sa PlantArchitecture::isShootPruned(), PlantArchitecture::pruneBranch()
     */
    [[nodiscard]] bool isPruned() const {
        return phytomers.empty();
    }

    /**
     * \brief Calculates the total leaf area of a shoot starting from a given node index.
     *
     * \param[in] start_node_index [optional] The index of the starting node in the shoot.
     * \return Total leaf area of the shoot and its child shoots starting from the given node.
     * \note Returns zero for a pruned shoot. Throws an error if start_node_index is out of range on a shoot that still has phytomers.
     */
    [[nodiscard]] float sumShootLeafArea(uint start_node_index = 0) const;

    //! Recompute the cached inflorescence area borne at or above every node of this shoot, and of its child shoots
    /**
     * An inflorescence keeps expanding after it is created, so the area it contributes cannot be accumulated once the way \ref Phytomer::downstream_leaf_area is; it has to be recomputed as the plant
     * grows. This does so for the whole shoot in a single sweep from the tip downward, filling \ref Phytomer::downstream_inflorescence_area at every node, and recurses into child shoots so their caches
     * are refreshed too. Call it once per shoot per time step, before reading any node's value with \ref sumDownstreamInflorescenceArea().
     *
     * Counts only those inflorescences that load the stem they are borne on, as determined by \ref Phytomer::inflorescenceLoadsStem(); a lateral flower cluster contributes nothing to the girth of the
     * branch it hangs from.
     *
     * \return Total inflorescence area supported by this shoot and everything above it
     */
    float refreshDownstreamInflorescenceArea();

    //! Recompute the live leaf area borne at or above every node of this shoot and of all shoots above it
    /**
     * Sets Phytomer::live_downstream_leaf_area for each phytomer by a single sweep from the shoot tip downward, recursing into child shoots.
     *
     * \return Total live leaf area (m^2) supported by this shoot and everything above it
     */
    float refreshLiveDownstreamLeafArea();

    //! Get the inflorescence area borne at or above a given node, including on child shoots
    /**
     * Returns the value cached by \ref refreshDownstreamInflorescenceArea(), which must have been called for this shoot at some point since the inflorescence geometry last changed.
     *
     * \param[in] start_node_index Node index to sum from
     * \return Surface area (m^2) of inflorescences supported at that node
     */
    [[nodiscard]] float sumDownstreamInflorescenceArea(uint start_node_index) const;

    /**
     * \brief Calculates the total volume of all child shoots starting from a specified node index.
     * \param[in] start_node_index The starting index of the node from which to sum child volumes.
     * \return The total volume of all child shoots starting from the given node index.
     * \note Returns zero for a pruned shoot. Throws an error if start_node_index is out of range on a shoot that still has phytomers.
     */
    [[nodiscard]] float sumChildVolume(uint start_node_index = 0) const;

    /**
     * \brief Propagates the given leaf area downstream through the specified shoot.
     *
     * \param[in] shoot Pointer to the shoot through which leaf area is propagated
     * \param[in] node_index Index of the node to start propagation from
     * \param[in] leaf_area Leaf area to add downstream
     */
    void propagateDownstreamLeafArea(const Shoot *shoot, uint node_index, float leaf_area);

    /**
     * \brief Updates the shoot node positions and their associated geometries.
     * \param[in] update_context_geometry Indicates whether the geometry context should be updated for the shoot nodes.
     */
    void updateShootNodes(bool update_context_geometry = true);

    uint current_node_number = 0;
    uint nodes_this_season = 0;
    //! Length (m) of the internodes this shoot added in its last completed season, recorded when it goes dormant. Sets the vigour of the laterals that break from the buds formed in that season (see
    //! ShootParameters::vegetative_bud_break_vigour_threshold and ShootParameters::lateral_vigour_ratio).
    float previous_season_extension_m = 0;

    helios::vec3 base_position;
    AxisRotation base_rotation;
    helios::vec3 radial_outward_axis;

    //! Offset that holds this shoot's base off the axis of its parent, and the parent state it was derived from.
    /**
     * The offset makes a branch emerge from the surface of its parent rather than from inside it, and
     * updateShootNodes() has to re-apply it whenever the parent's nodes are rebuilt. Deriving the
     * direction means reproducing the child's base rotation, which costs more than the node update it
     * feeds, so it is cached and recomputed only when the parent internode has turned or thickened.
     */
    helios::vec3 cached_base_offset = helios::make_vec3(0, 0, 0);
    //! Whether the side of the parent this shoot is seated on is held where it is, and carried with the parent, rather than worked out again from the parent's petiole or the shoot's base rotation
    /**
     * Set while the plant is posed and kept once a pose is made permanent (see PlantArchitecture::bakePlantPose()); put back if the pose is undone. The petiole and base rotation describe where the
     * shoot was seated as it grew; once the plant has been re-posed they no longer say where the shoot is, and working the seat out from them again would slide the shoot's base around its parent
     * the next time the parent thickened or the shoot was re-seated.
     */
    bool base_offset_pinned = false;
    helios::vec3 last_parent_internode_axis = helios::make_vec3(1e30f, 1e30f, 1e30f);
    float last_parent_node_radius = -1.f;

    //! Horizontal heading the shoot set out on, used as the target of the azimuthal restoring force.
    /** The tortuosity noise is applied to the shoot's direction with no memory of where it was going, so the
     * azimuth performs an unanchored random walk and long shoots wander far off their original bearing. This
     * records the initial horizontal heading so the walk can be pulled back towards it. Zero until the shoot
     * has a well-defined horizontal direction.
     */
    helios::vec3 initial_heading_horizontal = helios::make_vec3(0, 0, 0);

    const int ID;
    const int parent_shoot_ID;
    const uint plantID;
    const uint parent_node_index;
    const uint rank;
    const uint parent_petiole_index;

    //! Whether this shoot continues its parent's axis rather than branching off it.
    /**
     * True for a shoot created by appendShoot(), which carries the parent's axis onward from its tip and
     * keeps the parent's rank. False for a lateral branch from addChildShoot() or
     * addShootFromNodePositions(), which emerges from the side of the parent and is one rank deeper.
     *
     * The distinction matters when Shoot::updateShootNodes() re-seats the shoot's base: a lateral branch
     * is displaced one parent radius outward so it emerges from the parent's surface, whereas a
     * continuation belongs on the parent's axis and must not be displaced at all.
     */
    bool is_axis_continuation = false;

    float sugar_pool_molC = 0; // mol C
    float starch_pool_molC = 0;
    float total_carbohydrate_pool_molC = sugar_pool_molC + starch_pool_molC;

    //! Net photosynthesis credited to this shoot's own leaves since the last end-of-season branch evaluation (mol C)
    float season_net_photosynthesis_molC = 0;
    //! Wood maintenance respiration charged to this shoot since the last end-of-season branch evaluation (mol C)
    float season_maintenance_respiration_molC = 0;
    //! Construction of this shoot's wood, leaves, flowers and fruit, with growth respiration, since the last end-of-season branch evaluation (mol C)
    float season_growth_molC = 0;
    //! The part of season_growth_molC spent on this shoot's fruit (mol C)
    float season_fruit_cost_molC = 0;
    //! Probability that this shoot flowers when it next breaks dormancy, set at the season boundary (see ShootParameters::return_bloom_scale)
    float flowering_probability = 1;

    //! Per-leaf nitrogen tracking - maps leaf objID to nitrogen content per unit area (g N/m²)
    std::map<uint, float> leaf_nitrogen_gN_m2;
    //! Nitrogen (g N) each leaf below target, expanding or mature, wanted this timestep but did not receive from uptake; filled by remobilization. Rebuilt every timestep.
    std::map<uint, float> leaf_N_unmet_demand_gN;
    //! Leaf area (m²) at which each leaf's nitrogen per area was last stated, so that growth dilutes it rather than creating nitrogen
    std::map<uint, float> leaf_N_area_basis_m2;
    //! Nitrogen per area (g N/m²) of each senescing leaf when its senescence began; a leaf is in this map once it has started senescing
    std::map<uint, float> leaf_N_at_senescence_onset;

    float old_shoot_volume = 0;

    float phyllochron_increase = 5;
    float phyllochron_recovery = phyllochron_increase;

    float days_with_negative_carbon_balance = 0;
    //! Plant age (days) at which this tiller reached CarbonGrassParameters::tiller_establishment_leaves leaves and began to be judged on its own carbon balance; negative before then
    float tiller_established_age = -1;

    //! Break dormancy: wake the vegetative buds and resume growth, and wake the floral buds unless they have already woken this season
    void breakDormancy();
    //! Wake the floral buds of a dormant shoot without breaking its vegetative dormancy, starting the clocks of their phenological stages
    /**
     * Floral buds wake when the plant's dormancy breaks, while the shoot itself may break vegetative dormancy up to the plant's budbreak duration later (see PlantArchitecture::setPlantPhenologicalSpread()).
     */
    void wakeFloralBuds();
    void makeDormant();


    bool isdormant;
    //! Whether the floral buds have woken since the shoot last went dormant
    /**
     * Read only while the shoot is dormant: a dormant shoot whose floral buds are awake carries flowers and fruit while it waits to break vegetative dormancy.
     */
    bool floral_buds_awake = false;
    //! Days after the plant's dormancy break at which this shoot breaks vegetative dormancy
    /**
     * Drawn each season when the shoot's floral buds wake. See PlantArchitecture::setPlantPhenologicalSpread().
     */
    float budbreak_delay = 0;
    uint dormancy_cycles = 0;

    bool meristem_is_alive = true;

    float phyllochron_counter = 0;
    float elongation_max = 0.25;

    //! Ornstein-Uhlenbeck tortuosity curvature state (deg/m) applied about the shoot bending axis
    float curvature_perturbation = 0;
    //! Ornstein-Uhlenbeck tortuosity curvature state (deg/m) applied about the axis perpendicular to both the shoot axis and the bending axis
    /**
     * The name is kept for XML schema compatibility; this is not a rotation about the global vertical. Together with
     * curvature_perturbation it forms the two orthogonal components of an isotropic curvature perturbation in the shoot frame.
     */
    float yaw_perturbation = 0;

    //! Graviceptive gain beta (deg/m) -- rate of curvature production per unit inclination from vertical
    float gravitropic_curvature = 0;

    //! Target length of internodes grown at the apex; see PlantArchitecture::setShootInternodeLengthMax().
    float internode_length_max_shoot_initial;

    //! Sentinel value for internode_tube_objID meaning "no internode tube object exists in the Context"
    /**
     * Used both before the tube geometry has been built and after it has been deleted (e.g. by
     * pruneGroundCollisions()). Resetting the ID to this sentinel on deletion is what keeps a freed
     * object ID from being handed back to callers.
     */
    static constexpr uint no_internode_tube_objID = 4294967294;

    uint internode_tube_objID = no_internode_tube_objID;

    std::vector<std::vector<helios::vec3>> shoot_internode_vertices; // first index is phytomer within shoot, second index is segment within phytomer internode tube
    std::vector<std::vector<float>> shoot_internode_radii; // first index is phytomer within shoot, second index is segment within phytomer internode tube

    bool build_context_geometry_internode = true;

    // map of node number (key) to IDs of shoot children (value)
    std::map<int, std::vector<int>> childIDs;

    ShootParameters shoot_parameters;

    //! Growth state specific to this shoot, overriding the shoot type where set.
    ShootGrowthState growth_state;

    std::string shoot_type_label;

    float phyllochron_instantaneous;
    float elongation_rate_instantaneous;
    //! Rate at which this shoot's leaf blades are presently expanding, in the units of ShootParameters::leaf_expansion_rate_max.
    /**
     * Applies to the blades only. A petiole is a stem segment and elongates on elongation_rate_instantaneous, the
     * rate this shoot's internodes elongate at, so it goes on extending after the blade it carries has stopped.
     *
     * Resolved once when the shoot is created: it is the shoot type's ShootParameters::leaf_expansion_rate_max
     * where that is set, and otherwise elongation_rate_instantaneous, so a shoot type that does not distinguish
     * the two expands its leaves at the rate its internodes and petioles elongate. Held alongside elongation_rate_instantaneous
     * rather than read from the parameters each step so that the two rates can be modulated independently by a
     * growth model, in the way phyllochron_instantaneous is modulated by the carbohydrate model.
     */
    float leaf_expansion_rate_instantaneous;
    //! Internode elongation rate this shoot reaches when carbohydrate is not limiting, drawn when the shoot is created
    float elongation_rate_potential;
    //! Leaf expansion rate this shoot reaches when carbohydrate is not limiting, drawn when the shoot is created
    float leaf_expansion_rate_potential;

    //! Growth-priority multiplier awarded by PlantArchitecture::updateGrowthPriority().
    /** Sibling laterals do not grow alike in a real tree: a few well-lit ones become dominant, keep extending
     * for years and thicken into structural limbs, while the rest stay short twigs. This factor scales the
     * shoot's node ceiling and internode length so that the model can express that divergence. 1 is the
     * undifferentiated default, used by any plant that never calls updateGrowthPriority().
     */
    float growth_priority = 1.f;

    //! Shoot-type baseline that growth_priority scales, captured the first time the shoot is promoted.
    /** The promoted parameters must be derived from this fixed baseline rather than from their own current
     * values. Re-multiplying the live values each season compounds without bound -- capping growth_priority
     * alone does not stop it, because the parameters are what actually grow.
     */
    bool growth_priority_baseline_set = false;
    uint growth_priority_base_max_nodes = 0;
    uint growth_priority_base_max_nodes_per_season = 0;
    float growth_priority_base_internode_length_max = 0.f;
    float growth_priority_base_bud_break_max = 0.f;

    std::vector<std::shared_ptr<Phytomer>> phytomers;

    PlantArchitecture *plantarchitecture_ptr;

    helios::Context *context_ptr;
};

struct PlantInstance {

    PlantInstance(const helios::vec3 &a_base_position, float a_current_age, const std::string &a_plant_name, helios::Context *a_context_ptr) :
        base_position(a_base_position), current_age(a_current_age), plant_name(a_plant_name), context_ptr(a_context_ptr) {
    }
    std::vector<std::shared_ptr<Shoot>> shoot_tree;
    helios::vec3 base_position;
    float current_age;

    //! Pose of the plant set through PlantArchitecture::setPlantJointPose(), with the plant as it grew, which the pose is measured from. Null while the plant stands as it grew.
    std::shared_ptr<PlantPoseState> pose_state;

    //! Whether a pose has been made a permanent part of the plant (see PlantArchitecture::bakePlantPose()), so that its geometry is no longer what its growth parameters alone describe
    bool pose_baked = false;
    float time_since_dormancy = 0;
    helios::Context *context_ptr;
    std::string plant_name;
    //! Build parameters the plant was built with by \ref PlantArchitecture::buildPlantInstanceFromLibrary(): the value sampled for this plant of each structure parameter, and each organ parameter given as
    //! a fixed value. Kept for the life of the plant, because some of them, such as the tiller count of sorghum, are read as the plant grows.
    std::map<std::string, float> build_parameters;
    //! The build parameters that shape the plant's flowers and fruit, as distributions, a fixed value being a constant one. Its flower and fruit prototypes are sampled from them and cached by them, so
    //! plants that differ only in other parameters share their prototypes.
    std::map<std::string, BuildParameterDistribution> organ_build_parameters;
    std::pair<std::string, float> epicormic_shoot_probability_perlength_per_day; //.first is the epicormic shoot label string, .second is the probability

    // Phenological thresholds.
    //
    // These defaults encode "no phenology scheduled", mirroring disablePlantPhenology(). They are what
    // advanceTime() sees for any plant that never has setPlantPhenologicalThresholds() called on it --
    // i.e. a plant built through the manual API (addPlantInstance + addBaseStemShoot/appendShoot/
    // addChildShoot), or one restored from an XML file predating the phenology tags. All library
    // builders set every field explicitly, so they are unaffected by these values.
    //
    // A zero dd_to_dormancy made the dormancy predicate in advanceTime() true on the very first step
    // and, because time_since_dormancy is reset there, on every step thereafter -- repeatedly calling
    // Shoot::makeDormant(), which strips every leaf and marks all non-dormant buds BUD_DEAD. Since
    // breakDormancy() only revives buds that are not BUD_DEAD, the plant was permanently defoliated
    // and sterilized instead of growing.
    float dd_to_dormancy_break = 0; //!< 0 = no chilling requirement; a dormant shoot may break immediately
    float dd_to_flower_initiation = -1; //!< -1 = stage disabled (advanceTime() gates these on >= 0)
    float dd_to_flower_opening = -1; //!< -1 = stage disabled
    float dd_to_fruit_set = -1; //!< -1 = stage disabled
    //! Deliberately 1e6 rather than -1: unlike the stages above, this field is used as a divisor in the
    //! fruit-growth block of advanceTime(), and appendPhytomerToShoot() can set a bud BUD_FRUITING based
    //! purely on shoot structure without consulting dd_to_fruit_set. A negative value would then produce
    //! a negative scale factor and trip the assert in Phytomer::setInflorescenceScaleFraction().
    float dd_to_fruit_maturity = 1e6;
    float dd_to_dormancy = 1e6; //!< effectively "never", keeping the dormancy predicate false
    float max_leaf_lifespan = 1e6;
    bool is_evergreen = false;
    //! Days over which the shoots break vegetative dormancy, starting at dd_to_dormancy_break
    /**
     * Zero breaks every shoot at once. See PlantArchitecture::setPlantPhenologicalSpread().
     */
    float budbreak_duration = 0;
    //! Days over which the floral buds make their first reproductive transition, starting at its threshold
    /**
     * Zero makes every bud transition at once. See PlantArchitecture::setPlantPhenologicalSpread().
     */
    float flowering_duration = 0;
    //! Days by which the time fruit take to reach full size can exceed dd_to_fruit_maturity
    /**
     * Zero gives every fruit dd_to_fruit_maturity. See PlantArchitecture::setPlantPhenologicalSpread().
     */
    float fruit_maturity_duration = 0;
    //! The earliest image of LeafPrototype::leaf_texture_series that any leaf of the plant uses, whatever its age (see PlantArchitecture::setPlantMinimumLeafTextureStage())
    uint minimum_leaf_texture_stage = 0;

    float max_age = 999;

    //! Tracks whether the Context geometry has been synced since the plant reached max_age.
    //! Once a plant is static at max_age its geometry no longer changes, so it only needs to be
    //! pushed to the Context once rather than rebuilt every subsequent timestep.
    bool mature_geometry_synced = false;

    //! Running maximum whole-plant leaf area reached during the current season (m^2), used to decide which sweep's grid to
    //! keep as season_peak_shadow_grid. Reset to zero at dormancy onset.
    float season_peak_leaf_area = 0.f;

    //! Number of completed seasons (incremented at dormancy onset); a sapwood ring is owed per internode per season (CarbonWoodParameters::sapwood_area_per_leaf_area)
    int cambium_season_index = 0;

    //! Sparse shadow-propagation voxel grid, keyed by integer voxel coordinate, used by shade-based branch shedding. Each leaf
    //! deposits shadow into a pyramid of voxels below it, so branches deep inside the crown see a high accumulated shadow
    //! value while branches on the crown surface do not. Rebuilt once per advanceTime() sweep.
    /**
     * Implements the shadow-propagation model of Palubicki et al. (2009) "Self-organizing tree models for image synthesis",
     * ACM Transactions on Graphics 28(3), as a cheap surrogate for light interception: no ray tracing or visibility queries
     * are performed. A leaf in voxel (I,J,K) adds a*b^(-q) to every voxel (I+-p, J-q, K+-p) for q in [0,q_max], p in [0,q].
     */
    std::unordered_map<long long, float> shadow_grid;

    //! World position that the voxel keys of shadow_grid, crown_depth_grid and their season-peak copies are measured from.
    /**
     * Zero until the plant is moved by PlantArchitecture::setPlantBasePosition(), which shifts it by the same offset as the plant, so that the
     * cached grids keep describing the moved crown without having to be rebuilt.
     */
    helios::vec3 shadow_grid_origin = helios::make_vec3(0, 0, 0);

    //! Shadow grid captured at the seasonal peak of canopy density, for dormancy-requiring shoots.
    /**
     * Such shoots sample all of a season's buds at once when dormancy breaks, at which point the current canopy is empty and
     * the live shadow_grid would be empty too. They therefore read this snapshot of the densest canopy state reached during
     * the season just completed.
     */
    std::unordered_map<long long, float> season_peak_shadow_grid;

    //! Depth (in voxels) of each occupied voxel beneath the crown surface, keyed the same way as shadow_grid.
    /**
     * The vertical shadow pyramid alone measures how much canopy sits ABOVE a point, so it penalizes a low branch on the
     * outside of the crown just as heavily as one buried in the middle, even though the outer branch has open sky beside it.
     * This companion field measures how far a point is from the crown's outer surface in any direction, which is the quantity
     * that actually distinguishes "interior" from "exterior". A value of 0 means the voxel is on the crown surface.
     *
     * Computed by a breadth-first flood inward from the surface voxels, so the cost is linear in the number of occupied
     * voxels and no hull fitting or distance transform over empty space is required.
     */
    std::unordered_map<long long, int> crown_depth_grid;

    //! Crown-depth field captured alongside season_peak_shadow_grid.
    std::unordered_map<long long, int> season_peak_crown_depth_grid;

    //! Edge length (m) of the shadow grid voxels for this plant. Zero until the grid has been built at least once.
    float shadow_grid_voxel_size = 0.f;

    //! Number of leaves the shadow grid was last built from. Rebuilding is skipped while the leaf count is unchanged, since the
    //! grid only feeds branch shedding and growth priority and does not need to track sub-timestep leaf expansion exactly.
    size_t shadow_grid_leaf_count = SIZE_MAX;

    //! Voxel size of season_peak_shadow_grid, which may differ from the live grid as the plant grows.
    float season_peak_shadow_grid_voxel_size = 0.f;

    //! Position (m) of each leaf of the season's peak canopy, captured alongside season_peak_shadow_grid for branch shedding.
    std::vector<helios::vec3> season_peak_leaf_positions;

    //! One-sided area (m^2) of each leaf in season_peak_leaf_positions.
    std::vector<float> season_peak_leaf_areas;

    //! ID of the shoot bearing each leaf in season_peak_leaf_positions.
    std::vector<uint> season_peak_leaf_shootIDs;

    CarbohydrateParameters carb_parameters;

    //! Temperature factor applied to the 20 C maintenance respiration rates, set by PlantArchitecture::adjustPlantMaintenanceRespiration()
    float respiration_temperature_factor = 1.f;
    //! Soluble sugar held in the root system (mol C)
    float root_sugar_pool_molC = 0;
    //! Starch held in the root system (mol C)
    float root_starch_pool_molC = 0;
    //! Carbon fluxes over the most recent advanceTime() call
    PlantCarbonBudget carbon_budget;
    //! Calls to PlantArchitecture::accumulateHourlyLeafPhotosynthesis() since the carbohydrate model last credited photosynthesis
    uint photosynthesis_accumulations = 0;
    //! Leaf primitive-hours without "net_photosynthesis" data since the carbohydrate model last credited photosynthesis
    uint leaf_primitive_hours_missing_photosynthesis = 0;
    //! Net photosynthesis credited at the start of the current advanceTime() call, per day of the call (mol C / day)
    float photosynthesis_supply_rate_molC_day = 0;
    //! Thermal time supplied by PlantArchitecture::accumulateThermalTime() since the last advanceTime() call (C day)
    double pending_thermal_time_Cd = 0;
    //! Hours of air temperature supplied by PlantArchitecture::accumulateThermalTime() since the last advanceTime() call
    double pending_thermal_time_hours = 0;
    //! Thermal time on the grain-filling temperatures (CarbonGrassParameters::grain_fill_base_temperature) supplied since the last advanceTime() call (C day)
    double pending_grain_thermal_time_Cd = 0;
    //! Developmental days per calendar day of a panicle past anthesis over the current advanceTime() call (CARBON_SUBMODEL_GRASS; see CarbonGrassParameters::grain_fill_base_temperature)
    float grain_development_rate = 1.f;
    //! Thermal time accumulated since the plant was built under the carbohydrate model (C day)
    double cumulative_thermal_time_Cd = 0;
    //! Developmental days per calendar day over the current advanceTime() call: 1 without the carbohydrate model, otherwise the call's thermal time over that of the reference temperature
    float development_rate = 1.f;
    //! Maintenance respiration of wood and roots charged in the current sub-step (mol C)
    float step_maintenance_respiration_molC = 0;
    //! Recent photosynthate available to the reproductive sinks built from current photosynthate (fruit past its reserve-funded period; panicle structure) and their potential growth:
    //! {supply (mol C), demand (mol C), duration (days)} per sub-step, oldest first
    std::vector<std::array<float, 3>> reproductive_assimilate_history;
    //! Realized relative to potential growth of the reproductive sinks built from current photosynthate, averaged over supply_averaging_days (1 before any such sink has grown)
    float reproductive_supply_ratio = 1;
    //! Vegetative growth demand -- stem, leaf and root construction with growth respiration -- as a rate (mol C / day), averaged over supply_averaging_days; the demand of the tiller
    //! supply/demand index (CarbonGrassParameters::tiller_supply_demand_min)
    float vegetative_demand_rate_molC_day = 0;
    //! Structural carbon of the root system (mol C), grown by stage-dependent allocation under CARBON_SUBMODEL_GRASS. Under CARBON_SUBMODEL_TREE roots are a fixed fraction of the wood and this is unused.
    float root_structural_carbon_molC = 0;
    //! Whether the main stem has reached CarbonGrassParameters::stem_elongation_node (CARBON_SUBMODEL_GRASS)
    bool grass_stem_elongation_started = false;
    //! Plant age at which stem elongation began (CARBON_SUBMODEL_GRASS)
    float grass_stem_elongation_age = 0;
    //! Structural carbon laid down by pipe-model radial growth in the current sub-step (mol C)
    float step_girth_growth_molC = 0;
    //! Structural carbon pipe-model radial growth would have laid down in the current sub-step with photosynthate unlimited (mol C)
    float step_girth_potential_molC = 0;
    //! Recent photosynthate allotted to pipe-model radial growth and its potential cost: {allotted (mol C), potential (mol C), duration (days)} per sub-step, oldest first
    std::vector<std::array<float, 3>> cambium_supply_history;
    //! Allotted relative to potential cost of pipe-model radial growth, averaged over supply_averaging_days (see CarbonWoodParameters::radial_growth_rate)
    float cambium_supply_ratio = 1;

    // --- Nitrogen Model --- //

    //! Parameters for the nitrogen model
    NitrogenParameters nitrogen_parameters;

    //! Root nitrogen pool (g N)
    float root_nitrogen_pool_gN = 0;
    //! Available nitrogen pool for leaves to draw from (g N)
    float available_nitrogen_pool_gN = 0;
    //! Cumulative nitrogen uptake tracking (g N)
    float cumulative_N_uptake_gN = 0;
    //! Fruit nitrogen demand (g N) the available pool could not cover this timestep; filled by remobilization. Rebuilt every timestep.
    float unmet_fruit_N_demand_gN = 0;
    //! Nitrogen delivered to the plant's fruit since the nitrogen pools were initialized (g N)
    float fruit_nitrogen_pool_gN = 0;
    //! Plant nitrogen stress factor from the last nitrogen update (0 = no leaf nitrogen, 1 = at target)
    float nitrogen_stress_factor = 1.f;

    //! Snapshot of shoot parameters that were active when this plant was created
    //! This prevents parameter contamination between different plant types
    //! Prescribed leaf angle distribution this plant's leaves are steered toward as it grows, if any.
    /**
     * Set by PlantArchitecture::enablePlantLeafAngleDistributionTracking() and consulted once per growth
     * sub-step. Unlike the one-shot PlantArchitecture::setPlantLeafAngleDistribution(), which re-aims every
     * leaf on the plant, tracking steers only the leaves that are still expanding, so a mature leaf never
     * moves and the plant does not jitter as it grows.
     */
    struct LeafAngleDistributionTracker {
        //! Whether the inclination of new leaves is being steered toward a Beta distribution
        bool track_elevation = false;
        //! Whether the azimuth of new leaves is being steered toward an ellipsoidal distribution
        bool track_azimuth = false;
        //! First parameter of the target Beta distribution of leaf inclination
        float Beta_mu_inclination = 1.f;
        //! Second parameter of the target Beta distribution of leaf inclination
        float Beta_nu_inclination = 1.f;
        //! Eccentricity of the target ellipsoidal distribution of leaf azimuth
        float eccentricity_azimuth = 0.f;
        //! Rotation of the target azimuth ellipse (degrees)
        float ellipse_rotation_azimuth_degrees = 0.f;
        //! How strongly the population's shortfall in each angle bin outweighs keeping a leaf near the angle the model gave it (radians)
        /**
         * Zero leaves every leaf where the procedural model put it. Large values (a radian or more) pull the
         * population onto the target as hard as it can be pulled, at the cost of the plant's own form.
         */
        float lambda = 0.f;
        //! Leaf area accumulated in each angle bin, rebuilt every sub-step
        std::vector<float> bin_weight;
        //! Share of the total leaf area the target distribution puts in each bin
        std::vector<float> bin_target_fraction;
        //! Blade facets of a leaf, remembered between sub-steps so the histogram does not re-derive them every time
        struct BladeCacheEntry {
            //! Number of primitives the leaf object had when this entry was built; a change means the leaf was rebuilt or pruned
            size_t primitive_count = 0;
            //! The leaf's blade primitives, excluding the petiolule
            std::vector<uint> blade_UUIDs;
            //! Signed offset of each blade facet's inclination from the blade's mean inclination (radians)
            /**
             * A blade is not flat: a folded, curved leaf spreads its facets over a range of inclinations
             * about the direction it faces as a whole. Steering aims the blade's mean normal, but the
             * distribution being matched is over facets, so the tracker has to know how a leaf's facet area
             * will land about wherever its mean is aimed. Held as offsets from the mean rather than as
             * absolute angles so the spread can be re-centred on a proposed target without touching geometry.
             */
            std::vector<float> facet_inclination_offset;
            //! Area of each blade facet, in the same order as facet_inclination_offset
            std::vector<float> facet_area;
            //! Fraction of the blade's area landing in each inclination bin offset from the bin its mean normal occupies
            /**
             * The facet spread reduced to the form the target chooser needs: a normalized kernel over bin
             * offsets, so scoring a candidate bin is a short correlation rather than a pass over every facet.
             * Built alongside the facet arrays, since both come from the same walk over the blade.
             */
            std::vector<float> facet_bin_kernel;
            //! Index within facet_bin_kernel corresponding to zero offset
            int facet_bin_kernel_origin = 0;
        };
        /**
         * Selecting a leaf's blade facets means a string comparison against the primitive data of every
         * primitive in the leaf object, and the histogram visits every leaf on the plant on every sub-step.
         * The membership only changes when a leaf's geometry is rebuilt, so it is cached and re-derived only
         * when the cached entry no longer describes the object. Keyed by leaf object ID; the stored primitive
         * count is what detects a rebuilt or pruned leaf.
         */
        std::map<uint, BladeCacheEntry> blade_primitive_cache;

        //! Shoot ID and node index of each phytomer created this sub-step, whose leaves are still to be assigned a target
        std::vector<std::pair<int, uint>> pending_cohort;
        //! Drawn from the Context generator once when tracking is enabled, so that steering does not consume draws from the stream the plant is built with
        std::minstd_rand0 generator;
    };

    //! Leaf angle distribution this plant is being steered toward, if tracking is enabled
    LeafAngleDistributionTracker leaf_angle_tracker;
    //! Whether this plant's leaf angles are being steered as it grows
    bool leaf_angle_tracking_enabled = false;

    std::map<std::string, ShootParameters> shoot_types_snapshot;

    // --- Per-plant Attraction Points --- //

    //! Flag indicating if attraction points are enabled for this plant
    bool attraction_points_enabled = false;

    //! Vector of attraction point positions for this plant
    std::vector<helios::vec3> attraction_points;

    //! Attraction detection cone half-angle in radians for this plant
    float attraction_cone_half_angle_rad = 80.f * M_PI / 180.f;

    //! Attraction detection cone height in meters for this plant
    float attraction_cone_height = 0.1f;

    //! Weight factor for attraction vs natural growth for this plant (0.0 = ignore attraction, 1.0 = full attraction)
    float attraction_weight = 0.6f;

    //! Reduction factor for attraction weight when hard obstacles are present for this plant (0.0 = no attraction, 1.0 = full attraction)
    float attraction_obstacle_reduction_factor = 0.5f;
};

//! Options for PlantArchitecture::generateLeafTextureStages()
struct LeafTextureStageOptions {
    //! Colors a leaf passes through after its original color, in order, keeping the leaf's own veins and shading. Not used if senesced_textures is given.
    /**
     * \note The defaults are for appearance. They are not calibrated to measurements of any species.
     */
    std::vector<helios::RGBcolor> senescence_colors = {helios::make_RGBcolor(0.76f, 0.66f, 0.18f), helios::make_RGBcolor(0.36f, 0.22f, 0.10f)};

    //! Images a leaf passes through after its original image, in order, each with the same resolution as the leaf texture. If given, these are used in place of senescence_colors.
    std::vector<std::string> senesced_textures;

    //! Image giving when each pixel turns, with the same resolution as the leaf texture: black pixels turn first and white pixels last. If empty, a leaf turns from its margin inward with patches given by patchiness.
    std::string onset_map;

    //! Fraction of the generated order of turning that is random patches rather than distance from the leaf's margin, between 0 and 1. Not used if onset_map is given.
    float patchiness = 0.3f;

    //! Fraction of the leaf's aging over which any one pixel changes color, greater than 0 and at most 1. Smaller values give sharper fronts between changed and unchanged areas of the leaf.
    float transition_width = 0.25f;

    //! Selects the random patches, so that different variants of a leaf can turn in different patterns
    uint noise_seed = 0;
};

//! Parameters for USD articulated rigid body export for IsaacSim
struct USDExportParameters {
    //! Young's modulus (Pa) for joint stiffness calculation: K = E*I/L
    float elastic_modulus = 5e9f;
    //! Density (kg/m^3) for mass calculation from capsule volume
    float wood_density = 800.f;
    //! Damping time constant (s) for the joints between stem, petiole and peduncle segments. Damping = time constant * K
    float damping_time_constant = 0.02f;
    //! Cap on each joint's own natural frequency, as a multiple of the physics step rate, enforced by adding joint armature. Lower is more stable.
    float armature_stability_ratio = 8.f;
    //! Physics step rate (Hz) the simulation is expected to run at, used to size the joint armature
    float physics_steps_per_second = 60.f;
    //! Static friction coefficient for collision material
    float static_friction = 0.5f;
    //! Dynamic friction coefficient for collision material
    float dynamic_friction = 0.3f;
    //! Restitution (bounciness) for collision material
    float restitution = 0.1f;
    //! Spring stiffness (N*m/rad) for leaf/fruit/flower attachment joints
    float organ_spring_stiffness = 10.f;
    //! Damping (N*m*s/rad) for leaf/fruit/flower attachment joints
    float organ_spring_damping = 1.f;
    //! Leaf mass per unit area (kg/m^2) — used to compute leaf mass from leaf area
    float leaf_mass_per_area = 0.05f;
    //! Mass per fruit (kg)
    float fruit_mass = 0.01f;
    //! Mass per flower (kg)
    float flower_mass = 0.002f;
    //! PhysX articulation solver position iteration count
    uint solver_position_iterations = 32;
    //! Minimum segment length (m). Segments shorter than this are skipped.
    float min_segment_length = 0.001f;
};

//! Kind of joint in the skeleton of a plant (see PlantArchitecture::getPlantJoints())
enum PlantJointType {
    //! Base of the whole plant, at PlantArchitecture::getPlantBasePosition(). Carries every organ of the plant.
    PLANT_JOINT_ROOT,
    //! Attachment of a shoot to the node of the parent shoot it grows from. Carries the shoot and everything on it.
    PLANT_JOINT_SHOOT,
    //! Base of the internode of one phytomer. Carries that internode, the organs at its node, and every phytomer and child shoot further out along the shoot.
    PLANT_JOINT_INTERNODE,
    //! Base of a petiole, at the node it grows from. Carries the petiole and its leaves.
    PLANT_JOINT_PETIOLE,
    //! Base of the peduncle of a floral bud. Carries the peduncle and its flowers or fruit.
    PLANT_JOINT_PEDUNCLE
};

//! Identifier of one joint in the skeleton of a plant
/**
 * A joint is named by the organ it is the base of, using the same indices as the rest of the plant architecture interface, so an identifier stays valid for as long as that organ exists: growth adds
 * joints without renaming the ones there, while a petiole joint goes when its leaf is shed and a peduncle joint when its bud stops flowering or fruiting. Indices that do not apply to the joint's type
 * must be left at zero; an identifier with one set is refused.
 */
struct PlantJointID {
    //! Kind of joint
    PlantJointType type = PLANT_JOINT_ROOT;
    //! ID of the shoot the joint belongs to (not used by PLANT_JOINT_ROOT)
    uint shootID = 0;
    //! Index of the phytomer along the shoot (PLANT_JOINT_INTERNODE, PLANT_JOINT_PETIOLE, PLANT_JOINT_PEDUNCLE)
    uint node_index = 0;
    //! Index of the petiole within the phytomer (PLANT_JOINT_PETIOLE, PLANT_JOINT_PEDUNCLE)
    uint petiole_index = 0;
    //! Index of the floral bud within the petiole (PLANT_JOINT_PEDUNCLE)
    uint bud_index = 0;

    bool operator==(const PlantJointID &other) const {
        return type == other.type && shootID == other.shootID && node_index == other.node_index && petiole_index == other.petiole_index && bud_index == other.bud_index;
    }
    bool operator!=(const PlantJointID &other) const {
        return !(*this == other);
    }
    bool operator<(const PlantJointID &other) const {
        if (type != other.type) {
            return type < other.type;
        }
        if (shootID != other.shootID) {
            return shootID < other.shootID;
        }
        if (node_index != other.node_index) {
            return node_index < other.node_index;
        }
        if (petiole_index != other.petiole_index) {
            return petiole_index < other.petiole_index;
        }
        return bud_index < other.bud_index;
    }
};

//! Where a joint of a plant is and how the organ it carries is oriented (see PlantArchitecture::getPlantJointFrame())
struct PlantJointFrame {
    //! Point the joint turns about, in world coordinates
    helios::vec3 pivot;
    //! Unit direction of what the organ grows from, in world coordinates: the vertical for the plant root, the parent internode for a shoot, petiole or peduncle, and the previous internode for an internode.
    helios::vec3 parent_axis;
    //! Unit direction of the organ where it leaves the joint, in world coordinates
    helios::vec3 organ_axis;
    //! Unit axis, in world coordinates, about which the organ is tilted toward or away from parent_axis. It is perpendicular to both; when the two are parallel it is an arbitrary perpendicular.
    helios::vec3 pitch_axis;
    //! Length in meters of the organ along its centerline: the internode, petiole or peduncle. Zero for joints whose organ has no length of its own (the plant root and a shoot).
    float length = 0;
};

//! Pose of one joint of a plant: how far the organ it carries is turned from the way it grew (see PlantArchitecture::setPlantJointPose())
/**
 * The three angles are applied about the axes of the joint as the plant grew (see \ref PlantJointFrame), roll first, then pitch, then yaw. A default-constructed pose is the plant as it grew.
 */
struct PlantJointPose {
    //! Tilt of the organ away from what it grows from, in degrees, about PlantJointFrame::pitch_axis. Negative values tilt it toward its parent.
    float pitch_degrees = 0;
    //! Turn of the organ around what it grows from, in degrees, about PlantJointFrame::parent_axis
    float yaw_degrees = 0;
    //! Twist of the organ about its own axis, in degrees, about PlantJointFrame::organ_axis
    float roll_degrees = 0;
    //! Length of the organ as a multiple of the length it grew to. Only an internode or a petiole has a length to pose (PLANT_JOINT_INTERNODE, PLANT_JOINT_PETIOLE); it must be 1 for any other joint.
    /**
     * The organ's centerline is stretched about the joint, keeping its shape, and everything attached along it or beyond it is carried to where its point of attachment goes, without being stretched
     * itself. To set a length in meters, divide it by the organ's grown length, which is PlantJointFrame::length divided by the length_scale the joint currently has.
     */
    float length_scale = 1;
    //! Bend of a petiole along its length, in degrees, about PlantJointFrame::pitch_axis: positive curls it further from what it grows from, as a positive pitch_degrees tilts it. Only a petiole has a bend to pose (PLANT_JOINT_PETIOLE); it must be 0 for any other joint.
    /**
     * Where pitch_degrees turns the whole petiole rigidly about its base, a bend turns it progressively along its length, so it can be pushed or pulled into a new shape without a kink. The bend is the
     * change, from the way the petiole grew, in the angle between its direction at its base and its direction at its tip: the first segment of its centerline keeps its direction, the last is turned by the
     * whole bend, and the segments between are turned by shares that grow evenly with arclength. It is an offset like the angles, so zero leaves the curvature the petiole grew with. Each leaf on the
     * petiole is carried rigidly to where its point of attachment goes and turned with the centerline there. The bend is applied to the petiole as it grew, about the joint's axes as it grew, before the
     * joint is turned by its angles, so it is independent of them, and roll_degrees turns the plane of the bend with the petiole. A petiole whose centerline has a single segment has one direction and cannot
     * be bent. See \ref PlantArchPoseBend.
     */
    float bend_degrees = 0;
    //! Sideways bend of a petiole along its length, in degrees, about the axis perpendicular to both PlantJointFrame::organ_axis and PlantJointFrame::pitch_axis: positive turns it the way a positive yaw_degrees turns it around the stem. Only a petiole has a bend to pose; it must be 0 for any other joint.
    /**
     * Spread along the petiole as bend_degrees is. The two bends together are one bend, of the size of their vector sum, about the axis that combines them: the angle between the petiole's directions at
     * its base and its tip changes by that size.
     */
    float bend_lateral_degrees = 0;
};

//! Kind of plant organ represented by a PlantArticulationLink
enum PlantArticulationLinkType {
    //! Capsule segment of a shoot internode
    ARTICULATION_LINK_INTERNODE,
    //! Capsule segment of a leaf petiole
    ARTICULATION_LINK_PETIOLE,
    //! Capsule segment of a flower or fruit peduncle
    ARTICULATION_LINK_PEDUNCLE,
    //! Leaf blade mass body
    ARTICULATION_LINK_LEAF,
    //! Fruit mass body
    ARTICULATION_LINK_FRUIT,
    //! Flower mass body
    ARTICULATION_LINK_FLOWER
};

//! One rigid link of a plant articulation (see PlantArchitecture::getPlantArticulation())
struct PlantArticulationLink {
    //! Unique name of the link within the plant
    std::string name;
    //! Organ the link represents
    PlantArticulationLinkType type = ARTICULATION_LINK_INTERNODE;
    //! Index of the parent link, or -1 if the link is the root and is fixed to the world
    int parent_index = -1;
    //! World-space position of the link frame origin in the rest pose: the capsule midpoint, or the organ mesh centroid
    helios::vec3 position;
    //! World-space orientation of the link frame in the rest pose as a unit quaternion (w,x,y,z). The local +z axis is the capsule axis.
    float qw = 1, qx = 0, qy = 0, qz = 0;
    //! Capsule radius in meters (zero for leaf, fruit, and flower links)
    float radius = 0;
    //! Half-length of the capsule shaft in meters (zero for leaf, fruit, and flower links)
    float half_length = 0;
    //! Mass in kg
    float mass = 0;
    //! Principal moments of inertia about the link frame axes in kg*m^2
    helios::vec3 inertia_diagonal;
    //! Context object the link was built from: the tube object for internode, petiole, and peduncle segments, or the organ object. Zero if none.
    uint source_objID = 0;
    //! Innermost joint of the plant skeleton that carries this link (see PlantArchitecture::getPlantJoints()): the internode, petiole or peduncle a segment is part of, or the one a leaf, flower or fruit hangs from.
    PlantJointID joint;
};

//! Elastic joint connecting a plant articulation link to its parent
struct PlantArticulationJoint {
    //! Index of the parent link
    int parent_index = -1;
    //! Index of the child link
    int child_index = -1;
    //! Joint anchor in the parent link frame
    helios::vec3 anchor_in_parent;
    //! Joint anchor in the child link frame
    helios::vec3 anchor_in_child;
    //! Rotational spring stiffness in N*m/rad
    float stiffness = 0;
    //! Rotational damping in N*m*s/rad
    float damping = 0;
    //! Artificial joint inertia in kg*m^2 that caps the joint's own natural frequency
    float armature = 0;
};

//! Physics-ready articulated rigid-body description of a plant
struct PlantArticulation {
    //! Links ordered so that every parent precedes its children
    std::vector<PlantArticulationLink> links;
    //! One joint per non-root link
    std::vector<PlantArticulationJoint> joints;
};

//! Opaque storage for a single growth animation frame snapshot (defined in InputOutput.cpp)
struct GrowthFrameSnapshot;

//! Storage container for growth animation frame data
struct PlantGrowthAnimationStorage {
    //! Map from plantID to vector of frame snapshots
    std::map<uint, std::vector<std::shared_ptr<GrowthFrameSnapshot>>> plant_frames;
};

//! Kind of plant organ that a Context object of a plant belongs to
enum class PlantOrganType {
    //! Internode: the shoot's internode tube, which spans all phytomers of the shoot
    INTERNODE,
    //! Petiole tube
    PETIOLE,
    //! Leaf (or leaflet of a compound leaf)
    LEAF,
    //! Peduncle tube of a floral bud
    PEDUNCLE,
    //! Flower, open or closed, of a floral bud
    FLOWER,
    //! Fruit of a floral bud
    FRUIT
};

//! Where a Context primitive or object sits in the structure of a plant, as returned by PlantArchitecture::getPrimitiveOrganLocation()
/**
 * Fields that do not apply to the organ type are set to \ref PlantOrganLocation::not_applicable.
 */
struct PlantOrganLocation {
    //! Value of an index field that does not apply to the organ type
    static constexpr uint not_applicable = 0xFFFFFFFF;

    //! Identifier of the plant
    uint plantID;
    //! Identifier of the shoot within the plant
    uint shootID;
    //! Index of the phytomer (node) on the shoot, counted from 0 at the shoot base. For an internode primitive, the phytomer whose internode the primitive belongs to.
    uint node_index;
    //! Kind of organ
    PlantOrganType organ_type;
    //! Index of the petiole within the phytomer (petioles, leaves), or the first index of the floral bud in Phytomer::floral_buds (peduncles, flowers, fruit; see Phytomer::getFloralBudStorageIndex())
    uint petiole_index;
    //! Index of the leaf within its petiole (leaves only)
    uint leaf_index;
    //! Second index of the floral bud in Phytomer::floral_buds (peduncles, flowers and fruit only; see Phytomer::getFloralBudStorageIndex())
    uint floral_bud_index;
    //! Context object ID of the organ
    uint objID;
};

//! One straight segment of the centerline of a plant axis (internode, petiole or peduncle), as returned by PlantArchitecture::getPlantSkeletonSegments()
struct PlantSkeletonSegment {
    //! Start of the segment, nearer the base of the organ
    helios::vec3 start;
    //! End of the segment
    helios::vec3 end;
    //! Radius of the organ at the start of the segment
    float start_radius;
    //! Radius of the organ at the end of the segment
    float end_radius;
    //! Organ the segment belongs to. Its node_index is the phytomer the segment belongs to.
    PlantOrganLocation organ;
};

//! Analytic stand-in for one fruit object in fruit-branch contact tests (see PlantArchitecture::enableFruitBranchCollisionAvoidance())
/**
 * Built from the fruit mesh and stored in the frame of the fruit's attachment point, at the fruit scale it was built at. Positions scale linearly with the fruit's growth about that point and are
 * rotated along with any swing of the peduncle, so the proxy follows the fruit without being rebuilt.
 */
struct FruitContactProxy {
    //! FloralBud::current_fruit_scale_factor at which the proxy was built
    float build_scale = 1;
    //! Center of the sphere enclosing the whole fruit, relative to the attachment point
    helios::vec3 bounding_center;
    //! Radius of that sphere
    float bounding_radius = 0;
    //! Centers of the chain of spheres covering the fruit along its axis, relative to the attachment point
    std::vector<helios::vec3> sphere_centers;
    //! Radii of those spheres
    std::vector<float> sphere_radii;
    //! Triangle vertices (three per triangle) relative to the attachment point; empty when the mesh is too large and the sphere chain is used instead
    std::vector<helios::vec3> triangle_vertices;
    //! Largest distance of any mesh vertex from the attachment point, at build_scale
    float max_extent = 0;
};

//! Internode segments near one fruiting floral bud, cached between refreshes of the fruit-branch contact tests
struct FruitContactCandidates {
    //! Candidate internode segments as (shootID, phytomer index within the shoot, segment index within the phytomer's internode)
    std::vector<std::array<uint, 3>> segments;
    //! Plant age (days) at which the list was built
    float refreshed_at_age = -1;
    //! Number of shoots the plant had when the list was built
    size_t shoot_count = 0;
    //! Fruit scale at which the bud was last left in contact (wedged); it is not retried until the fruit have grown, or the list is rebuilt. Negative when not wedged.
    float unresolved_at_scale = -1;
    //! Fruit scale at which the bud last tried and failed to settle back toward its grown pose; it is not retried until the fruit have grown, or the list is rebuilt. Negative when not blocked.
    float settle_blocked_at_scale = -1;
};

class PlantArchitecture {
public:
    //! Main architectural model class constructor
    /**
     * \param[in] context_ptr Pointer to the Helios context.
     */
    explicit PlantArchitecture(helios::Context *context_ptr);

    //! Destructor - cleans up internal CollisionDetection instance if owned
    ~PlantArchitecture();

    //! Unit test routines
    static int selfTest(int argc, char **argv);

    //! Set a callback function to receive progress updates during long-running operations.
    /**
     * \param[in] callback Function that receives (progress_fraction, message_string).
     *   progress_fraction is in [0, 1]. message describes the current operation phase.
     *   Pass nullptr or an empty std::function to clear the callback.
     */
    void setProgressCallback(std::function<void(float, const std::string&)> callback);

    //! Register an external cancellation flag polled during long plant builds.
    /**
     * When the pointed-to int becomes non-zero, the canopy per-plant build loop
     * and the advanceTime() growth loop stop early (returning whatever has been
     * built so far), so a long generation can be aborted mid-build instead of
     * running to completion. The flag is owned by the caller (e.g. a ctypes int
     * shared with Python, written under the GIL while the build runs GIL-free)
     * and must outlive the build; pass nullptr to clear. The per-iteration read
     * is effectively free. Set this before the build call.
     * \param[in] flag Pointer to a 0/non-zero cancellation flag, or nullptr.
     */
    void setCancelFlag(volatile int *flag);

    //! Add optional output object data values to the Context
    /**
     * \param[in] object_data_label Name of object data (e.g., "age", "rank")
     */
    void optionalOutputObjectData(const std::string &object_data_label);

    //! Add optional output object data values to the Context
    /**
     * \param[in] object_data_labels Vector of names of object data (e.g., {"age", "rank"})
     */
    void optionalOutputObjectData(const std::vector<std::string> &object_data_labels);

    // ********* Methods for Building Plants from Existing Library ********* //

    //! Load an existing plant model from the library
    /**
     * \param[in] plant_label User-defined label for the plant model to be loaded.
     * \note Not available inside a phytomer callback (\ref PhytomerParameters::phytomer_creation_function or \ref PhytomerParameters::phytomer_callback_function): a call made from one does nothing, and is reported once, as an aggregated warning, at the end of the enclosing call. See \ref PlantArchCallbackEdits.
     */
    void loadPlantModelFromLibrary(const std::string &plant_label);

    //! Get list of all available plant models in the library
    [[nodiscard]] std::vector<std::string> getAvailablePlantModels() const;

    //! Get the names of the build parameters a library plant model accepts
    /**
     * These are the keys that may be given to \ref buildPlantInstanceFromLibrary() for the model. Their meanings and ranges are listed in the plant library documentation (see \ref PlantArchBuildParams).
     * \param[in] plant_model_name Name of the plant model. See \ref getAvailablePlantModels().
     * \return Names of the model's structure parameters followed by its organ parameters. Empty if the model accepts none.
     */
    [[nodiscard]] std::vector<std::string> getPlantModelBuildParameterNames(const std::string &plant_model_name) const;

    //! Get the empirical leaf inclination distribution a library plant model declares
    /**
     * A plant model may declare the leaf inclination distribution measured for its species. Plants built from
     * such a model by \ref buildPlantInstanceFromLibrary() are steered toward that distribution as they grow,
     * without the caller asking for it. See \ref PlantArchLeafAngleLibrary.
     *
     * \param[in] plant_model_name Name of the plant model to query. See \ref getAvailablePlantModels().
     * \return Beta distribution parameters of leaf inclination (x=mu, y=nu), or (0,0) if the model declares no measured distribution.
     */
    [[nodiscard]] helios::vec2 getPlantModelLeafInclinationDistribution(const std::string &plant_model_name) const;

    //! Set or replace the empirical leaf inclination distribution of a library plant model
    /**
     * Affects plants built after this call; plants that already exist are not changed. Pass (0,0) to clear the
     * distribution, after which plants of this model are built with the leaf angles the model generates.
     *
     * \param[in] plant_model_name Name of the plant model. See \ref getAvailablePlantModels().
     * \param[in] Beta_mu_inclination First parameter of the Beta distribution of leaf inclination. Must be positive, or zero together with Beta_nu_inclination.
     * \param[in] Beta_nu_inclination Second parameter of the Beta distribution of leaf inclination. Must be positive, or zero together with Beta_mu_inclination.
     */
    void setPlantModelLeafInclinationDistribution(const std::string &plant_model_name, float Beta_mu_inclination, float Beta_nu_inclination);

    //! Whether a library plant model declares an empirical leaf inclination distribution
    /**
     * \param[in] plant_model_name Name of the plant model to query. See \ref getAvailablePlantModels().
     * \return True if the model declares a measured leaf inclination distribution.
     */
    [[nodiscard]] bool doesPlantModelDeclareLeafInclinationDistribution(const std::string &plant_model_name) const;

    //! Build a plant instance based on the model currently loaded from the library
    /**
     * \param[in] base_position Cartesian coordinates of the base of the plant.
     * \param[in] age Age of the plant in days.
     * \param[in] build_parameters [optional] Map of parameter names to values overriding species-specific defaults of the model, such as trellis dimensions, trunk height and scaffold count, or the size and
     * form of grape clusters and sorghum panicles. Parameter names are species-specific and documented in the plant library documentation (see \ref PlantArchBuildParams). A name the model does not accept throws.
     * \return ID of the plant instance.
     * \note Cannot be used inside a phytomer callback (\ref PhytomerParameters::phytomer_creation_function or \ref PhytomerParameters::phytomer_callback_function): a call made from one raises an error. See \ref PlantArchCallbackEdits.
     */
    uint buildPlantInstanceFromLibrary(const helios::vec3 &base_position, float age, const std::map<std::string, float> &build_parameters = {});

    //! Build a plant instance based on the model currently loaded from the library, with build parameters drawn from distributions
    /**
     * A structure parameter, such as "tiller_count" or "trunk_height", is sampled once for the plant. An organ parameter, such as "berry_diameter", is sampled once for each of the model's flower and fruit
     * prototypes, which are shared by every plant given the same organ distributions. See \ref PlantArchBuildParamDistributions.
     * \param[in] base_position Cartesian coordinates of the base of the plant.
     * \param[in] age Age of the plant in days.
     * \param[in] build_parameters Map of parameter names to distributions overriding species-specific defaults of the model (see \ref PlantArchBuildParams). A name the model does not accept, a distribution
     * reaching outside the valid range of its parameter, or a parameter that counts something given other than whole-number constant or discrete values throws.
     * \return ID of the plant instance.
     * \note Cannot be used inside a phytomer callback (\ref PhytomerParameters::phytomer_creation_function or \ref PhytomerParameters::phytomer_callback_function): a call made from one raises an error. See \ref PlantArchCallbackEdits.
     */
    uint buildPlantInstanceFromLibrary(const helios::vec3 &base_position, float age, const std::map<std::string, BuildParameterDistribution> &build_parameters);

    //! Build a canopy of regularly spaced plants based on the model currently loaded from the library
    /**
     * \param[in] canopy_center_position Cartesian coordinates of the center of the canopy.
     * \param[in] plant_spacing_xy Spacing between plants in the canopy in the x- and y-directions.
     * \param[in] plant_count_xy Number of plants in the canopy in the x- and y-directions.
     * \param[in] age Age of the plants in the canopy in days.
     * \param[in] germination_rate [optional] Probability that a plant in the canopy germinates and a plant is created.
     * \param[in] build_parameters [optional] Map of parameter names to values overriding species-specific defaults of the model, such as trellis dimensions, trunk height and scaffold count, or the size and
     * form of grape clusters and sorghum panicles. Parameter names are species-specific and documented in the plant library documentation (see \ref PlantArchBuildParams). A name the model does not accept throws.
     * \return Vector of plant instance IDs.
     * \note Cannot be used inside a phytomer callback (\ref PhytomerParameters::phytomer_creation_function or \ref PhytomerParameters::phytomer_callback_function): a call made from one raises an error. See \ref PlantArchCallbackEdits.
     */
    std::vector<uint> buildPlantCanopyFromLibrary(const helios::vec3 &canopy_center_position, const helios::vec2 &plant_spacing_xy, const helios::int2 &plant_count_xy, float age, float germination_rate = 1.f,
                                                  const std::map<std::string, float> &build_parameters = {});

    //! Build a canopy of randomly scattered plants based on the model currently loaded from the library
    /**
     * \param[in] canopy_center_position Cartesian coordinates of the center of the canopy boundaries.
     * \param[in] canopy_extent_xy Size/extent of the canopy boundaries in the x- and y-directions.
     * \param[in] plant_count Number of plants to randomly generate inside canopy bounds.
     * \param[in] age Age of the plants in the canopy in days.
     * \param[in] build_parameters [optional] Map of parameter names to values overriding species-specific defaults of the model, such as trellis dimensions, trunk height and scaffold count, or the size and
     * form of grape clusters and sorghum panicles. Parameter names are species-specific and documented in the plant library documentation (see \ref PlantArchBuildParams). A name the model does not accept throws.
     * \return Vector of plant instance IDs.
     * \note Cannot be used inside a phytomer callback (\ref PhytomerParameters::phytomer_creation_function or \ref PhytomerParameters::phytomer_callback_function): a call made from one raises an error. See \ref PlantArchCallbackEdits.
     */
    std::vector<uint> buildPlantCanopyFromLibrary(const helios::vec3 &canopy_center_position, const helios::vec2 &canopy_extent_xy, uint plant_count, float age, const std::map<std::string, float> &build_parameters = {});

    //! Build a canopy of regularly spaced plants based on the model currently loaded from the library, with build parameters drawn from distributions
    /**
     * Each plant is given its own sample of every structure parameter, such as "tiller_count", and the plants share one set of flower and fruit prototypes sampled from the organ parameters. See
     * \ref PlantArchBuildParamDistributions.
     * \param[in] canopy_center_position Cartesian coordinates of the center of the canopy.
     * \param[in] plant_spacing_xy Spacing between plants in the canopy in the x- and y-directions.
     * \param[in] plant_count_xy Number of plants in the canopy in the x- and y-directions.
     * \param[in] age Age of the plants in the canopy in days.
     * \param[in] germination_rate Probability that a plant in the canopy germinates and a plant is created.
     * \param[in] build_parameters Map of parameter names to distributions overriding species-specific defaults of the model (see \ref PlantArchBuildParams), checked as by
     * \ref buildPlantInstanceFromLibrary(const helios::vec3 &, float, const std::map<std::string, BuildParameterDistribution> &).
     * \return Vector of plant instance IDs.
     * \note Cannot be used inside a phytomer callback (\ref PhytomerParameters::phytomer_creation_function or \ref PhytomerParameters::phytomer_callback_function): a call made from one raises an error. See \ref PlantArchCallbackEdits.
     */
    std::vector<uint> buildPlantCanopyFromLibrary(const helios::vec3 &canopy_center_position, const helios::vec2 &plant_spacing_xy, const helios::int2 &plant_count_xy, float age, float germination_rate,
                                                  const std::map<std::string, BuildParameterDistribution> &build_parameters);

    //! Build a canopy of randomly scattered plants based on the model currently loaded from the library, with build parameters drawn from distributions
    /**
     * Each plant is given its own sample of every structure parameter, such as "tiller_count", and the plants share one set of flower and fruit prototypes sampled from the organ parameters. See
     * \ref PlantArchBuildParamDistributions.
     * \param[in] canopy_center_position Cartesian coordinates of the center of the canopy boundaries.
     * \param[in] canopy_extent_xy Size/extent of the canopy boundaries in the x- and y-directions.
     * \param[in] plant_count Number of plants to randomly generate inside canopy bounds.
     * \param[in] age Age of the plants in the canopy in days.
     * \param[in] build_parameters Map of parameter names to distributions overriding species-specific defaults of the model (see \ref PlantArchBuildParams), checked as by
     * \ref buildPlantInstanceFromLibrary(const helios::vec3 &, float, const std::map<std::string, BuildParameterDistribution> &).
     * \return Vector of plant instance IDs.
     * \note Cannot be used inside a phytomer callback (\ref PhytomerParameters::phytomer_creation_function or \ref PhytomerParameters::phytomer_callback_function): a call made from one raises an error. See \ref PlantArchCallbackEdits.
     */
    std::vector<uint> buildPlantCanopyFromLibrary(const helios::vec3 &canopy_center_position, const helios::vec2 &canopy_extent_xy, uint plant_count, float age,
                                                  const std::map<std::string, BuildParameterDistribution> &build_parameters);

    //! Get the shoot parameters structure for a specific shoot type in the current plant model
    /**
     * \param[in] shoot_type_label User-defined label for the shoot type.
     * \return ShootParameters structure for the specified shoot type.
     */
    ShootParameters getCurrentShootParameters(const std::string &shoot_type_label);

    //! Get the shoot parameters structure for all shoot types in the current plant model
    /**
     * \return Map of shoot type labels to ShootParameters structures for all shoot types in the current plant model. The key is the user-defined label string for the shoot type, and the value is the corresponding ShootParameters structure.
     */
    std::map<std::string, ShootParameters> getCurrentShootParameters();

    //! Get the phytomer parameters structure for all shoot types in the current plant model
    /**
     * \return Map of phytomer parameters for all type labels to ShootParameters structures for all shoot types in the current plant model. The key is the user-defined label string for the shoot type, and the value is the corresponding
     * PhytomerParameters structure.
     */
    std::map<std::string, PhytomerParameters> getCurrentPhytomerParameters();

    //! Get the list of shoot type labels for the currently loaded plant model
    /**
     * \return Vector of shoot type label strings for the currently loaded plant model.
     * \note This method only works if a plant model has been loaded via loadPlantModelFromLibrary(). Use the overload listShootTypeLabels(plant_model_name) to query a specific model without loading, or listShootTypeLabels(plantID) to query a plant
     * instance.
     */
    [[nodiscard]] std::vector<std::string> listShootTypeLabels() const;

    //! Get the list of shoot type labels for a specific plant model without changing current state
    /**
     * \param[in] plant_model_name Name of the plant model to query (e.g., "bean", "tomato", "almond"). Use getAvailablePlantModels() to see available plant model names.
     * \return Vector of shoot type label strings for the specified plant model.
     * \note This method temporarily loads the plant model to extract shoot type information, then restores the original plant state. The current plant model remains unchanged after calling this method.
     */
    [[nodiscard]] std::vector<std::string> listShootTypeLabels(const std::string &plant_model_name);

    //! Get the list of shoot type labels for a specific plant instance
    /**
     * \param[in] plantID Unique identifier for the plant instance.
     * \return Vector of shoot type label strings for the specified plant instance.
     * \note This method returns the shoot types that were defined when the plant instance was created. The shoot types are captured from the plant model at creation time.
     */
    [[nodiscard]] std::vector<std::string> listShootTypeLabels(uint plantID) const;

    //! Get the parameters of a shoot type as held by an existing plant
    /**
     * A plant keeps its own copy of the shoot types it was built with, from which every new shoot on it is created; \ref getCurrentShootParameters() reads the plant model's definitions instead.
     * \param[in] plantID ID of the plant
     * \param[in] shoot_type_label Label of the shoot type
     * \return The plant's parameters for that shoot type
     */
    [[nodiscard]] ShootParameters getPlantShootTypeParameters(uint plantID, const std::string &shoot_type_label) const;

    //! Update the parameters of a shoot type on an existing plant, for the shoots it grows from now on
    /**
     * Shoots that already exist keep their own copies of the parameters and are not changed.
     * \param[in] plantID ID of the plant
     * \param[in] shoot_type_label Label of a shoot type the plant already has
     * \param[in] params New parameters for that shoot type
     */
    void updatePlantShootTypeParameters(uint plantID, const std::string &shoot_type_label, const ShootParameters &params);

    //! Update the parameters of a single shoot type in the current plant model
    /**
     * \param[in] shoot_type_label User-defined label for the shoot type to be updated.
     * \param[in] params Updated parameters structure for the shoot type.
     * \note This will overwrite any existing shoot parameter definitions in their entirety rather than merging into them. The stored parameters are the values passed in: shoot-level random parameters
     * are not resampled, so reading a shoot type and writing it straight back is a no-op.
     */
    void updateCurrentShootParameters(const std::string &shoot_type_label, const ShootParameters &params);

    //! Update the parameters of all shoot types in the current plant model
    /**
     * \param[in] params Updated parameters structure for the shoot type.
     * \note This replaces the full set of shoot types, so any shoot type not present in \a params is removed. Shoot-level random parameters are not resampled.
     * \note Not available inside a phytomer callback (\ref PhytomerParameters::phytomer_creation_function or \ref PhytomerParameters::phytomer_callback_function): a call made from one does nothing, and is reported once, as an aggregated warning, at the end of the enclosing call. See \ref PlantArchCallbackEdits.
     */
    void updateCurrentShootParameters(const std::map<std::string, ShootParameters> &params);

    // ********* Methods for Building Custom Plant Geometry from Scratch ********* //

    //! Create an instance of a plant
    /** This is the first step of the plant building process. It creates an empty plant instance that can be built up manually, or using a pre-defined plant type in the library.
     * \param[in] base_position Cartesian coordinates of the base of the plant.
     * \param[in] current_age Age of the plant in days.
     * \return ID of the plant instance.
     */
    uint addPlantInstance(const helios::vec3 &base_position, float current_age);

    //! Duplicate an existing plant instance and specify its base position and age
    /**
     * The copy is an independent plant instance carrying the source's configuration: its name, shoot parameters, phenological thresholds
     * and spread, maximum age, epicormic shoot probability, and carbohydrate and nitrogen parameters. Per-plant attraction points are
     * translated onto the new base position, since they are absolute world coordinates.
     *
     * It also carries where the source is in its season, so that it goes on growing as the source would: the dormancy and growth state
     * of every shoot, including its own shoot parameters and the girth of its internodes; the age, dormancy and leaves of every
     * phytomer; and every bud, with its state, clocks and phenological delays, the flowers and fruit of a floral bud being rebuilt at the
     * size the source's had reached.
     *
     * The shape is rebuilt rather than copied: the architecture, internode lengths and leaf and organ sizes match the source, but the
     * angles and curvatures that the growth model draws at random, such as leaf orientation and peduncle pitch, are drawn anew. The
     * carbohydrate and nitrogen pools are not copied, as they are simulation state rather than configuration; the copy starts with the
     * pools of a newly created plant of the given age.
     *
     * \param[in] plantID ID of the existing plant instance to be duplicated.
     * \param[in] base_position Cartesian coordinates of the base of the new plant copy.
     * \param[in] base_rotation Rotation of the new plant copy.
     * \param[in] current_age Age of the new plant copy in days.
     * \return ID of the new plant instance.
     * \note Cannot be used inside a phytomer callback (\ref PhytomerParameters::phytomer_creation_function or \ref PhytomerParameters::phytomer_callback_function): a call made from one raises an error. See \ref PlantArchCallbackEdits.
     */
    uint duplicatePlantInstance(uint plantID, const helios::vec3 &base_position, const AxisRotation &base_rotation, float current_age);

    //! Delete an existing plant instance
    /**
     * \param[in] plantID ID of the plant instance to be deleted.
     * \note When all plant instances have been deleted, hidden prototype objects are also cleaned up from the Context.
     * \note Not available inside a phytomer callback (\ref PhytomerParameters::phytomer_creation_function or \ref PhytomerParameters::phytomer_callback_function): a call made from one does nothing, and is reported once, as an aggregated warning, at the end of the enclosing call. See \ref PlantArchCallbackEdits.
     */
    void deletePlantInstance(uint plantID);

    //! Delete multiple existing plant instances
    /**
     * \param[in] plantIDs IDs of the plant instances to be deleted.
     * \note Not available inside a phytomer callback (\ref PhytomerParameters::phytomer_creation_function or \ref PhytomerParameters::phytomer_callback_function): a call made from one does nothing, and is reported once, as an aggregated warning, at the end of the enclosing call. See \ref PlantArchCallbackEdits.
     */
    void deletePlantInstance(const std::vector<uint> &plantIDs);

    //! Specify the threshold values for plant phenological stages. All time values have units of days.
    /**
     * \param[in] plantID ID of the plant.
     * \param[in] time_to_dormancy_break Length of the dormancy period.
     * \param[in] time_to_flower_initiation Time from emergence/dormancy required to reach flower creation (closed flowers).
     * \param[in] time_to_flower_opening Time from flower initiation to flower opening.
     * \param[in] time_to_fruit_set Time from flower opening required to reach fruit set (i.e., flower dies and fruit is created).
     * \param[in] time_to_fruit_maturity Time from fruit set date required to reach fruit maturity.
     * \param[in] time_to_dormancy Time from emergence/dormancy break required to enter the next dormancy period.
     * \param[in] max_leaf_lifespan [optional] Maximum lifespan of a leaf in days.
     * \param[in] is_evergreen [optional] True if the plant is evergreen (i.e., does not lose all leaves during senescence).
     * \note Any phenological stage can be skipped by specifying a negative threshold value. In this case, the stage will be skipped and the threshold for the next stage will be relative to the previous stage.
     */
    void setPlantPhenologicalThresholds(uint plantID, float time_to_dormancy_break, float time_to_flower_initiation, float time_to_flower_opening, float time_to_fruit_set, float time_to_fruit_maturity, float time_to_dormancy,
                                        float max_leaf_lifespan = 1e6, bool is_evergreen = false);

    //! Spread the timing of phenological events across the shoots and buds of a plant. All values are durations in days.
    /**
     * Without a spread, every shoot breaks dormancy on the same day and every bud reaches each stage on the same day. With one, each shoot or bud is given its own delay, drawn from a normal distribution
     * truncated to the window [0, duration] with mean duration/2 and standard deviation duration/4. The thresholds set by \ref setPlantPhenologicalThresholds() then mark the start of each event, when the first
     * shoot or bud reaches it, and the event continues for the given duration. A duration of 0 restores the default, in which no delays are drawn.
     *
     * \param[in] plantID ID of the plant.
     * \param[in] budbreak_duration Days over which the shoots break vegetative dormancy (leaf-out), starting time_to_dormancy_break after the plant entered dormancy. The floral buds of every shoot wake at the start
     * of the window whatever its shoot's delay, so flowering can come before leaf-out. Must be less than time_to_dormancy, so that every shoot breaks before the season ends.
     * \param[in] flowering_duration Days over which the floral buds make their first reproductive transition after they become active: flower initiation, or flower opening, or fruit set, whichever is the first
     * stage enabled by \ref setPlantPhenologicalThresholds(). The stages after it follow at their usual intervals, so a bud that flowers late also sets fruit late. Measured, this is the time from the first to the
     * last flower opening, which is about the reported bloom period (onset to petal fall) less time_to_fruit_set.
     * \param[in] fruit_maturity_duration Days by which the time each fruit takes to reach full size can exceed time_to_fruit_maturity. An inflorescence given its own maturity period through
     * PhytomerParameters::inflorescence.inflorescence_maturity_period is not varied.
     * \note With the carbohydrate model enabled, the flowering and fruit maturity spreads are in developmental days, like the bud clocks they delay (see CarbonDevelopmentParameters), while
     * the budbreak spread is in calendar days, since the dormancy cycle runs on the calendar.
     * \note A delay is drawn when the shoot or bud reaches the point at which it applies, so setting a spread affects only events still to come. Buds created during the season (on species whose flowers do not
     * require dormancy) are already spread by the node they sit on, and the flowering delay is added to that.
     */
    void setPlantPhenologicalSpread(uint plantID, float budbreak_duration, float flowering_duration, float fruit_maturity_duration);

    //! Sets the carbohydrate model parameters for a specific plant.
    /**
     * \param[in] plantID Identifier for the plant whose parameters are being set.
     * \param[in] carb_parameters Reference to the carbohydrate parameters to assign to the plant.
     */
    void setPlantCarbohydrateModelParameters(uint plantID, const CarbohydrateParameters &carb_parameters);

    //! Sets carbohydrate model parameters for specified plants.
    /**
     * \param[in] plantIDs A vector of plant IDs for which the parameters will be set.
     * \param[in] carb_parameters The carbohydrate model parameters to apply.
     */
    void setPlantCarbohydrateModelParameters(const std::vector<uint> &plantIDs, const CarbohydrateParameters &carb_parameters);

    //! Get the carbohydrate model parameters of a plant
    /**
     * \param[in] plantID ID of the plant.
     * \return The plant's carbohydrate model parameters: those of its library model, or those last given to setPlantCarbohydrateModelParameters().
     */
    [[nodiscard]] CarbohydrateParameters getPlantCarbohydrateModelParameters(uint plantID) const;

    /**
     * \brief Disables the phenological progression of a specified plant instance.
     *
     * \param[in] plantID Identifier of the plant whose phenology is to be disabled.
     */
    void disablePlantPhenology(uint plantID);

    //! Make every leaf of a plant use at least a given image of LeafPrototype::leaf_texture_series, whatever its age
    /**
     * A leaf normally moves through its images according to its own age. This is for changes that do not follow leaf age, such as a whole canopy turning color in autumn or senescing at the end of a crop's season. A leaf older than the stage for its age stays on the image for its age, and a shoot type with fewer images uses its last one.
     * \param[in] plantID Identifier of the plant.
     * \param[in] stage Index of the image. Setting it back to 0 returns every leaf to the image for its age.
     */
    void setPlantMinimumLeafTextureStage(uint plantID, uint stage);

    //! Get the image of LeafPrototype::leaf_texture_series that every leaf of a plant uses at minimum
    /**
     * \param[in] plantID Identifier of the plant.
     * \return Index of the image given by setPlantMinimumLeafTextureStage(), which is 0 by default.
     */
    [[nodiscard]] uint getPlantMinimumLeafTextureStage(uint plantID) const;

    //! Advance plant growth by a specified time interval for all plants
    /**
     * \param[in] time_step_days Time interval in days.
     * \note Not available inside a phytomer callback (\ref PhytomerParameters::phytomer_creation_function or \ref PhytomerParameters::phytomer_callback_function): a call made from one does nothing, and is reported once, as an aggregated warning, at the end of the enclosing call. See \ref PlantArchCallbackEdits.
     */
    void advanceTime(float time_step_days);

    //! Advance plant growth by a specified time interval for all plants
    /**
     * \param[in] time_step_years Number of years to advance.
     * \param[in] time_step_days Number of days to advance (added to number of years).
     * \note Not available inside a phytomer callback (\ref PhytomerParameters::phytomer_creation_function or \ref PhytomerParameters::phytomer_callback_function): a call made from one does nothing, and is reported once, as an aggregated warning, at the end of the enclosing call. See \ref PlantArchCallbackEdits.
     */
    void advanceTime(int time_step_years, float time_step_days);

    //! Advance plant growth by a specified time interval for a single plant
    /**
     * \param[in] plantID ID of the plant instance.
     * \param[in] time_step_days Time interval in days.
     * \note Not available inside a phytomer callback (\ref PhytomerParameters::phytomer_creation_function or \ref PhytomerParameters::phytomer_callback_function): a call made from one does nothing, and is reported once, as an aggregated warning, at the end of the enclosing call. See \ref PlantArchCallbackEdits.
     */
    void advanceTime(uint plantID, float time_step_days);

    //! Advance plant growth by a specified time interval for a specified set of plants
    /**
     * \param[in] plantIDs IDs of the plant instances.
     * \param[in] time_step_days Time interval in days.
     * \note Not available inside a phytomer callback (\ref PhytomerParameters::phytomer_creation_function or \ref PhytomerParameters::phytomer_callback_function): a call made from one does nothing, and is reported once, as an aggregated warning, at the end of the enclosing call. See \ref PlantArchCallbackEdits.
     */
    void advanceTime(const std::vector<uint> &plantIDs, float time_step_days);

    //! Adjust fruit objects to prevent collision with solid obstacles via rotation
    /**
     * This method checks all fruit in the plant architecture for collisions with solid obstacles
     * and rotates them about their base position to prevent intersection. The rotation is applied
     * in the opposite direction of the original pitch to simulate natural fruit lifting.
     *
     * \note This method should be called after all plant growth updates are complete.
     */
    void adjustFruitForObstacleCollision();

    //! Accumulates one hour of net photosynthesis from the leaves of every plant
    /**
     * Reads the primitive data "net_photosynthesis" (umol CO2 m^-2 s^-1) of every leaf primitive on a growing (non-dormant) shoot, multiplies it by the
     * primitive's area and one hour, and adds the result (mol C) to the running total of the phytomer bearing the leaf. The next call to advanceTime()
     * credits the totals to the shoots' carbohydrate pools and resets them. Call once per simulated hour, after the photosynthesis model has run.
     */
    void accumulateHourlyLeafPhotosynthesis();

    //! Accumulates the thermal time of a period of air temperature for every plant, to set how far the next advanceTime() call develops each plant under the carbohydrate model
    /**
     * With the carbohydrate model enabled, plant development runs in thermal time (see CarbonDevelopmentParameters): each advanceTime() call develops a plant by the thermal time supplied
     * here since the previous call, in units of the thermal time of one day at CarbohydrateParameters::development's reference temperature, and then resets it. Call it once per simulated
     * hour with the hour's mean air temperature (hours = 1), or once for a longer period at its mean temperature; the hours supplied between two advanceTime() calls must add up to the
     * length of the call. Plants built at an age with buildPlantInstanceFromLibrary() or buildPlantCanopyFromLibrary() are grown to that age at the reference temperature and need none.
     * \param[in] air_temperature_C Mean air temperature over the period (C).
     * \param[in] hours Length of the period (hours, > 0).
     */
    void accumulateThermalTime(float air_temperature_C, float hours);

    //! Accumulates the thermal time of a period of air temperature for the given plants (see accumulateThermalTime(float, float))
    /**
     * \param[in] plantIDs IDs of the plants.
     * \param[in] air_temperature_C Mean air temperature over the period (C).
     * \param[in] hours Length of the period (hours, > 0).
     */
    void accumulateThermalTime(const std::vector<uint> &plantIDs, float air_temperature_C, float hours);

    //! Thermal time a plant has accumulated under the carbohydrate model since it was built (C day; see CarbonDevelopmentParameters)
    /**
     * \param[in] plantID ID of the plant.
     * \return Thermal time credited by advanceTime() calls so far (C day), excluding any growth to the plant's build age.
     */
    [[nodiscard]] double getPlantThermalTime(uint plantID) const;

    //! Development rate of a plant at an air temperature (C day of thermal time per day of that temperature; see CarbonDevelopmentParameters)
    /**
     * \param[in] development Temperature response of development.
     * \param[in] air_temperature_C Air temperature (C).
     * \return 0 at or below the base and at or above the maximum temperature, rising linearly to (optimum - base) at the optimum temperature and falling linearly beyond it.
     */
    [[nodiscard]] static float developmentThermalRate(const CarbonDevelopmentParameters &development, float air_temperature_C);

    //! Adjusts the plant's maintenance respiration rate based on the prevailing air temperature
    /**
     * This function iterates through all the plants in the architecture.
     * It sets the plant's maintenance respiration rate for this timestep depending on the prevailing timeseries air temperature (K), using each plant's
     * CarbohydrateParameters::respiration_Q10. The temperature also sets the sugar-starch partitioning of the carbohydrate pools.
     *
     * \note Respiration rate defaults to that which occurs at 20C if not specified.
     */
    void adjustPlantMaintenanceRespiration(float Ta);

    // -- plant building methods -- //

    //! Define a new shoot type based on a set of ShootParameters
    /**
     * \param[in] shoot_type_label User-defined label for the new shoot type. This string is used later to reference this type of shoot.
     * \param[in] shoot_params Parameters structure for the new shoot type.
     * \note If a shoot type with this label already exists, its definition is replaced in its entirety rather than merged into. The stored parameters are the values passed in: shoot-level random
     * parameters are not resampled, so reading a shoot type and writing it straight back is a no-op.
     */
    void defineShootType(const std::string &shoot_type_label, const ShootParameters &shoot_params);

    //! Define the stem/trunk shoot (base of plant) to start a new plant. This requires a plant instance has already been created using the addPlantInstance() method.
    /**
     * \param[in] plantID ID of the plant instance.
     * \param[in] current_node_number Number of nodes of the stem shoot.
     * \param[in] base_rotation AxisRotation object (pitch, yaw, roll) specifying the orientation of the base of the shoot.
     * \param[in] internode_radius Radius of the internodes along the shoot.
     * \param[in] internode_length_max Maximum length (i.e., fully elongated) of the internodes along the shoot.
     * \param[in] internode_length_scale_factor_fraction Scaling factor of the maximum internode length to determine the actual initial internode length at the time of creation (=1 applies no scaling).
     * \param[in] leaf_scale_factor_fraction Scaling factor of the leaf/petiole to determine the actual initial leaf size at the time of creation (=1 applies no scaling).
     * \param[in] radius_taper Tapering factor of the internode radius along the shoot (0=constant radius, 1=linear taper to zero radius).
     * \param[in] shoot_type_label Label of the shoot type to be used for the base stem shoot. This requires that the shoot type has already been defined using the defineShootType() method.
     * \return ID of the new shoot to be used to reference it later.
     * \note Cannot be used inside a phytomer callback (\ref PhytomerParameters::phytomer_creation_function or \ref PhytomerParameters::phytomer_callback_function): a call made from one raises an error. See \ref PlantArchCallbackEdits.
     */
    uint addBaseStemShoot(uint plantID, uint current_node_number, const AxisRotation &base_rotation, float internode_radius, float internode_length_max, float internode_length_scale_factor_fraction, float leaf_scale_factor_fraction,
                          float radius_taper, const std::string &shoot_type_label);

    //! Manually append a new shoot at the end of an existing shoot. This is used when the characteristics of a shoot change along its length (e.g., from a unifoliate to trifoliate leaf).
    /**
     * The new shoot continues the parent's axis rather than branching off it: its base is seated on the tip
     * of the parent's last node and it keeps the parent's rank. It is therefore not given the outward offset
     * that addChildShoot() applies to seat a lateral branch on the parent's surface. Appending also stops the
     * parent from producing further phytomers at its apex.
     *
     * \param[in] plantID ID of the plant instance.
     * \param[in] parent_shoot_ID ID of the shoot to which the new shoot will be appended.
     * \param[in] current_node_number Number of nodes/phytomers of the newly appended shoot.
     * \param[in] base_rotation AxisRotation object (pitch, yaw, roll) specifying the orientation of the base of the shoot relative to the parent shoot.
     * \param[in] internode_radius Initial radius of the internodes along the shoot.
     * \param[in] internode_length_max Length of the internode of the newly appended shoot.
     * \param[in] internode_length_scale_factor_fraction Scaling factor of the maximum internode length to determine the actual initial internode length at the time of creation (=1 applies no scaling).
     * \param[in] leaf_scale_factor_fraction Scaling factor of the leaf/petiole to determine the actual initial leaf size at the time of creation (=1 applies no scaling).
     * \param[in] radius_taper Tapering factor of the internode radius along the shoot (0=constant radius, 1=linear taper to zero radius).
     * \param[in] shoot_type_label Label of the shoot type to be used for the new shoot. This requires that the shoot type has already been defined using the defineShootType() method.
     * \return ID of the new shoot to be used to reference it later.
     * \note The new shoot inherits the parent's bound on the radius of phytomers grown at its apex (ShootGrowthState::internode_radius_cap), if the parent has one.
     * \note Cannot be used inside a phytomer callback (\ref PhytomerParameters::phytomer_creation_function or \ref PhytomerParameters::phytomer_callback_function): a call made from one raises an error. See \ref PlantArchCallbackEdits.
     */
    uint appendShoot(uint plantID, int parent_shoot_ID, uint current_node_number, const AxisRotation &base_rotation, float internode_radius, float internode_length_max, float internode_length_scale_factor_fraction, float leaf_scale_factor_fraction,
                     float radius_taper, const std::string &shoot_type_label);

    //! Manually add a child shoot at the axillary bud of a phytomer.
    /**
     * \param[in] plantID ID of the plant instance.
     * \param[in] parent_shoot_ID ID of the shoot to which the new shoot will be added.
     * \param[in] parent_node_index Number of the node of the parent shoot at which the new shoot will be added.
     * \param[in] current_node_number Number of nodes of the newly added shoot.
     * \param[in] shoot_base_rotation AxisRotation object (pitch, yaw, roll) specifying the orientation of the base of the shoot.
     * \param[in] internode_radius Initial radius of the internodes along the shoot.
     * \param[in] internode_length_max Length of the internode of the newly appended shoot.
     * \param[in] internode_length_scale_factor_fraction Scaling factor of the maximum internode length to determine the actual initial internode length at the time of creation (=1 applies no scaling).
     * \param[in] leaf_scale_factor_fraction Scaling factor of the leaf/petiole to determine the actual initial leaf size at the time of creation (=1 applies no scaling).
     * \param[in] radius_taper Tapering factor of the internode radius along the shoot (0=constant radius, 1=linear taper to zero radius).
     * \param[in] shoot_type_label Label of the shoot type to be used for the new shoot. This requires that the shoot type has already been defined using the defineShootType() method.
     * \param[in] petiole_index [optional] Index of the petiole within the internode to which the new shoot will be attached (when there are multiple petioles per internode)
     * \return ID of the newly generated shoot.
     * \note Cannot be used inside a phytomer callback (\ref PhytomerParameters::phytomer_creation_function or \ref PhytomerParameters::phytomer_callback_function): a call made from one raises an error. See \ref PlantArchCallbackEdits.
     */
    uint addChildShoot(uint plantID, int parent_shoot_ID, uint parent_node_index, uint current_node_number, const AxisRotation &shoot_base_rotation, float internode_radius, float internode_length_max, float internode_length_scale_factor_fraction,
                       float leaf_scale_factor_fraction, float radius_taper, const std::string &shoot_type_label, uint petiole_index = 0);

    //! Manually add a shoot whose internode geometry is prescribed by measured node positions.
    /**
     * This builds a single shoot, rendered as a single continuous internode tube, that follows a path given by the caller rather than one generated from the shoot type's curvature and tortuosity parameters. It is
     * intended for reconstructing a plant from measured geometry such as a QSM, a digitized skeleton or photogrammetry. A shoot built through addBaseStemShoot(), appendShoot() or addChildShoot() is an
     * extrapolation from its base rotation and so cannot follow a measured curve; approximating one by chaining many short shoots produces a separate tube object per link, which leaves visible gaps at every bend.
     *
     * The supplied positions are the phytomer endpoints: N+1 positions define N internodes and therefore N phytomers. The phytomer parameter internode.length_segments still controls how finely each internode is
     * subdivided, with the intermediate nodes interpolated along the straight segment between the two prescribed endpoints. The caller controls internode length by choosing how many nodes to supply.
     *
     * The prescribed phytomers are created fully elongated and are therefore not re-scaled or re-curved by subsequent calls to advanceTime(). New phytomers added at the shoot apex as the plant grows are generated
     * normally from the shoot type's own parameters, continuing from the direction of the final prescribed internode, and use the mean of the prescribed internode lengths as their target length (change it with setShootInternodeLengthMax()). Prescribed radii
     * act as a lower bound: if the shoot type has a non-zero ShootParameters::girth_area_factor the pipe model may thicken an internode during growth, but never thins one, so a shoot type with a girth area factor
     * of zero preserves the prescribed radii exactly.
     *
     * When parent_shoot_ID is non-negative the base of the shoot is seated on the parent in the same way as addChildShoot(), i.e. offset from the attachment node to the surface of the parent internode, and the whole
     * path is translated onto that point. All relative geometry -- directions, internode lengths and curvature -- is preserved exactly; only the absolute position of the shoot changes, and only by the discrepancy
     * between the first supplied position and the attachment point. An error is raised if that discrepancy is large enough that the shoot would not be connected to its parent.
     *
     * \param[in] plantID ID of the plant instance.
     * \param[in] parent_shoot_ID ID of the shoot to which the new shoot will be attached, or -1 to create a base stem shoot at the start of a new plant.
     * \param[in] parent_node_index Number of the node of the parent shoot at which the new shoot will be added. Ignored when parent_shoot_ID is -1.
     * \param[in] internode_node_positions Positions of the internode nodes in world coordinates, ordered from the base of the shoot to its tip. Element 0 is the base of the shoot. At least two positions are required,
     * and no two consecutive positions may be coincident.
     * \param[in] internode_radii Radius of the shoot at each node, one per entry of internode_node_positions. All radii must be greater than zero.
     * \param[in] shoot_type_label Label of the shoot type to be used for the new shoot. This requires that the shoot type has already been defined using the defineShootType() method.
     * \param[in] petiole_index [optional] Index of the petiole within the internode to which the new shoot will be attached (when there are multiple petioles per internode)
     * \return ID of the newly generated shoot.
     * \note Like every other manually-added shoot, the new shoot is created dormant. Call breakPlantDormancy() before advanceTime() if it is to grow.
     * \note The axillary vegetative buds are created exactly as by addChildShoot(), from the shoot type's ShootParameters::vegetative_bud_break_probability_min and companions. If the type has
     * ShootParameters::growth_requires_dormancy false, they are sampled at once and those that do not break are set dead, and a later call to makePlantDormant() kills the rest; set them dormant first with
     * \ref Phytomer::setVegetativeBudState(BudState) if they are to break after a dormancy period.
     * \note The prescribed geometry is protected from the growth model, but not from the caller. The phytomers are created fully elongated, which is what exempts them from the elongation performed by advanceTime();
     * calling Phytomer::setInternodeLengthScaleFraction() on one of them directly overrides that and rescales the internode away from its measured length.
     * \note This form builds the measured geometry and grows the shoot afterwards with the same shoot type, so any parameter set to make the measured section buildable -- a node cap raised to fit it, curvature zeroed
     * so the measured path is not fought -- also governs the shoot's future growth. Where those two roles call for different values, use the overload taking a separate growth shoot type.
     * \note Cannot be used inside a phytomer callback (\ref PhytomerParameters::phytomer_creation_function or \ref PhytomerParameters::phytomer_callback_function): a call made from one raises an error. See \ref PlantArchCallbackEdits.
     */
    uint addShootFromNodePositions(uint plantID, int parent_shoot_ID, uint parent_node_index, const std::vector<helios::vec3> &internode_node_positions, const std::vector<float> &internode_radii, const std::string &shoot_type_label,
                                   uint petiole_index = 0);

    //! Manually add a shoot whose internode geometry is prescribed by measured node positions, growing afterwards as a different shoot type.
    /**
     * This behaves exactly as the overload above, except that the parameters used to *build* the measured geometry and the parameters governing how the shoot *grows next* are taken from two different shoot types.
     *
     * The two roles genuinely differ for a reconstruction. Building measured wood calls for curvature and tortuosity of zero so the measured path is not fought, a node cap at least as large as the longest measured
     * branch, and often a girth area factor of zero so the measured radii are preserved. None of those describe how the plant should grow: a shoot inheriting them extends perfectly straight and never reaches its node
     * cap. Supplying the growth type separately is what keeps the measurement from dictating the growth model, and it removes the need to maintain two hand-synchronised shoot types that differ only in these respects.
     *
     * The growth type governs the node caps consulted as the shoot extends, the gravitropic curvature of phytomers added at the apex, and the type of the shoots this shoot's vegetative buds produce. The build type
     * still supplies the phytomer parameters the measured section is constructed from, and remains the label reported by the shoot.
     *
     * \param[in] plantID ID of the plant instance.
     * \param[in] parent_shoot_ID ID of the shoot to which the new shoot will be attached, or -1 to create a base stem shoot at the start of a new plant.
     * \param[in] parent_node_index Number of the node of the parent shoot at which the new shoot will be added. Ignored when parent_shoot_ID is -1.
     * \param[in] internode_node_positions Positions of the internode nodes in world coordinates, ordered from the base of the shoot to its tip. Element 0 is the base of the shoot. At least two positions are required,
     * and no two consecutive positions may be coincident.
     * \param[in] internode_radii Radius of the shoot at each node, one per entry of internode_node_positions. All radii must be greater than zero.
     * \param[in] shoot_type_label Label of the shoot type whose parameters are used to build the measured geometry. This requires that the shoot type has already been defined using the defineShootType() method.
     * \param[in] growth_shoot_type_label Label of the shoot type whose parameters govern the shoot's future growth. This requires that the shoot type has already been defined using the defineShootType() method.
     * \param[in] petiole_index [optional] Index of the petiole within the internode to which the new shoot will be attached (when there are multiple petioles per internode)
     * \return ID of the newly generated shoot.
     * \note A measured branch longer than the growth type's ShootParameters::max_nodes is accepted rather than rejected -- its length is a property of the plant, not a caller error -- and simply stops extending.
     * \note ShootParameters::girth_area_factor is taken from the build type, not the growth type. A reconstruction sets it near zero precisely to preserve the measured radii, and because the pipe model only ever
     * increases a radius, taking it from the growth type instead would silently thicken measured wood. See \ref ShootGrowthState.
     * \note Cannot be used inside a phytomer callback (\ref PhytomerParameters::phytomer_creation_function or \ref PhytomerParameters::phytomer_callback_function): a call made from one raises an error. See \ref PlantArchCallbackEdits.
     */
    uint addShootFromNodePositions(uint plantID, int parent_shoot_ID, uint parent_node_index, const std::vector<helios::vec3> &internode_node_positions, const std::vector<float> &internode_radii, const std::string &shoot_type_label,
                                   const std::string &growth_shoot_type_label, uint petiole_index = 0);

    //! Manually append a shoot whose internode geometry is prescribed by measured node positions, continuing the axis of its parent shoot.
    /**
     * The counterpart of appendShoot() for measured geometry, as addShootFromNodePositions() is of addChildShoot(). The new shoot carries on from the tip of the parent shoot's last node along the same
     * botanical axis: its base is seated on that tip, on the parent's axis, rather than on the parent's surface as a lateral branch is, and the parent stops producing phytomers at its apex. Use it where
     * one axis is built as two shoots because its parts are of different shoot types - a hypocotyl and the stem above its cotyledons, for example. Built with addShootFromNodePositions() instead, the upper
     * shoot is a lateral and stands nearly a parent radius beside the axis it continues.
     *
     * The whole path is translated so that its first position lies on the parent's tip, so all relative geometry is preserved exactly, and an error is raised if the first position is further from that
     * tip than a shoot connected to it could be. Everything else is as for addShootFromNodePositions().
     *
     * \param[in] plantID ID of the plant instance.
     * \param[in] parent_shoot_ID ID of the shoot whose axis the new shoot continues.
     * \param[in] internode_node_positions Positions of the internode nodes in world coordinates, ordered from the base of the shoot to its tip. Element 0 is the base of the shoot, at the parent's tip. At
     * least two positions are required, and no two consecutive positions may be coincident.
     * \param[in] internode_radii Radius of the shoot at each node, one per entry of internode_node_positions. All radii must be greater than zero.
     * \param[in] shoot_type_label Label of the shoot type to be used for the new shoot. This requires that the shoot type has already been defined using the defineShootType() method.
     * \return ID of the newly generated shoot.
     * \note Cannot be used inside a phytomer callback (\ref PhytomerParameters::phytomer_creation_function or \ref PhytomerParameters::phytomer_callback_function): a call made from one raises an error. See \ref PlantArchCallbackEdits.
     */
    uint appendShootFromNodePositions(uint plantID, int parent_shoot_ID, const std::vector<helios::vec3> &internode_node_positions, const std::vector<float> &internode_radii, const std::string &shoot_type_label);

    //! Manually append a shoot whose internode geometry is prescribed by measured node positions, continuing the axis of its parent shoot and growing afterwards as a different shoot type.
    /**
     * As the overload above, with the shoot type that governs the shoot's future growth given separately from the one its measured geometry is built with. See the corresponding overload of
     * addShootFromNodePositions() for what each of the two types supplies.
     *
     * \param[in] plantID ID of the plant instance.
     * \param[in] parent_shoot_ID ID of the shoot whose axis the new shoot continues.
     * \param[in] internode_node_positions Positions of the internode nodes in world coordinates, ordered from the base of the shoot to its tip. Element 0 is the base of the shoot, at the parent's tip.
     * \param[in] internode_radii Radius of the shoot at each node, one per entry of internode_node_positions. All radii must be greater than zero.
     * \param[in] shoot_type_label Label of the shoot type whose parameters are used to build the measured geometry.
     * \param[in] growth_shoot_type_label Label of the shoot type whose parameters govern the shoot's future growth.
     * \return ID of the newly generated shoot.
     */
    uint appendShootFromNodePositions(uint plantID, int parent_shoot_ID, const std::vector<helios::vec3> &internode_node_positions, const std::vector<float> &internode_radii, const std::string &shoot_type_label,
                                      const std::string &growth_shoot_type_label);

    //! Manually add a child epicormic shoot (water sprout) at an arbitrary position along the shoot
    /**
     * \param[in] plantID ID of the plant instance.
     * \param[in] parent_shoot_ID ID of the shoot to which the new shoot will be added.
     * \param[in] parent_position_fraction Position along the parent shoot to add the epicormic shoot as a fraction of the parent shoot length.
     * \param[in] current_node_number Number of nodes of the newly added shoot.
     * \param[in] zenith_perturbation_degrees Pitch angle of epicormic shoot base away from parent shoot axis (degrees).
     * \param[in] internode_radius Initial radius of the internodes along the shoot. Values larger than the current radius of the parent internode the shoot emerges from are reduced to that radius.
     * \param[in] internode_length_max Length of the internode of the newly appended shoot.
     * \param[in] internode_length_scale_factor_fraction Scaling factor of the maximum internode length to determine the actual initial internode length at the time of creation (=1 applies no scaling).
     * \param[in] leaf_scale_factor_fraction Scaling factor of the leaf/petiole to determine the actual initial leaf size at the time of creation (=1 applies no scaling).
     * \param[in] radius_taper Tapering factor of the internode radius along the shoot (0=constant radius, 1=linear taper to zero radius).
     * \param[in] shoot_type_label Label of the shoot type to be used for the new shoot. This requires that the shoot type has already been defined using the defineShootType() method.
     * \return ID of the newly generated shoot.
     * \note Phytomers the shoot later grows during advanceTime() are likewise built no thicker than the parent internode's radius at the time the shoot emerged (see ShootGrowthState::internode_radius_cap).
     * \note Cannot be used inside a phytomer callback (\ref PhytomerParameters::phytomer_creation_function or \ref PhytomerParameters::phytomer_callback_function): a call made from one raises an error. See \ref PlantArchCallbackEdits.
     */
    uint addEpicormicShoot(uint plantID, int parent_shoot_ID, float parent_position_fraction, uint current_node_number, float zenith_perturbation_degrees, float internode_radius, float internode_length_max,
                           float internode_length_scale_factor_fraction, float leaf_scale_factor_fraction, float radius_taper, const std::string &shoot_type_label);

    //! Add a new phytomer at the terminal bud of a shoot.
    /**
     * \param[in] plantID ID of the plant instance.
     * \param[in] shootID ID of the shoot to which the phytomer will be added
     * \param[in] phytomer_parameters Parameters of the phytomer to be added
     * \param[in] internode_radius Radius of the phytomer internode at the time of creation
     * \param[in] internode_length_max Maximum internode length at full elongation
     * \param[in] internode_length_scale_factor_fraction Scaling factor of the maximum internode length to determine the actual initial internode length at the time of creation (=1 applies no scaling).
     * \param[in] leaf_scale_factor_fraction Scaling factor of the leaf/petiole to determine the actual initial leaf size at the time of creation (=1 applies no scaling).
     * \return ID of generated phytomer
     * \note Cannot be used inside a phytomer callback (\ref PhytomerParameters::phytomer_creation_function or \ref PhytomerParameters::phytomer_callback_function): a call made from one raises an error. See \ref PlantArchCallbackEdits.
     */
    int appendPhytomerToShoot(uint plantID, uint shootID, const PhytomerParameters &phytomer_parameters, float internode_radius, float internode_length_max, float internode_length_scale_factor_fraction, float leaf_scale_factor_fraction);

    //! Enable shoot type to produce epicormic child shoots (water sprouts)
    /**
     * \note The probability that the shoot produces an epicormic shoot over period of dt is P = (epicormic_probability_perlength_perday * shoot_length * dt)*sin(shoot_inclinantion)
     * \param[in] plantID ID of the plant instance.
     * \param[in] epicormic_shoot_type_label Label of the shoot type corresponding to epicormic shoots.
     * \param[in] epicormic_probability_perlength_perday Probability of epicormic shoot formation per unit length of the parent shoot per day.
     */
    void enableEpicormicChildShoots(uint plantID, const std::string &epicormic_shoot_type_label, float epicormic_probability_perlength_perday);

    //! Do not build internode primitive geometry in the Context
    void disableInternodeContextBuild();

    //! Do not build petiole primitive geometry in the Context
    void disablePetioleContextBuild();

    //! Do not build peduncle primitive geometry in the Context
    void disablePeduncleContextBuild();

    //! Enable automatic removal of organs that are below the ground plane
    /**
     * \param[in] ground_height [optional] Height of the ground plane (default = 0).
     */
    void enableGroundClipping(float ground_height = 0.f);

    // -- collision detection methods -- //

    //! Enable collision detection for plant growth avoidance (creates internal CollisionDetection instance)
    /**
     * \param[in] target_object_UUIDs [optional] Vector of specific UUIDs to avoid (empty = avoid all geometry)
     * \param[in] target_object_IDs [optional] Vector of specific object IDs to avoid (empty = avoid all objects)
     * \param[in] enable_petiole_collision [optional] Enable collision detection for petioles (default: false)
     * \param[in] enable_fruit_collision [optional] Enable collision detection for fruits (default: false)
     */
    void enableSoftCollisionAvoidance(const std::vector<uint> &target_object_UUIDs = {}, const std::vector<uint> &target_object_IDs = {}, bool enable_petiole_collision = false, bool enable_fruit_collision = false);

    //! Disable collision detection for plant growth and clean up internal CollisionDetection instance
    void disableCollisionDetection();

    //! Set collision avoidance parameters
    /**
     * \param[in] view_half_angle_deg Half-angle of the collision detection view cone in degrees (default = 80)
     * \param[in] look_ahead_distance How far ahead collisions will be considered in meters (default = 0.1)
     * \param[in] sample_count Number of directional samples within the cone (default = 256)
     * \param[in] inertia_weight Weight factor for directional inertia vs collision avoidance (0.0 = use optimal direction, 1.0 = ignore collision avoidance) (default = 0.4)
     */
    void setSoftCollisionAvoidanceParameters(float view_half_angle_deg, float look_ahead_distance, int sample_count, float inertia_weight);

    //! Mark specific geometry as static for collision detection efficiency
    /**
     * \param[in] target_UUIDs Vector of primitive UUIDs representing static obstacles (buildings, fixed structures, etc.)
     */
    void setStaticObstacles(const std::vector<uint> &target_UUIDs);

    //! Get access to the internal CollisionDetection instance (for advanced usage)
    /**
     * \return Pointer to internal CollisionDetection instance, or nullptr if not enabled
     */
    [[nodiscard]] CollisionDetection *getCollisionDetection() const;

    //! Set which organ types should participate in collision detection
    /**
     * \param[in] include_internodes Include internode segments in collision detection (default: false)
     * \param[in] include_leaves Include leaf surfaces in collision detection (default: true)
     * \param[in] include_petioles Include petiole segments in collision detection (default: false)
     * \param[in] include_flowers Include flower geometry in collision detection (default: false)
     * \param[in] include_fruit Include fruit geometry in collision detection (default: false)
     */
    void setCollisionRelevantOrgans(bool include_internodes, bool include_leaves, bool include_petioles, bool include_flowers, bool include_fruit);

    //! Enable solid obstacle avoidance for plant growth
    /**
     * \brief Designates certain primitives as solid obstacles that plant growth must avoid.
     *
     * When enabled, plant growth will gradually redirect as it approaches these obstacles,
     * with increasingly strong curvature as distance decreases. The plant will eventually
     * grow parallel to the obstacle surface at a marginal distance.
     *
     * \param[in] obstacle_UUIDs Vector of primitive UUIDs that represent solid obstacles
     * \param[in] avoidance_distance Distance at which obstacle avoidance begins (meters)
     * \param[in] enable_fruit_adjustment Enable automatic fruit rotation adjustment to avoid obstacles (default: false)
     * \param[in] enable_obstacle_pruning Enable solid obstacle collision pruning (default: true)
     */
    void enableSolidObstacleAvoidance(const std::vector<uint> &obstacle_UUIDs, float avoidance_distance = 0.5f, bool enable_fruit_adjustment = false, bool enable_obstacle_pruning = false);

    //! Configure Context geometry update scheduling for efficiency
    /**
     * \param[in] update_frequency How often to update Context geometry (1=every timestep, 2=every 2 timesteps, etc.)
     * \param[in] force_update_on_collision Force Context update when collision avoidance is triggered
     */
    void setGeometryUpdateScheduling(int update_frequency = 3, bool force_update_on_collision = true);

    //! Keep growing fruit from intersecting the internodes of their own plant
    /**
     * Once enabled, every time step of advanceTime() tests each fruiting floral bud's fruit against the nearby internodes of the same plant. A fruit found inside a branch is moved out at one of two
     * hinges, whichever moves the contact out with the smaller rotation: the peduncle swings, with everything it carries, rigidly about the node it grows from, or the fruit tilts about its own attachment
     * point on the peduncle -- the way a pedicel bends and a fruit leans when it comes to rest against a limb. Fruit are tested through
     * analytic proxies built from their meshes (an enclosing sphere, a chain of spheres along the fruit, and the triangles themselves for small meshes) against the plant's internode skeleton, so the
     * cost is a small fraction of a growth step and no collision BVH is involved. The swing is incremental: fruit that are already clear cost one bounding-sphere test per nearby segment, and a fruit that
     * cannot be cleared within the allowed swing, for example one wedged between spurs, is left at the least-penetrating pose reached. A fruit that has been moved and is clear of every branch,
     * because the branch has grown away or the fruit was pushed further than it needed, settles back toward its grown, hanging pose at a set rate, stopping where it would touch wood again. Swings persist through writePlantStructureXML() /
     * readPlantStructureXML(). Only internodes of the fruit's own plant are considered; leaves, petioles, flowers and other plants are not. See also setFruitBranchCollisionParameters().
     */
    void enableFruitBranchCollisionAvoidance();

    //! Stop testing fruit against branches; peduncles already swung keep their pose
    void disableFruitBranchCollisionAvoidance();

    //! Set the parameters of fruit-branch collision avoidance (see enableFruitBranchCollisionAvoidance())
    /**
     * \param[in] max_swing_deg Largest total angle (degrees) a peduncle may be swung from its grown pose, and likewise a fruit tilted about its attachment point; a fruit still in contact at these angles stays there (default = 90)
     * \param[in] candidate_refresh_days Days between rebuilds of each bud's list of nearby internode segments; new shoots appearing between rebuilds are only seen at the next one (default = 10)
     * \param[in] clearance Gap (meters) to leave between a fruit and a branch surface; 0 lets them touch (default = 0)
     * \param[in] settle_rate_deg_per_day [optional] Rate (degrees per day) at which a swung peduncle or tilted fruit that is clear of every branch turns back toward its grown, hanging pose; 0 leaves fruit where a contact put them (default = 10)
     */
    void setFruitBranchCollisionParameters(float max_swing_deg, float candidate_refresh_days, float clearance, float settle_rate_deg_per_day = 10.f);

    // -- attraction points methods -- //

    //! Enable attraction points to guide plant growth toward specific targets
    /**
     * \param[in] attraction_points Vector of 3D positions that plants should grow toward
     * \param[in] view_half_angle_deg Half-angle of the attraction detection view cone in degrees (default = 80)
     * \param[in] look_ahead_distance How far ahead attraction points will be considered in meters (default = 0.1)
     * \param[in] attraction_weight Weight factor for attraction vs natural growth (0.0 = ignore attraction, 1.0 = full attraction) (default = 0.6)
     */
    void enableAttractionPoints(const std::vector<helios::vec3> &attraction_points, float view_half_angle_deg = 45.0f, float look_ahead_distance = 0.1f, float attraction_weight = 0.6f);

    //! Disable attraction points and revert to natural growth
    void disableAttractionPoints();

    //! Update attraction points positions (overwrites existing points)
    /**
     * \param[in] attraction_points Updated vector of 3D positions that plants should grow toward
     */
    void updateAttractionPoints(const std::vector<helios::vec3> &attraction_points);

    //! Update attraction points positions by appending to any existing points
    /**
     * \param[in] attraction_points Updated vector of 3D positions that plants should grow toward
     */
    void appendAttractionPoints(const std::vector<helios::vec3> &attraction_points);

    //! Set attraction parameters
    /**
     * \param[in] view_half_angle_deg Half-angle of the attraction detection view cone in degrees
     * \param[in] look_ahead_distance How far ahead attraction points will be considered in meters
     * \param[in] attraction_weight Weight factor for attraction vs natural growth (0.0 = ignore attraction, 1.0 = full attraction)
     * \param[in] obstacle_reduction_factor Reduction factor for attraction when hard obstacles are present (default = 0.75)
     */
    void setAttractionParameters(float view_half_angle_deg, float look_ahead_distance, float attraction_weight, float obstacle_reduction_factor = 0.75f);

    //! Enable attraction points for a specific plant
    /**
     * \param[in] plantID Identifier of the plant to apply attraction points to
     * \param[in] attraction_points Vector of 3D positions that the plant should grow toward
     * \param[in] view_half_angle_deg Half-angle of the attraction detection view cone in degrees (default = 80)
     * \param[in] look_ahead_distance How far ahead attraction points will be considered in meters (default = 0.1m)
     * \param[in] attraction_weight Weight factor for attraction vs natural growth (0.0 = ignore attraction, 1.0 = full attraction) (default = 0.6)
     */
    void enableAttractionPoints(uint plantID, const std::vector<helios::vec3> &attraction_points, float view_half_angle_deg = 80.0f, float look_ahead_distance = 0.1f, float attraction_weight = 0.6f);

    //! Disable attraction points for a specific plant
    /**
     * \param[in] plantID Identifier of the plant to disable attraction points for
     */
    void disableAttractionPoints(uint plantID);

    //! Update attraction points positions for a specific plant (overwrites existing points)
    /**
     * \param[in] plantID Identifier of the plant to update attraction points for
     * \param[in] attraction_points Updated vector of 3D positions that the plant should grow toward
     */
    void updateAttractionPoints(uint plantID, const std::vector<helios::vec3> &attraction_points);

    //! Append attraction points for a specific plant
    /**
     * \param[in] plantID Identifier of the plant to append attraction points for
     * \param[in] attraction_points Vector of 3D positions to append to existing attraction points
     */
    void appendAttractionPoints(uint plantID, const std::vector<helios::vec3> &attraction_points);

    //! Set attraction parameters for a specific plant
    /**
     * \param[in] plantID Identifier of the plant to set attraction parameters for
     * \param[in] view_half_angle_deg Half-angle of the attraction detection view cone in degrees
     * \param[in] look_ahead_distance How far ahead attraction points will be considered in meters
     * \param[in] attraction_weight Weight factor for attraction vs natural growth (0.0 = ignore attraction, 1.0 = full attraction)
     * \param[in] obstacle_reduction_factor Reduction factor for attraction when hard obstacles are present (default = 0.75)
     */
    void setAttractionParameters(uint plantID, float view_half_angle_deg, float look_ahead_distance, float attraction_weight, float obstacle_reduction_factor = 0.75f);

    /**
     * \brief Detects attraction points within a viewing cone from a given position.
     *
     * This method finds the closest attraction point within a cone defined by the look direction,
     * half-angle, and look-ahead distance. It performs pure geometric calculations without
     * requiring collision detection infrastructure.
     *
     * \param[in] vertex Starting position (apex of the viewing cone)
     * \param[in] look_direction Direction the cone is pointing (will be normalized)
     * \param[in] look_ahead_distance Maximum distance to look for attraction points
     * \param[in] half_angle_degrees Half-angle of the viewing cone in degrees (0-180)
     * \param[out] direction_to_closest Unit vector pointing to the closest attraction point (if found)
     * \return True if an attraction point was found within the cone, false otherwise
     */
    bool detectAttractionPointsInCone(const helios::vec3 &vertex, const helios::vec3 &look_direction, float look_ahead_distance, float half_angle_degrees, helios::vec3 &direction_to_closest) const;

    //! Overloaded version that accepts attraction points as a parameter (for plant-specific attraction points)
    bool detectAttractionPointsInCone(const std::vector<helios::vec3> &attraction_points, const helios::vec3 &vertex, const helios::vec3 &look_direction, float look_ahead_distance, float half_angle_degrees, helios::vec3 &direction_to_closest) const;

    // -- methods for modifying the current plant state -- //

    /**
     * \brief Initializes the carbohydrate pool for all shoots of all plant instances
     * \param[in] carbohydrate_concentration_molC_m3 Concentration of carbohydrates in molC per cubic meter of wood
     * \note See initializePlantCarbohydratePool().
     */
    void initializeCarbohydratePool(float carbohydrate_concentration_molC_m3);

    /**
     * \brief Initializes the carbohydrate pool for a specific plant.
     *
     * Every shoot is given the concentration times its internode volume, and the root system is filled to the same fraction of its storage capacity
     * (the concentration times the root volume times CarbohydrateParameters::root_storage_ratio). The plant's current structure is taken as already built:
     * only growth after initialization is charged to the pools, so a plant can be grown with the carbohydrate model off and the model enabled afterwards.
     * \param[in] plantID Unique identifier of the plant.
     * \param[in] carbohydrate_concentration_molC_m3 Initial carbohydrate concentration in moles of carbon per cubic meter of wood.
     * \note The plant with the specified ID must exist, and the carbohydrate concentration must be non-negative.
     */
    void initializePlantCarbohydratePool(uint plantID, float carbohydrate_concentration_molC_m3);

    /**
     * \brief Initializes the carbohydrate pool for a specific shoot of a plant.
     *
     * \param[in] plantID Identifier for the plant whose shoot's carbohydrate pool is being initialized.
     * \param[in] shootID Identifier for the shoot of the plant.
     * \param[in] carbohydrate_concentration_molC_m3 Carbohydrate concentration in moles of carbon per cubic meter; must be non-negative.
     * \note Throws an error if the plant or shoot ID does not exist, or if the carbohydrate concentration is negative. The shoot's current structure is taken as already built.
     */
    void initializeShootCarbohydratePool(uint plantID, uint shootID, float carbohydrate_concentration_molC_m3);

    //! Get the non-structural carbohydrate (sugar + starch) held in a plant's root system
    /**
     * \param[in] plantID Identifier of the plant.
     * \return Root carbohydrate pool (mol C).
     */
    [[nodiscard]] float getPlantRootCarbohydratePool(uint plantID) const;

    //! Get the total non-structural carbohydrate (sugar + starch) of a plant: every living shoot plus the root system
    /**
     * \param[in] plantID Identifier of the plant.
     * \return Whole-plant carbohydrate pool (mol C).
     */
    [[nodiscard]] float getPlantTotalCarbohydratePool(uint plantID) const;

    //! Get a plant's non-structural carbohydrate as a fraction of its storage capacity
    /**
     * The storage capacity is CarbohydrateParameters::storage_capacity times the dry mass of every living shoot's wood, plus root_storage_ratio times as much per unit dry mass of roots.
     * The fill never exceeds 1 after a time step, because carbohydrate above capacity is released below ground.
     * \param[in] plantID Identifier of the plant.
     * \return Whole-plant carbohydrate divided by storage capacity (0 for a plant with no wood).
     */
    [[nodiscard]] float getPlantCarbohydrateStorageFill(uint plantID) const;

    //! Get a plant's carbon fluxes summed over the most recent call to advanceTime()
    /**
     * \param[in] plantID Identifier of the plant.
     * \return Photosynthesis, respiration and construction fluxes (mol C). The change in getPlantTotalCarbohydratePool() over the call equals
     * net photosynthesis minus the respiration, construction, storage_overflow and shed_branch_carbohydrate terms, unless shoots were pruned by other
     * means during it. photosynthesis_downregulation (not part of net photosynthesis) and fruit_dropped (a count) are diagnostics.
     */
    [[nodiscard]] PlantCarbonBudget getPlantCarbonBudget(uint plantID) const;

    //! Get the grain sink state of a plant under the grass sub-model, summed over its culms
    /**
     * \param[in] plantID Identifier of the plant.
     * \return The number of culms bearing a panicle, their grain number, the carbon built into grain and into panicle structure, the carbon the grain could have received at its potential
     * filling rate, and the carbohydrate held in the culms. Every culm counts whether or not its grain number has been fixed yet; a plant without panicles returns zeros.
     * \note The plant must exist and run CARBON_SUBMODEL_GRASS.
     */
    [[nodiscard]] PlantGrainStatus getPlantGrainStatus(uint plantID) const;

    /**
     * \brief Adjusts the leaf scaling factor (length as a fraction of its maximal length) for a specific phytomer on a plant shoot.
     *
     * \param[in] plantID Identifier of the plant.
     * \param[in] shootID Identifier of the shoot on the specified plant.
     * \param[in] node_number Node number of the phytomer to adjust.
     * \param[in] leaf_scale_factor_fraction Fractional scaling factor for the leaf, must be in the range [0, 1].
     */
    void setPhytomerLeafScale(uint plantID, uint shootID, uint node_number, float leaf_scale_factor_fraction);

    //! Set the target length of internodes grown at the apex of an existing shoot.
    /**
     * Each phytomer added at the shoot apex elongates toward this length. A shoot built by addBaseStemShoot(), appendShoot() or addChildShoot() starts with the internode length it was created with; a shoot built
     * by addShootFromNodePositions() starts with the mean of its prescribed internode lengths, which for a measured seedling is mostly hypocotyl and says little about how long its next internodes will be.
     * Internodes that already exist are not changed.
     * \param[in] plantID ID of the plant instance.
     * \param[in] shootID ID of the shoot within the plant.
     * \param[in] internode_length_max Target internode length in metres; must be greater than zero.
     */
    void setShootInternodeLengthMax(uint plantID, uint shootID, float internode_length_max);

    //! Prescribe the path of a petiole on an existing phytomer from measured node positions.
    /**
     * This is the organ-level counterpart of addShootFromNodePositions(). Where that method prescribes the internode skeleton of a shoot, this one prescribes the centerline of a single petiole hanging off it, so
     * that a reconstruction from labelled measurements -- a segmented point cloud, a digitized plant -- can follow the measured petiole rather than the path the shoot type's petiole pitch and curvature would
     * generate.
     *
     * The supplied positions are the nodes of the petiole tube, ordered from the base outward. Their number is free and need not match the shoot type's PetioleParameters::length_segments; the petiole tube is
     * rebuilt to match. The caller controls petiole length by choosing where to put the nodes.
     *
     * The petiole's leaves are carried to the new path: each stays at the fraction of the petiole's length it was attached at, and is turned as the petiole's direction there has turned. They are the same
     * leaf objects, moved rather than rebuilt. (The leaves of a petiole that had no length, such as that of a sessile leaf, have no position along it to keep and are left where they are.)
     *
     * The first supplied position is snapped onto the tip of the internode the petiole grows from, and the rest of the path is translated by the same amount. All relative geometry -- directions, segment lengths
     * and curvature -- is preserved exactly; only the absolute position changes, and only by the discrepancy between the first supplied position and the internode tip. An error is raised if that discrepancy is
     * large enough that the petiole would not be attached to the stem. The snap is not cosmetic: every petiole is re-seated onto its internode tip on each timestep, by translating the whole phytomer by the
     * discrepancy, so a base that merely sits close to the tip would be shifted again on every timestep.
     *
     * The prescribed petiole is not re-scaled by subsequent calls to advanceTime(). Its radii are held as given rather than treated as a lower bound, since a petiole is not subject to the pipe model that thickens
     * internodes.
     *
     * \param[in] plantID ID of the plant instance.
     * \param[in] shootID ID of the shoot carrying the phytomer.
     * \param[in] node_index Index of the phytomer within the shoot, counted from the base of the shoot.
     * \param[in] petiole_index Index of the petiole within the phytomer (for a species with more than one petiole per internode).
     * \param[in] petiole_node_positions Positions of the petiole nodes in world coordinates, ordered from the base of the petiole to its tip. At least two positions are required, and no two consecutive positions
     * may be coincident.
     * \param[in] petiole_node_radii Radius of the petiole at each node, one per entry of petiole_node_positions. All radii must be greater than zero.
     * \note Call this before setPetioleLeafGeometry() for the same petiole, since leaf placement is oriented from the petiole axis.
     * \note The prescribed geometry is protected from the growth model, but not from the caller. Calling Phytomer::setLeafScaleFraction() on the phytomer directly will rescale the petiole away from its measured
     * path, in the same way that Phytomer::setInternodeLengthScaleFraction() overrides a prescribed internode.
     */
    void setPetioleNodePositions(uint plantID, uint shootID, uint node_index, uint petiole_index, const std::vector<helios::vec3> &petiole_node_positions, const std::vector<float> &petiole_node_radii);

    //! Prescribe the base position, orientation and size of every leaf on a petiole from measured geometry.
    /**
     * This is the leaf-level counterpart of setPetioleNodePositions(), intended for the same reconstruction workflow. Every leaf on the petiole is prescribed in one call: for a compound leaf the leaflets are not
     * independent, since a leaflet's roll and yaw signs and the prototype it is a copy of all follow from its position along the petiole. A species with one leaf per petiole passes one-element vectors.
     *
     * Each leaf is rebuilt from its prototype and oriented from the supplied angles, which are intrinsic rotations in the leaf's rest frame on the petiole rather than inputs to the procedural rotation chain
     * a grown leaf goes through. The frame is built from the petiole tip axis t, the horizontal direction h perpendicular to it (world +y when t is vertical) and the upward normal of the petiole plane
     * n = t x h. At zero angles the leaf's midrib is t turned about n by the leaflet's position on a compound leaf (t for a terminal or single leaf, h or -h for a lateral leaflet held off the tip, evenly fanned
     * for leaflets attached at the tip), and its blade normal is n. Yaw then turns the leaf about n, pitch raises its tip toward n, and roll turns the blade about its own midrib, right-handed, in that order.
     * The convention is the same for every leaf whatever its position on the petiole or the node it hangs from, and involves no size-dependent correction, so the angles follow directly from a measured midrib
     * and blade normal expressed in that frame.
     *
     * The prescribed base, orientation and size are held exactly and are not changed by subsequent calls to advanceTime(). Prescribed leaves are additionally exempt from the self-weight droop, so the blade keeps
     * the prototype's rest shape rather than continuing to bend as the leaf ages.
     *
     * The prescribed pose, the base positions and a prescribed petiole path are written by writePlantStructureXML() and restored by readPlantStructureXML().
     *
     * \param[in] plantID ID of the plant instance.
     * \param[in] shootID ID of the shoot carrying the phytomer.
     * \param[in] node_index Index of the phytomer within the shoot, counted from the base of the shoot.
     * \param[in] petiole_index Index of the petiole within the phytomer (for a species with more than one petiole per internode).
     * \param[in] leaf_bases Base position of each leaf in world coordinates, one per leaf on the petiole.
     * \param[in] leaf_rotations Roll, pitch and yaw of each leaf in radians, one per leaf on the petiole.
     * \param[in] leaf_sizes Fully-elongated size of each leaf in meters, one per leaf on the petiole. All sizes must be greater than zero.
     * \param[in] leaf_stalk_lengths [optional] Length of each leaf's stalk (petiolule) in meters, for a leaf prototype that builds one: the distance from the leaf's base to where its blade begins. One per
     * leaf, all greater than zero; with none given each stalk is the species' share of its blade's length. A stalk is as thick as the petiole is where the leaf's base lies along it, whatever the leaf's size.
     * \note Supplying a different number of leaves than the petiole carries raises an error rather than adding or removing leaves; change the count with setPetioleLeafCount() first.
     * \note Rebuilding each leaf discards primitive data a caller has attached to it. The object label and material are restored; other primitive data is not.
     * \param[in] leaf_stalk_bases [optional] Where each leaf's stalk leaves the petiole, in world coordinates, one per leaf. A stalk is otherwise built back along its blade's own line from the leaf's base, and a
     * measured stalk seldom lies on that line: it rises from the petiole to a blade held a few millimetres off it. With these the stalk runs straight from the given point to where the blade begins and
     * bends there into the join under the blade, which is unchanged; the blade stays where its base, angles and size put it. A stalk may be no longer than its blade and must reach the blade from behind
     * its base, and it is as thick as the petiole where it leaves it; anything else raises an error. Needs a leaf prototype that builds a stalk.
     */
    void setPetioleLeafGeometry(uint plantID, uint shootID, uint node_index, uint petiole_index, const std::vector<helios::vec3> &leaf_bases, const std::vector<AxisRotation> &leaf_rotations,
                                const std::vector<float> &leaf_sizes, const std::vector<float> &leaf_stalk_lengths = {}, const std::vector<helios::vec3> &leaf_stalk_bases = {});

    //! Change the number of leaves (leaflets) on a petiole of an existing phytomer.
    /**
     * A shoot type fixes leaf.leaves_per_petiole for every phytomer it builds, but a measured compound leaf carries whatever number of leaflets it has: a young tomato leaf three or five, a mature one seven
     * or more, a cotyledon one. This deletes the petiole's leaves and rebuilds leaf_count of them procedurally from the leaf prototype, sized by leaflet_scale and prototype_scale, oriented with the
     * phytomer's leaf angles and spaced along the petiole by leaflet_offset, exactly as a new phytomer's leaves are. Call it before setPetioleLeafGeometry(), which then expects leaf_count entries. The rebuilt
     * leaves are exempt from drooping with age and, like prescribed leaves, are not re-posed by advanceTime().
     * \param[in] plantID ID of the plant instance.
     * \param[in] shootID ID of the shoot within the plant.
     * \param[in] node_index Index of the phytomer along the shoot.
     * \param[in] petiole_index Index of the petiole within the phytomer.
     * \param[in] leaf_count New number of leaves on the petiole; must be at least one.
     */
    void setPetioleLeafCount(uint plantID, uint shootID, uint node_index, uint petiole_index, uint leaf_count);

    //! Regenerate one petiole of an existing phytomer from its own pitch, curvature, length and lean, and rebuild its leaves on it.
    /**
     * A shoot type gives every phytomer the same distribution of petiole pitch and curvature. A plant built leaf by leaf from measurements, or fitted one leaf at a time, needs each petiole to have its own.
     * This generates the petiole by the phytomer constructor's rules from the values given, prescribes it with setPetioleNodePositions() (so it is held as given from then on), and rebuilds the petiole's
     * leaves on it with the phytomer's leaf parameters. See Phytomer::setPetioleShape() for the conventions.
     * \param[in] plantID ID of the plant instance.
     * \param[in] shootID ID of the shoot within the plant.
     * \param[in] node_index Index of the phytomer along the shoot.
     * \param[in] petiole_index Index of the petiole within the phytomer.
     * \param[in] outward_direction Direction the petiole leans toward, in world coordinates; any vector that is not parallel to the internode axis.
     * \param[in] pitch_degrees Angle between the petiole's base and the internode axis, in degrees.
     * \param[in] curvature Change in the petiole's direction along its length, in degrees per meter.
     * \param[in] length Length of the petiole in meters; must be greater than zero.
     */
    void setPetioleShape(uint plantID, uint shootID, uint node_index, uint petiole_index, const helios::vec3 &outward_direction, float pitch_degrees, float curvature, float length);

    //! Rebuild the leaflets of one petiole of an existing phytomer with their own count, spacing and sizes.
    /**
     * The per-phytomer counterpart of the shoot parameters leaf.leaves_per_petiole, leaf.leaflet_offset, leaf.leaflet_scale and leaf.prototype_scale. See Phytomer::setPetioleLeaflets().
     * \param[in] plantID ID of the plant instance.
     * \param[in] shootID ID of the shoot within the plant.
     * \param[in] node_index Index of the phytomer along the shoot.
     * \param[in] petiole_index Index of the petiole within the phytomer.
     * \param[in] leaf_count Number of leaves on the petiole; must be at least one.
     * \param[in] leaflet_offset Spacing between lateral leaflet pairs as a fraction of the petiole length; zero or greater.
     * \param[in] leaflet_scale Size of each lateral pair relative to the pair nearer the tip; greater than zero.
     * \param[in] prototype_scale Fully-elongated size of the terminal leaflet in meters; greater than zero.
     * \param[in] intercalary_leaflet_scale Size of the intercalary leaflets relative to the primary pair nearer the tip; zero for a leaf with none. Negative (the default) keeps the phytomer's current value,
     * which a leaf measured with no intercalary leaflets would otherwise inherit from its shoot type.
     */
    void setPetioleLeaflets(uint plantID, uint shootID, uint node_index, uint petiole_index, uint leaf_count, float leaflet_offset, float leaflet_scale, float prototype_scale,
                            float intercalary_leaflet_scale = -1.f);

    //! Rebuild the leaflets of one petiole of an existing phytomer turned by their own pitch, yaw and roll.
    /**
     * The per-phytomer counterpart of the shoot parameters leaf.pitch, leaf.yaw and leaf.roll. See Phytomer::setPetioleLeafletAngles().
     * \param[in] plantID ID of the plant instance.
     * \param[in] shootID ID of the shoot within the plant.
     * \param[in] node_index Index of the phytomer along the shoot.
     * \param[in] petiole_index Index of the petiole within the phytomer.
     * \param[in] pitch_degrees Leaflet pitch in degrees.
     * \param[in] yaw_degrees Leaflet yaw in degrees.
     * \param[in] roll_degrees Leaflet roll in degrees.
     */
    void setPetioleLeafletAngles(uint plantID, uint shootID, uint node_index, uint petiole_index, float pitch_degrees, float yaw_degrees, float roll_degrees);

    //! Reshape the leaflets of one petiole of an existing phytomer with their own blade shape.
    /**
     * The per-phytomer counterpart of the leaf prototype's leaf_aspect_ratio, longitudinal_curvature, lateral_curvature and midrib_fold_fraction. The leaves are reshaped in place, keeping their object IDs
     * and object data. See Phytomer::setPetioleLeafShape().
     * \param[in] plantID ID of the plant instance.
     * \param[in] shootID ID of the shoot within the plant.
     * \param[in] node_index Index of the phytomer along the shoot.
     * \param[in] petiole_index Index of the petiole within the phytomer.
     * \param[in] aspect_ratio Ratio of blade width to length; greater than zero.
     * \param[in] longitudinal_curvature Curvature along the blade's length (+ up, - down).
     * \param[in] lateral_curvature Curvature across the blade's width (+ up, - down).
     * \param[in] midrib_fold_fraction Fold along the midrib, from -1 (halves folded down shut) through 0 (flat) to 1 (halves folded up shut).
     */
    void setPetioleLeafShape(uint plantID, uint shootID, uint node_index, uint petiole_index, float aspect_ratio, float longitudinal_curvature, float lateral_curvature, float midrib_fold_fraction);

    //! Roll the leaves of one petiole of an existing plant about their midribs
    /**
     * Wraps each blade's cross-section onto a circular arc about its midrib, toward its upper side (see LeafPrototype::roll_fraction), without changing its area: grasses roll their leaves this way under
     * water stress, and unroll them again as it eases, over the course of a day. The leaves are reshaped in place, keeping their object IDs and object data (for example the nitrogen model's
     * "leaf_nitrogen_gN_m2", "age" and "leaf_texture_stage"), their size, orientation and position, and the rest of their shape; a drooping leaf is deflected again from its new rest shape. A blade with too
     * few cells across its width to form the arc is re-meshed finer the first time it is rolled, keeping its object but replacing its blade primitives (see Phytomer::reshapeLeaf()). The phytomer's own
     * copy of the leaf prototype takes the value, so leaves it builds later are rolled too, and the roll is recorded by writePlantStructureXML().
     * \param[in] plantID ID of the plant instance.
     * \param[in] shootID ID of the shoot within the plant.
     * \param[in] node_index Index of the phytomer along the shoot.
     * \param[in] petiole_index Index of the petiole within the phytomer.
     * \param[in] roll_fraction Roll as a fraction of a closed tube, from 0 (flat) to 1 (edges meeting).
     * \note Only leaves built by the generic leaf prototype can be rolled; an OBJ-loaded or custom-built leaf raises an error.
     */
    void setPetioleLeafRoll(uint plantID, uint shootID, uint node_index, uint petiole_index, float roll_fraction);

    //! Roll every leaf of an existing plant about its midrib
    /**
     * Applies \ref setPetioleLeafRoll() to every petiole of the plant that carries leaves.
     * \param[in] plantID ID of the plant instance.
     * \param[in] roll_fraction Roll as a fraction of a closed tube, from 0 (flat) to 1 (edges meeting).
     */
    void setPlantLeafRoll(uint plantID, float roll_fraction);

    //! Set whether one leaf bends under its own weight
    /**
     * A leaf of a species with a non-zero LeafPrototype::flexibility bends under its own weight as it grows, but a leaf placed with setPetioleLeafGeometry() or rebuilt by setPetioleLeafCount() keeps its
     * prototype's rest shape. Turning droop on gives such a leaf the bend a grown leaf has: the plug-in's own self-weight deflection, worked out from the leaf's orientation as it is placed, its size taken as
     * its fully expanded size, and its compliance aged with its phytomer, exactly as for a grown leaf. The leaf keeps bending as it would if it had grown there: it is bent again when its orientation changes
     * (for example by a joint pose) and as its phytomer ages. Turning droop off gives a leaf, grown or placed, its rest shape. The leaf keeps its object and primitives; only its vertices move. The setting is
     * recorded by writePlantStructureXML(). See \ref PlantArch_leafdroop.
     * \param[in] plantID ID of the plant instance.
     * \param[in] shootID ID of the shoot carrying the phytomer.
     * \param[in] node_index Index of the phytomer within the shoot.
     * \param[in] petiole_index Index of the petiole within the phytomer.
     * \param[in] leaf_index Index of the leaf on the petiole.
     * \param[in] droops True to bend the blade under its own weight, false to give it its rest shape.
     * \note Only a leaf built by the generic leaf prototype has a rest shape to bend from; an OBJ-loaded or custom-built leaf raises an error. Turning droop on for a leaf of a species whose flexibility is
     * zero, or for a leaf whose blade follows a midrib prescribed by setLeafMidrib(), raises an error.
     */
    void setLeafDroop(uint plantID, uint shootID, uint node_index, uint petiole_index, uint leaf_index, bool droops);

    //! Whether one leaf bends under its own weight (see setLeafDroop())
    /**
     * \param[in] plantID ID of the plant instance.
     * \param[in] shootID ID of the shoot carrying the phytomer.
     * \param[in] node_index Index of the phytomer within the shoot.
     * \param[in] petiole_index Index of the petiole within the phytomer.
     * \param[in] leaf_index Index of the leaf on the petiole.
     * \return True if the blade is bent under its own weight.
     */
    [[nodiscard]] bool doesLeafDroop(uint plantID, uint shootID, uint node_index, uint petiole_index, uint leaf_index) const;

    //! Lay one leaf's blade along a prescribed midrib, twisted about it by a given roll at each point
    /**
     * Bends the blade so that its midrib follows a polyline, as when a blade is fitted to a midrib traced in a point cloud: a grass blade that rises, arches over and hangs, a curvature that changes along
     * the blade, or a blade that twists, none of which the blade's own shape terms can give. Each row of the blade's lattice across its width is carried rigidly with its station on the midrib, so the
     * blade keeps its outline and its cross-section (fold, lateral curvature, roll, waves) and is bent without being stretched.
     *
     * The midrib is given in world coordinates from the base of the blade (where the blade begins, beyond any stalk) to its tip, by any number of points of at least two. The first point is moved onto the
     * blade's base, and the rest of the polyline with it, and an error is raised if it is further from it than 1 mm or 2% of the polyline's length, whichever is larger. The blade's stations are laid along
     * the polyline keeping the proportions of the blade's own midrib segments, and the blade takes the polyline's length: a longer or shorter midrib scales the blade, its width with it.
     *
     * Each row turns with the midrib segment that ends at its station, carried from the blade's base without twisting (a rotation-minimizing frame, which has no special direction, so a blade pointing
     * straight up is no different from any other); the row at the base keeps the orientation the blade has where it leaves its stalk or petiole. The roll at each point then twists the blade about the
     * midrib from there, and is interpolated along the midrib between points. A blade given its own midrib as read by getLeafMidrib(), or the midrib of the same blade drooped under its own weight, takes
     * that shape.
     *
     * The midrib replaces the blade's droop, which is turned off while it is prescribed, and its longitudinal curvature. It is held in the leaf's own frame, so moving the leaf (a joint pose,
     * setPlantBasePosition()) carries it, and reshapeLeaf() lays the new cross-section along it. The petiole's growth is pinned as setPetioleLeafGeometry() pins it, so that advanceTime() does not rescale
     * the blade. The leaf keeps its object and primitives. setPetioleLeafGeometry() and setPetioleLeafCount() rebuild the leaf and so discard its midrib. The midrib is recorded by
     * writePlantStructureXML(). See \ref PlantArch_blademidrib.
     * \param[in] plantID ID of the plant instance.
     * \param[in] shootID ID of the shoot carrying the phytomer.
     * \param[in] node_index Index of the phytomer within the shoot.
     * \param[in] petiole_index Index of the petiole within the phytomer.
     * \param[in] leaf_index Index of the leaf on the petiole.
     * \param[in] midrib Points of the midrib in world coordinates, from the base of the blade to its tip; at least two, no two consecutive ones coincident.
     * \param[in] rolls Twist of the blade about the midrib at each point, in radians, right-handed about the midrib's direction from base to tip, measured from the blade bent along the midrib without
     * twisting; one per point, or none for an untwisted blade.
     * \note Only a leaf built by the generic leaf prototype has a lattice to lay along a midrib; an OBJ-loaded or custom-built leaf raises an error.
     */
    void setLeafMidrib(uint plantID, uint shootID, uint node_index, uint petiole_index, uint leaf_index, const std::vector<helios::vec3> &midrib, const std::vector<float> &rolls = {});

    //! Lay one leaf's blade along a prescribed midrib, turned about it to face a given normal at each point
    /**
     * As setLeafMidrib(uint, uint, uint, uint, uint, const std::vector<helios::vec3>&, const std::vector<float>&), but the blade's twist is given by the normal of its upper side at each point, as a point
     * cloud gives it, rather than by a roll. The normals are interpolated along the midrib between points and taken perpendicular to it; a normal that is (nearly) parallel to the midrib raises an error,
     * since it does not fix how the blade is turned. The normal is that of the blade ignoring its fold and curvature across the midrib.
     * \param[in] plantID ID of the plant instance.
     * \param[in] shootID ID of the shoot carrying the phytomer.
     * \param[in] node_index Index of the phytomer within the shoot.
     * \param[in] petiole_index Index of the petiole within the phytomer.
     * \param[in] leaf_index Index of the leaf on the petiole.
     * \param[in] midrib Points of the midrib in world coordinates, from the base of the blade to its tip; at least two, no two consecutive ones coincident.
     * \param[in] normals Normal of the blade's upper side at each point, in world coordinates; one per point. They need not be unit length.
     */
    void setLeafMidrib(uint plantID, uint shootID, uint node_index, uint petiole_index, uint leaf_index, const std::vector<helios::vec3> &midrib, const std::vector<helios::vec3> &normals);

    //! Get the midrib of one leaf's blade as the blade now is
    /**
     * Works for any leaf built by the generic leaf prototype, whether its midrib was prescribed or not. For a blade with no prescribed midrib the midrib is read from the blade itself and its rolls are
     * measured against its rest shape laid untwisted along the same midrib, so a grown or drooped blade reads rolls of zero. Passing the result back to setLeafMidrib() leaves the blade as it is.
     * \param[in] plantID ID of the plant instance.
     * \param[in] shootID ID of the shoot carrying the phytomer.
     * \param[in] node_index Index of the phytomer within the shoot.
     * \param[in] petiole_index Index of the petiole within the phytomer.
     * \param[in] leaf_index Index of the leaf on the petiole.
     * \return Position, normal and roll at each station of the blade's lattice along its length, from base to tip.
     */
    [[nodiscard]] LeafMidrib getLeafMidrib(uint plantID, uint shootID, uint node_index, uint petiole_index, uint leaf_index) const;

    //! Return one leaf's blade from a prescribed midrib to its own shape
    /**
     * The blade takes its rest shape again, at the size it had before its midrib was prescribed, and droops again if it drooped then. Does nothing for a leaf with no prescribed midrib.
     * \param[in] plantID ID of the plant instance.
     * \param[in] shootID ID of the shoot carrying the phytomer.
     * \param[in] node_index Index of the phytomer within the shoot.
     * \param[in] petiole_index Index of the petiole within the phytomer.
     * \param[in] leaf_index Index of the leaf on the petiole.
     */
    void clearLeafMidrib(uint plantID, uint shootID, uint node_index, uint petiole_index, uint leaf_index);

    //! Whether one leaf's blade is laid along a midrib prescribed by setLeafMidrib()
    /**
     * \param[in] plantID ID of the plant instance.
     * \param[in] shootID ID of the shoot carrying the phytomer.
     * \param[in] node_index Index of the phytomer within the shoot.
     * \param[in] petiole_index Index of the petiole within the phytomer.
     * \param[in] leaf_index Index of the leaf on the petiole.
     * \return True if the blade follows a prescribed midrib.
     */
    [[nodiscard]] bool hasLeafMidrib(uint plantID, uint shootID, uint node_index, uint petiole_index, uint leaf_index) const;

    //! Rebuild the leaflets of one petiole of an existing phytomer with each turned by its own offset from the leaf's angles.
    /**
     * See Phytomer::setPetioleLeafletAngleOffsets(). Every leaflet of a leaf is otherwise turned by one set of angles, mirrored across the petiole.
     * \param[in] plantID ID of the plant instance.
     * \param[in] shootID ID of the shoot within the plant.
     * \param[in] node_index Index of the phytomer along the shoot.
     * \param[in] petiole_index Index of the petiole within the phytomer.
     * \param[in] pitch_offsets_degrees Pitch offset of each leaflet in degrees, in the petiole's leaf order; one per leaflet.
     * \param[in] yaw_offsets_degrees Yaw offset of each leaflet in degrees, in the leaf's mirrored sense; one per leaflet.
     * \param[in] roll_offsets_degrees Roll offset of each leaflet in degrees; one per leaflet.
     */
    void setPetioleLeafletAngleOffsets(uint plantID, uint shootID, uint node_index, uint petiole_index, const std::vector<float> &pitch_offsets_degrees, const std::vector<float> &yaw_offsets_degrees,
                                       const std::vector<float> &roll_offsets_degrees);

    /**
     * \brief Sets the base position of a plant with the specified ID, moving the plant if it has already been built.
     *
     * For a plant with no shoots, this only stores the position, and the first shoot added with addBaseStemShoot() is placed there. For a plant that already has shoots,
     * the whole plant is translated by the difference between the new and the old base position: its Context geometry, the positions stored by its shoots, phytomers,
     * leaves, peduncles, flowers and fruit, its per-plant attraction points, and the cached shading state used by branch shedding. Later growth continues from the moved
     * position. Attraction points set for all plants through enableAttractionPoints(const std::vector<helios::vec3>&, float, float, float) are copied into each plant,
     * so the moved plant's copy moves with it while the shared list does not.
     *
     * \note Do not also translate the plant's objects in the Context, or they will be moved twice. This cannot be called from inside a phytomer callback.
     * \param[in] plantID Unique identifier of the plant
     * \param[in] base_position Coordinates representing the new base position of the plant
     */
    void setPlantBasePosition(uint plantID, const helios::vec3 &base_position);

    //! List the joints of a plant's skeleton
    /**
     * The skeleton is the plant's own structure: one joint at the base of the plant, one where each shoot is attached to its parent, one at the base of each internode, one at the base of each petiole
     * and one at the base of each peduncle. Joints are listed from the base of the plant outward, so that every joint comes after the joints that carry it.
     *
     * The base shoot of a plant has no PLANT_JOINT_SHOOT of its own, since PLANT_JOINT_ROOT already turns it. The first internode of every shoot does have a PLANT_JOINT_INTERNODE, whose pivot
     * is the shoot's attachment point. Petioles of zero length (sessile leaves) and peduncles that have not been built have no joint. Pruned shoots contribute nothing.
     *
     * \param[in] plantID ID of the plant instance.
     * \return Identifiers of the plant's joints, from the base outward.
     */
    [[nodiscard]] std::vector<PlantJointID> getPlantJoints(uint plantID) const;

    //! Get the position and orientation of one joint of a plant's skeleton, as the plant currently stands
    /**
     * \param[in] plantID ID of the plant instance.
     * \param[in] joint Identifier of the joint, as listed by getPlantJoints().
     * \return Pivot point, axes and organ length of the joint in world coordinates.
     */
    [[nodiscard]] PlantJointFrame getPlantJointFrame(uint plantID, const PlantJointID &joint) const;

    //! Re-pose a grown plant by turning one of its joints
    /**
     * Turns the organ the joint carries, together with everything growing from it, as one rigid body about the joint, and leaves the rest of the plant where it is. An internode or a petiole can also be
     * made longer or shorter (PlantJointPose::length_scale), which carries everything beyond it along its axis. The geometry in the Context and
     * everything the plant architecture model stores about the plant are moved together, so later calls see the plant as it now stands. No organ is rebuilt: object IDs and primitive UUIDs are unchanged.
     *
     * The pose is an offset from the plant as it grew, not an increment on its current pose. Setting a joint, then other joints, then the first joint again gives the same plant as setting them in any
     * other order, and setting every joint back to zero (or calling resetPlantPose()) restores the grown plant exactly.
     *
     * Petioles and leaves that bend under their own weight (a non-zero flexibility) settle under gravity again in their new orientation, so they are not carried perfectly rigidly; those of a rigid
     * species are. Changes made to a posed plant by other calls that leave its structure alone, such as \ref Phytomer::reshapeLeaf() or \ref setPlantBasePosition(), are kept, and become part of what
     * the pose is measured from: resetting the pose then gives the grown plant with those changes.
     *
     * A pose lasts until the plant next grows or changes structure: advanceTime() first makes the pose permanent, as bakePlantPose() does, and growth continues from the posed plant. A change of
     * structure made by another call while the plant is posed -- an organ added or removed, or a petiole given a different number of nodes with \ref setPetioleNodePositions() -- likewise makes the pose
     * permanent the next time a pose is set, and resetPlantPose() then raises an error, because the plant as it grew can no longer be told apart from the pose.
     *
     * A flower or fruit is rebuilt from its bud whenever the bud changes state, for example from flower to fruit, so the pose of a peduncle lasts only until then.
     *
     * If a pose cannot be applied, for example a turn about an internode that has no length, an error is raised and the plant is left with the poses it had.
     *
     * \note Do not also rotate the plant's objects in the Context, or the plant architecture model will no longer describe them. This cannot be called from inside a phytomer callback.
     * \note A plant that is posed, or whose pose has been made permanent, cannot yet be written by writePlantStructureXML() or copied by duplicatePlantInstance(), which raise an error.
     * \param[in] plantID ID of the plant instance.
     * \param[in] joint Identifier of the joint, as listed by getPlantJoints().
     * \param[in] pose Offset of the joint from the way the plant grew.
     */
    void setPlantJointPose(uint plantID, const PlantJointID &joint, const PlantJointPose &pose);

    //! Re-pose a grown plant by turning several of its joints at once
    /**
     * As setPlantJointPose(), but the plant is moved once for the whole set, which is much faster than setting the joints one at a time. Joints that are not named keep the pose they have.
     * \param[in] plantID ID of the plant instance.
     * \param[in] joints Identifiers of the joints, as listed by getPlantJoints().
     * \param[in] poses Offset of each joint from the way the plant grew, in the order of \p joints.
     */
    void setPlantJointPoses(uint plantID, const std::vector<PlantJointID> &joints, const std::vector<PlantJointPose> &poses);

    //! Set the length of an internode or a petiole of a grown plant in meters
    /**
     * A convenience for setPlantJointPose(): the length is converted to PlantJointPose::length_scale using the length the organ grew to, and the angles already set for the joint are kept.
     * \param[in] plantID ID of the plant instance.
     * \param[in] joint Identifier of a PLANT_JOINT_INTERNODE or PLANT_JOINT_PETIOLE joint, as listed by getPlantJoints().
     * \param[in] length Length of the organ along its centerline, in meters.
     */
    void setPlantJointLength(uint plantID, const PlantJointID &joint, float length);

    //! Push or pull the tip of a petiole of a grown plant to a given position, bending the petiole smoothly along its length
    /**
     * A convenience for setPlantJointPose() on the PLANT_JOINT_PETIOLE joint of the petiole: its bend (PlantJointPose::bend_degrees and PlantJointPose::bend_lateral_degrees) and its length
     * (PlantJointPose::length_scale) are worked out so that the tip of its centerline lands on \p tip_position, while the angles already set for the joint (pitch, yaw and roll) are kept. The petiole
     * leaves its base in the direction it already had and curves evenly toward the target, so there is no kink, and its leaves are carried with it as for any other pose. A petiole that bends under the
     * weight of its leaves is aimed again until its tip settles on the target. The result can be read back with getPlantJointPose() and undone with resetPlantPose().
     *
     * A target is out of reach, and an error is raised with the plant left as it was, when no bend brings the tip to it, when reaching it would bend the petiole by more than 180 degrees from base to
     * tip, or when it would scale the petiole's length by less than 0.25 or more than 4; a target behind the petiole's base is usually the first of these. Turning the joint toward the target first
     * (PlantJointPose::pitch_degrees and yaw_degrees) brings it within reach. A petiole whose centerline has a single segment cannot be bent and also raises an error.
     * \param[in] plantID ID of the plant instance.
     * \param[in] shootID ID of the shoot carrying the phytomer.
     * \param[in] node_index Index of the phytomer within the shoot.
     * \param[in] petiole_index Index of the petiole within the phytomer.
     * \param[in] tip_position Position in world coordinates for the tip of the petiole's centerline.
     */
    void setPetioleTipPosition(uint plantID, uint shootID, uint node_index, uint petiole_index, const helios::vec3 &tip_position);

    //! Re-pose a grown shoot by giving the positions of its nodes
    /**
     * Bends and stretches the shoot at its internode joints so that its nodes are where they are given, as when a cane is laid along a trellis wire or a stem is fitted to a measured centerline. It is a
     * convenience for setPlantJointPoses(): the pose of each of the shoot's PLANT_JOINT_INTERNODE joints is worked out and set, replacing any pose those joints had, and everything on the shoot is
     * carried as for any other pose. The poses of all other joints are kept, and the result can be read back with getPlantJointPose() and undone with resetPlantPose().
     *
     * One position is given for the base of the shoot and one for the tip of each of its internodes, so a shoot with N nodes takes N+1 positions. The shoot stays attached where it is: the first position
     * must be at its base, and a small discrepancy there moves the whole path onto the base. Each internode keeps its own shape, and is turned and scaled so that its tip is at its position.
     *
     * The positions fix where each internode points but not how it is twisted about its own length. An internode is reached by a pitch away from, and a yaw around, the internode before it, which is what
     * decides which way its leaves end up facing.
     *
     * \param[in] plantID ID of the plant instance.
     * \param[in] shootID ID of the shoot.
     * \param[in] node_positions Position in world coordinates of the base of the shoot, followed by the tip of each of its internodes from the base outward.
     */
    void setShootNodePositions(uint plantID, uint shootID, const std::vector<helios::vec3> &node_positions);

    //! Get the pose of one joint of a plant, as an offset from the way the plant grew
    /**
     * \param[in] plantID ID of the plant instance.
     * \param[in] joint Identifier of the joint, as listed by getPlantJoints().
     * \return Offset last set for the joint with setPlantJointPose(). A default-constructed pose (zero angles, a length scale of 1) for a joint that has not been posed, or after the pose has been made permanent.
     */
    [[nodiscard]] PlantJointPose getPlantJointPose(uint plantID, const PlantJointID &joint) const;

    //! Whether a plant currently has a pose, that is, a joint turned from the way the plant grew
    /**
     * \param[in] plantID ID of the plant instance.
     * \return True if any joint has a non-zero pose that has not been made permanent.
     */
    [[nodiscard]] bool isPlantPosed(uint plantID) const;

    //! Put a plant back exactly as it grew, undoing every joint pose
    /**
     * Changes made to the posed plant by other calls (see setPlantJointPose()) are kept. Does nothing for a plant that has no pose. An error is raised if the structure of the plant was changed while
     * it was posed, since the plant as it grew can then no longer be restored.
     * \param[in] plantID ID of the plant instance.
     */
    void resetPlantPose(uint plantID);

    //! Make the pose of a plant a permanent part of it
    /**
     * The plant stays as it is posed, and that becomes the plant "as it grew": every joint then reads zero, and later poses are measured from it. advanceTime() does this for a posed plant before it grows
     * it, so that, for example, a shoot tied to a trellis wire stays there and goes on growing from where it was tied. An internode or petiole whose length was posed keeps that length, and the length it
     * is growing toward is moved by the same factor, so that later growth does not undo it. Does nothing for a plant that has no pose.
     * \param[in] plantID ID of the plant instance.
     */
    void bakePlantPose(uint plantID);

    /**
     * \brief Sets the leaf elevation angle distribution for a specific plant.
     *
     * This method modifies the elevation angles of leaves in the plant such that they follow a Beta distribution.
     * The methodology does not randomly sample an angle for each leaf. The leaves are ordered by their current inclination and each is given the angle at its own position in the target distribution, so that a leaf held more steeply than another is still the steeper of the two afterwards and the total rotation applied is the least that realizes the distribution.
     * Leaves are ordered by leaf area rather than by count, matching the way \ref getPlantLeafInclinationAngleDistribution() reports the distribution.
     * Each leaf is re-aimed about its own base, so it keeps pointing out along its petiole, and the imposed angle is recorded on the phytomer so that it survives a \ref writePlantStructureXML() / \ref readPlantStructureXML() round trip.
     * The more leaves in the plant the more accurate the distribution will be, and the more the plant will look like the original.
     *
     * \param[in] plantID Identifier for the plant.
     * \param[in] Beta_mu_inclination Mean value parameter for the Beta distribution of inclination angles.
     * \param[in] Beta_nu_inclination Shape parameter for the Beta distribution of inclination angles.
     */
    void setPlantLeafElevationAngleDistribution(uint plantID, float Beta_mu_inclination, float Beta_nu_inclination) const;

    /**
     * \brief Sets the leaf elevation angle distribution for a list of plants.
     *
     * This method modifies the elevation angles of leaves in the plants such that they follow a Beta distribution.
     * The methodology does not randomly sample an angle for each leaf. The leaves of all the plants are pooled and ordered by their current inclination, and each is given the angle at its own position in the target distribution, so that a leaf held more steeply than another is still the steeper of the two afterwards and the total rotation applied is the least that realizes the distribution.
     * Leaves are ordered by leaf area rather than by count, matching the way \ref getPlantLeafInclinationAngleDistribution() reports the distribution.
     * Each leaf is re-aimed about its own base, so it keeps pointing out along its petiole, and the imposed angle is recorded on the phytomer so that it survives a \ref writePlantStructureXML() / \ref readPlantStructureXML() round trip.
     * The distribution is realized over the canopy as a whole, so an individual plant within it need not follow the distribution on its own.
     * The more leaves in the canopy the more accurate the distribution will be, and the more the plants will look like the originals.
     *
     * \param[in] plantIDs List of plant IDs for which to set the elevation angle distribution.
     * \param[in] Beta_mu_inclination Mean value parameter for the Beta distribution of inclination angles.
     * \param[in] Beta_nu_inclination Shape parameter for the Beta distribution of inclination angles.
     */
    void setPlantLeafElevationAngleDistribution(const std::vector<uint> &plantIDs, float Beta_mu_inclination, float Beta_nu_inclination) const;

    /**
     * \brief Sets the azimuth angle distribution for plant leaves.
     *
     * This method modifies the azimuth angles of leaves in the plant such that they follow an ellipsoidal distribution.
     * The methodology does not randomly sample an angle for each leaf. The leaves are ordered by their current azimuth and each is given the angle at its own position in the target distribution, which keeps their arrangement around the stem and turns the leaves as little as the distribution allows.
     * Because azimuth is periodic, the ordering starts from the direction the ellipse is rotated to rather than from an arbitrary world axis.
     * Leaves are ordered by leaf area rather than by count, matching the way \ref getPlantLeafAzimuthAngleDistribution() reports the distribution.
     * Each leaf is re-aimed about its own base, so it keeps pointing out along its petiole, and the imposed angle is recorded on the phytomer so that it survives a \ref writePlantStructureXML() / \ref readPlantStructureXML() round trip.
     * The more leaves in the plant the more accurate the distribution will be, and the more the plant will look like the original.
     *
     * \param[in] plantID Identifier of the plant whose leaf azimuth angle distribution is being set.
     * \param[in] eccentricity Eccentricity value for the ellipse defining the azimuth distribution.
     * \param[in] ellipse_rotation_degrees Rotation angle of the ellipse in degrees.
     */
    void setPlantLeafAzimuthAngleDistribution(uint plantID, float eccentricity, float ellipse_rotation_degrees) const;

    /**
     * \brief Sets the azimuth angle distribution of plant leaves.
     *
     * This method modifies the azimuth angles of leaves in the plants such that they follow an ellipsoidal distribution.
     * The methodology does not randomly sample an angle for each leaf. The leaves of all the plants are pooled and ordered by their current azimuth, and each is given the angle at its own position in the target distribution, which keeps their arrangement around the stem and turns the leaves as little as the distribution allows.
     * Because azimuth is periodic, the ordering starts from the direction the ellipse is rotated to rather than from an arbitrary world axis.
     * Leaves are ordered by leaf area rather than by count, matching the way \ref getPlantLeafAzimuthAngleDistribution() reports the distribution.
     * Each leaf is re-aimed about its own base, so it keeps pointing out along its petiole, and the imposed angle is recorded on the phytomer so that it survives a \ref writePlantStructureXML() / \ref readPlantStructureXML() round trip.
     * The distribution is realized over the canopy as a whole, so an individual plant within it need not follow the distribution on its own.
     * The more leaves in the canopy the more accurate the distribution will be, and the more the plants will look like the originals.
     *
     * \param[in] plantIDs List of plant IDs to which the angle distribution will be applied.
     * \param[in] eccentricity Eccentricity value for the ellipse defining the azimuth distribution.
     * \param[in] ellipse_rotation_degrees Rotation angle of the ellipse in degrees.
     */
    void setPlantLeafAzimuthAngleDistribution(const std::vector<uint> &plantIDs, float eccentricity, float ellipse_rotation_degrees) const;

    /**
     * \brief Sets the leaf angle distribution (both elevation and azimuth) for a specific plant
     *
     * This method modifies the elevation angles of leaves in the plant such that they follow a Beta distribution, and the azimuth angles such that they follow an ellipsoidal distribution.
     * The methodology does not randomly sample angles for each leaf. The leaves are ordered by their current angle and each is given the angle at its own position in the target distribution, so that the leaf arrangement of the original plant is preserved and the leaves are moved as little as the distributions allow.
     * The inclination and the azimuth are assigned independently in this way and then imposed together, so each leaf is re-aimed once.
     * Leaves are ordered by leaf area rather than by count, matching the way \ref getPlantLeafInclinationAngleDistribution() and \ref getPlantLeafAzimuthAngleDistribution() report the distributions.
     * Each leaf is re-aimed about its own base, so it keeps pointing out along its petiole, and the imposed angles are recorded on the phytomer so that they survive a \ref writePlantStructureXML() / \ref readPlantStructureXML() round trip.
     * The more leaves in the plant the more accurate the distribution will be, and the more the plant will look like the original.
     *
     * \param[in] plantID The unique identifier of the plant.
     * \param[in] Beta_mu_inclination The mean inclination angle parameter (Beta distribution).
     * \param[in] Beta_nu_inclination The shape parameter nu of the distribution (Beta distribution).
     * \param[in] eccentricity Eccentricity value for the ellipse defining the azimuth distribution.
     * \param[in] ellipse_rotation_degrees Rotation angle of the ellipse in degrees.
     */
    void setPlantLeafAngleDistribution(uint plantID, float Beta_mu_inclination, float Beta_nu_inclination, float eccentricity, float ellipse_rotation_degrees) const;

    /**
     * \brief Sets the leaf angle distribution (both elevation and azimuth) for a list of specified plants
     *
     * This method modifies the elevation angles of leaves in the plants such that they follow a Beta distribution, and the azimuth angles such that they follow an ellipsoidal distribution.
     * The methodology does not randomly sample angles for each leaf. The leaves of all the plants are pooled and ordered by their current angle, and each is given the angle at its own position in the target distribution, so that the leaf arrangement of the original plants is preserved and the leaves are moved as little as the distributions allow.
     * The inclination and the azimuth are assigned independently in this way and then imposed together, so each leaf is re-aimed once.
     * Leaves are ordered by leaf area rather than by count, matching the way \ref getPlantLeafInclinationAngleDistribution() and \ref getPlantLeafAzimuthAngleDistribution() report the distributions.
     * Each leaf is re-aimed about its own base, so it keeps pointing out along its petiole, and the imposed angles are recorded on the phytomer so that they survive a \ref writePlantStructureXML() / \ref readPlantStructureXML() round trip.
     * The distribution is realized over the canopy as a whole, so an individual plant within it need not follow the distribution on its own.
     * The more leaves in the canopy the more accurate the distribution will be, and the more the plants will look like the originals.
     *
     * \param[in] plantIDs Vector of plant IDs to which the leaf angle distribution is to be applied.
     * \param[in] Beta_mu_inclination Mean parameter for the beta distribution of inclination angles.
     * \param[in] Beta_nu_inclination Shape parameter for the beta distribution of inclination angles.
     * \param[in] eccentricity Eccentricity value for the ellipse defining the azimuth distribution.
     * \param[in] ellipse_rotation_degrees Rotation angle of the ellipse in degrees.
     */
    void setPlantLeafAngleDistribution(const std::vector<uint> &plantIDs, float Beta_mu_inclination, float Beta_nu_inclination, float eccentricity, float ellipse_rotation_degrees) const;

    //! Steer a plant's leaf angles toward a prescribed distribution as it grows
    /**
     * Where \ref setPlantLeafAngleDistribution() re-aims every leaf on a finished plant, this follows the
     * distribution through growth: each leaf is given a target as it emerges and turns onto it while it
     * expands, so that a fully grown leaf never moves again. The plant therefore matches the distribution at
     * every stage without the leaves shifting about from one timestep to the next.
     *
     * Targets are not drawn independently per leaf, which would reproduce the distribution while destroying
     * the arrangement the model generated. Each emerging leaf is assigned the angle bin that minimizes
     * <code>angular distance from the angle the model gave it</code> minus <code>lambda</code> times
     * <code>how far that bin is below its share of the plant's leaf area</code>. The bins the population is
     * short of therefore attract leaves, but only leaves whose own orientation is already close to them.
     *
     * \param[in] plantID Identifier for the plant.
     * \param[in] Beta_mu_inclination Mean value parameter for the Beta distribution of inclination angles.
     * \param[in] Beta_nu_inclination Shape parameter for the Beta distribution of inclination angles.
     * \param[in] eccentricity Eccentricity value for the ellipse defining the azimuth distribution.
     * \param[in] ellipse_rotation_degrees Rotation angle of the ellipse in degrees.
     * \param[in] lambda_degrees How strongly to favour filling the distribution over keeping each leaf near the angle the model gave it. Zero leaves the plant unchanged; values of order 180 match the
     * distribution as closely as the growing plant allows.
     * \note Enabling tracking on a plant that is already being tracked replaces the target, so the target may be varied over the plant's life.
     * \note The distribution is realized within each plant, from that plant's own leaves, in every tracking mode (also when several plants are given at once). A plant with few leaves, such as a
     * grass culm with 7-14, can only approximate it: each leaf fills a large share of the distribution, and the mean angle stays between the target and the angles the model gave the leaves.
     * To realize a distribution over a canopy as a whole once the plants have grown, use \ref setPlantLeafAngleDistribution(const std::vector<uint> &, float, float, float, float) const, which pools
     * the leaves of all the plants.
     */
    void enablePlantLeafAngleDistributionTracking(uint plantID, float Beta_mu_inclination, float Beta_nu_inclination, float eccentricity, float ellipse_rotation_degrees, float lambda_degrees);

    //! Steer the leaf angles of several plants toward a prescribed distribution as they grow
    /**
     * Each plant is steered toward the distribution on its own, as by \ref enablePlantLeafAngleDistributionTracking(uint, float, float, float, float, float); the distribution is not pooled over the plants.
     * \param[in] plantIDs List of plant IDs to steer.
     * \param[in] Beta_mu_inclination Mean value parameter for the Beta distribution of inclination angles.
     * \param[in] Beta_nu_inclination Shape parameter for the Beta distribution of inclination angles.
     * \param[in] eccentricity Eccentricity value for the ellipse defining the azimuth distribution.
     * \param[in] ellipse_rotation_degrees Rotation angle of the ellipse in degrees.
     * \param[in] lambda_degrees How strongly to favour filling the distribution over keeping each leaf near the angle the model gave it.
     */
    void enablePlantLeafAngleDistributionTracking(const std::vector<uint> &plantIDs, float Beta_mu_inclination, float Beta_nu_inclination, float eccentricity, float ellipse_rotation_degrees, float lambda_degrees);

    //! Steer a plant's leaf inclination angles toward a Beta distribution as it grows, leaving azimuth to the model
    /**
     * \param[in] plantID Identifier for the plant.
     * \param[in] Beta_mu_inclination Mean value parameter for the Beta distribution of inclination angles.
     * \param[in] Beta_nu_inclination Shape parameter for the Beta distribution of inclination angles.
     * \param[in] lambda_degrees How strongly to favour filling the distribution over keeping each leaf near the angle the model gave it.
     */
    void enablePlantLeafElevationAngleDistributionTracking(uint plantID, float Beta_mu_inclination, float Beta_nu_inclination, float lambda_degrees);

    //! Steer the leaf inclination angles of several plants toward a Beta distribution as they grow, leaving azimuth to the model
    /**
     * Each plant is steered toward the distribution on its own, as by \ref enablePlantLeafElevationAngleDistributionTracking(uint, float, float, float) (see the note there on plants with few leaves).
     * \param[in] plantIDs List of plant IDs to steer.
     * \param[in] Beta_mu_inclination Mean value parameter for the Beta distribution of inclination angles.
     * \param[in] Beta_nu_inclination Shape parameter for the Beta distribution of inclination angles.
     * \param[in] lambda_degrees How strongly to favour filling the distribution over keeping each leaf near the angle the model gave it.
     */
    void enablePlantLeafElevationAngleDistributionTracking(const std::vector<uint> &plantIDs, float Beta_mu_inclination, float Beta_nu_inclination, float lambda_degrees);

    //! Steer a plant's leaf azimuth angles toward an ellipsoidal distribution as it grows, leaving inclination to the model
    /**
     * \param[in] plantID Identifier for the plant.
     * \param[in] eccentricity Eccentricity value for the ellipse defining the azimuth distribution.
     * \param[in] ellipse_rotation_degrees Rotation angle of the ellipse in degrees.
     * \param[in] lambda_degrees How strongly to favour filling the distribution over keeping each leaf near the angle the model gave it.
     */
    void enablePlantLeafAzimuthAngleDistributionTracking(uint plantID, float eccentricity, float ellipse_rotation_degrees, float lambda_degrees);

    //! Stop steering a plant's leaf angles toward a prescribed distribution
    /**
     * Leaves already steered keep the orientation they have reached; leaves emerging afterward are left where
     * the procedural model puts them.
     * \param[in] plantID Identifier for the plant.
     */
    void disablePlantLeafAngleDistributionTracking(uint plantID);

    //! Whether a plant's leaf angles are being steered toward a prescribed distribution
    /**
     * \param[in] plantID Identifier for the plant.
     * \return True if PlantArchitecture::enablePlantLeafAngleDistributionTracking() is in effect for this plant.
     */
    [[nodiscard]] bool isPlantLeafAngleDistributionTrackingEnabled(uint plantID) const;

    /**
     * \brief Sets the maximum age of a plant, beyond which it stops growing.
     *
     * Once a plant's age reaches this value, PlantArchitecture::advanceTime() stops advancing it and its geometry becomes static.
     * The default is 999 days. Every plant model in the library sets its own value as part of its builder (an apple tree, for example, uses 1460 days),
     * but a plant assembled manually with PlantArchitecture::addPlantInstance() keeps the default, and so silently stops growing after 999 days.
     *
     * Setting a maximum age below the plant's current age is permitted, and freezes the plant at its current form.
     *
     * \param[in] plantID ID of the plant instance.
     * \param[in] max_age Maximum age of the plant in days. Must be greater than or equal to zero.
     */
    void setPlantMaxAge(uint plantID, float max_age);

    /**
     * \brief Harvests a plant by removing all leaves and fruit.
     *
     * \param[in] plantID The unique identifier of the plant to be harvested.
     */
    void harvestPlant(uint plantID);

    /**
     * \brief Removes all leaves from a specified shoot in a specified plant.
     *
     * \param[in] plantID ID of the plant from which leaves are to be removed.
     * \param[in] shootID ID of the shoot within the plant whose leaves are to be removed.
     */
    void removeShootLeaves(uint plantID, uint shootID);

    /**
     * \brief Removes all vegetative buds from a specified shoot in a specified plant.
     *
     * \param[in] plantID ID of the plant from which buds are to be removed.
     * \param[in] shootID ID of the shoot within the plant whose buds are to be removed.
     */
    void removeShootVegetativeBuds(uint plantID, uint shootID);

    /**
     * \brief Removes all floral buds from a specified shoot in a specified plant.
     *
     * \param[in] plantID ID of the plant from which floral buds are to be removed.
     * \param[in] shootID ID of the shoot within the plant whose floral buds are to be removed.
     */
    void removeShootFloralBuds(uint plantID, uint shootID);

    /**
     * \brief Removes all leaves from the plant with the specified ID.
     *
     * \param[in] plantID The unique identifier of the plant whose leaves are to be removed.
     */
    void removePlantLeaves(uint plantID);

    /**
     * \brief Makes the specified plant enter a dormant state.
     *
     * Every shoot is made dormant as at the end of a season: every vegetative and floral bud that is not already dormant is set dead, leaves are removed unless the plant is evergreen, and the time since
     * dormancy is reset to zero. Buds that are active but have not yet broken are killed too. These are the buds of any shoot whose type has ShootParameters::growth_requires_dormancy false, which are
     * sampled as their phytomers are created, so a manually built shoot of such a type keeps no buds through this call unless they are set dormant first with \ref Phytomer::setVegetativeBudState(BudState).
     *
     * \param[in] plantID ID of the plant to be made dormant.
     */
    void makePlantDormant(uint plantID);

    /**
     * \brief Breaks the dormancy of all shoots in the specified plant.
     *
     * \param[in] plantID Identifier of the plant whose shoots' dormancy should be broken.
     */
    void breakPlantDormancy(uint plantID);

    //! Prunes a branch from a specific plant at a designated node index
    /**
     * The node at node_index and every node above it are removed, along with their leaves,
     * inflorescences and internode geometry, and the same is done recursively to every shoot
     * descending from those nodes. Pruning at node_index 0 therefore removes the shoot entirely.
     *
     * A shoot removed entirely keeps its slot in the plant's shoot tree so that shoot IDs are not
     * renumbered, but it becomes inert: it has zero nodes, contributes no leaf area or volume, and
     * is unlinked from its parent so that traversals of the plant structure no longer reach it. Use
     * isShootPruned() to detect such shoots when iterating over getAllShootIDs().
     *
     * Pruning is idempotent -- calling this on a shoot that has already been pruned away does
     * nothing. This is what allows a whole branch system to be pruned by looping over shoot IDs,
     * since removing a shoot also empties its descendants.
     *
     * Called from inside a phytomer callback (PhytomerParameters::phytomer_creation_function or
     * PhytomerParameters::phytomer_callback_function), the arguments are checked immediately but the
     * cut is deferred: it is applied at the end of the current time step of advanceTime(), or before
     * the build call that ran the callback returns. Until then the shoot still looks uncut, including
     * to the callbacks of the remaining phytomers in that time step. Deferred cuts are applied in the
     * order they were requested; a cut whose shoot or node has already been removed by an earlier one
     * is already satisfied and does nothing. See \ref PlantArchCallbackEdits.
     *
     * With \a keep_trimmed_material set, the removed Context objects are not deleted but left in place, detached from the plant, and returned, for example to hand them to a physics simulation. They form one
     * connected piece: the removed section of the shoot, whose internode tube is split off from the part that stays on the plant (see helios::Context::splitTubeObject()), with everything it bears and every
     * shoot descending from it. The objects keep their object and primitive data (e.g., "plantID", "object_label"), but the plant no longer refers to them, so later growth, deletePlantInstance() and other
     * plant operations do not touch them; deleting them is up to the caller. Keeping the material cannot be requested from inside a phytomer callback, where the cut is deferred.
     *
     * \param[in] plantID Unique identifier of the plant to prune.
     * \param[in] shootID Identifier of the shoot to prune from within the plant.
     * \param[in] node_index Index of the node on the shoot where the branch will be pruned.
     * \param[in] keep_trimmed_material [optional] If true, the removed material is left in the Context and returned; if false (default), it is deleted.
     * \return Object IDs of the removed material. Empty when \a keep_trimmed_material is false, or when the shoot had already been pruned away.
     * \note Throws an error if node_index is out of range for a shoot that still has nodes, or if \a keep_trimmed_material is true inside a phytomer callback.
     * \sa isShootPruned(), pruneBranchAtPrimitive(), prunePlantsByPlane()
     */
    std::vector<uint> pruneBranch(uint plantID, uint shootID, uint node_index, bool keep_trimmed_material = false);

    //! Prune the branch at the node of an internode primitive, for example the primitive hit by a ray from a pruning tool
    /**
     * Equivalent to pruneBranch() at the plant, shoot and node given by getPrimitiveOrganLocation() for \a UUID.
     * \param[in] UUID Unique identifier of a primitive of a shoot's internode tube.
     * \param[in] keep_trimmed_material [optional] If true, the removed material is left in the Context and returned; if false (default), it is deleted. See pruneBranch().
     * \return Object IDs of the removed material. Empty when \a keep_trimmed_material is false.
     * \note Throws an error if the primitive is not part of a plant's internode tube (a leaf, petiole or flower primitive, or one that is not part of any plant).
     */
    std::vector<uint> pruneBranchAtPrimitive(uint UUID, bool keep_trimmed_material = false);

    //! Find the plant organ that a Context primitive belongs to
    /**
     * Maps a primitive, for example one hit by a ray, a LiDAR beam or a collision query, to its plant, shoot, phytomer (node) and organ. For an internode primitive, node_index is the phytomer whose
     * internode the primitive belongs to. Lookups use an index that is built on the first lookup and rebuilt automatically, on the next lookup, after any operation that changes the plants' organ objects
     * (growth, pruning, leaf removal, harvesting, XML reading, deleting plants and so on), so a lookup costs O(1) on average.
     * \param[in] UUID Unique identifier of the primitive.
     * \return Location of the primitive's organ.
     * \note Throws an error if the primitive is not part of an organ of a plant of this PlantArchitecture instance, including material detached by pruning. Use isPlantPrimitive() to filter primitives first.
     * \note Not thread-safe: the first lookup after a change rebuilds the index.
     */
    [[nodiscard]] PlantOrganLocation getPrimitiveOrganLocation(uint UUID) const;

    //! Find the plant organ that each of several Context primitives belongs to
    /**
     * See \ref getPrimitiveOrganLocation(uint) const.
     * \param[in] UUIDs Unique identifiers of the primitives.
     * \return Location of each primitive's organ, in the order of \a UUIDs.
     * \note Throws an error if any of the primitives is not part of an organ of a plant. Use isPlantPrimitive() to filter primitives first.
     */
    [[nodiscard]] std::vector<PlantOrganLocation> getPrimitiveOrganLocations(const std::vector<uint> &UUIDs) const;

    //! Find the plant organ that a Context object is
    /**
     * See \ref getPrimitiveOrganLocation(uint) const. A shoot's internode tube is a single object spanning all of the shoot's phytomers, so for it node_index is PlantOrganLocation::not_applicable.
     * \param[in] objID Identifier of the object.
     * \return Location of the organ.
     * \note Throws an error if the object is not an organ of a plant.
     */
    [[nodiscard]] PlantOrganLocation getObjectOrganLocation(uint objID) const;

    //! Check whether a Context primitive is part of an organ of a plant
    /**
     * \param[in] UUID Unique identifier of the primitive.
     * \return True if getPrimitiveOrganLocation() would find the primitive's organ; false if the primitive does not exist or is not part of a plant of this PlantArchitecture instance.
     */
    [[nodiscard]] bool isPlantPrimitive(uint UUID) const;

    //! Get the centerline segments of a plant's axes, with their radii, for example for proximity or collision tests against the plant's skeleton
    /**
     * Each internode, petiole or peduncle centerline is returned as the straight segments between its successive nodes, in order from base to tip, with the radius of the organ at both ends and the organ
     * the segment belongs to. Internode segments come from Phytomer::getInternodeNodePositions(). Shoots that have been pruned away contribute nothing.
     * \param[in] plantID Identifier of the plant.
     * \param[in] organ_types [optional] Organs to include: any of PlantOrganType::INTERNODE (default), PlantOrganType::PETIOLE and PlantOrganType::PEDUNCLE.
     * \return Segments of the requested organs.
     * \note Throws an error if the plant does not exist, or if \a organ_types includes an organ without a centerline (leaves, flowers or fruit).
     */
    [[nodiscard]] std::vector<PlantSkeletonSegment> getPlantSkeletonSegments(uint plantID, const std::vector<PlantOrganType> &organ_types = {PlantOrganType::INTERNODE}) const;

    //! Cut plants across a plane, as in a hedging pass
    /**
     * Everything on the side of the plane that \a plane_normal points toward is trimmed. Only the axes of the plant are cut:
     * - A shoot whose internode crosses the plane is pruned at that phytomer, exactly as by pruneBranch(), so the node, its leaves, inflorescences and child shoots, and every phytomer above it are
     *   removed. The cut snaps to nodes: the remaining stub ends at the last node short of the plane, up to one internode length from it.
     * - A phytomer whose petiole crosses the plane loses its leaves and petioles, as by Phytomer::removeLeaf(). This removes every petiole on that phytomer, including any that do not cross the plane.
     * - A floral bud whose peduncle crosses the plane loses its peduncle and its flowers or fruit, and the bud is set to BUD_DEAD.
     *
     * Leaf blades and fruit that reach across the plane from an axis that does not cross it are not trimmed, and neither are flowers or fruit whose peduncle geometry is not built.
     *
     * With \a keep_trimmed_material set, the trimmed Context objects are not deleted but left in place, detached from the plant, for example to hand them to a physics simulation. They are returned
     * grouped into fragments, each of which is one connected piece: a pruned shoot section with everything it bears, the leaves and petioles of one petiole, or one peduncle with its flowers or fruit.
     * The trimmed objects keep their object and primitive data (e.g., "plantID", "object_label"), but the plant no longer refers to them, so later growth, deletePlantInstance() and other plant
     * operations do not touch them; deleting them is up to the caller.
     *
     * \param[in] plantIDs Identifiers of the plants to cut.
     * \param[in] plane_point Any point on the cutting plane.
     * \param[in] plane_normal Normal of the cutting plane, pointing toward the side to be trimmed. It need not be unit length.
     * \param[in] keep_trimmed_material [optional] If true, the trimmed material is left in the Context and returned; if false (default), it is deleted.
     * \return Object IDs of the trimmed material, one inner vector per fragment. Empty when \a keep_trimmed_material is false.
     * \note Cannot be called from inside a phytomer callback.
     * \sa pruneBranch()
     */
    std::vector<std::vector<uint>> prunePlantsByPlane(const std::vector<uint> &plantIDs, const helios::vec3 &plane_point, const helios::vec3 &plane_normal, bool keep_trimmed_material = false);

    //! Cut a plant across a plane, as in a hedging pass
    /**
     * See \ref prunePlantsByPlane(const std::vector<uint>&, const helios::vec3&, const helios::vec3&, bool) for details.
     * \param[in] plantID Identifier of the plant to cut.
     * \param[in] plane_point Any point on the cutting plane.
     * \param[in] plane_normal Normal of the cutting plane, pointing toward the side to be trimmed. It need not be unit length.
     * \param[in] keep_trimmed_material [optional] If true, the trimmed material is left in the Context and returned; if false (default), it is deleted.
     * \return Object IDs of the trimmed material, one inner vector per fragment. Empty when \a keep_trimmed_material is false.
     */
    std::vector<std::vector<uint>> prunePlantsByPlane(uint plantID, const helios::vec3 &plane_point, const helios::vec3 &plane_normal, bool keep_trimmed_material = false);

    // -- methods for querying information about the plant -- //

    /**
     * \brief Retrieves the name of the plant associated with a given plant ID.
     *
     * \param[in] plantID The unique identifier of the plant.
     * \return The name of the plant corresponding to the provided plant ID.
     */
    [[nodiscard]] std::string getPlantName(uint plantID) const;

    //! Get the build parameters a plant was built with
    /**
     * These are the parameters given to \ref buildPlantInstanceFromLibrary() (see \ref PlantArchBuildParams). A parameter given as a \ref BuildParameterDistribution appears with the value sampled for this
     * plant, except an organ parameter, which is sampled for each flower and fruit prototype rather than for the plant and appears only in \ref getPlantOrganBuildParameters(). A plant keeps its parameters
     * for its life, and they are restored with it from XML. Parameters that were not given are absent, and take the model's default where they are read.
     * \param[in] plantID Unique identifier of the plant.
     * \return Map of build parameter names to values, empty for a plant not built from the library or built without parameters.
     */
    [[nodiscard]] const std::map<std::string, float> &getPlantBuildParameters(uint plantID) const;

    //! Get the build parameters that shape a plant's flowers and fruit, as distributions
    /**
     * These are the organ parameters given to \ref buildPlantInstanceFromLibrary(), such as "berry_diameter" (see \ref PlantArchBuildParams), with a fixed value given as a constant distribution. Each
     * flower and fruit prototype of the plant is built from its own sample of them, and plants with the same organ parameters share their prototypes.
     * \param[in] plantID Unique identifier of the plant.
     * \return Map of organ parameter names to distributions, empty if none were given.
     */
    [[nodiscard]] const std::map<std::string, BuildParameterDistribution> &getPlantOrganBuildParameters(uint plantID) const;

    /**
     * \brief Retrieves the age of a specific plant.
     *
     * \param[in] plantID Unique identifier of the plant.
     * \return The age of the plant in days associated with the given plantID.
     */
    [[nodiscard]] float getPlantAge(uint plantID) const;

    /**
     * \brief Retrieves the maximum age of a specific plant, beyond which it stops growing.
     *
     * \param[in] plantID Unique identifier of the plant.
     * \return The maximum age of the plant in days. See PlantArchitecture::setPlantMaxAge().
     */
    [[nodiscard]] float getPlantMaxAge(uint plantID) const;

    /**
     * \brief Retrieves the number of nodes in a specific shoot of a specific plant.
     *
     * \param[in] plantID The unique identifier of the plant.
     * \param[in] shootID The index of the shoot within the plant.
     * \return The current number of nodes in the specified shoot.
     */
    [[nodiscard]] uint getShootNodeCount(uint plantID, uint shootID) const;

    //! Reports whether a shoot's existing geometry was prescribed by the caller rather than generated
    /**
     * True for a shoot built by \ref PlantArchitecture::addShootFromNodePositions(), whose internode path
     * follows measured node positions. Such a shoot's existing phytomers are exempt from the re-scaling and
     * re-curving performed by \ref PlantArchitecture::advanceTime(), so a caller reading geometry back can
     * tell which parts of a plant are measurement and which are model output.
     *
     * \param[in] plantID The unique identifier of the plant.
     * \param[in] shootID The index of the shoot within the plant.
     * \return True if the shoot was built from prescribed node positions.
     */
    [[nodiscard]] bool isShootGeometryPrescribed(uint plantID, uint shootID) const;

    //! Reports whether a shoot of a plant has been pruned away entirely
    /**
     * PlantArchitecture::pruneBranch() called with a node index of 0 removes all of a shoot's
     * phytomers and geometry, but keeps the shoot in the plant's shoot tree so that shoot IDs stay
     * stable across a prune. The shoot ID therefore remains valid and is still returned by
     * getAllShootIDs(), but the shoot itself is inert: it has zero nodes, contributes no leaf area
     * or volume, is no longer listed as a child of its parent, and cannot be queried for geometry
     * (getShootTaper() throws for it). Use this to skip such shoots when walking getAllShootIDs().
     *
     * \param[in] plantID The unique identifier of the plant.
     * \param[in] shootID The index of the shoot within the plant (see getAllShootIDs()).
     * \return True if the shoot was pruned away and no longer forms part of the plant.
     */
    [[nodiscard]] bool isShootPruned(uint plantID, uint shootID) const;

    /**
     * \brief Whether a shoot is structural wood -- the trunk or a scaffold -- rather than fine distal growth.
     *
     * Determined by shoot type: every type other than "proleptic", "sylleptic" and "spur" is treated as structural. Branching
     * order does not work here, because the order-0 axis runs the whole length of the trunk, so every lateral borne on it
     * is order 1 however small.
     *
     * \note A species whose fine shoots use other type names has all of its wood treated as structural, and so is never
     * shed or reprioritised.
     *
     * \param[in] plantID ID of the plant.
     * \param[in] shootID ID of the shoot within the plant.
     * \return True if the shoot is structural wood and must not be shed or reprioritised.
     */
    [[nodiscard]] bool isStructuralShoot(uint plantID, uint shootID) const;

    /**
     * \brief Rank sibling shoots by light exposure and promote the best-lit few to a higher growth priority.
     *
     * Applies apical dominance among siblings: within each parent shoot, the laterals are ranked by the light
     * their tips receive and the leading fraction is promoted while the shaded remainder is suppressed,
     * changing both their node ceiling and their internode length. Promotion is cumulative across seasons, so
     * a branch that leads repeatedly compounds into a structural limb while its siblings remain twigs.
     * Structural shoots (see isStructuralShoot()) and spurs are excluded.
     *
     * \param[in] plantID ID of the plant to update.
     */
    void updateGrowthPriority(uint plantID);

    /**
     * \brief Shed the branches that did not pay for their own upkeep over the past year (carbohydrate model).
     *
     * Self-pruning by carbon starvation, after the branch autonomy of Sprugel et al. (1991): a branch is judged by its own
     * carbon balance, not by the tree's reserves. Each shoot records the net photosynthesis of its own leaves, the
     * maintenance respiration of its own wood, and the construction of its own wood, leaves, flowers and fruit (with
     * growth respiration) since the last evaluation. A branch -- a shoot together with every descendant that is not itself
     * shed -- is shed when its leaves' net photosynthesis is less than CarbonWoodParameters::branch_shedding_carbon_ratio
     * times the carbon it spent, i.e. when it has had to import carbon from the rest of the tree. Shoots are judged
     * from the tips inward, so a shaded twig can be shed while the limb that carries it survives on its other branches,
     * and a limb survives only if what remains of it still pays its way. Structural wood (see isStructuralShoot()) and
     * shoots that have not yet been through a full season are never shed. The records are reset afterwards, and the
     * carbohydrate held in shed branches is reported in PlantCarbonBudget::shed_branch_carbohydrate.
     *
     * With the carbohydrate model enabled this replaces the shade-based shedding at each dormancy boundary, so it only
     * needs to be called directly to evaluate a plant at some other time.
     *
     * \param[in] plantID ID of the plant.
     */
    void shedCarbonStarvedBranches(uint plantID);

    /**
     * \brief Retrieves the IDs of all shoots belonging to a specific plant.
     *
     * Shoot IDs are sequential indices (0-based) into the plant's shoot tree, in the order the shoots were created.
     * The trunk/base-stem shoot is always shoot ID 0. The returned IDs can be passed to getPlantShoot(), getShootNodeCount(), getShootTaper(), etc.
     *
     * \param[in] plantID The unique identifier of the plant.
     * \return Vector of shoot IDs for all shoots in the plant.
     */
    [[nodiscard]] std::vector<uint> getAllShootIDs(uint plantID) const;

    //! Retrieves the parent shoot of a shoot
    /**
     * \param[in] plantID The unique identifier of the plant.
     * \param[in] shootID The index of the shoot within the plant.
     * \return ID of the shoot this shoot grew from, or -1 if it is the base stem shoot.
     * \note A pruned shoot still reports the parent it grew from, even though it is no longer listed among that parent's children. Its parent is always a live shoot, since pruning a shoot also prunes everything descending from it.
     */
    [[nodiscard]] int getParentShootID(uint plantID, uint shootID) const;

    //! Retrieves the branching rank of a shoot
    /**
     * Rank is the botanical branching order: the base stem shoot is rank 0, a branch off it is rank 1,
     * a branch off that is rank 2, and so on. A shoot created by appendShoot() continues its parent's
     * axis rather than branching from it, so it keeps the parent's rank. Rank is therefore not the same
     * as the number of steps to the base shoot -- see getShootDepth() for that.
     *
     * \param[in] plantID The unique identifier of the plant.
     * \param[in] shootID The index of the shoot within the plant.
     * \return Branching rank of the shoot.
     */
    [[nodiscard]] uint getShootRank(uint plantID, uint shootID) const;

    //! Retrieves the number of shoots between a shoot and the base stem shoot
    /**
     * The base stem shoot has depth 0, its children depth 1, and so on. Unlike getShootRank() this
     * counts every step in the shoot tree, including axis continuations created by appendShoot().
     *
     * \param[in] plantID The unique identifier of the plant.
     * \param[in] shootID The index of the shoot within the plant.
     * \return Number of steps from this shoot to the base stem shoot.
     */
    [[nodiscard]] uint getShootDepth(uint plantID, uint shootID) const;

    //! Retrieves the chain of shoots connecting a shoot to the base stem shoot
    /**
     * \param[in] plantID The unique identifier of the plant.
     * \param[in] shootID The index of the shoot within the plant.
     * \return Shoot IDs ordered from the given shoot to the base stem shoot, including both. For the base stem shoot this is a single element.
     */
    [[nodiscard]] std::vector<uint> getPathToRoot(uint plantID, uint shootID) const;

    //! Retrieves the shoots that grew directly out of a given shoot
    /**
     * Ordered by the node they attach to. This includes shoots created by appendShoot(), which continue
     * the parent's axis rather than branching from it; compare their getShootRank() with the parent's to
     * tell the two apart. Shoots that have been pruned away are not included.
     *
     * \param[in] plantID The unique identifier of the plant.
     * \param[in] shootID The index of the shoot within the plant.
     * \return IDs of the direct children of the shoot, empty if it has none.
     */
    [[nodiscard]] std::vector<uint> getChildShootIDs(uint plantID, uint shootID) const;

    //! Retrieves every shoot descending from a given shoot
    /**
     * Collected depth-first, so a shoot is always listed before its own descendants. The shoot itself is
     * not included. Shoots that have been pruned away are not included.
     *
     * \param[in] plantID The unique identifier of the plant.
     * \param[in] shootID The index of the shoot within the plant.
     * \return IDs of all descendants of the shoot, empty if it has none.
     */
    [[nodiscard]] std::vector<uint> getAllDescendantShootIDs(uint plantID, uint shootID) const;

    //! Retrieves the shoots of a plant grouped by branching rank
    /**
     * The index into the outer vector is the rank, so element 0 holds the base stem shoot and element 2
     * holds every second-order branch. A rank with no live shoots is an empty entry rather than being
     * skipped, so the index always equals the rank. Shoots that have been pruned away are not included.
     *
     * \param[in] plantID The unique identifier of the plant.
     * \return Shoot IDs grouped by rank, indexed by rank.
     */
    [[nodiscard]] std::vector<std::vector<uint>> getShootIDsByRank(uint plantID) const;

    //! Retrieves the parent-to-children structure of a plant
    /**
     * Only shoots that actually have children appear as keys. Shoots that have been pruned away appear
     * neither as keys nor among the children.
     *
     * \param[in] plantID The unique identifier of the plant.
     * \return Map of shoot ID to the IDs of its direct children.
     */
    [[nodiscard]] std::map<uint, std::vector<uint>> getShootHierarchyMap(uint plantID) const;

    //! Retrieves the shoots of a plant that have no children
    /**
     * These are the tips of the shoot tree. Note that this is a topological test rather than a botanical
     * one: a shoot whose axis is continued by appendShoot() has that continuation as a child and so is
     * not terminal. Shoots that have been pruned away are not included -- an empty pruned shell has no
     * children but is not a tip of the plant.
     *
     * \param[in] plantID The unique identifier of the plant.
     * \return IDs of all shoots with no children.
     */
    [[nodiscard]] std::vector<uint> getTerminalShootIDs(uint plantID) const;

    /**
     * \brief Retrieves a read-only handle to a specific shoot of a specific plant.
     *
     * Provides direct access to the Shoot structure, including its topology (rank, parent_shoot_ID, parent_node_index, childIDs) and woody internode geometry
     * (shoot_internode_vertices, shoot_internode_radii). Intended for read-only inspection of the plant's ground-truth structure; the returned shoot should not be mutated.
     *
     * \param[in] plantID The unique identifier of the plant.
     * \param[in] shootID The index of the shoot within the plant (see getAllShootIDs()).
     * \return Const reference to the shared pointer holding the requested shoot.
     */
    [[nodiscard]] const std::shared_ptr<Shoot> &getPlantShoot(uint plantID, uint shootID) const;

    /**
     * \brief Retrieves the taper of a shoot based on its radius measurements.
     *
     * \param[in] plantID The unique identifier of the plant.
     * \param[in] shootID The unique identifier of the shoot within the specified plant.
     * \return The taper of the shoot as a float value, constrained between 0 and 1.
     * \note Throws an error if the shoot has no internodes because it was pruned away. Check isShootPruned() first when iterating over getAllShootIDs().
     */
    [[nodiscard]] float getShootTaper(uint plantID, uint shootID) const;

    /**
     * \brief Retrieves the base position of the specified plant.
     *
     * For a plant with no shoots yet, this is the position at which its first shoot will be placed.
     *
     * \param[in] plantID Identifier of the plant whose base position is being queried.
     * \return Base position of the plant as a helios::vec3 object.
     */
    [[nodiscard]] helios::vec3 getPlantBasePosition(uint plantID) const;

    /**
     * \brief Retrieves the base positions of multiple plants.
     *
     * \param[in] plantIDs A vector containing the IDs of the plants for which the base positions are required.
     * \return A vector of vec3 objects representing the base positions of the specified plants.
     */
    [[nodiscard]] std::vector<helios::vec3> getPlantBasePosition(const std::vector<uint> &plantIDs) const;

    //! Sum the one-sided leaf area of all leaves in the plant
    /**
     * Blade area only: a leaflet's petiolule (see \ref LeafPrototype::build_petiolule) belongs to the leaf object but is not counted.
     * \param[in] plantID ID of the plant instance.
     * \return Total one-sided leaf area of all leaves in the plant.
     */
    [[nodiscard]] float sumPlantLeafArea(uint plantID) const;

    //! Get the one-sided blade area of a leaf object
    /**
     * A leaf object can carry a petiolule, the stalk joining a leaflet to its rachis (see \ref LeafPrototype::build_petiolule), whose primitives are
     * labelled "petiolule" rather than "leaf". This reports the area of the object without them, which is what every leaf-area quantity of the plugin means.
     * \param[in] leaf_objID Object ID of the leaf.
     * \return One-sided blade area of the leaf (m^2).
     */
    [[nodiscard]] float getLeafBladeArea(uint leaf_objID) const;

    //! Get the local light exposure at a point from a plant's shadow-propagation grid
    /** Exposure runs from 0 (fully shaded) to 1 (unshaded), and is the quantity that drives shade-based branch shedding at the season
        boundary: branches deep in the crown interior see a low exposure while branches on the crown surface
        do not. Returns 1 when the plant has no canopy yet, or when the shadow grid is disabled because the carbon model is
        active. Intended for inspection and testing.
        \param[in] plantID Identifier of the plant.
        \param[in] position Query position in world coordinates.
        \return Local light exposure in [0,1].
    */
    [[nodiscard]] float getShadowLightExposureAtPoint(uint plantID, const helios::vec3 &position) const;

    //! Get the local light exposure at a point under the canopy of the plant's most recent seasonal peak
    /** Like \ref getShadowLightExposureAtPoint(), but read from the shadow grid captured when the plant's leaf area last reached its
        seasonal maximum, rather than from the current canopy. Through dormancy, when a deciduous plant has no leaves and the current
        exposure is 1 everywhere, this still gives the light each point stood in over the past summer. Returns 1 before the plant's
        first season in leaf, or when the carbon model is active.
        \param[in] plantID Identifier of the plant.
        \param[in] position Query position in world coordinates.
        \return Local light exposure in [0,1].
    */
    [[nodiscard]] float getSeasonPeakLightExposureAtPoint(uint plantID, const helios::vec3 &position) const;

    //! Calculate the height of the last internode on the base stem/shoot
    /**
     * \param[in] plantID ID of the plant instance.
     * \return Height of the last internode on the base stem/shoot.
     */
    [[nodiscard]] float getPlantStemHeight(uint plantID) const;

    //! Calculate the height of the highest element in the plant
    /**
     * \param[in] plantID ID of the plant instance.
     * \return Height of the highest element in the plant.
     */
    [[nodiscard]] float getPlantHeight(uint plantID) const;

    //! Calculate the leaf inclination angle distribution of all leaves in the plant.
    /**
     * \param[in] plantID ID of the plant instance.
     * \param[in] Nbins Number of bins for the histogram.
     * \param[in] normalize [optional] Normalize the histogram (default = true).
     * \return Histogram of leaf inclination angles. Bins are evenly spaced between 0 and 90 degrees.
     */
    [[nodiscard]] std::vector<float> getPlantLeafInclinationAngleDistribution(uint plantID, uint Nbins, bool normalize = true) const;

    //! Calculate the leaf inclination angle distribution of all leaves in multiple plants.
    /**
     * \param[in] plantIDs Vector of IDs of the plant instances.
     * \param[in] Nbins Number of bins for the histogram.
     * \param[in] normalize [optional] Normalize the histogram (default = true).
     * \return Histogram of leaf inclination angles. Bins are evenly spaced between 0 and 90 degrees.
     */
    [[nodiscard]] std::vector<float> getPlantLeafInclinationAngleDistribution(const std::vector<uint> &plantIDs, uint Nbins, bool normalize = true) const;

    //! Calculate the leaf azimuth angle distribution of all leaves in the plant.
    /**
     * \param[in] plantID ID of the plant instance.
     * \param[in] Nbins Number of bins for the histogram.
     * \param[in] normalize [optional] Normalize the histogram (default = true).
     * \return Histogram of leaf azimuth angles. Bins are evenly spaced between 0 and 360 degrees.
     */
    [[nodiscard]] std::vector<float> getPlantLeafAzimuthAngleDistribution(uint plantID, uint Nbins, bool normalize = true) const;

    //! Calculate the leaf azimuth angle distribution of all leaves in multiple plants.
    /**
     * \param[in] plantIDs Vector of ID of the plant instances.
     * \param[in] Nbins Number of bins for the histogram.
     * \param[in] normalize [optional] Normalize the histogram (default = true).
     * \return Histogram of leaf azimuth angles. Bins are evenly spaced between 0 and 360 degrees.
     */
    [[nodiscard]] std::vector<float> getPlantLeafAzimuthAngleDistribution(const std::vector<uint> &plantIDs, uint Nbins, bool normalize = true) const;

    //! Get the total number of leaves on the plant
    /**
     * \param[in] plantID ID of the plant instance.
     * \return Total number of leaves on the plant.
     */
    [[nodiscard]] uint getPlantLeafCount(uint plantID) const;

    //! Get the base positions of all leaves on the plant
    /**
     * \param[in] plantID ID of the plant instance.
     * \return Vector of base positions of all leaves on the plant.
     */
    [[nodiscard]] std::vector<helios::vec3> getPlantLeafBases(uint plantID) const;

    //! Get the base positions of all leaves for a list of plants
    /**
     * \param[in] plantIDs List of IDs of the plant instances.
     * \return Vector of base positions of all leaves on the plants.
     */
    [[nodiscard]] std::vector<helios::vec3> getPlantLeafBases(const std::vector<uint> &plantIDs) const;

    //! Get the built one-sided surface area of every leaf on the plant
    /**
     * Measured from the geometry that was actually built, rather than reported from the shoot type's
     * ShootParameters. The two answer different questions: the shoot type gives the distribution a
     * parameter was drawn from, while this gives what the plant ended up with. Comparing them is the
     * point -- a plant whose leaf parameters carry a wide spread can still deliver leaves of a single
     * size, and nothing in the parameters themselves would reveal that.
     *
     * Unlike \ref Phytomer::getLeafArea(), which reports the area a growing leaf is expanding toward,
     * this reports present area, so a leaf part-way through its growth is counted at its current size.
     *
     * Leaves are visited shoot by shoot and then phytomer by phytomer, the same order as
     * \ref getPlantLeafObjectIDs(). Leaves whose geometry does not exist -- removed, senesced, or never
     * built -- are omitted rather than reported as zero, so the result can be shorter than
     * \ref getPlantLeafCount().
     *
     * \param[in] plantID ID of the plant instance.
     * \return One-sided blade area (m^2) of each leaf on the plant, excluding any petiolule (see \ref getLeafBladeArea()).
     */
    [[nodiscard]] std::vector<float> getPlantLeafAreas(uint plantID) const;

    //! Get the built length of every internode on the plant
    /**
     * Measured along the internode's node positions as they were built, so a shoot whose geometry was
     * prescribed by \ref addShootFromNodePositions() reports its measured lengths and a grown shoot
     * reports what growth produced. See \ref getPlantLeafAreas() for why this differs from reading
     * ShootParameters::internode_length_max.
     *
     * Internodes are visited shoot by shoot and then phytomer by phytomer, so the result has one entry
     * per phytomer on the plant.
     *
     * \param[in] plantID ID of the plant instance.
     * \return Length (m) of each internode on the plant.
     */
    [[nodiscard]] std::vector<float> getPlantInternodeLengths(uint plantID) const;

    //! Get the inclination angle of every leaf on the plant
    /**
     * The angle between each leaf blade and the horizontal, computed from its area-weighted normal so
     * that a curved or folded blade is summarized by the direction it mostly faces. 0 degrees is a
     * horizontal blade and 90 degrees a vertical one; because a blade is a surface, a normal pointing
     * down describes the same inclination as its opposite pointing up, so the angle is folded about the
     * horizontal and never exceeds 90 degrees.
     *
     * Ordering and the treatment of missing geometry match \ref getPlantLeafAreas(). A blade whose facet
     * normals cancel exactly is additionally omitted, since it faces no single direction.
     *
     * \param[in] plantID ID of the plant instance.
     * \return Inclination angle (degrees) of each leaf on the plant.
     */
    [[nodiscard]] std::vector<float> getPlantLeafInclinations(uint plantID) const;

    //! Checks if the plant with the given ID is dormant
    /**
     * \param[in] plantID The ID of the plant to check.
     * \return True if all shoots on the plant are dormant, false otherwise.
     */
    [[nodiscard]] bool isPlantDormant(uint plantID) const;

    //! Determine the phenological stage of a plant
    /**
     * \param[in] plantID The ID of the plant to check.
     * \return Phenological stage: "dormant", "vegetative", "reproductive", or "senescent".
     */
    [[nodiscard]] std::string determinePhenologyStage(uint plantID) const;

    //! Write all vertices in the plant to a file for external processing (e.g., bounding volume, convex hull)
    /**
     * \param[in] plantID ID of the plant instance.
     * \param[in] filename Name/path of the output file.
     */
    void writePlantMeshVertices(uint plantID, const std::string &filename) const;

    //! Get IDs for all plant instances
    /**
     * \return Vector of plant IDs for all plant instances
     */
    [[nodiscard]] std::vector<uint> getAllPlantIDs() const;

    //! Get object IDs for all organs objects for a given plant
    /**
     * \param[in] plantID ID of the plant instance.
     * \return Vector of object IDs for all organs in the plant.
     */
    [[nodiscard]] std::vector<uint> getAllPlantObjectIDs(uint plantID) const;

    //! Get primitive UUIDs for all primitives in a given plant
    /**
     * \param[in] plantID ID of the plant instance.
     * \param[in] include_hidden If true, also include UUIDs of hidden prototype primitives managed by this PlantArchitecture instance (prototypes are shared across all plant instances, not specific to one plant).
     * \return Vector of primitive UUIDs for all primitives in the plant (and optionally hidden prototypes).
     */
    [[nodiscard]] std::vector<uint> getAllPlantUUIDs(uint plantID, bool include_hidden = false) const;

    //! Get object IDs for all internode (Tube) objects for a given plant
    /**
     * \param[in] plantID ID of the plant instance.
     * \return Vector of object IDs for all internodes in the plant.
     */
    [[nodiscard]] std::vector<uint> getPlantInternodeObjectIDs(uint plantID) const;

    //! Get object IDs for internode (Tube) objects matching a specified shoot type label
    /**
     * \param[in] plantID ID of the plant instance.
     * \param[in] shoot_type_label Label of the shoot type to filter internodes by.
     * \return Vector of object IDs for all internodes matching the specified shoot type label.
     * \note Throws a helios_runtime_error if the plant does not exist or if no shoots with the specified shoot type label are found.
     */
    [[nodiscard]] std::vector<uint> getPlantInternodeObjectIDs(uint plantID, const std::string &shoot_type_label) const;

    //! Get object IDs for all petiole (Tube) objects for a given plant
    /**
     * \param[in] plantID ID of the plant instance.
     * \return Vector of object IDs for all petioles in the plant.
     */
    [[nodiscard]] std::vector<uint> getPlantPetioleObjectIDs(uint plantID) const;

    //! Get object IDs for all leaf objects for a given plant
    /**
     * \param[in] plantID ID of the plant instance.
     * \return Vector of object IDs for all leaves in the plant.
     */
    [[nodiscard]] std::vector<uint> getPlantLeafObjectIDs(uint plantID) const;

    //! Get object IDs for all leaf objects for a list of plants
    /**
     * \param[in] plantIDs List of IDs of the plant instances.
     * \return Vector of object IDs for all leaves in the plants.
     */
    [[nodiscard]] std::vector<uint> getPlantLeafObjectIDs(const std::vector<uint> &plantIDs) const;

    //! Get object IDs for all peduncle (Tube) objects for a given plant
    /**
     * \param[in] plantID ID of the plant instance.
     * \return Vector of object IDs for all peduncles in the plant.
     */
    [[nodiscard]] std::vector<uint> getPlantPeduncleObjectIDs(uint plantID) const;

    //! Get object IDs for all inflorescence objects for a given plant
    /**
     * \param[in] plantID ID of the plant instance.
     * \return Vector of object IDs for all inflorescences in the plant.
     */
    [[nodiscard]] std::vector<uint> getPlantFlowerObjectIDs(uint plantID) const;

    //! Get object IDs for all fruit objects for a given plant
    /**
     * \param[in] plantID ID of the plant instance.
     * \return Vector of object IDs for all fruits in the plant.
     */
    [[nodiscard]] std::vector<uint> getPlantFruitObjectIDs(uint plantID) const;

    //! Assign a fruit count value for each shoot on a plant
    /**
     * \param[in] plantID ID of the plant instance.
     */

    void updateShootFruitCounts(uint plantID) const;


    //! Get internode object IDs for all shoots for a given plant
    /**
     * \param[in] plantID ID of the plant instance.
     * \return Vector of object IDs for all shoots in the plant.
     */

    [[nodiscard]] std::vector<uint> getShootInternodeObjectIDs(uint plantID) const;

    //! Get collision-relevant object IDs for a specific plant
    /**
     * \param[in] plantID Plant ID for which to query collision-relevant object IDs
     * \return Vector of object IDs for collision-relevant organs belonging to specified plant
     *
     * Returns object IDs for plant organs that are currently enabled for collision detection
     * based on the settings from setCollisionRelevantOrgans().
     */
    [[nodiscard]] std::vector<uint> getPlantCollisionRelevantObjectIDs(uint plantID) const;

    //! Get UUIDs for all existing plant primitives
    /**
     * \return Vector of UUIDs for all plant primitives.
     */
    [[nodiscard]] std::vector<uint> getAllUUIDs() const;

    //! Get UUIDs for all existing leaf primitives
    /**
     * \return Vector of UUIDs for all leaf primitives.
     */
    [[nodiscard]] std::vector<uint> getAllLeafUUIDs() const;

    //! Get UUIDs for all existing internode primitives
    /**
     * \return Vector of UUIDs for all internode primitives.
     */
    [[nodiscard]] std::vector<uint> getAllInternodeUUIDs() const;

    //! Get UUIDs for all existing petiole primitives
    /**
     * \return Vector of UUIDs for all petiole primitives.
     */
    [[nodiscard]] std::vector<uint> getAllPetioleUUIDs() const;

    //! Get UUIDs for all existing peduncle primitives
    /**
     * \return Vector of UUIDs for all peduncle primitives.
     */
    [[nodiscard]] std::vector<uint> getAllPeduncleUUIDs() const;

    //! Get UUIDs for all existing flower primitives
    /**
     * \return Vector of UUIDs for all flower primitives.
     */
    [[nodiscard]] std::vector<uint> getAllFlowerUUIDs() const;

    //! Get UUIDs for all existing fruit primitives
    /**
     * \return Vector of UUIDs for all fruit primitives.
     */
    [[nodiscard]] std::vector<uint> getAllFruitUUIDs() const;

    //! Get object IDs for all existing plant compound objects
    /**
     * \return Vector of object IDs for all plant compound objects.
     */
    [[nodiscard]] std::vector<uint> getAllObjectIDs() const;

    // -- carbohydrate model -- //

    /**
     * \brief Enables the carbohydrate model in the plant architecture development.
     */
    void enableCarbohydrateModel();

    /**
     * \brief Disables the carbohydrate model
     */
    void disableCarbohydrateModel();

    // -- Nitrogen Model Methods -- //

    /**
     * \brief Enable the nitrogen model for tracking plant nitrogen status and stress
     *
     * The nitrogen model simulates nitrogen uptake, allocation to leaves, remobilization
     * from old to young leaves, and calculates a nitrogen stress factor (0-1) that other
     * plugins can optionally use to modify photosynthesis and growth.
     *
     * \see disableNitrogenModel(), addPlantNitrogen(), setPlantNitrogenParameters()
     */
    void enableNitrogenModel();

    /**
     * \brief Disable the nitrogen model
     */
    void disableNitrogenModel();

    /**
     * \brief Check if the nitrogen model is enabled
     * \return True if nitrogen model is enabled, false otherwise
     */
    [[nodiscard]] bool isNitrogenModelEnabled() const;

    /**
     * \brief Set nitrogen model parameters for a single plant
     * \param[in] plantID Plant ID to set parameters for
     * \param[in] params Nitrogen parameters struct
     * \throws helios_runtime_error if plant does not exist
     */
    void setPlantNitrogenParameters(uint plantID, const NitrogenParameters &params);

    /**
     * \brief Set nitrogen model parameters for multiple plants
     * \param[in] plantIDs Vector of plant IDs
     * \param[in] params Nitrogen parameters struct
     * \throws helios_runtime_error if any plant does not exist
     */
    void setPlantNitrogenParameters(const std::vector<uint> &plantIDs, const NitrogenParameters &params);

    /**
     * \brief Initialize nitrogen pools for all plants, setting the nitrogen content per unit area of every existing leaf and emptying the root and available pools
     * \param[in] initial_leaf_N_area Initial leaf nitrogen content per unit leaf area (g N/m²) given to every existing leaf
     * \throws helios_runtime_error if initial_leaf_N_area is negative
     */
    void initializeNitrogenPools(float initial_leaf_N_area);

    /**
     * \brief Initialize nitrogen pools for a specific plant, setting the nitrogen content per unit area of every existing leaf and emptying the root and available pools
     * \param[in] plantID Plant ID to initialize
     * \param[in] initial_leaf_N_area Initial leaf nitrogen content per unit leaf area (g N/m²) given to every existing leaf
     * \throws helios_runtime_error if plant does not exist or initial_leaf_N_area is negative
     */
    void initializePlantNitrogenPools(uint plantID, float initial_leaf_N_area);

    /**
     * \brief Add nitrogen to a single plant (immediate application to root and available pools)
     * \param[in] plantID Plant ID to receive nitrogen
     * \param[in] amount_gN Amount of nitrogen to add (g N)
     * \throws helios_runtime_error if plant does not exist or amount is negative
     */
    void addPlantNitrogen(uint plantID, float amount_gN);

    /**
     * \brief Add nitrogen to multiple plants (immediate application)
     * \param[in] plantIDs Vector of plant IDs to receive nitrogen
     * \param[in] amount_gN Amount of nitrogen to add to each plant (g N)
     * \throws helios_runtime_error if any plant does not exist or amount is negative
     */
    void addPlantNitrogen(const std::vector<uint> &plantIDs, float amount_gN);

    /**
     * \brief Get the nitrogen currently in a plant's available pool, awaiting allocation to its organs
     * \param[in] plantID Plant ID to query
     * \return Nitrogen in the available pool (g N)
     * \note Nitrogen enters the pool from addPlantNitrogen() (less the root allocation) and from leaves resorbed before they are shed, and leaves it as uptake by growing leaves and fruit.
     */
    [[nodiscard]] float getPlantAvailableNitrogen(uint plantID) const;

    /**
     * \brief Get the nitrogen a plant has delivered to its fruit since its nitrogen pools were initialized
     * \param[in] plantID Plant ID to query
     * \return Nitrogen delivered to fruit (g N), from the available pool and remobilized from leaves, including that of fruit since removed from the plant
     * \note With the root pool (getPlantRootNitrogen()), the available pool (getPlantAvailableNitrogen()) and the nitrogen of the leaves ("leaf_nitrogen_gN_m2" object data times leaf area),
     * this accounts for all the nitrogen given to the plant, less what has left with leaves that were shed.
     */
    [[nodiscard]] float getPlantFruitNitrogen(uint plantID) const;

    /**
     * \brief Get the nitrogen allocated to a plant's roots since its nitrogen pools were initialized
     * \param[in] plantID Plant ID to query
     * \return Nitrogen in the root pool (g N): NitrogenParameters::root_allocation_fraction of the nitrogen given with addPlantNitrogen()
     */
    [[nodiscard]] float getPlantRootNitrogen(uint plantID) const;

    // -- manual plant generation from input string -- //

    /**
     * \brief Retrieves a string representation of a plant based on its ID.
     *
     * \param[in] plantID The unique identifier of the plant.
     * \return A string encoding of the plant structure.
     */
    [[nodiscard]] std::string getPlantString(uint plantID) const;

    /**
     * \brief Generates a plant model based on the given generation string and phytomer parameters.
     *
     * \param[in] generation_string Input string describing the plant generation rules and structure.
     * \param[in] phytomer_parameters Parameters that define the characteristics of a single phytomer.
     * \return The total number of phytomers generated in the plant architecture.
     * \note Cannot be used inside a phytomer callback (\ref PhytomerParameters::phytomer_creation_function or \ref PhytomerParameters::phytomer_callback_function): a call made from one raises an error. See \ref PlantArchCallbackEdits.
     */
    uint generatePlantFromString(const std::string &generation_string, const PhytomerParameters &phytomer_parameters);

    /**
     * \brief Generates a plant based on the provided description string and phytomer parameters.
     *
     * \param[in] generation_string A string encoding of the plant structure.
     * \param[in] phytomer_parameters A map containing parameter configurations for each type of phytomer.
     * \return A unique identifier for the generated plant.
     * \note The input string must begin with '{', and valid phytomer parameters must be provided.
     * \note Cannot be used inside a phytomer callback (\ref PhytomerParameters::phytomer_creation_function or \ref PhytomerParameters::phytomer_callback_function): a call made from one raises an error. See \ref PlantArchCallbackEdits.
     */
    uint generatePlantFromString(const std::string &generation_string, const std::map<std::string, PhytomerParameters> &phytomer_parameters);

    /**
     * \brief Writes the structure of a plant instance to an XML file.
     *
     * \param[in] plantID The unique identifier for the plant instance.
     * \param[in] filename Path to the XML file where the plant structure will be saved.
     * \note The function checks if the plant instance exists and if the output file path
     * is valid and writable. Errors related to invalid plant ID or file issues will throw exceptions.
     * \note In addition to the shoot structure, the plant's phenological thresholds are written
     * (dd_to_dormancy_break, dd_to_flower_initiation, dd_to_flower_opening, dd_to_fruit_set,
     * dd_to_fruit_maturity, dd_to_dormancy, max_leaf_lifespan, is_evergreen). These are not derivable
     * from the shoot structure, so without them a restored plant would fall back to the default
     * thresholds rather than the ones it was built with.
     */
    void writePlantStructureXML(uint plantID, const std::string &filename) const;

    //! Writes plant structure to TreeQSM cylinder format.
    /**
     * Writes the plant structure as a series of cylinders in the TreeQSM format.
     * Each row represents one cylinder with columns for radius, length, start position,
     * axis direction, parent ID, extension ID, branch ID, branch order, position in branch,
     * mean absolute distance, surface coverage, added flag, and unmodified radius.
     *
     * \param[in] plantID Identifier of the plant to export.
     * \param[in] filename Path to the output file (typically .txt extension).
     * \throws helios::runtime_error Throws an error if the plant ID does not exist or if the file cannot be opened.
     * \note The output follows the TreeQSM format (Raumonen et al., 2013) with tab-separated values.
     */
    void writeQSMCylinderFile(uint plantID, const std::string &filename) const;

    //! Write plant as a USD articulated rigid body for NVIDIA IsaacSim physics simulation.
    /**
     * Exports the plant structure (internodes, petioles, peduncles, leaves, fruits, flowers) as a
     * PhysX articulation in USDA (ASCII USD) format. Each tube segment becomes a rigid link with a
     * capsule collision shape, connected by D6 joints with rotational spring/damper drives derived
     * from beam bending stiffness (E*I/L). Leaves, fruits, and flowers are represented as mass
     * bodies attached by spring links at their base positions.
     *
     * Every joint locks translation, limits bending to +/-60 degrees about the two axes perpendicular to the parent
     * segment, and carries the same spring on all three rotation axes. Segment joints are damped in proportion to
     * their stiffness (USDExportParameters::damping_time_constant).
     *
     * Beam stiffness on a short, light segment is far stiffer than a physics solver can integrate at a typical step
     * rate, and a plant exported with the physical values alone collapses as soon as the simulation starts. Each joint
     * is therefore given armature, an artificial inertia, sized so that the joint's own natural frequency does not
     * exceed USDExportParameters::armature_stability_ratio times USDExportParameters::physics_steps_per_second. The
     * armature adds no weight, so deflection under gravity is unchanged, and it is small next to the inertia of the
     * limb beyond the joint. If the simulation runs at a coarser step than physics_steps_per_second, set that
     * parameter to match or the plant may be unstable.
     *
     * \param[in] plantID ID of the plant instance to export.
     * \param[in] filename Path to the output file (should have .usda extension).
     * \param[in] params Export parameters controlling physics properties and scope.
     */
    void writePlantStructureUSD(uint plantID, const std::string &filename, const USDExportParameters &params = USDExportParameters()) const;

    //! Get the articulated rigid-body description of a plant used for physics simulation
    /**
     * Builds the same links and joints that writePlantStructureUSD() exports, without writing a file, so that a physics
     * engine can be driven from them directly. Each internode, petiole, and peduncle segment becomes a capsule link, and
     * each leaf, fruit, and flower becomes a mass link, connected to its parent by an elastic joint whose stiffness,
     * damping, and armature are computed from the export parameters.
     *
     * \param[in] plantID ID of the plant instance.
     * \param[in] params Parameters controlling the physical properties of links and joints.
     * \return Links and joints of the plant articulation, with every parent link preceding its children.
     */
    [[nodiscard]] PlantArticulation getPlantArticulation(uint plantID, const USDExportParameters &params = USDExportParameters()) const;

    //! Register the current plant state as a growth animation frame.
    /**
     * Captures a snapshot of the plant's geometry (all links, transforms, meshes, and materials) at the
     * current point in the growth simulation. Call this after each \ref advanceTime() step to record the
     * plant's state for later animation export via \ref writePlantGrowthUSD().
     *
     * \param[in] plantID ID of the plant instance to capture.
     * \param[in] min_segment_length Minimum segment length (m). Segments shorter than this are skipped.
     */
    void registerGrowthFrame(uint plantID, float min_segment_length = 0.001f);

    //! Write all registered growth frames as a time-sampled USD animation file for Blender.
    /**
     * Exports a USDA file containing all growth frames registered via \ref registerGrowthFrame() as
     * time-sampled Xform transforms. Each plant organ is an independent prim with animated position and
     * orientation. Organs that appear during growth are toggled from invisible to visible at the
     * appropriate time code. The resulting file can be imported directly into Blender.
     *
     * This is a visual-only export — no physics prims, joints, or collision shapes are written.
     *
     * \param[in] plantID ID of the plant instance to export.
     * \param[in] filename Path to the output file (should have .usda extension).
     * \param[in] seconds_per_frame Duration in seconds that each growth frame occupies in the animation (default: 1 second).
     *   For example, with 6 frames and seconds_per_frame=1.0, the total animation is 5 seconds long.
     */
    void writePlantGrowthUSD(uint plantID, const std::string &filename, float seconds_per_frame = 1.0f) const;

    //! Clear stored growth animation frames for a plant.
    /**
     * \param[in] plantID ID of the plant instance whose frames should be cleared.
     */
    void clearGrowthFrames(uint plantID);

    //! Get the number of registered growth frames for a plant.
    /**
     * \param[in] plantID ID of the plant instance to query.
     * \return Number of frames registered via \ref registerGrowthFrame().
     */
    [[nodiscard]] uint getGrowthFrameCount(uint plantID) const;

    /**
     * \brief Reads plant structure data from an XML file.
     *
     * Parses the specified XML file containing plant architecture information and extracts
     * relevant data.
     *
     * \param[in] filename The path to the XML file to load.
     * \param[in] quiet [optional] If true, suppresses console output of status messages.
     * \return A vector of unsigned integers representing the plant IDs parsed from the XML file.
     * \note Throws an exception if the file cannot be parsed or is missing required tags.
     * \note The phenological threshold tags written by writePlantStructureXML() are optional on read,
     * so files written before those tags existed still load. When they are absent, the restored plant
     * keeps the default thresholds, which schedule no phenology.
     * \note Cannot be used inside a phytomer callback (\ref PhytomerParameters::phytomer_creation_function or \ref PhytomerParameters::phytomer_callback_function): a call made from one raises an error. See \ref PlantArchCallbackEdits.
     */
    std::vector<uint> readPlantStructureXML(const std::string &filename, bool quiet = false);

    //! Disable standard output from this plug-in
    void disableMessages();

    //! Re-enable standard output from this plug-in
    void enableMessages();

    //! Generate images of a leaf at successive stages of aging, for use in LeafPrototype::leaf_texture_series
    /**
     * Each pixel of the leaf changes color at its own point in the aging, so that color spreads across the leaf as an aging front rather than fading in everywhere at once. The images have exactly the transparency of the leaf texture, as LeafPrototype::leaf_texture_series requires.
     * \param[in] leaf_texture Path to the image of the young leaf, which can be a simple file name as for LeafPrototype::leaf_texture_file.
     * \param[in] output_directory Directory to write the images to, which is created if it does not exist. Images named after the leaf texture are overwritten, so give different directories for different options.
     * \param[in] stage_count Number of stages, including the leaf texture as the first, which must be at least 2.
     * \param[in] options How the leaf ages.
     * \return Paths to the image for each stage, starting with the leaf texture itself. These are PNG files if the leaf texture has transparency and JPEG files if not.
     */
    static std::vector<std::string> generateLeafTextureStages(const std::string &leaf_texture, const std::string &output_directory, uint stage_count, const LeafTextureStageOptions &options = LeafTextureStageOptions());

    //! Resolve an asset file path for the plantarchitecture plugin
    /**
     * This method resolves asset file paths (textures and OBJ models) for the plantarchitecture plugin,
     * allowing users to specify simple paths like "OliveBark.jpg" or "MyLeaf.obj" instead of using the
     * verbose helios::resolvePluginAsset() wrapper.
     *
     * Resolution order:
     * 1. Empty path returns empty string
     * 2. Absolute paths that exist are returned as-is
     * 3. Try resolving as a general file path (for already-resolved paths)
     * 4. Try resolving as a plugin asset path (e.g., "assets/textures/OliveBark.jpg")
     * 5. If path doesn't have "assets/" prefix, determine subdirectory from file extension:
     *    - .obj/.mtl files: try "assets/obj/" + filename
     *    - Other files: try "assets/textures/" + filename
     *
     * \param[in] texture_file The asset file path (can be simple filename, relative path, or absolute path)
     * \return Resolved absolute path to the asset file
     * \throws helios_runtime_error if the asset file cannot be found
     */
    static std::string resolveTextureFile(const std::string &texture_file);

    //! Record a class map of a texture on the primitives mapped with it, as the primitive data "spectral_class_map"
    /**
     * The class map is checked against the texture (resolution, and class 0 marking exactly the transparent texels of a texture with transparency, or absent for one without), and recorded on each
     * of the given primitives that is mapped with the texture and covers a class (\ref helios::classMapAreaFractions()); primitives that already carry the data are left as they are.
     * \param[in] context Context the primitives belong to
     * \param[in] UUIDs Primitives to consider
     * \param[in] texture_file Resolved path of the texture the class map was made for, or empty to take every textured primitive among the UUIDs (which must then share one texture)
     * \param[in] class_map_file Class map, resolved as a texture file is (see \ref resolveTextureFile())
     * \param[in] organ Name of the organ, for error messages (for example "leaf")
     * \param[in] remedy Advice appended to error messages about a class map that does not match the texture
     */
    static void recordSpectralClassMap(helios::Context *context, const std::vector<uint> &UUIDs, const std::string &texture_file, const std::string &class_map_file, const std::string &organ, const std::string &remedy);

    friend struct Phytomer;
    friend struct Shoot;

    //! Grants the plug-in's self-tests access to private per-plant state.
    /**
     * Defined solely in the plug-in's selfTest.cpp, so nothing test-related is exposed to users
     * of the library.
     */
    friend class PlantArchitectureTestHelper;

private:

    //! Draw the delay of one shoot or bud within a phenological window of setPlantPhenologicalSpread
    /**
     * \param[in] duration Length of the window in days.
     * \return Delay in days, from a normal distribution with mean duration/2 and standard deviation duration/4 truncated to [0, duration]. 0 without a draw when duration is 0, so that a plant with no spread
     * consumes no random numbers.
     */
    [[nodiscard]] float samplePhenologicalDelay(float duration) const;

    //! Clear BVH cache (called at start of each growth cycle)
    void clearBVHCache() const;

    //! Rebuild BVH once per timestep with current target geometry
    void rebuildBVHForTimestep();

    //! Internal helper method for plant library functions to set attraction points
    /**
     * This method is for use within plant library methods to set plant-specific attraction points
     * without affecting the global state or other plants.
     * \param[in] plantID Identifier of the plant to set attraction points for
     * \param[in] attraction_points Vector of 3D positions that the plant should grow toward
     * \param[in] view_half_angle_deg Half-angle of the attraction detection view cone in degrees
     * \param[in] look_ahead_distance How far ahead attraction points will be considered in meters
     * \param[in] attraction_weight Weight factor for attraction vs natural growth
     * \param[in] obstacle_reduction_factor Reduction factor for attraction when hard obstacles are present
     */
    void setPlantAttractionPoints(uint plantID, const std::vector<helios::vec3> &attraction_points, float view_half_angle_deg = 80.0f, float look_ahead_distance = 0.1f, float attraction_weight = 0.6f, float obstacle_reduction_factor = 0.75f);

    //! Ensure inflorescence prototype maps are initialized for given phytomer parameters
    /**
     * This method checks if unique inflorescence prototypes (flowers and fruit) have been created and cached.
     * If not, it creates and hides the prototype objects. This is necessary before calling createInflorescenceGeometry()
     * when unique_prototypes > 0, otherwise map::at() will throw an out_of_range exception.
     * \param[in] params Phytomer parameters containing inflorescence configuration
     * \param[in] plantID ID of the plant bearing the inflorescence. Its name labels the prototype materials, and its build parameters are passed to the prototype functions and select the cached set.
     */
    void ensureInflorescencePrototypesInitialized(const PhytomerParameters &params, uint plantID);

    // --- Structural edits requested from inside phytomer callbacks --- //

    //! A pruneBranch() call made from inside a phytomer callback, held until no loop over the plant structure is active
    struct DeferredPrune {
        uint plantID;
        uint shootID;
        uint node_index;
    };

    //! Marks the duration of one phytomer callback invocation (creation or per-time-step)
    /**
     * Exception-safe: the depth is restored even if the callback throws.
     */
    class PhytomerCallbackScope {
    public:
        explicit PhytomerCallbackScope(PlantArchitecture *plantarchitecture_ptr);
        ~PhytomerCallbackScope();
        PhytomerCallbackScope(const PhytomerCallbackScope &) = delete;
        PhytomerCallbackScope &operator=(const PhytomerCallbackScope &) = delete;

    private:
        PlantArchitecture *plantarchitecture_ptr;
    };

    //! Marks the duration of a public operation during which phytomer callbacks can run
    /**
     * Scopes nest. When the outermost one ends normally, the pruneBranch() calls deferred during the operation are applied and the
     * warnings collected for operations skipped inside callbacks are reported. When it ends because an exception is propagating, both
     * are discarded instead, since the operation has already failed. Either way, nothing deferred outlives the public call.
     */
    class StructuralOperationScope {
    public:
        explicit StructuralOperationScope(PlantArchitecture *plantarchitecture_ptr);
        ~StructuralOperationScope() noexcept(false);
        StructuralOperationScope(const StructuralOperationScope &) = delete;
        StructuralOperationScope &operator=(const StructuralOperationScope &) = delete;

    private:
        PlantArchitecture *plantarchitecture_ptr;
        int uncaught_exceptions_at_entry;
    };

    //! True while a phytomer creation or per-time-step callback is running
    [[nodiscard]] bool isInsidePhytomerCallback() const;

    //! Record that a void structural operation was skipped because it was called from inside a phytomer callback
    /**
     * \param[in] operation Name of the skipped operation, e.g. "deletePlantInstance"
     * \return True if the caller is inside a phytomer callback and must return without doing anything.
     */
    bool skipStructuralOperationInCallback(const std::string &operation);

    //! Raise an error if a structural operation that returns a value is called from inside a phytomer callback
    /**
     * \param[in] operation Name of the rejected operation, e.g. "addChildShoot"
     */
    void rejectStructuralOperationInCallback(const std::string &operation) const;

    //! Build a shoot along prescribed node positions: what addShootFromNodePositions() and appendShootFromNodePositions() both do
    /**
     * \param[in] continues_parent_axis True to seat the base on the tip of the parent's last node, on its axis, as an axis continuation; false to seat it on the parent's surface at parent_node_index, as a lateral.
     * \param[in] caller Name of the public method, for error messages.
     */
    uint prescribedShoot(uint plantID, int parent_shoot_ID, uint parent_node_index, const std::vector<helios::vec3> &internode_node_positions, const std::vector<float> &internode_radii, const std::string &shoot_type_label,
                         const std::string &growth_shoot_type_label, uint petiole_index, bool continues_parent_axis, const std::string &caller);

    //! Apply a validated cut: the removal performed by pruneBranch()
    void applyPruneBranch(uint plantID, uint shootID, uint node_index, std::vector<uint> *detached_objIDs = nullptr);

    //! Apply, in call order, every pruneBranch() call deferred from inside a phytomer callback
    void flushDeferredPrunes();

    //! Mark the primitive-to-organ index out of date, so that the next organ lookup rebuilds it
    /**
     * Called wherever organ objects of a plant are created, deleted, detached or have primitives added, i.e. wherever Shoot::internode_tube_objID, Phytomer::petiole_objIDs, Phytomer::leaf_objIDs,
     * FloralBud::peduncle_objIDs or FloralBud::inflorescence_objIDs change or an internode tube gains, loses or moves primitives.
     */
    void invalidateOrganIndex() const;

    //! Rebuild the primitive-to-organ index from the structure of every plant, if it is out of date
    void ensureOrganIndexCurrent() const;

    //! Look up a primitive in the organ index (which must be current)
    /**
     * \param[in] UUID Unique identifier of the primitive.
     * \param[out] node_index Node of the primitive: the record's node_index, or for an internode primitive the phytomer its triangle belongs to.
     * \return The organ record of the primitive's parent object, or nullptr if the primitive does not exist or is not part of a plant.
     */
    [[nodiscard]] const PlantOrganLocation *findPrimitiveOrganLocation(uint UUID, uint &node_index) const;

    //! Organ record of every live organ object of every plant, keyed by object ID. For internode tubes node_index is PlantOrganLocation::not_applicable.
    mutable std::unordered_map<uint, PlantOrganLocation> organ_index_by_objID;

    //! Phytomer (node) index of every primitive of every live internode tube, keyed by primitive UUID
    mutable std::unordered_map<uint, uint> internode_node_by_UUID;

    //! False when the organ index must be rebuilt before the next lookup
    mutable bool organ_index_current = false;

    //! Number of phytomer callbacks currently running (nested invocations count separately)
    int phytomer_callback_depth = 0;

    //! Number of nested public operations during which phytomer callbacks can run
    int structural_operation_depth = 0;

    //! pruneBranch() calls made from inside phytomer callbacks, waiting to be applied
    std::vector<DeferredPrune> deferred_prunes;

    //! Warnings for structural operations skipped inside phytomer callbacks, reported once per public call
    helios::WarningAggregator callback_warnings;

protected:
    helios::Context *context_ptr;

    std::minstd_rand0 *generator = nullptr;

    uint plant_count = 0;

    std::string current_plant_model;

    std::function<void(float, const std::string&)> progress_callback;

    //! External cancellation flag polled during long builds (nullptr = none).
    //! See setCancelFlag(); owned by the caller, read via volatile in the canopy
    //! build loop and advanceTime().
    volatile int *cancel_flag = nullptr;

    //! Build parameters declared for a library plant model, see \ref registerPlantModelBuildParameters()
    struct PlantModelBuildParameters {
        std::vector<BuildParameterDeclaration> structure_parameters;
        std::vector<BuildParameterDeclaration> organ_parameters;
        std::function<void(const std::map<std::string, float> &)> validate_organ_parameters;
    };

    // Build parameters of the plant being built by buildPlantInstanceFromLibrary(), set only for the duration of that call. The builder reads current_build_parameters, which hold the values sampled for
    // the plant, and addPlantInstance() attaches both maps to the plant the builder creates.
    std::map<std::string, float> current_build_parameters;
    std::map<std::string, BuildParameterDistribution> current_organ_build_parameters;
    // Name of the library model being built by buildPlantInstanceFromLibrary(), set only for the duration of that call, given by addPlantInstance() to the plant it creates so that the organs built
    // during the build are named for the model ("custom" otherwise).
    std::string current_build_plant_name;

    // Function pointer maps for plant model registration
    std::map<std::string, std::function<void()>> shoot_initializers;
    std::map<std::string, std::function<uint(const helios::vec3 &)>> plant_builders;
    std::map<std::string, std::string> plant_type_map;
    //! Build parameters declared by each library plant model, keyed by model name. Populated by \ref registerPlantModelBuildParameters().
    std::map<std::string, PlantModelBuildParameters> plant_model_build_parameters;
    //! Empirical leaf inclination Beta distribution declared by each library plant model, keyed by model name
    /**
     * Populated by \ref registerPlantModel(). The x-component is the Beta distribution parameter mu and the
     * y-component is nu, in the parameterization of \ref helios::sample_Beta_distribution().
     *
     * A model absent from this map declares no measured distribution, and its plants are not steered.
     * Absence is the "unset" representation deliberately: a numeric sentinel
     * stored against every model would switch leaf angle distribution tracking on for species that have no
     * distribution to track, which is observable through \ref isPlantLeafAngleDistributionTrackingEnabled(),
     * \ref duplicatePlantInstance() and \ref writePlantStructureXML().
     */
    std::map<std::string, helios::vec2> plant_leaf_inclination_distribution_map;
    //! Steering strength used for the leaf inclination distribution a library plant model declares (degrees)
    /**
     * Deliberately not a library parameter. A species' measured leaf angle distribution is a statement about
     * the angles its leaves take, not about how hard the model should pull toward them, so it is not something
     * a species definition has data for. This is the value the tracking documentation gives as matching the
     * distribution as closely as a growing plant allows, which is what a plant carrying a measured
     * distribution should do. A caller wanting a different trade-off overrides it per plant with
     * \ref enablePlantLeafElevationAngleDistributionTracking().
     */
    static constexpr float library_leaf_angle_lambda_degrees = 180.f;

    std::map<uint, PlantInstance> plant_instances;

    [[nodiscard]] std::string makeShootString(const std::string &current_string, const std::shared_ptr<Shoot> &shoot, const std::vector<std::shared_ptr<Shoot>> &shoot_tree) const;

    std::map<std::string, ShootParameters> shoot_types;

    //! Key of the cached leaf prototypes: the leaf prototype's unique identifier paired with the number of leaflets the cached set was built for
    /**
     * A blade's shape is built from its position along the compound leaf, which is measured from the leaflet count, so a set of blades built for one leaflet count is not usable - not even as a prefix - by a
     * petiole carrying a different number of leaflets. The count therefore belongs in the key and not just the identifier, because two shoot types can legitimately share an identifier while carrying
     * different numbers of leaflets (a one-leaflet cotyledon type whose leaf prototype was copied from a compound type), and a single shoot type whose leaves_per_petiole is a random parameter draws a
     * different count on different phytomers.
     */
    using LeafPrototypeCacheKey = std::pair<uint, uint>;

    // Value first index is the unique leaf prototype, second index is the leaflet along a compound leaf (if applicable)
    std::map<LeafPrototypeCacheKey, std::vector<std::vector<uint>>> unique_leaf_prototype_objIDs;

    //! Undeformed blade geometry of each cached leaf prototype, parallel to \ref unique_leaf_prototype_objIDs
    /**
     * A drooping leaf is re-deflected from its rest shape every time it grows rather than bent further from its current shape, so that the deflection cannot accumulate and creep the leaf downward. The rest
     * shape is therefore kept once per prototype - not once per leaf - which is what lets thousands of leaves share it without giving up the memory savings of the prototype cache. Empty for species whose
     * leaves do not droop.
     */
    struct LeafRestGeometry {
        //! Blade lattice, which is what bends
        std::vector<helios::vec3> vertices;
        //! Vertices that follow the lattice in the leaf's mesh - its petiolule, if it has one - which do not bend but have to be written back along with the blade
        std::vector<helios::vec3> appended_vertices;
        uint subdivisions_x = 0;
        uint subdivisions_y = 0;
        //! Shape the prototype blade was built with, recorded for every blade built by GenericLeafPrototype() whether or not it droops
        LeafShapeValues shape;
        bool has_shape = false;
        //! Position, in the frame buildLeafBladeLattice() builds the blade in, of the origin of the prototype object's own frame
        /**
         * Context::addPolymeshObject() puts an object's origin at the first vertex of its first primitive, not at the origin the blade was built about, so a vertex of the object's own frame (which every
         * copy of the prototype shares) is a lattice vertex minus this offset. Recorded for every blade, drooping or not.
         */
        helios::vec3 lattice_frame_offset;
    };
    std::map<LeafPrototypeCacheKey, std::vector<std::vector<LeafRestGeometry>>> unique_leaf_prototype_rest_geometry;

    //! Record the undeformed geometry of one leaf prototype, so leaves copied from it can be deflected from their rest shape
    /**
     * Appends one entry to \ref unique_leaf_prototype_rest_geometry for the given prototype, capturing the blade lattice in the prototype's own local frame, and separately the petiolule vertices that follow
     * it in a leaf built with one. A rigid species, or a mesh that is not a lattice the deflection understands, records an empty entry and its leaves stay rigid. Called both when a phytomer first builds its prototypes and when readPlantStructureXML() rebuilds them, so that a
     * reloaded plant droops like a grown one.
     * \param[in] prototype_params Leaf prototype parameters the prototype was built from.
     * \param[in] leaves_per_petiole Number of leaflets the prototype set is being built for, which is part of the cache key.
     * \param[in] prototype_index Index of this prototype among the species' unique prototypes.
     * \param[in] objID_leaf Object ID of the prototype leaf.
     * \param[in] leaf_flexibility Resolved flexibility of the phytomer the prototype is being built for. Zero records an empty entry.
     */
    void recordLeafPrototypeRestGeometry(const LeafPrototype &prototype_params, int leaves_per_petiole, int prototype_index, uint objID_leaf, float leaf_flexibility);

    //! Give every leaf of a plant the image of LeafPrototype::leaf_texture_series for its age
    /**
     * \param[in] plantID Identifier of the plant.
     * \param[in] reapply If true, the image is applied to every leaf even where the phytomer's stage has not changed, as is needed when its leaves have been replaced.
     */
    void updatePlantLeafTextureStages(uint plantID, bool reapply = false);

    //! Write the "leaf_texture_stage" object data of every existing leaf, after it has been enabled with optionalOutputObjectData()
    void refreshLeafTextureStageOutput();

    //! Look up one cached leaf blade prototype, to be copied for a leaf
    /**
     * Fails with an explicit error rather than indexing the cache out of range when the requested blade was never cached, which means the caller is asking for a leaflet count or a prototype index the set
     * was not built for.
     * \param[in] prototype_identifier Unique identifier of the leaf prototype the blade was built from.
     * \param[in] leaves_per_petiole Number of leaflets the petiole carries, which selects the set of blades built for that count.
     * \param[in] prototype_index Index of the wanted prototype among the species' unique prototypes.
     * \param[in] leaf_index Position of the leaflet along the compound leaf.
     * \param[in] calling_function Name of the calling function, quoted in the error message.
     * \return Object ID of the cached prototype leaf, which the caller copies to make the leaf.
     */
    [[nodiscard]] uint getCachedLeafPrototypeObjID(uint prototype_identifier, uint leaves_per_petiole, uint prototype_index, uint leaf_index, const std::string &calling_function) const;

    // Key is the prototype function pointer and the organ build parameter distributions of the plant it was built for, so plants built with different parameters get their own prototypes, and each
    // prototype of a set is built from its own sample of the distributions; value index is the unique flower prototype
    std::map<std::pair<FlowerPrototypeFunction, std::map<std::string, BuildParameterDistribution>>, std::vector<uint>> unique_open_flower_prototype_objIDs;
    // Key is the prototype function pointer and the plant's organ build parameter distributions; value index is the unique flower prototype
    std::map<std::pair<FlowerPrototypeFunction, std::map<std::string, BuildParameterDistribution>>, std::vector<uint>> unique_closed_flower_prototype_objIDs;
    // Key is the prototype function pointer and the plant's organ build parameter distributions; value index is the unique fruit prototype
    std::map<std::pair<FruitPrototypeFunction, std::map<std::string, BuildParameterDistribution>>, std::vector<uint>> unique_fruit_prototype_objIDs;

    //! A material named for organs by nameOrganMaterials(), with the appearance of the automatic material it was named from
    struct OrganMaterial {
        std::string label;
        helios::RGBAcolor color;
        std::string texture_file;
        bool texture_color_overridden = false;
        std::string normal_map_file;
        helios::NormalMapKind normal_map_kind = helios::NORMAL_MAP_NONE;
        float normal_map_strength = 1.f;
        bool normal_map_flip_green = false;
    };
    //! Material named for each automatic material label and organ material base name, given to the organs of that kind and appearance created later (see nameOrganMaterials())
    std::map<std::pair<std::string, std::string>, OrganMaterial> organ_material_labels;

    //! Give an organ's automatic materials descriptive labels starting with desired_base_name, sharing the material already named for organs of the same kind and appearance
    void nameOrganMaterials(uint objID, const std::string &desired_base_name);

    //! Give the automatic materials of several organs descriptive labels (see nameOrganMaterials(uint, const std::string &))
    void nameOrganMaterials(const std::vector<uint> &objIDs, const std::string &desired_base_name);

    //! Draw one value of each organ build parameter of a plant, for building one flower or fruit prototype
    /**
     * \param[in] organ_build_parameters Organ build parameter distributions of the plant.
     * \return Map of organ parameter names to sampled values. Constants are returned without drawing from the random generator.
     */
    [[nodiscard]] std::map<std::string, float> sampleOrganBuildParameters(const std::map<std::string, BuildParameterDistribution> &organ_build_parameters) const;

    //! Get object IDs for all hidden prototype objects managed by this PlantArchitecture instance
    [[nodiscard]] std::vector<uint> getAllPrototypeObjectIDs() const;

    //! Delete all hidden prototype objects from the Context and clear the prototype maps
    void deleteAllPrototypes();

    bool build_context_geometry_internode = true;
    bool build_context_geometry_petiole = true;
    bool build_context_geometry_peduncle = true;

    float ground_clipping_height = -99999;

    void validateShootTypes(ShootParameters &shoot_parameters, const std::map<std::string, ShootParameters> &shoot_types_ref) const;

    //! Register a plant model with its initialization and build functions
    /**
     * \param[in] name Plant model label used to look up the model.
     * \param[in] shoot_init Function that initializes the model's shoot definitions.
     * \param[in] plant_build Function that builds a plant instance at the given base position and returns its plant ID.
     * \param[in] plant_type [optional] Category label written as "plant_type" object data when enabled (e.g. "tree", "weed"). Defaults to "herbaceous" when omitted.
     * \param[in] leaf_inclination_Beta [optional] Empirical leaf inclination distribution measured for this species, as Beta distribution parameters (x=mu, y=nu). Plants built from this model by
     * \ref buildPlantInstanceFromLibrary() are steered toward it as they grow. The default of (0,0) means the model declares no measured distribution and its plants keep the leaf angles the model
     * generates.
     * \note Either both components are positive, or both are zero. A model that gives one without the other is a half-filled entry and throws.
     */
    void registerPlantModel(const std::string &name, std::function<void()> shoot_init, std::function<uint(const helios::vec3 &)> plant_build, const std::string &plant_type = "herbaceous",
                            const helios::vec2 &leaf_inclination_Beta = helios::make_vec2(0.f, 0.f));

    //! Declare the build parameters a registered plant model accepts
    /**
     * \ref buildPlantInstanceFromLibrary() rejects any key not declared for the model, so a misspelled parameter fails instead of silently leaving the default in place. A model with no declaration accepts none.
     *
     * Every declared parameter given to \ref buildPlantInstanceFromLibrary() is kept with the plant for its life. They are declared in two groups. Structure parameters are read by the model's builder, and
     * possibly by its phytomer creation function as the plant grows. Organ parameters shape the model's flowers and fruit: they are passed to its prototype functions and identify its cached prototypes, so
     * plants that differ only in structure parameters share their flower and fruit prototypes.
     * Each parameter is declared with its valid range, which must be the declaration the code reading it checks against, so that a \ref BuildParameterDistribution can be checked for its whole range
     * before any plant is built.
     * \param[in] name Plant model label, which must already be registered with \ref registerPlantModel().
     * \param[in] structure_parameters Declarations of every build parameter read by the model's builder.
     * \param[in] organ_parameters [optional] Declarations of every build parameter read by the model's flower and fruit prototype functions.
     * \param[in] validate_organ_parameters [optional] Function that reads every organ parameter and throws if one is invalid. It is called with the organ parameters given as fixed values before the plant
     * is built and when organ parameters are read from XML, so an invalid value fails then rather than when the first flower or fruit forms. Required when organ parameters are declared.
     */
    void registerPlantModelBuildParameters(const std::string &name, const std::vector<BuildParameterDeclaration> &structure_parameters, const std::vector<BuildParameterDeclaration> &organ_parameters = {},
                                           std::function<void(const std::map<std::string, float> &)> validate_organ_parameters = nullptr);

    //! Check that every entry of a build parameter map is declared for a plant model, that every distribution lies within its parameter's range, and that its fixed organ parameters are valid
    /**
     * A fixed structure parameter is checked by the builder that reads it, and a fixed organ parameter by the model's organ validator. A distribution is checked here against its declaration, since the
     * code reading it sees only one sample at a time.
     * \param[in] plant_model_name Name of the library plant model.
     * \param[in] build_parameters Map of build parameter names to distributions.
     * \param[in] calling_function Name of the calling function, quoted in error messages.
     * \return The organ parameters among the entries of build_parameters.
     */
    [[nodiscard]] std::map<std::string, BuildParameterDistribution> validateBuildParameters(const std::string &plant_model_name, const std::map<std::string, BuildParameterDistribution> &build_parameters,
                                                                                            const std::string &calling_function) const;

    //! Initialize all plant model registrations
    void initializePlantModelRegistrations();

    void parseStringShoot(const std::string &LString_shoot, uint plantID, int parentID, uint parent_node, const std::map<std::string, PhytomerParameters> &phytomer_parameters, ShootParameters &shoot_parameters);

    void parseShootArgument(const std::string &shoot_argument, const std::map<std::string, PhytomerParameters> &phytomer_parameters, ShootParameters &shoot_parameters, AxisRotation &base_rotation, std::string &phytomer_label);

    void parseInternodeArgument(const std::string &internode_argument, float &internode_radius, float &internode_length, PhytomerParameters &phytomer_parameters);

    void parsePetioleArgument(const std::string &petiole_argument, PhytomerParameters &phytomer_parameters);

    void parseLeafArgument(const std::string &leaf_argument, PhytomerParameters &phytomer_parameters);

    void initializeDefaultShoots(const std::string &plant_label);

    [[nodiscard]] bool detectGroundCollision(uint objID);

    [[nodiscard]] bool detectGroundCollision(const std::vector<uint> &objID) const;

    //! Throws if the given plant does not exist or the shoot ID is out of range for it
    /**
     * Shared by the shoot topology accessors so that they all report a bad ID the same way.
     * \param[in] plantID Plant to validate.
     * \param[in] shootID Shoot to validate within that plant.
     * \param[in] function_name Name of the calling function, used in the error message.
     */
    void validateShootID(uint plantID, uint shootID, const std::string &function_name) const;

    //! Resolve a joint identifier to the shoot and phytomer it names, raising an error that names the calling function if the joint does not exist
    /**
     * \param[in] plantID ID of the plant instance.
     * \param[in] joint Identifier of the joint.
     * \param[in] function_name Name of the calling function, for the error message.
     * \return Phytomer the joint belongs to, or nullptr for PLANT_JOINT_ROOT and PLANT_JOINT_SHOOT, which belong to no single phytomer.
     */
    Phytomer *validatePlantJoint(uint plantID, const PlantJointID &joint, const std::string &function_name) const;

    //! Call a function for every rigid part of a plant, in an order that depends only on the plant's structure
    /**
     * \param[in] plant_instance Plant whose parts are visited.
     * \param[in,out] pose_state Pose of the plant.
     * \param[in] create_missing If true, entries of \p pose_state are created to match the plant's structure. If false, they must already match.
     * \param[in] callback Function called for each part.
     * \return False, without having called the callback for every part, if \p create_missing is false and the plant's structure does not match \p pose_state.
     */
    bool forEachPoseGroup(PlantInstance &plant_instance, PlantPoseState &pose_state, bool create_missing, const std::function<void(PoseGroupReference &)> &callback) const;

    //! Record a plant as it stands, as what its pose is measured from
    void capturePlantRestPose(PlantInstance &plant_instance) const;

    //! Fold changes made to a posed plant by anything other than the pose into what the pose is measured from
    /**
     * \return False if the plant's structure has changed since the pose was recorded, in which case nothing can be folded in and the pose has to be made permanent as it stands.
     */
    bool rebindPlantRestPose(PlantInstance &plant_instance) const;

    //! Motion of each rigid part of a shoot, and of every shoot growing from it, under the joint poses that are set. The plant's stored geometry must be as the plant grew.
    void computeShootPoseTransforms(uint plantID, uint shootID, const PoseRigidTransform &carrier_transform) const;

    //! Put a plant's stored geometry back as the plant grew, and work out the motion of each of its rigid parts under the joint poses that are set
    void computePlantPoseTransforms(uint plantID);

    //! Get a plant ready for a pose to be set: keep what else has been done to it since it was last posed, and record it as it grew if that has not been done
    void preparePlantForPosing(uint plantID);

    //! Forget the record of a plant as it grew, once every joint is back at zero and the plant stands exactly as recorded
    void releasePlantPoseState(PlantInstance &plant_instance) const;

    //! Move a plant to the joint poses that are set, or, if they cannot be applied, put the previous poses back and raise the error again
    /**
     * \param[in] plantID ID of the plant instance.
     * \param[in] previous_joint_poses Joint poses the plant had before the ones now set, which are known to apply.
     */
    void evaluatePlantPoseOrRestore(uint plantID, const std::map<PlantJointID, PlantJointPose> &previous_joint_poses);

    //! Move a plant to the joint poses that are set
    void evaluatePlantPose(uint plantID);

    //! Carry a translation of a whole posed plant into what its pose is measured from (see setPlantBasePosition())
    void translatePlantRestPose(PlantInstance &plant_instance, const helios::vec3 &shift) const;

    //! Raise an error naming the calling function if a plant is posed, or has had a pose made permanent, for functions that rebuild a plant from its growth parameters alone
    void rejectPosedPlant(uint plantID, const std::string &function_name) const;

    //! Shared implementation of the leaf angle distribution tracking setters
    /**
     * \param[in] plantID Identifier for the plant.
     * \param[in] Beta_mu_inclination Mean value parameter for the Beta distribution of inclination angles.
     * \param[in] Beta_nu_inclination Shape parameter for the Beta distribution of inclination angles.
     * \param[in] eccentricity Eccentricity value for the ellipse defining the azimuth distribution.
     * \param[in] ellipse_rotation_degrees Rotation angle of the ellipse in degrees.
     * \param[in] lambda_degrees How strongly to favour filling the distribution over keeping each leaf near the angle the model gave it.
     * \param[in] track_elevation Whether to steer leaf inclination.
     * \param[in] track_azimuth Whether to steer leaf azimuth.
     */
    //! Give every leaf that emerged this growth sub-step a target angle, and turn the expanding ones toward the targets they already have
    /**
     * Called once per plant per sub-step of PlantArchitecture::advanceTime(), after the sub-step's new
     * phytomers have been created. Does nothing for a plant that is not being tracked.
     * \param[in] plantID Identifier for the plant.
     */
    void updatePlantLeafAngleDistributionTracking(uint plantID);

    void enablePlantLeafAngleDistributionTracking_private(uint plantID, float Beta_mu_inclination, float Beta_nu_inclination, float eccentricity, float ellipse_rotation_degrees, float lambda_degrees, bool track_elevation, bool track_azimuth);

    void setPlantLeafAngleDistribution_private(const std::vector<uint> &plantIDs, float Beta_mu_inclination, float Beta_nu_inclination, float eccentricity_azimuth, float ellipse_rotation_azimuth_degrees, bool set_elevation, bool set_azimuth) const;

    //! Collect leaf object IDs together with their matching attachment base positions
    /**
     * Gathers both values in a single traversal of the shoot tree so that the object ID at a given
     * index and the base position at the same index always refer to the same leaf. Callers that
     * need the two in correspondence must use this rather than pairing the results of
     * getPlantLeafObjectIDs() and getPlantLeafBases(), which are built by independent traversals
     * and would silently desynchronize if either ever started filtering.
     *
     * \param[in] plantIDs List of IDs of the plant instances.
     * \param[out] leaf_objIDs Object IDs of the leaves.
     * \param[out] leaf_bases Attachment base position of each leaf, index-matched to leaf_objIDs.
     */
    void getPlantLeafObjectIDsAndBases(const std::vector<uint> &plantIDs, std::vector<uint> &leaf_objIDs, std::vector<helios::vec3> &leaf_bases) const;

    static float interpolateTube(const std::vector<float> &P, float frac);

    static helios::vec3 interpolateTube(const std::vector<helios::vec3> &P, float frac);

    //! Names of additional object data to add to the Context
    std::map<std::string, bool> output_object_data;

    // --- Plant Growth --- //

    void incrementPhytomerInternodeGirth(uint plantID, uint shootID, uint node_number, float dt, bool update_context_geometry);
    //! Increment an internode's girth toward its pipe-model target, limited by the shoot's carbohydrate (carbohydrate model enabled, CARBON_SUBMODEL_TREE)
    /**
     * \param[in] plantID ID of the plant
     * \param[in] shootID ID of the shoot
     * \param[in] node_number Index of the internode within the shoot
     * \param[in] dt Length of the sub-step (days)
     * \param[in] update_context_geometry If true, the shoot's tube object radii are updated in the Context
     * \param[in] shoot_length Length of the shoot (m), computed once per shoot per sub-step by the caller
     * \param[in] shoot_volume Internode volume of the shoot (m^3), computed once per shoot per sub-step by the caller
     */
    void incrementPhytomerInternodeGirth_carb(uint plantID, uint shootID, uint node_number, float dt, bool update_context_geometry, float shoot_length, float shoot_volume);

    //! Mark every organ of a shoot as already paid for, so that the carbohydrate model charges only growth that happens from now on
    /**
     * \param[in] shoot Shoot whose internodes, leaves and fruit are marked as charged
     */
    void markShootConstructionCharged(Shoot &shoot) const;

    //! Rebuild the shadow-propagation voxel grid for a plant from its current leaf positions
    /** Each leaf stamps a downward-widening pyramid of shadow into the grid, following Palubicki et al. (2009). This is a
        cheap surrogate for light interception -- an additive scatter into a sparse array, with no ray tracing, visibility
        queries or radiative solve. The voxel size is derived from the plant's current crown extent so that the grid
        resolution tracks the plant as it grows.
        \param[in] plantID Identifier of the plant.
    */
    void updateShadowGrid(uint plantID);

    //! Local light exposure Q at a point, from a plant's shadow grid
    /** Combines two signals: the vertical shadow accumulated from the canopy above the point, and how deep the point sits
        beneath the crown surface. The first alone treats a low outer branch the same as a buried interior one; the second
        supplies the lateral (diffuse side light) component that distinguishes them.
        \param[in] grid Shadow grid to query.
        \param[in] depth_grid Companion crown-depth field, giving each occupied voxel's depth beneath the crown surface.
        \param[in] voxel_size Edge length (m) of the grid voxels.
        \param[in] grid_origin World position the grid's voxel keys are measured from (PlantInstance::shadow_grid_origin).
        \param[in] position Query position in world coordinates.
        \return Light exposure in [0, C]; the unshaded value C when the grid is empty.
    */
    [[nodiscard]] float shadowGridLightExposure(const std::unordered_map<long long, float> &grid, const std::unordered_map<long long, int> &depth_grid, float voxel_size, const helios::vec3 &grid_origin, const helios::vec3 &position) const;

    //! Record the position, area and bearing shoot of every leaf on a plant, as the canopy branch shedding will judge
    /** Called when the plant's leaf area reaches a new maximum for the season, so that what is kept is the season's peak canopy.
        \param[in] plantID Identifier of the plant.
    */
    void captureSeasonPeakLeaves(uint plantID);

    //! Shed branches that are too shaded to pay for themselves
    /** Self-pruning. Parameter-based growth has no branch mortality of its own, so without this every twig
        ever produced survives indefinitely and the branching hierarchy carries far too fat a tail. Evaluated
        at the dormancy boundary, when a branch has completed a full season in leaf, on the leaves of the season's
        peak canopy. A branch is judged by the mean light reaching all the foliage it supports, relative to the mean
        over the whole plant. Structural wood is never shed -- see \ref isStructuralShoot().
        \param[in] plantID ID of the plant to prune.
    */
    void shedShadedBranches(uint plantID);

    void pruneGroundCollisions(uint plantID);

    void pruneSolidBoundaryCollisions();

    // --- Carbohydrate Model --- //
    // Each of these operates on a single plant. They are called from inside advanceTime()'s per-plant
    // loop, so iterating over every plant instance here would apply the carbon balance N_plants times
    // per sub-step and let advancing one plant alter the pools of every other plant.

    //! Move the photosynthate accumulated on a plant's leaves into its shoot sugar pools
    /**
     * \param[in] plantID ID of the plant whose shoots are credited
     */
    void accumulateShootPhotosynthesis(uint plantID);

    //! Debit stem and root maintenance respiration from each shoot of a plant
    /**
     * \param[in] plantID ID of the plant whose shoots are debited
     * \param[in] dt Length of the sub-step (days)
     */
    void subtractShootMaintenanceCarbon(uint plantID, float dt);

    //! Debit the structural growth and growth respiration of each shoot's stem, leaves and roots from the plant's pools, then the reproductive growth of its sub-model
    /**
     * The stem, leaf and root charges are shared by every sub-model. The reproductive sink is then charged by the plant's CarbonSubmodel (for CARBON_SUBMODEL_TREE, subtractFruitGrowthCarbon()),
     * which is told what the sub-step has spent from reserves so far.
     * \param[in] plantID ID of the plant whose shoots are debited
     * \param[in] dt Length of the sub-step (calendar days)
     */
    void subtractShootGrowthCarbon(uint plantID, float dt);

    //! Abort floral buds and prune shoots of a plant that cannot sustain their carbon balance
    /**
     * \param[in] plantID ID of the plant to check
     * \param[in] dt Length of the sub-step (days)
     */
    void checkCarbonPool_abortOrgans(uint plantID, float dt);
    //! Dry mass (g DW) weighted by relative storage capacity: every living shoot's stem, plus root_storage_ratio times the root dry mass
    [[nodiscard]] float plantStorageCapacityMass(uint plantID) const;
    //! Slow the phyllochron, internode elongation and leaf expansion of each shoot of a plant according to its carbon status
    /**
     * Below CarbohydrateParameters::carbohydrate_growth_threshold, elongation and leaf expansion are scaled from the shoot's potential rates in
     * proportion to its carbohydrate concentration, so that a plant running short of carbohydrate slows its demand.
     * \param[in] plantID ID of the plant to check
     * \param[in] dt Length of the sub-step (days)
     */
    void checkCarbonPool_adjustPhyllochron(uint plantID, float dt);
    //! Move carbohydrate between a plant's shoots and roots and between sugar and starch
    /**
     * Phloem transport relaxes every compartment (each living shoot and the root system) toward a common relative fill,
     * the plant's carbohydrate per unit of storage capacity, at carbon_mixing_rate. Any compartment still in deficit is then
     * brought up to that mean fill from the others. Carbohydrate above the plant's storage capacity is released below ground (PlantCarbonBudget::storage_overflow), and sugar
     * and starch in each compartment relax toward a temperature-dependent partition.
     * \param[in] plantID ID of the plant to redistribute
     * \param[in] dt Length of the sub-step (days)
     */
    void checkCarbonPool_redistributeCarbon(uint plantID, float dt);

    //! Share the photosynthate left after maintenance, vegetative growth and fruit between storage and radial growth, and forgo any beyond both
    /**
     * The remainder is the net photosynthesis credited for the sub-step less everything charged in it since budget_before, with the sub-step's radial growth added back. Storage demands
     * storage_refill_rate x dt x (capacity - pool before the remainder) and radial growth its potential cost; each receives the remainder in proportion to its demand, up to that demand, which
     * sets the cambium supply ratio for the following sub-steps. Photosynthate beyond both demands is withdrawn from the pools as forgone photosynthesis.
     * \param[in] plantID ID of the plant
     * \param[in] dt Length of the sub-step (days)
     * \param[in] budget_before The plant's carbon budget at the start of the sub-step
     */
    void allocateSurplusCarbon(uint plantID, float dt, const PlantCarbonBudget &budget_before);

    //! Check that a plant's carbohydrate parameters name a sub-model this build implements, and that the parameters of that sub-model are consistent
    /**
     * \param[in] carb_parameters Parameters to check
     * \param[in] caller Name of the calling function, for the error message
     */
    static void validateCarbonSubmodel(const CarbohydrateParameters &carb_parameters, const char *caller);

    //! Record a sub-step's photosynthate available to the reproductive sinks built from current photosynthate and their potential growth, and return the supply ratio averaged over supply_averaging_days
    /**
     * \param[in] plantID ID of the plant
     * \param[in] supply_molC Photosynthate left for those sinks this sub-step (mol C)
     * \param[in] demand_molC Their potential growth this sub-step, with growth respiration (mol C)
     * \param[in] dt Length of the sub-step (days)
     * \return Supply relative to demand over the averaging period, capped at 1; 1 when there has been no demand
     */
    float recordReproductiveSupply(uint plantID, float supply_molC, float demand_molC, float dt);

    // --- Carbohydrate model: CARBON_SUBMODEL_TREE (CarbohydrateModelTree.cpp) --- //

    //! Charge the growth of a plant's fruit along the dry-matter curve of CarbonFruitParameters
    /**
     * Fruit up to CarbonFruitParameters::reserve_funded_days after set grows at its potential from the pools. Older fruit is built from current photosynthate: it grows at its potential times
     * the fruit supply ratio, the photosynthate left after maintenance respiration and the reserve-funded spending of the sub-step relative to that fruit's potential growth, averaged over
     * CarbohydrateParameters::supply_averaging_days.
     * \param[in] plantID ID of the plant
     * \param[in] dt Length of the sub-step (days)
     * \param[in] reserve_funded_spending_molC Stem, leaf and root growth (with growth respiration) already charged to the pools this sub-step (mol C)
     */
    void subtractFruitGrowthCarbon(uint plantID, float dt, double reserve_funded_spending_molC);
    //! Drop young fruit that current photosynthate cannot supply
    /**
     * Fruit past its reserve-funded period and up to CarbonFruitParameters::drop_window_end days after set drops while the growth it has received since set is below
     * CarbonFruitParameters::abscission_growth_ratio of its potential, most recently set first, until the remaining young fruit can be fed.
     * \param[in] plantID ID of the plant
     */
    void checkCarbonPool_dropFruit(uint plantID);

    // --- Carbohydrate model: CARBON_SUBMODEL_GRASS (CarbohydrateModelGrass.cpp) --- //

    //! Charge the growth of a plant's panicles and grain (CarbonGrassParameters)
    /**
     * For each culm with a panicle: during the critical period the panicle structure grows at its potential times the reproductive supply ratio (the panicles' share, by demand, of the
     * photosynthate left after maintenance respiration and leaf and root growth, which they divide with the growing stems, averaged over supply_averaging_days); at the end of the critical period the grain number is
     * fixed from the panicle carbon; during filling the grain grows at its potential, paid from the sub-step's remaining photosynthate first and then from the carbohydrate the culms and roots hold
     * above CarbonGrassParameters::culm_reserve_floor, the latter reported as PlantCarbonBudget::reserve_remobilization.
     * \param[in] plantID ID of the plant
     * \param[in] dt Length of the sub-step (calendar days); each panicle's clock has already been advanced by its developmental days
     * \param[in] reserve_funded_spending_molC Stem, leaf and root growth (with growth respiration) already charged to the pools this sub-step (mol C)
     * \param[in] stem_spending_molC The stem growth (with growth respiration) within reserve_funded_spending_molC (mol C)
     */
    void subtractGrainGrowthCarbon(uint plantID, float dt, double reserve_funded_spending_molC, double stem_spending_molC);
    //! Lay surplus photosynthate down as stem in the culms of a grass that have not yet reached anthesis, thickening their internodes up to CarbonGrassParameters::culm_thickening_max times their pipe-model radius
    /**
     * \param[in] plantID ID of the plant
     * \param[in] surplus_molC Photosynthate with no other sink this sub-step (mol C)
     * \return The part of the surplus spent, growth respiration included (mol C)
     */
    float growGrassCulmsFromSurplus(uint plantID, float surplus_molC);
    //! Radius the pipe model gives an internode: that of the cross-section its shoot type's girth_area_factor allots to the leaf area, and the inflorescence area borne on the stem's axis, above it
    /**
     * \param[in] shoot Shoot of the internode
     * \param[in] node_number Index of the internode along the shoot
     * \return Pipe-model radius (m)
     */
    [[nodiscard]] static float pipeModelInternodeRadius(const Shoot &shoot, uint node_number);
    //! Run the grass sub-model's shoot economy for a sub-step: note the onset of stem elongation and shed tillers that do not pay for themselves while the stem elongates
    /**
     * \param[in] plantID ID of the plant
     */
    void updateGrassTillers(uint plantID);
    //! Shed the leaves of culms past physiological maturity under the grass sub-model, oldest first, at an even rate over CarbonGrassParameters::post_maturity_senescence_days
    /**
     * \param[in] plantID ID of the plant
     */
    void senesceMatureGrassCulms(uint plantID);
    //! Factor by which the grass sub-model scales the probability that a tiller bud breaks, from the plant's carbon supply/demand index (CarbonGrassParameters::tiller_supply_demand_min)
    /**
     * \param[in] plantID ID of the plant
     * \return A factor between 0 and 1; 1 while no vegetative demand has been recorded or once stem elongation has begun (no further buds are sampled then)
     */
    [[nodiscard]] float grassTillerBudBreakFactor(uint plantID) const;
    //! Fraction of the main stem's development from emergence (0) to flag leaf (1), from its node count (CARBON_SUBMODEL_GRASS)
    [[nodiscard]] float grassMainStemDevelopmentFraction(uint plantID) const;
    //! Panicle age (developmental days since the flag leaf, the panicle's FloralBud::time_counter) at anthesis under CARBON_SUBMODEL_GRASS: the plant's time to fruit set
    [[nodiscard]] static float grassAnthesisAge(const PlantInstance &plant_instance);
    //! Panicle age at physiological maturity under CARBON_SUBMODEL_GRASS: the plant's time to fruit maturity
    [[nodiscard]] static float grassMaturityAge(const PlantInstance &plant_instance);
    //! Whether a floral bud is a culm's panicle under CARBON_SUBMODEL_GRASS: a fruiting terminal bud, which a culm bears from its flag leaf
    [[nodiscard]] static bool isGrassPanicle(const FloralBud &fbud);
    //! Panicle age of a culm under CARBON_SUBMODEL_GRASS (developmental days since its flag leaf), or -1 while it has no panicle
    [[nodiscard]] static float grassCulmPanicleAge(const Shoot &shoot);
    //! Whether a culm under CARBON_SUBMODEL_GRASS has died back after maturity: its panicle has passed physiological maturity plus CarbonGrassParameters::post_maturity_senescence_days. A dead
    //! culm no longer respires, and neither it nor its grain is subject to the starvation rules.
    [[nodiscard]] static bool isSenescedGrassCulm(const PlantInstance &plant_instance, const Shoot &shoot);
    //! Structural carbon of a plant's root system (mol C): the wood's divided by shoot_root_ratio under CARBON_SUBMODEL_TREE, or the carbon grown by allocation under CARBON_SUBMODEL_GRASS
    [[nodiscard]] float plantRootStructuralCarbon(uint plantID) const;
    //! Carbon (with growth respiration) a floral bud's reproductive organ would still need to reach its potential, which aborting it saves: the fruit's remaining dry matter under
    //! CARBON_SUBMODEL_TREE, the panicle's and grain's remaining growth under CARBON_SUBMODEL_GRASS
    [[nodiscard]] float remainingReproductiveCarbon(const Phytomer &phytomer, const FloralBud &fbud) const;

    bool carbon_model_enabled = false;

    //! True while buildPlantInstanceFromLibrary() or buildPlantCanopyFromLibrary() grows plants to their build age, which runs at the reference temperature with no thermal time supplied
    bool growing_to_build_age = false;

    // --- Nitrogen Model --- //

    // Each of these acts only on the plants given, which are the plants being advanced by advanceTime()

    //! Return nitrogen from senescing leaves to the plant's available pool, starting senescence in leaves that have reached its onset
    void senesceLeafNitrogen(const std::vector<uint> &plantIDs, float dt);
    void accumulateLeafNitrogen(const std::vector<uint> &plantIDs, float dt);
    void remobilizeNitrogen(const std::vector<uint> &plantIDs, float dt);
    void updateNitrogenStressFactor(const std::vector<uint> &plantIDs);
    void removeFruitNitrogen(const std::vector<uint> &plantIDs);
    //! Return what remains of the resorbable nitrogen of a phytomer's senescing leaves to the plant's available pool before the leaves are shed
    void resorbLeafNitrogen(PlantInstance &plant, Shoot &shoot, const Phytomer &phytomer);

    bool nitrogen_model_enabled = false;

    // --- Collision Detection --- //

    //! Internal collision detection instance (owned by PlantArchitecture)
    CollisionDetection *collision_detection_ptr = nullptr;

    //! Flag to track if we own the CollisionDetection instance (for cleanup)
    bool owns_collision_detection = false;

    //! Flag indicating if collision detection is enabled
    bool collision_detection_enabled = false;

    //! Target UUIDs to avoid during growth
    std::vector<uint> collision_target_UUIDs;

    //! Target object IDs to avoid during growth
    std::vector<uint> collision_target_object_IDs;

    //! Collision detection cone half-angle in radians
    float collision_cone_half_angle_rad = 80.f * M_PI / 180.f;

    //! Collision detection cone height in meters
    float collision_cone_height = 0.1f;

    //! Number of directional samples within collision detection cone
    int collision_sample_count = 256;

    //! Inertia weight for balancing directional preference vs collision avoidance
    float collision_inertia_weight = 0.4f;

    //! Frequency of Context geometry updates (1=every timestep, 2=every 2 timesteps, etc.)
    int geometry_update_frequency = 3;

    //! Force Context geometry update when collision avoidance is triggered
    bool force_update_on_collision = true;

    //! Organ filtering configuration for collision detection
    bool collision_include_internodes = false;
    bool collision_include_leaves = true;
    bool collision_include_petioles = false;
    bool collision_include_flowers = false;
    bool collision_include_fruit = false;

    //! Enable petiole collision detection (separate from general petiole inclusion)
    bool petiole_collision_detection_enabled = false;

    //! Enable fruit collision detection (separate from general fruit inclusion)
    bool fruit_collision_detection_enabled = false;

    //! Counter to track timesteps for geometry update scheduling
    int geometry_update_counter = 0;

    //! Flag to track if collision avoidance was applied in current timestep
    mutable bool collision_avoidance_applied = false;

    //! Enable spatial filtering for collision detection performance
    bool spatial_filtering_enabled = false;

    //! Maximum distance for collision detection queries (meters)
    float spatial_max_distance = 5.0f;

    //! BVH caching for within-timestep efficiency
    mutable bool bvh_cached_for_current_growth = false;
    mutable std::vector<uint> cached_target_geometry;
    mutable std::vector<uint> cached_filtered_geometry;

    //! Solid obstacle avoidance parameters
    bool solid_obstacle_avoidance_enabled = false;
    std::vector<uint> solid_obstacle_UUIDs;
    float solid_obstacle_avoidance_distance = 0.5f;
    float solid_obstacle_minimum_distance = 0.05f;
    bool solid_obstacle_fruit_adjustment_enabled = false;
    bool solid_obstacle_pruning_enabled = false;

    // --- Fruit-Branch Collision Avoidance --- //

    //! Whether fruit are swung out of the internodes of their own plant each time step (see enableFruitBranchCollisionAvoidance())
    bool fruit_branch_collision_enabled = false;
    //! Largest total swing (radians) of a peduncle out of its grown pose
    float fruit_branch_collision_max_swing_rad = 0.5f * PI_F;
    //! Days between rebuilds of the candidate internode lists
    float fruit_branch_collision_refresh_days = 10.f;
    //! Gap (meters) kept between fruit and branch surfaces
    float fruit_branch_collision_clearance = 0.f;
    //! Rate (radians per day) at which swings and tilts relax back toward the grown pose while the fruit stays clear
    float fruit_branch_collision_settle_rad_per_day = 10.f * PI_F / 180.f;
    //! Fruit meshes with more primitives than this are tested through their sphere chain rather than their triangles
    static constexpr size_t fruit_contact_exact_primitive_limit = 1000;
    //! Contact proxies of the fruit, keyed by fruit object ID; entries whose object is gone are dropped at the next check
    std::unordered_map<uint, FruitContactProxy> fruit_contact_proxies;
    //! Candidate internode segments per fruiting bud, keyed by the bud's first fruit object ID
    std::unordered_map<uint, FruitContactCandidates> fruit_contact_candidates;

    //! Build the contact proxy of one fruit object from its mesh
    /**
     * \param[in] fbud Floral bud carrying the fruit
     * \param[in] fruit_index Index of the fruit within fbud.inflorescence_objIDs
     * 
eturn Proxy in the frame of the fruit's attachment point
     */
    [[nodiscard]] FruitContactProxy buildFruitContactProxy(const FloralBud &fbud, size_t fruit_index) const;

    //! Collect the internode segments of a plant that lie within reach of a fruiting bud
    /**
     * \param[in] plantID Plant the bud belongs to
     * \param[in] phytomer Phytomer carrying the bud
     * \param[in] fbud The fruiting bud
     * \param[in] reach Distance (meters) from fbud.base_position within which a segment's capsule is kept
     * \param[in] shoot_lower Per-shoot lower corners of the bounding boxes of the internode capsules, indexed by shoot ID
     * \param[in] shoot_upper Per-shoot upper corners of the same boxes
     * 
eturn Candidate list with its refresh age and shoot count filled in
     */
    [[nodiscard]] FruitContactCandidates collectFruitContactCandidates(uint plantID, const Phytomer &phytomer, const FloralBud &fbud, float reach, const std::vector<helios::vec3> &shoot_lower, const std::vector<helios::vec3> &shoot_upper) const;

    //! Swing the fruiting buds of a plant out of the plant's internodes, and let those already clear settle back toward hanging (one time step of fruit-branch collision avoidance)
    /**
     * \param[in] plantID Plant to process
     * \param[in] dt_days Length of the time step (days), which scales the settling
     */
    void resolveFruitBranchCollisions(uint plantID, float dt_days);

    //! Drop the fruit-branch contact proxies and candidate lists of fruit objects that no longer exist
    void pruneFruitContactCaches();

    // --- Attraction Points --- //

    //! Flag indicating if attraction points are enabled
    bool attraction_points_enabled = false;

    //! Vector of attraction point positions
    std::vector<helios::vec3> attraction_points;

    //! Attraction detection cone half-angle in radians
    float attraction_cone_half_angle_rad = 80.f * M_PI / 180.f;

    //! Attraction detection cone height in meters
    float attraction_cone_height = 0.1f;

    //! Weight factor for attraction vs natural growth (0.0 = ignore attraction, 1.0 = full attraction)
    float attraction_weight = 0.6f;

    //! Reduction factor for attraction weight when hard obstacles are present (0.0 = no attraction, 1.0 = full attraction)
    float attraction_obstacle_reduction_factor = 0.5f;

    //! Flag to enable/disable console output messages
    bool printmessages = true;

    // --- Growth Animation Frame Storage --- //

    //! Storage for growth animation frames registered via registerGrowthFrame()
    PlantGrowthAnimationStorage growth_animation_storage;

    // --- Plant Library --- //

    void initializeAlmondTreeShoots();

    uint buildAlmondTree(const helios::vec3 &base_position);

    void initializeAlmondTreeAldrichShoots();

    uint buildAlmondTreeAldrich(const helios::vec3 &base_position);

    void initializeAlmondTreeIndependenceShoots();

    uint buildAlmondTreeIndependence(const helios::vec3 &base_position);

    void initializeAlmondTreeWoodColonyShoots();

    uint buildAlmondTreeWoodColony(const helios::vec3 &base_position);

    void initializeAppleTreeShoots();

    uint buildAppleTree(const helios::vec3 &base_position);

    //! Define the shoot types of the "apple_fruitingwall" model: a tall-spindle apple fruiting wall trained from its phytomer callback.
    void initializeAppleFruitingWallShoots();

    //! Build the "apple_fruitingwall" model as a feathered nursery tree, which the training callback grows into a wall over about five seasons.
    uint buildAppleFruitingWall(const helios::vec3 &base_position);

    void initializeAsparagusShoots();

    uint buildAsparagusPlant(const helios::vec3 &base_position);

    void initializeBindweedShoots();

    uint buildBindweedPlant(const helios::vec3 &base_position);

    void initializeBeanShoots();

    uint buildBeanPlant(const helios::vec3 &base_position);

    void initializeBougainvilleaShoots();

    uint buildBougainvilleaPlant(const helios::vec3 &base_position);

    void initializeCapsicumShoots();

    uint buildCapsicumPlant(const helios::vec3 &base_position);

    void initializeCheeseweedShoots();

    uint buildCheeseweedPlant(const helios::vec3 &base_position);

    void initializeCowpeaShoots();

    uint buildCowpeaPlant(const helios::vec3 &base_position);

    void initializeGrapevineVSPShoots();

    uint buildGrapevineVSP(const helios::vec3 &base_position);

    void initializeGrapevineWyeShoots();

    uint buildGrapevineWye(const helios::vec3 &base_position);

    void initializeGroundCherryWeedShoots();

    uint buildGroundCherryWeedPlant(const helios::vec3 &base_position);

    void initializeMaizeShoots();

    uint buildMaizePlant(const helios::vec3 &base_position);

    void initializeOliveTreeShoots();

    uint buildOliveTree(const helios::vec3 &base_position);

    void initializePistachioTreeShoots();

    uint buildPistachioTree(const helios::vec3 &base_position);

    void initializePuncturevineShoots();

    uint buildPuncturevinePlant(const helios::vec3 &base_position);

    void initializeEasternRedbudShoots();

    uint buildEasternRedbudPlant(const helios::vec3 &base_position);

    void initializeRiceShoots();

    uint buildRicePlant(const helios::vec3 &base_position);

    void initializeButterLettuceShoots();

    uint buildButterLettucePlant(const helios::vec3 &base_position);

    void initializeSoybeanShoots();

    uint buildSoybeanPlant(const helios::vec3 &base_position);

    void initializeSorghumShoots();

    uint buildSorghumPlant(const helios::vec3 &base_position);

    void initializeStrawberryShoots();

    uint buildStrawberryPlant(const helios::vec3 &base_position);

    void initializeSugarbeetShoots();

    uint buildSugarbeetPlant(const helios::vec3 &base_position);

    void initializeTomatoShoots();

    uint buildTomatoPlant(const helios::vec3 &base_position);

    void initializeCherryTomatoShoots();

    uint buildCherryTomatoPlant(const helios::vec3 &base_position);

    void initializeWalnutTreeShoots();

    uint buildWalnutTree(const helios::vec3 &base_position);

    void initializeWheatShoots();

    uint buildWheatPlant(const helios::vec3 &base_position);

    void initializeCucumberShoots();

    uint buildCucumberPlant(const helios::vec3 &base_position);
};

#include "Assets.h"

#endif // PLANT_ARCHITECTURE
