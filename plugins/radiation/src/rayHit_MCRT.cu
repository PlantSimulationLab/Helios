#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>

#include "RayTracing.cu.h"

using namespace optix;

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PerRayData_MCRT, prd, rtPayload, );
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(unsigned int, direct_ray_type_MCRT, , );
rtDeclareVariable(unsigned int, diffuse_ray_type_MCRT, , );
rtDeclareVariable(unsigned int, emission_ray_type_MCRT, , );

rtDeclareVariable(unsigned int, max_scatters, , );

rtDeclareVariable(unsigned int, UUID, attribute UUID, );

rtBuffer<float, 1> rho, tau, eps;

rtBuffer<unsigned int, 1> primitive_type;

rtBuffer<float, 1> radiation_in;
rtBuffer<bool, 1> twosided_flag;

rtBuffer<float, 1> Rsky;

rtBuffer<float, 2> transform_matrix;

rtDeclareVariable(float2, bound_box_x, , );
rtDeclareVariable(float2, bound_box_y, , );
rtDeclareVariable(float2, bound_box_z, , );

RT_PROGRAM void closest_hit_MCRT() {
    // method #2//
    //  if( prd.boundary_hits>=20 ){
    //    atomicAdd( &Rsky[UUID], prd.strength );
    //    return;
    //  }
    // method #2//

    float m[16];
    for (uint i = 0; i < 16; i++) {
        m[i] = transform_matrix[optix::make_uint2(i, UUID)];
    }

    float3 normal = make_float3(0, 0, 1);
    if (primitive_type[UUID] == 0 || primitive_type[UUID] == 3) {
        float3 s0 = make_float3(0, 0, 0);
        float3 s1 = make_float3(1, 0, 0);
        float3 s2 = make_float3(0, 1, 0);
        s0 = d_transformPoint(m, s0);
        s1 = d_transformPoint(m, s1);
        s2 = d_transformPoint(m, s2);
        normal = cross(s1 - s0, s2 - s0);
    } else if (primitive_type[UUID] == 1) {
        float3 v0 = d_transformPoint(m, make_float3(0, 0, 0));
        float3 v1 = d_transformPoint(m, make_float3(0, 1, 0));
        float3 v2 = d_transformPoint(m, make_float3(1, 1, 0));
        normal = cross(v1 - v0, v2 - v1);
    } else if (primitive_type[UUID] == 2) {
        float3 v0 = d_transformPoint(m, make_float3(0, 0, 0));
        float3 v1 = d_transformPoint(m, make_float3(1, 0, 0));
        float3 v2 = d_transformPoint(m, make_float3(0, 1, 0));
        normal = cross(v1 - v0, v2 - v0);
    } else {
    }
    normal = normalize(normal);

    // if primitive is not "two-sided", reject if hit is on back side
    if (twosided_flag[UUID] == 0) {
        if (dot(normal, ray.direction) > 0) {
            atomicAdd(&Rsky[UUID], prd.strength);
            return;
        }
    }

    float t_rho = rho[UUID];
    float t_tau = tau[UUID];

    // random number to determine absorption/reflection/transmission
    //  float R = rnd(prd.seed);

    // if( R<t_rho && prd.scatter_depth<max_scatters ){//reflection
    //   //if( prd.strength>1e-8 ){ //launch reflection ray

    //     float Rt = rnd(prd.seed);
    //     float Rp = rnd(prd.seed);
    //     float t = asin_safe(sqrtf(Rt));
    //     float p = 2.f*M_PI*Rp;

    //     float3 ray_direction;
    //     ray_direction.x = sin(t)*cos(p);
    //     ray_direction.y = sin(t)*sin(p);
    //     ray_direction.z = cos(t);

    //     ray_direction = d_rotatePoint( ray_direction, acos_safe(normal.z), atan2(normal.y,normal.x) );

    //     float3 ray_origin = ray.origin + t_hit*ray.direction;

    //     //rtPrintf("Launching reflection ray from prim #%d with strength %f, position (%f,%f,%f), direction
    //     (%f,%f,%f)\n",UUID,prd.strength,ray_origin.x,ray_origin.y,ray_origin.z,ray_direction.x,ray_direction.y,ray_direction.z);

    //     optix::Ray ray_reflect = optix::make_Ray(ray_origin, ray_direction, ray.ray_type, 1e-4, RT_DEFAULT_MAX);

    //     PerRayData_MCRT prd_reflect = prd;
    //     prd_reflect.scatter_depth ++;

    //     rtTrace( top_object, ray_reflect, prd_reflect);

    //     //}else{ //deposit what's left
    //     //atomicAdd( &radiation_in[UUID], prd.strength );
    //     //}
    // }else if( R<t_rho+t_tau  && prd.scatter_depth<max_scatters ){//transmission
    //   //if( prd.strength>1e-4 ){ //launch transmission ray

    //     float Rt = rnd(prd.seed);
    //     float Rp = rnd(prd.seed);
    //     float t = acos_safe(1.f-Rt);
    //     float p = 2.f*M_PI*Rp;

    //     float3 ray_direction;
    //     ray_direction.x = sin(t)*cos(p);
    //     ray_direction.y = sin(t)*sin(p);
    //     ray_direction.z = cos(t);

    //     ray_direction = d_rotatePoint( ray_direction, acos_safe(-normal.z), atan2(-normal.y,-normal.x) );

    //     float3 ray_origin = ray.origin + t_hit*ray.direction;

    //     rtPrintf("Launching transmission ray from prim #%d with strength %f, position (%f,%f,%f), direction
    //     (%f,%f,%f)\n",UUID,prd.strength,ray_origin.x,ray_origin.y,ray_origin.z,ray_direction.x,ray_direction.y,ray_direction.z);

    //     optix::Ray ray_reflect = optix::make_Ray(ray_origin, ray_direction, ray.ray_type, 1e-4, RT_DEFAULT_MAX);

    //     PerRayData_MCRT prd_reflect = prd;
    //     prd_reflect.scatter_depth ++;

    //     rtTrace( top_object, ray_reflect, prd_reflect);

    //     //}else{ //deposit what's left
    //     //atomicAdd( &radiation_in[UUID], prd.strength );
    //     //}
    // }else{ //absorption
    //   atomicAdd( &radiation_in[UUID], prd.strength );
    // }

    if (prd.scatter_depth < max_scatters) {  // reflection
        // if( prd.strength>1e-8 ){ //launch reflection ray

        float Rt = rnd(prd.seed);
        float Rp = rnd(prd.seed);
        float t = asin_safe(sqrtf(Rt));
        float p = 2.f * M_PI * Rp;

        float3 ray_direction;
        ray_direction.x = sin(t) * cos(p);
        ray_direction.y = sin(t) * sin(p);
        ray_direction.z = cos(t);

        ray_direction = d_rotatePoint(ray_direction, acos_safe(normal.z), atan2(normal.y, normal.x));

        float3 ray_origin = ray.origin + t_hit * ray.direction;

        optix::Ray ray_reflect = optix::make_Ray(ray_origin, ray_direction, ray.ray_type, 1e-4, RT_DEFAULT_MAX);

        PerRayData_MCRT prd_reflect = prd;
        prd_reflect.scatter_depth++;
        prd_reflect.strength *= t_rho;

        rtTrace(top_object, ray_reflect, prd_reflect);

        atomicAdd(&radiation_in[UUID], prd.strength * (1.f - t_rho));

    } else {  // absorption
        atomicAdd(&radiation_in[UUID], prd.strength);
    }
};

RT_PROGRAM void miss_MCRT() { atomicAdd(&Rsky[prd.origin_UUID], prd.strength); }
