#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <omp.h>

#include "vec3.hpp"
#include "zmorton.hpp"

#include "params.hpp"
#include "state.hpp"
#include "interact.hpp"
#include "binhash.hpp"

#include "stats.hpp"

/* Define this to use the bucketing version of the code */
/* #define USE_BUCKETING */

/*@T
 * \subsection{Density computations}
 * 
 * The formula for density is
 * \[
 *   \rho_i = \sum_j m_j W_{p6}(r_i-r_j,h)
 *          = \frac{315 m}{64 \pi h^9} \sum_{j \in N_i} (h^2 - r^2)^3.
 * \]
 * We search for neighbors of node $i$ by checking every particle,
 * which is not very efficient.  We do at least take advange of
 * the symmetry of the update ($i$ contributes to $j$ in the same
 * way that $j$ contributes to $i$).
 *@c*/

inline
void update_density(particle_t* pi, particle_t* pj, float h2, float C)
{
    float r2 = vec3_dist2(pi->x, pj->x);
    float z  = h2-r2;
    if (z > 0) {
        float rho_ij = C*z*z*z;
        pi->rho += rho_ij;
        pj->rho += rho_ij;
    }
}

inline
void update_density_i(particle_t* pi, particle_t* pj, float h2, float C)
{
    float r2 = vec3_dist2(pi->x, pj->x);
    float z  = h2-r2;
    if (z > 0) {
        float rho_ij = C*z*z*z;
        pi->rho += rho_ij;
    }
}

#ifdef AVX
inline
void neighbor_xyz(
        std::vector <float>& x, std::vector <float>& y, std::vector <float>& z,
        sim_state_t *s, std::vector <unsigned> &neighbors, int i) {
   for (const unsigned& ni:neighbors)
        for (const int &j:s->buckets[ni]) 
            if (i != j) {
                particle_t *pj = s->part + j;
                x.push_back(pj->x[0]);
                y.push_back(pj->x[1]);
                z.push_back(pj->x[2]);
            }
} 
float compute_rho_delta(
        std::vector <float>& X, std::vector <float>& Y, std::vector <float>& Z,
        float x, float y, float z, float h2, float C) {
    float rho = 0.0f;
    for (size_t i = 0; i < X.size(); i ++) {
        float dx = X[i] - x;
        float dy = Y[i] - y;
        float dz = Z[i] - z;
        float r2 = h2 - (dx*dx+dy*dy+dz*dz);
        if (r2 > 0) rho += r2*r2*r2;
    }
    return rho*C;
}

#include<immintrin.h>
inline
float compute_rho_delta_avx2(
        std::vector <float>& X, std::vector <float>& Y, std::vector <float>& Z,
        float x, float y, float z, float h2, float C) {
    float rho = 0.0f;
    const size_t vec_size = 8;
    const size_t n = X.size();

    __m256 vec_rho = _mm256_setzero_ps();
    __m256 vec_x = _mm256_set1_ps(x);
    __m256 vec_y = _mm256_set1_ps(y);
    __m256 vec_z = _mm256_set1_ps(z);
    __m256 vec_h2 = _mm256_set1_ps(h2);
    size_t i = 0;
    if (n>=vec_size) {
        for (; i <= n-vec_size; i += vec_size) {
            __m256 vec_X = _mm256_loadu_ps(&X[i]);
            __m256 vec_Y = _mm256_loadu_ps(&Y[i]);
            __m256 vec_Z = _mm256_loadu_ps(&Z[i]);
            __m256 vec_dx = _mm256_sub_ps(vec_X, vec_x);
            __m256 vec_dy = _mm256_sub_ps(vec_Y, vec_y);
            __m256 vec_dz = _mm256_sub_ps(vec_Z, vec_z);
            __m256 vec_dx2 = _mm256_mul_ps(vec_dx, vec_dx);
            __m256 vec_dy2 = _mm256_mul_ps(vec_dy, vec_dy);
            __m256 vec_dz2 = _mm256_mul_ps(vec_dz, vec_dz);
            __m256 vec_d2 = _mm256_sub_ps(
                    vec_h2, _mm256_add_ps(vec_dx2, _mm256_add_ps(vec_dy2, vec_dz2)));
            __m256 mask = _mm256_cmp_ps(vec_d2, _mm256_setzero_ps(), _CMP_GT_OS);
            __m256 vec_d2_cubed = _mm256_mul_ps(vec_d2, _mm256_mul_ps(vec_d2, vec_d2));
            vec_d2_cubed = _mm256_and_ps(mask, vec_d2_cubed);
            vec_rho = _mm256_add_ps(vec_rho, vec_d2_cubed);
        }
        float rho_array[vec_size];
        _mm256_storeu_ps(rho_array, vec_rho);
        for (size_t j = 0; j < vec_size; j++) {
            rho += rho_array[j];
        }
    }

    for (; i < n; i++) {
        float dx = X[i] - x;
        float dy = Y[i] - y;
        float dz = Z[i] - z;
        float d2 = h2 - (dx*dx + dy*dy + dz*dz);
        if (d2 > 0) {
            rho += d2 * d2 * d2;
        }
    }

    return rho*C;
}

#endif

void compute_density(sim_state_t* s, sim_param_t* params)
{
    int n = s->n;
    particle_t* p = s->part;
#ifndef VEC_BIN
    particle_t** hash = s->hash;
#endif

    float h  = params->h;
    float h2 = h*h;
    float h3 = h2*h;
    float h9 = h3*h3*h3;
    float C  = ( 315.0/64.0/M_PI ) * s->mass / h9;

    // Clear densities
    //for (int i = 0; i < n; ++i)
    //    p[i].rho = 0;

    // Accumulate density info
#ifdef USE_BUCKETING
    /* BEGIN TASK */
#pragma omp parallel for 
    for (int i = 0; i < n; i++) {
        particle_t* pi = p+i;
        pi->rho = ( 315.0/64.0/M_PI ) * s->mass / h3;

#ifdef VEC_BIN
        std::vector <unsigned> neighbors = neighbor_buckets(pi, h);
#ifdef AVX
        size_t est_size = neighbors.size() * s->buckets[neighbors[0]].size();
        if (est_size > AVX) {
            std::vector <float> x, y, z;
            x.reserve(est_size*2);
            y.reserve(est_size*2);
            z.reserve(est_size*2);

            neighbor_xyz(x, y, z, s, neighbors, i);
            pi->rho += compute_rho_delta_avx2(
                    x, y, z, pi->x[0], pi->x[1], pi->x[2], h2, C); 
        } else {
            for (const unsigned& ni:neighbors) {
                for (const int &j:s->buckets[ni]) 
                    if (i != j) {
                        particle_t *pj = p + j;
                        update_density_i(pi, pj, h2, C);
                    }
            }
        }

#else
        for (const unsigned& ni:neighbors) {
            for (const int &j:s->buckets[ni]) 
                if (i != j) {
                    particle_t *pj = p + j;
                    update_density_i(pi, pj, h2, C);
                }
        }
#endif
#else
        unsigned buckets[MAX_NBR_BINS];
        unsigned nbr = particle_neighborhood(buckets, pi, h);
        for (unsigned j = 0; j < nbr; j++) {
            for (particle_t *pj = hash[buckets[j]]; pj; pj = pj->next)
                if(pi != pj) update_density_i(pi, pj, h2, C);
        }
#endif

    }
    /* END TASK */
#else
    for (int i = 0; i < n; ++i) {
        particle_t* pi = s->part+i;
        pi->rho += ( 315.0/64.0/M_PI ) * s->mass / h3;
        for (int j = i+1; j < n; ++j) {
            particle_t* pj = s->part+j;
            update_density(pi, pj, h2, C);
        }
    }
#endif
}


/*@T
 * \subsection{Computing forces}
 * 
 * The acceleration is computed by the rule
 * \[
 *   \bfa_i = \frac{1}{\rho_i} \sum_{j \in N_i} 
 *     \bff_{ij}^{\mathrm{interact}} + \bfg,
 * \]
 * where the pair interaction formula is as previously described.
 * Like [[compute_density]], the [[compute_accel]] routine takes
 * advantage of the symmetry of the interaction forces
 * ($\bff_{ij}^{\mathrm{interact}} = -\bff_{ji}^{\mathrm{interact}}$)
 * but it does a very expensive brute force search for neighbors.
 *@c*/

inline
void update_forces(particle_t* pi, particle_t* pj, float h2,
                   float rho0, float C0, float Cp, float Cv)
{
    float dx[3];
    vec3_diff(dx, pi->x, pj->x);
    float r2 = vec3_len2(dx);
    if (r2 < h2) {
        const float rhoi = pi->rho;
        const float rhoj = pj->rho;
        float q = sqrt(r2/h2);
        float u = 1-q;
        float w0 = C0 * u/rhoi/rhoj;
        float wp = w0 * Cp * (rhoi+rhoj-2*rho0) * u/q;
        float wv = w0 * Cv;
        float dv[3];
        vec3_diff(dv, pi->v, pj->v);

        // Equal and opposite pressure forces
        vec3_saxpy(pi->a,  wp, dx);
        vec3_saxpy(pj->a, -wp, dx);
        
        // Equal and opposite viscosity forces
        vec3_saxpy(pi->a,  wv, dv);
        vec3_saxpy(pj->a, -wv, dv);
    }
}

inline
void update_forces_i(particle_t* pi, particle_t* pj, float h2,
                   float rho0, float C0, float Cp, float Cv)
{
    float dx[3];
    vec3_diff(dx, pi->x, pj->x);
    float r2 = vec3_len2(dx);
    if (r2 < h2) {
        const float rhoi = pi->rho;
        const float rhoj = pj->rho;
        float q = sqrt(r2/h2);
        float u = 1-q;
        float w0 = C0 * u/rhoi/rhoj;
        float wp = w0 * Cp * (rhoi+rhoj-2*rho0) * u/q;
        float wv = w0 * Cv;
        float dv[3];
        vec3_diff(dv, pi->v, pj->v);

        // Equal and opposite pressure forces
        vec3_saxpy(pi->a,  wp, dx);
        
        // Equal and opposite viscosity forces
        vec3_saxpy(pi->a,  wv, dv);
    }
}

void compute_accel(sim_state_t* state, sim_param_t* params)
{
    // Unpack basic parameters
    const float h    = params->h;
    const float rho0 = params->rho0;
    const float k    = params->k;
    const float mu   = params->mu;
    const float g    = params->g;
    const float mass = state->mass;
    const float h2   = h*h;

    // Unpack system state
    particle_t* p = state->part;
#ifndef VEC_BIN
    particle_t** hash = state->hash;
#endif
    int n = state->n;

    // Rehash the particles
#ifdef STATS
    double t0 = omp_get_wtime();
#endif

#ifdef VEC_BIN
    binhash_particles(state, h);
#else
    hash_particles(state, h);
#endif

#ifdef STATS
    double t1 = omp_get_wtime();
#endif
    // Compute density and color
   compute_density(state, params);
#ifdef STATS
    double t2 = omp_get_wtime();
#endif

    // Start with gravity and surface forces
    for (int i = 0; i < n; ++i)
        vec3_set(p[i].a,  0, -g, 0);

    // Constants for interaction term
    float C0 = 45 * mass / M_PI / ( (h2)*(h2)*h );
    float Cp = k/2;
    float Cv = -mu;

    // Accumulate forces
#ifdef USE_BUCKETING
    /* BEGIN TASK */
#pragma omp parallel for 
    for (int i = 0; i < n; i++) {
        particle_t* pi = p+i;
#ifdef VEC_BIN
        std::vector <unsigned> neighbors = neighbor_buckets(pi, h);
        for (const unsigned& ni:neighbors)
            for (const int &j:state->buckets[ni]) 
                if (i != j) {
                    particle_t *pj = p + j;
                    update_forces_i(pi, pj, h2, rho0, C0, Cp, Cv);
                }
#else
        unsigned buckets[MAX_NBR_BINS];
        unsigned nbr = particle_neighborhood(buckets, pi, h);
        for (unsigned j = 0; j < nbr; j++) {
            for (particle_t *pj = hash[buckets[j]]; pj; pj = pj->next) 
                if (pi != pj)
                    update_forces_i(pi, pj, h2, rho0, C0, Cp, Cv);
        }
#endif
    }
    /* END TASK */
#else
    for (int i = 0; i < n; ++i) {
        particle_t* pi = p+i;
        for (int j = i+1; j < n; ++j) {
            particle_t* pj = p+j;
            update_forces(pi, pj, h2, rho0, C0, Cp, Cv);
        }
    }
#endif

#ifdef STATS
    double t3 = omp_get_wtime();
    stats& stat = stats::get_stats();
    stat.accu_time(0, 1, t1-t0);
    stat.accu_time(0, 2, t2-t1);
    stat.accu_time(0, 3, t3-t2);
#endif
}
