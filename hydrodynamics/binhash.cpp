#include <string.h>

#include "zmorton.hpp"
#include "binhash.hpp"

/*@q
 * ====================================================================
 */

/*@T
 * \subsection{Spatial hashing implementation}
 * 
 * In the current implementation, we assume [[HASH_DIM]] is $2^b$,
 * so that computing a bitwise of an integer with [[HASH_DIM]] extracts
 * the $b$ lowest-order bits.  We could make [[HASH_DIM]] be something
 * other than a power of two, but we would then need to compute an integer
 * modulus or something of that sort.
 * 
 *@c*/

#define HASH_MASK (HASH_DIM-1)

unsigned particle_bucket(particle_t* p, float h)
{
    unsigned ix = p->x[0]/h;
    unsigned iy = p->x[1]/h;
    unsigned iz = p->x[2]/h;
    return zm_encode(ix & HASH_MASK, iy & HASH_MASK, iz & HASH_MASK);
}

#ifndef VEC_BIN
unsigned particle_neighborhood(unsigned* buckets, particle_t* p, float h)
{
    /* BEGIN TASK */
    int ix = p->x[0]/h;
    int iy = p->x[1]/h;
    int iz = p->x[2]/h;

    int ndim = 1.0/h;
    unsigned len = 0;
    for (int x = ix-1; x <= ix+1; x++) if (x >= 0 && x <= ndim)
        for (int y = iy-1; y <= iy+1; y++) if (y >= 0 && y <= ndim)
            for (int z = iz-1; z <= iz+1; z++) if (z >= 0 && z <= ndim)
                buckets[len ++] = zm_encode(unsigned(x) & HASH_MASK,
                        unsigned(y) & HASH_MASK, unsigned(z) & HASH_MASK);
    return len;
    /* END TASK */
}

void hash_particles(sim_state_t* s, float h)
{
    /* BEGIN TASK */
    particle_t* p = s->part;
    particle_t** hash = s->hash;
    int n = s->n;
    memset(hash, 0, HASH_SIZE*sizeof(particle_t*));

    for (int i = 0; i < n; i++) {
        particle_t* pi = p+i;
        unsigned b = particle_bucket(pi, h);
        pi->next = hash[b];
        hash[b] = pi;
    }
    /* END TASK */
}

#else

void binhash_particles(sim_state_t* s, float h) {
    for (size_t i = 0; i < HASH_SIZE; i ++)
        s->buckets[i].clear();
    for (int i = 0; i < s->n; i ++) {
        particle_t* pi = s->part+i;
        unsigned b = particle_bucket(pi, h);
        s->buckets[b].push_back(i);
    }
}

std::vector <unsigned> neighbor_buckets(particle_t *p, float h) {
    std::vector <unsigned> neighbors;
    neighbors.reserve(MAX_NBR_BINS);
    int ix = p->x[0]/h;
    int iy = p->x[1]/h;
    int iz = p->x[2]/h;

    int ndim = 1.0/h;
    for (int x = ix-1; x <= ix+1; x++) if (x >= 0 && x <= ndim)
        for (int y = iy-1; y <= iy+1; y++) if (y >= 0 && y <= ndim)
            for (int z = iz-1; z <= iz+1; z++) if (z >= 0 && z <= ndim)
                neighbors.push_back(zm_encode(unsigned(x) & HASH_MASK,
                        unsigned(y) & HASH_MASK, unsigned(z) & HASH_MASK));
  
    return neighbors;
}

#endif
