#include <string.h>
#include <math.h>
#include <stdlib.h>

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

unsigned particle_neighborhood(unsigned* buckets, particle_t* p, float h)
{
    /* BEGIN TASK */
    int ix = p->x[0]/h;
    int iy = p->x[1]/h;
    int iz = p->x[2]/h;

    int max_bucket_per_dim = 1.0/h;
    
    int last_index = 0;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
                int nx_bin = ix + dx;
                int ny_bin = iy + dy;
                int nz_bin = iz + dz;

                if (nx_bin >= 0 && nx_bin <= max_bucket_per_dim &&
                    ny_bin >= 0 && ny_bin <= max_bucket_per_dim &&
                    nz_bin >= 0 && nz_bin <= max_bucket_per_dim) {
                    unsigned bucket = zm_encode(unsigned(nx_bin) & HASH_MASK, 
                            unsigned(ny_bin) & HASH_MASK, unsigned(nz_bin) & HASH_MASK);
                    buckets[last_index] = bucket;
                    last_index++; 
                }
            } 
        }
    }
    return unsigned(last_index);
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
        int b = int(particle_bucket(pi, h));
        pi->next = hash[b];
        hash[b] = pi;
    }
    /* END TASK */
}
