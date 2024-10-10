#include <stdlib.h>
#include "state.hpp"
#include "binhash.hpp"

sim_state_t* alloc_state(int n)
{
    sim_state_t* s = (sim_state_t*) calloc(1, sizeof(sim_state_t));
    s->n     = n;
    s->part  = (particle_t*) calloc(n, sizeof(particle_t));
#ifdef VEC_BIN
    s->buckets = std::vector <std::vector <int>>(HASH_SIZE);
#else
    s->hash  = (particle_t**) calloc(HASH_SIZE, sizeof(particle_t*));
#endif
    return s;
}

void free_state(sim_state_t* s)
{
#ifndef VEC_BIN
    free(s->hash);
#endif
    free(s->part);
    free(s);
}
