#ifndef KEEP_AUTORESET
#define PUF_PARITY_NO_RESET
#endif
#include "build/map_validation/craftax_map.h"
static CraftaxClassic state;
static float obs[OBS_SIZE],action,reward,done;
void bridge_set(const uint8_t* map,const int* x,const float* f,const int* inv,const int* ach){
    memset(&state,0,sizeof(state));state.agents[0].observations=obs;state.agents[0].actions=&action;state.agents[0].rewards=&reward;state.agents[0].terminals=&done;
    c_init(&state);memcpy(state.map_packed,map,4096);
    state.player_r=x[0];state.player_c=x[1];state.player_dir=x[2];state.health=x[3];state.food=x[4];state.drink=x[5];state.energy=x[6];state.is_sleeping=x[7];state.timestep=x[8];
    state.recover=f[0];state.hunger=f[1];state.thirst=f[2];state.fatigue=f[3];state.light_level=f[4];
    for(int i=0;i<12;i++)state.inv[i]=inv[i];for(int i=0;i<22;i++)state.achievements[i]=ach[i];
    map_clear(&state.memory);compute_observations(&state);map_augment(&state.memory,obs,state.player_r,state.player_c,-1,1);
}
void bridge_get(int* x,float* f,int* inv,int* ach,float* observation,uint8_t* map){
    int values[]={state.player_r,state.player_c,state.player_dir,state.health,state.food,state.drink,state.energy,state.is_sleeping,state.timestep};memcpy(x,values,sizeof(values));
    float values_f[]={state.recover,state.hunger,state.thirst,state.fatigue,state.light_level,reward,done};memcpy(f,values_f,sizeof(values_f));
    for(int i=0;i<12;i++)inv[i]=state.inv[i];for(int i=0;i<22;i++)ach[i]=state.achievements[i];memcpy(observation,obs,sizeof(obs));memcpy(map,state.map_packed,4096);
}
void bridge_step(int a){action=a;puf_step(&state);}
void bridge_reset(){puf_reset(&state);}
void bridge_mob(int kind,int r,int c,int hp,int cooldown){
    uint64_t bit=1ULL<<c;state.mob_bits[r]|=bit;
    if(kind==0){state.zombie_r[0]=r;state.zombie_c[0]=c;state.zombie_hp[0]=hp;state.zombie_cd[0]=cooldown;state.zombie_mask[0]=1;state.zombie_bits[r]|=bit;}
    if(kind==1){state.cow_r[0]=r;state.cow_c[0]=c;state.cow_hp[0]=hp;state.cow_mask[0]=1;state.cow_bits[r]|=bit;}
    if(kind==2){state.skel_r[0]=r;state.skel_c[0]=c;state.skel_hp[0]=hp;state.skel_cd[0]=cooldown;state.skel_mask[0]=1;state.skel_bits[r]|=bit;}
    compute_observations(&state);
}
void bridge_plant(int r,int c,int age){state.plant_r[0]=r;state.plant_c[0]=c;state.plant_age[0]=age;state.plant_mask[0]=1;}
// Feature-only entry points feed exactly the same observed frames to C and JAX.
static ObservedMap feature_state;
void features_clear(){map_clear(&feature_state);}
void features_step(float* observation,int r,int c,int a,int mode){map_augment(&feature_state,observation,r,c,a,mode);}
