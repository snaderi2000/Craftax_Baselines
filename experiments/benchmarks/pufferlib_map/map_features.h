#pragma once
#include <stdint.h>
#include <string.h>
#include <math.h>
#define MAP_LOCAL 1345
#define MAP_CELLS 1089
#define MAP_FEATURE_OBS (1345+2*1089+40)
typedef struct {
    uint8_t terrain[4096], distance[4096];
    int seen[4096], elapsed, depleted[3], blocked_streak;
    int last_r,last_c,initialized,last_sleeping,route_valid;
} ObservedMap;
static const int map_dr[5]={0,0,0,-1,1};
static const int map_dc[5]={0,-1,1,0,0};
static int map_inside(int r,int c){return r>=0&&r<64&&c>=0&&c<64;}
static int map_walk(uint8_t tile){return tile==3||tile==7||tile==8||tile==14;}
static void map_clear(ObservedMap* m){memset(m,0,sizeof(*m));for(int i=0;i<4096;i++)m->seen[i]=-1;}
static void map_routes(ObservedMap* m){
    int queue[4096],head=0,tail=0;memset(m->distance,129,4096);
    for(int r=0;r<64;r++)for(int c=0;c<64;c++){
        int k=r*64+c;if(!map_walk(m->terrain[k]))continue;
        for(int d=1;d<=4;d++){int rr=r+map_dr[d],cc=c+map_dc[d];
            if(map_inside(rr,cc)&&m->terrain[rr*64+cc]==4){m->distance[k]=0;queue[tail++]=k;break;}}
    }
    while(head<tail){int k=queue[head++],r=k/64,c=k%64,dist=m->distance[k];if(dist>=128)continue;
        for(int d=1;d<=4;d++){int rr=r+map_dr[d],cc=c+map_dc[d];if(!map_inside(rr,cc))continue;int j=rr*64+cc;
            if(map_walk(m->terrain[j])&&m->distance[j]>dist+1){m->distance[j]=dist+1;queue[tail++]=j;}}
    }
    m->route_valid=1;
}
// Reads only the supplied local observation; positions supply existing odometry.
static void map_augment(ObservedMap* m,float* obs,int r,int c,int action,int mode){
    int reset=!m->initialized,dr=reset?0:r-m->last_r,dc=reset?0:c-m->last_c;
    int blocked=!reset&&action>=1&&action<=4&&dr==0&&dc==0&&!m->last_sleeping;
    m->elapsed+=!reset;m->blocked_streak=blocked?(m->blocked_streak<64?m->blocked_streak+1:64):0;
    for(int k=0;k<3;k++)m->depleted[k]=obs[1336+k]<=0?(m->depleted[k]<256?m->depleted[k]+1:256):0;
    for(int i=0;i<63;i++){int rr=r+i/9-3,cc=c+i%9-4;if(!map_inside(rr,cc))continue;
        int code=0;for(int b=1;b<17;b++)if(obs[i*21+b]>obs[i*21+code])code=b;
        int k=rr*64+cc;if(m->terrain[k]!=code+1)m->route_valid=0;
        m->terrain[k]=code+1;m->seen[k]=m->elapsed;
    }
    for(int i=0;i<MAP_CELLS;i++){int rr=r+i/33-16,cc=c+i%33-16,k=rr*64+cc;
        int valid=map_inside(rr,cc),seen=valid?m->seen[k]:-1;
        obs[MAP_LOCAL+i]=valid?m->terrain[k]:0;
        int age=seen<0?10000:m->elapsed-seen;if(age>10000)age=10000;
        obs[MAP_LOCAL+MAP_CELLS+i]=log1pf((float)age)/logf(10001.0f);
    }
    float* f=obs+MAP_LOCAL+2*MAP_CELLS;memset(f,0,40*sizeof(float));
    if(mode){
        if(!m->route_valid)map_routes(m);
        for(int i=0;i<5;i++){int rr=r+map_dr[i],cc=c+map_dc[i],valid=map_inside(rr,cc);
            int dist=valid?m->distance[rr*64+cc]:129;f[i]=dist<128?dist/128.0f:1;f[5+i]=dist<=128;
            if(i)f[9+i]=valid&&m->terrain[rr*64+cc]==4;
        }
        int facing=0;for(int d=1;d<4;d++)if(obs[1339+d]>obs[1339+facing])facing=d;
        f[14]=f[10+facing];for(int k=0;k<4096;k++)if(m->terrain[k]==4){f[15]=1;break;}
        for(int k=0;k<3;k++)f[16+k]=m->depleted[k]/256.0f;
        if(!reset&&action>=0&&action<17)f[19+action]=1;
        f[36]=dr;f[37]=dc;f[38]=blocked;f[39]=m->blocked_streak/64.0f;
    }
    m->initialized=1;m->last_r=r;m->last_c=c;m->last_sleeping=obs[1344]>0;
}
