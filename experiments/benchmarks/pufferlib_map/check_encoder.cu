#include "build/map_validation/pufferl.cu"
int main(){
    constexpr int B=2,H=4; cublas_init_handle();
    Encoder e={};e.out_dim=H;create_map_encoder(&e);auto* w=(MapWeights*)e.create_weights(&e);
    Allocator params={},acts={},grads={};e.reg_params(w,&params);MapActivations a={};e.reg_train(w,&a,&acts,&grads,B);
    alloc_create(&params);alloc_create(&acts);alloc_create(&grads);
    float raw[B*MAP_FEATURE_OBS],table[72],weight[H*MAP_EMBED_INPUT],up[B*H];
    for(int k=0;k<B*MAP_FEATURE_OBS;k++)raw[k]=(k%13)*.01f;
    for(int b=0;b<B;b++)for(int c=0;c<1089;c++)raw[b*MAP_FEATURE_OBS+1345+c]=(b+c)%18;
    for(int k=0;k<72;k++)table[k]=(k-35)*.01f;
    for(int k=0;k<H*MAP_EMBED_INPUT;k++)weight[k]=(k%17-8)*.001f;
    for(int k=0;k<B*H;k++)up[k]=(k+1)*.03f;
    Prec input={.shape={B,MAP_FEATURE_OBS}},gradient={.shape={B,H}};
    cudaMalloc(&input.data,sizeof(raw));cudaMalloc(&gradient.data,sizeof(up));
    cudaMemcpy(input.data,raw,sizeof(raw),cudaMemcpyHostToDevice);cudaMemcpy(gradient.data,up,sizeof(up),cudaMemcpyHostToDevice);
    cudaMemcpy(w->embedding.data,table,sizeof(table),cudaMemcpyHostToDevice);cudaMemcpy(w->weight.data,weight,sizeof(weight),cudaMemcpyHostToDevice);
    e.forward(w,&a,input,0);e.backward(w,&a,gradient,0);cudaDeviceSynchronize();assert(cudaGetLastError()==cudaSuccess);
    float out[B*H],dt[72],dw[H*MAP_EMBED_INPUT];cudaMemcpy(out,a.out.data,sizeof(out),cudaMemcpyDeviceToHost);
    cudaMemcpy(dt,a.dembedding.data,sizeof(dt),cudaMemcpyDeviceToHost);cudaMemcpy(dw,a.dweight.data,sizeof(dw),cudaMemcpyDeviceToHost);
    double ref_table[72]={},ref_dw[H*MAP_EMBED_INPUT]={},ref_out[B*H]={};
    for(int b=0;b<B;b++)for(int f=0;f<MAP_EMBED_INPUT;f++){
      int code=-1,component=-1;double value;
      if(f<1345)value=raw[b*MAP_FEATURE_OBS+f];
      else if(f<1345+1089*4){int j=f-1345;code=(int)raw[b*MAP_FEATURE_OBS+1345+j/4];component=j%4;value=table[code*4+component];}
      else value=raw[b*MAP_FEATURE_OBS+1345+1089+f-1345-1089*4];
      for(int h=0;h<H;h++){ref_out[b*H+h]+=value*weight[h*MAP_EMBED_INPUT+f];ref_dw[h*MAP_EMBED_INPUT+f]+=up[b*H+h]*value;
        if(code>=0)ref_table[code*4+component]+=up[b*H+h]*weight[h*MAP_EMBED_INPUT+f];}
    }
    double max_err=0;int nonzero=0;
    for(int k=0;k<B*H;k++){double err=fabs(out[k]-ref_out[k]);max_err=fmax(max_err,err);assert(err<1e-4*(1+fabs(ref_out[k])));}
    for(int k=0;k<72;k++){double err=fabs(dt[k]-ref_table[k]);max_err=fmax(max_err,err);assert(err<1e-4*(1+fabs(ref_table[k])));nonzero+=fabs(dt[k])>1e-9;}
    for(int k=0;k<H*MAP_EMBED_INPUT;k++){double err=fabs(dw[k]-ref_dw[k]);max_err=fmax(max_err,err);assert(err<1e-4*(1+fabs(ref_dw[k])));}
    assert(nonzero>0);printf("{\"passed\":true,\"max_absolute_error\":%.9g,\"embedding_gradient_nonzero\":%d}\n",max_err,nonzero);return 0;
}
