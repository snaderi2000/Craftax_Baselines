// Learned categorical map embedding (18 x 4), followed by Puffer's projection.
// float32 benchmark only; gradients scatter into the shared embedding table.
#define MAP_EMBED_INPUT (1345+1089*4+1089+40)
struct MapWeights {Prec embedding,weight;int hidden;};
struct MapActivations {Prec raw,features,out,dfeatures,dembedding,dweight;};
__global__ void map_pack(const float* raw,const float* table,float* features,int B){
    int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=B*MAP_EMBED_INPUT)return;
    int b=k/MAP_EMBED_INPUT,f=k%MAP_EMBED_INPUT;float value;
    if(f<1345)value=raw[b*MAP_FEATURE_OBS+f];
    else if(f<1345+1089*4){int j=f-1345,code=(int)raw[b*MAP_FEATURE_OBS+1345+j/4];value=table[code*4+j%4];}
    else value=raw[b*MAP_FEATURE_OBS+1345+1089+(f-1345-1089*4)];
    features[k]=value;
}
__global__ void map_scatter(const float* raw,const float* df,float* dt,int B){
    int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=B*1089*4)return;
    int b=k/(1089*4),j=k%(1089*4),code=(int)raw[b*MAP_FEATURE_OBS+1345+j/4];
    atomicAdd(dt+code*4+j%4,df[b*MAP_EMBED_INPUT+1345+j]);
}
Prec map_encoder_forward(void* weights,void* acts,Prec input,cudaStream_t stream){
    auto* w=(MapWeights*)weights;auto* a=(MapActivations*)acts;
    int B=numel(input.shape)/MAP_FEATURE_OBS;
    if(a->raw.data)puf_copy(&a->raw,&input,stream);
    map_pack<<<grid_size(B*MAP_EMBED_INPUT),BLOCK_SIZE,0,stream>>>(input.data,w->embedding.data,a->features.data,B);
    puf_mm(&a->features,&w->weight,&a->out,stream);return a->out;
}
void map_encoder_backward(void* weights,void* acts,Prec grad,cudaStream_t stream){
    auto* w=(MapWeights*)weights;auto* a=(MapActivations*)acts;
    puf_mm_tn(&grad,&a->features,&a->dweight,stream);
    puf_mm_nn(&grad,&w->weight,&a->dfeatures,stream);
    cudaMemsetAsync(a->dembedding.data,0,72*sizeof(float),stream);
    int B=a->features.shape[0];
    map_scatter<<<grid_size(B*1089*4),BLOCK_SIZE,0,stream>>>(a->raw.data,a->dfeatures.data,a->dembedding.data,B);
}
void map_encoder_init(void* weights,ulong* seed,cudaStream_t stream){
    auto* w=(MapWeights*)weights;puf_normal_init(&w->embedding,0.5f,(*seed)++,stream);
    puf_kaiming_init(&w->weight,sqrtf(2.f),(*seed)++,stream);
}
void map_encoder_params(void* weights,Allocator* alloc){
    auto* w=(MapWeights*)weights;w->embedding={.shape={18,4}};w->weight={.shape={w->hidden,MAP_EMBED_INPUT}};
    alloc_register(alloc,&w->embedding);alloc_register(alloc,&w->weight);
}
void map_encoder_train(void* weights,void* acts,Allocator* alloc,Allocator* grads,int B){
    auto* w=(MapWeights*)weights;auto* a=(MapActivations*)acts;
    a->raw={.shape={B,MAP_FEATURE_OBS}};a->features={.shape={B,MAP_EMBED_INPUT}};a->out={.shape={B,w->hidden}};
    a->dfeatures={.shape={B,MAP_EMBED_INPUT}};a->dembedding={.shape={18,4}};a->dweight={.shape={w->hidden,MAP_EMBED_INPUT}};
    alloc_register(alloc,&a->raw);alloc_register(alloc,&a->features);alloc_register(alloc,&a->out);alloc_register(alloc,&a->dfeatures);
    alloc_register(grads,&a->dembedding);alloc_register(grads,&a->dweight);
}
void map_encoder_rollout(void* weights,void* acts,Allocator* alloc,int B){
    auto* w=(MapWeights*)weights;auto* a=(MapActivations*)acts;
    a->features={.shape={B,MAP_EMBED_INPUT}};a->out={.shape={B,w->hidden}};
    alloc_register(alloc,&a->features);alloc_register(alloc,&a->out);
}
void* map_encoder_weights(void* self){auto* e=(Encoder*)self;auto* w=(MapWeights*)calloc(1,sizeof(MapWeights));w->hidden=e->out_dim;return w;}
static void create_map_encoder(Encoder* enc){
    enc->forward=map_encoder_forward;enc->backward=map_encoder_backward;enc->init_weights=map_encoder_init;
    enc->reg_params=map_encoder_params;enc->reg_train=map_encoder_train;enc->reg_rollout=map_encoder_rollout;
    enc->create_weights=map_encoder_weights;enc->activation_size=sizeof(MapActivations);
}
