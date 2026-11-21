#define INK_ROUTE_IMPL
#include "inkling-device-routes.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>
#include "inkling-reference-lru.inc"

struct ggml_backend_cuda_context;
void ggml_cuda_op_get_rows(ggml_backend_cuda_context &,ggml_tensor *);
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(1);}}while(0)
static void *alloc(size_t n){void *v=nullptr;CK(cudaMalloc(&v,n));return v;}
struct Pool {
    int slots;
    size_t stride[2];
    void *state,*source[2]{},*bank[2]{};
    ink_dev_layer layer[3]{};
    moe_cache_lru cpu;
    std::array<std::array<int,128>,48> maps{};
    ink_route_stats expected{};
    void init(cudaStream_t stream,int count,size_t stride0,size_t stride1) {
        slots=count;stride[0]=stride0;stride[1]=stride1;state=alloc(ink_route_bytes());
        for(int i=0;i<2;++i) {
            CK(cudaMallocHost(&source[i],3*128*stride[i]));bank[i]=alloc((slots+1)*stride[i]);
            CK(cudaMemsetAsync(bank[i],0xAD,(slots+1)*stride[i],stream));
            auto *host=static_cast<unsigned char*>(source[i]);
            for(size_t j=0;j<3*128*stride[i];++j)host[j]=uint8_t(j*17+(j>>4)+(j>>8)+(j>>15)+i*31);
            cudaPointerAttributes a{};CK(cudaPointerGetAttributes(&a,source[i]));
            if(!a.devicePointer)std::abort();
            for(int l=0;l<3;++l) {
                layer[l].source[i]=static_cast<unsigned char*>(a.devicePointer)+l*128*stride[i];
                layer[l].bank[i]=bank[i];layer[l].stride[i]=stride[i];
            }
        }
        reset(stream);
    }
    void reset(cudaStream_t stream) {
        expected={};cpu=moe_cache_lru{};cpu.init(48,128,slots);
        for(int i=0;i<48;++i){maps[i].fill(slots);cpu.slot_map[i]=maps[i].data();}
        ink_route_init(stream,state,slots);
    }
    void finish(cudaStream_t stream) {
        ink_route_stats result{};ink_route_finish(stream,state,&result);
        assert(memcmp(&expected,&result,sizeof(result))==0);
    }
    void release() {
        for(int i=0;i<2;++i){CK(cudaFreeHost(source[i]));CK(cudaFree(bank[i]));}
        CK(cudaFree(state));
    }
};
struct Visit {
    int physical;Pool *pool;
    int *route; int *output;
    void *bank_snap[2]{},*map_snap{};
    ggml_tensor map{},ids{},result{};
    std::array<int,8> values{};
    void init(Pool &p,int layer) {
        pool=&p;physical=layer;
        for(int j=0;j<2;++j)bank_snap[j]=alloc((p.slots+1)*p.stride[j]);
        map_snap=alloc(128*sizeof(int));
        route=static_cast<int*>(alloc(sizeof(values)));output=static_cast<int*>(alloc(sizeof(values)));
        map.data=ink_route_graph_map(p.state,layer);map.type=ids.type=result.type=GGML_TYPE_I32;
        ids.data=route;result.data=output;result.op=GGML_OP_GET_ROWS;
        result.ne[0]=1;result.ne[1]=8;result.ne[2]=result.ne[3]=1;
        result.src[0]=&map;result.src[1]=&ids;
        ink_route_graph_mark(&result,p.state,layer,&p.layer[layer]);
    }
    void record(ggml_backend_cuda_context &ctx,cudaStream_t stream) {
        ggml_cuda_op_get_rows(ctx,&result);
        for(int j=0;j<2;++j)CK(cudaMemcpyAsync(bank_snap[j],pool->bank[j],(pool->slots+1)*pool->stride[j],cudaMemcpyDeviceToDevice,stream));
        CK(cudaMemcpyAsync(map_snap,map.data,128*sizeof(int),cudaMemcpyDeviceToDevice,stream));
    }
    void verify(cudaStream_t stream) {
        auto &p=*pool;size_t hits=0;
        std::vector<moe_cache_lru::fill> fills,cpu;
        assert(p.cpu.plan(physical,values.data(),values.size(),1,8,0,[](size_t){return size_t(0);},fills,cpu,hits));
        std::array<int,8> selected;std::array<int,128> mapped;
        CK(cudaMemcpyAsync(selected.data(),output,sizeof(selected),cudaMemcpyDeviceToHost,stream));
        CK(cudaMemcpyAsync(mapped.data(),map_snap,sizeof(mapped),cudaMemcpyDeviceToHost,stream));CK(cudaStreamSynchronize(stream));
        assert(mapped==p.maps[physical]);for(size_t k=0;k<8;++k)assert(selected[k]==mapped[values[k]]);
        ++p.expected.visits;p.expected.all_hits+=fills.empty();p.expected.hits+=hits;p.expected.misses+=fills.size();p.expected.bytes+=fills.size()*(p.stride[0]+p.stride[1]);
        for(int j=0;j<2;++j){
            std::vector<uint8_t> data((p.slots+1)*p.stride[j]);
            CK(cudaMemcpyAsync(data.data(),bank_snap[j],data.size(),cudaMemcpyDeviceToHost,stream));CK(cudaStreamSynchronize(stream));
            for(int slot=0;slot<p.slots;++slot) {
                int key=p.cpu.key_of[slot];if(key<0)continue;
                assert(memcmp(data.data()+slot*p.stride[j],static_cast<const uint8_t*>(p.source[j])+key*p.stride[j],p.stride[j])==0);
            }
            assert(std::all_of(data.end()-p.stride[j],data.end(),[](uint8_t v){return v==0xad;}));
        }
    }
    void release(){for(auto b:bank_snap)CK(cudaFree(b));CK(cudaFree(map_snap));CK(cudaFree(route));CK(cudaFree(output));}
};
int main() {
    ggml_backend_t be=ggml_backend_cuda_init(0);if(!be || !ink_route_supported() || !ink_route_graph_available())return 77;
    ggml_backend_synchronize(be);auto s=static_cast<cudaStream_t>(ink_route_stream(be));
    auto &ctx=*static_cast<ggml_backend_cuda_context*>(be->context);
    Pool p[2]{};p[0].init(s,8,384,192);p[1].init(s,23,320,256);
    Visit v[6]{};const int groups[]={0,1,0,0,1,0},layers[]={0,1,0,2,1,0};
    for(int i=0;i<6;++i)v[i].init(p[groups[i]],layers[i]);
    CK(cudaStreamSynchronize(s));
    CK(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));
    for(auto &step:v)step.record(ctx,s);
    cudaGraph_t graph=nullptr;CK(cudaStreamEndCapture(s,&graph));
    size_t n=0;CK(cudaGraphGetNodes(graph,nullptr,&n));
    std::vector<cudaGraphNode_t> nodes(n);CK(cudaGraphGetNodes(graph,nodes.data(),&n));int kernels=0;
    for(auto node:nodes){cudaGraphNodeType type;CK(cudaGraphNodeGetType(node,&type));kernels+=type==cudaGraphNodeTypeKernel;}
    assert(kernels==12);
    cudaGraphExec_t exec=nullptr;CK(cudaGraphInstantiate(&exec,graph,0));
    for(int epoch=0;epoch<2;++epoch) {
        for(int trial=0;trial<48;++trial) {
            for(int i=0;i<6;++i){
                for(int k=0;k<8;++k)v[i].values[k]=((trial/3+epoch*4)*7+v[i].physical*13+k*11)%128;
                if(trial%5==0)v[i].values[7]=v[i].values[0];
                CK(cudaMemcpyAsync(v[i].route,v[i].values.data(),sizeof(v[i].values),cudaMemcpyHostToDevice,s));
            }
            CK(cudaGraphLaunch(exec,s));CK(cudaStreamSynchronize(s));
            for(auto &step:v)step.verify(s);
        }
        for(auto &pool:p){
            pool.finish(s);
            printf("epoch=%d slots=%d visits=%llu hits=%llu misses=%llu all_hit=%llu selected+map+bank+guard=exact\n",
                epoch,pool.slots,(unsigned long long)pool.expected.visits,(unsigned long long)pool.expected.hits,
                (unsigned long long)pool.expected.misses,(unsigned long long)pool.expected.all_hits);
            if(epoch==0)pool.reset(s);
        }
    }
    CK(cudaGraphExecDestroy(exec));CK(cudaGraphDestroy(graph));
    for(auto &step:v)step.release();for(auto &pool:p)pool.release();
    ggml_backend_free(be);std::puts("One captured graph, 96 replays, six ordered consumers, stable pointers, changed IDs, two fresh owners.");
}
