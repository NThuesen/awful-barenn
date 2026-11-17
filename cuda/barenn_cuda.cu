// Treinamento SGD CUDA: um bloco processa as amostras em ordem a cada época.
#include "barenn_cuda_suporte.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <limits>

static void check(cudaError_t status, const char* operacao) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operacao) + ": " + cudaGetErrorString(status));
}
#define CUDA(call) check((call), #call)

static double copy_h2d_seconds = 0, copy_d2h_seconds = 0;
static cudaError_t timed_copy(void* dst, const void* src, size_t bytes, cudaMemcpyKind direction) {
    auto begin = std::chrono::steady_clock::now();
    cudaError_t result = cudaMemcpy(dst,src,bytes,direction);
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    if (direction == cudaMemcpyHostToDevice) copy_h2d_seconds += seconds;
    if (direction == cudaMemcpyDeviceToHost) copy_d2h_seconds += seconds;
    return result;
}

// Libera o buffer também quando uma chamada CUDA ou a leitura do oráculo falha.
template<class T> class DeviceBuffer {
    T* ptr = nullptr;
public:
    explicit DeviceBuffer(size_t n) {
        CUDA(cudaMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(T)));
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    T* get() const { return ptr; }
    void release() {
        T* old = ptr;
        ptr = nullptr;
        CUDA(cudaFree(old));
    }
};

struct Layer {
    int in, out;
    float *w, *b, *z, *a, *delta, *grad_in;
};

__global__ void forward(Layer l, const float* input, bool final) {
    int i = threadIdx.x;
    if (i < l.out) {
        float sum = l.b[i];
        for (int j = 0; j < l.in; ++j) sum += input[j] * l.w[i*l.in+j];
        l.z[i] = sum;
        l.a[i] = final ? sum : fmaxf(0.0f, sum);
    }
}

// Todas as threads do bloco participam de todas as barreiras.
__device__ void fused_forward(Layer l, const float* input, bool final) {
    int i=threadIdx.x;
    if (i<l.out) {
        float sum=l.b[i];
        for(int j=0;j<l.in;++j) sum+=input[j]*l.w[i*l.in+j];
        l.z[i]=sum; l.a[i]=final ? sum : fmaxf(0.0f,sum);
    }
    __syncthreads();
}
__device__ void fused_backward(Layer l, const float* input, const float* gout, bool final) {
    int i=threadIdx.x;
    if (i<l.out) l.delta[i]=final ? gout[i] : gout[i]*(l.z[i]>0.0f ? 1.0f : 0.0f);
    __syncthreads();
    if (i<l.in) {
        float sum=0.0f;
        for(int j=0;j<l.out;++j) sum+=l.delta[j]*l.w[j*l.in+i];
        l.grad_in[i]=sum;
    }
    __syncthreads(); // grad_in inteiro calculado com pesos antigos.
    if (i<l.out) {
        for(int j=0;j<l.in;++j) l.w[i*l.in+j]-=0.02f*(l.delta[i]*input[j]);
        l.b[i]-=0.02f*l.delta[i];
    }
    __syncthreads(); // atualização concluída antes de avançar de camada.
}
// Captura somente diagnóstica das duas primeiras amostras selecionadas.
struct Snapshot {
    float before_w[2720], after_w[2720], before_b[58], after_b[58];
    float z[58], a[58], delta[58], gin[112], grad[10];
};
#ifdef BARENN_ORACLE
__device__ void snapshot_params(Snapshot* snap, Layer a, Layer b, Layer c, bool before) {
    int t=threadIdx.x;
    float* w=before ? snap->before_w : snap->after_w;
    float* bias=before ? snap->before_b : snap->after_b;
    for(int k=t;k<2048;k+=blockDim.x) w[k]=a.w[k];
    for(int k=t;k<512;k+=blockDim.x) w[2048+k]=b.w[k];
    for(int k=t;k<160;k+=blockDim.x) w[2560+k]=c.w[k];
    if(t<32) bias[t]=a.b[t];
    if(t<16) bias[32+t]=b.b[t];
    if(t<10) bias[48+t]=c.b[t];
    __syncthreads();
}
#endif
// Um único bloco percorre a época na ordem original, com pesos em memória compartilhada.
__global__ void train_epoch(Layer d1, Layer d2, Layer d3, const float* images,
                            const int* labels, int n, float* losses, Snapshot* traces) {
    __shared__ float w1[2048],w2[512],w3[160],b1[32],b2[16],b3[10];
    __shared__ float z1[32],z2[16],z3[10],a1[32],a2[16],a3[10];
    __shared__ float dlt1[32],dlt2[16],dlt3[10],gi1[64],gi2[32],gi3[16],grad[10];
    Layer l1{64,32,w1,b1,z1,a1,dlt1,gi1};
    Layer l2{32,16,w2,b2,z2,a2,dlt2,gi2};
    Layer l3{16,10,w3,b3,z3,a3,dlt3,gi3};
    int t=threadIdx.x;
    for(int k=t;k<2048;k+=blockDim.x) w1[k]=d1.w[k];
    for(int k=t;k<512;k+=blockDim.x) w2[k]=d2.w[k];
    for(int k=t;k<160;k+=blockDim.x) w3[k]=d3.w[k];
    if(t<32)b1[t]=d1.b[t]; if(t<16)b2[t]=d2.b[t]; if(t<10)b3[t]=d3.b[t];
    __syncthreads();
    for(int i=0;i<n;++i) {
        const float* input=images+static_cast<size_t>(i)*64;
#ifdef BARENN_ORACLE
        if(traces && i<2) snapshot_params(&traces[i],l1,l2,l3,true);
#endif
        fused_forward(l1,input,false);
        fused_forward(l2,l1.a,false);
        fused_forward(l3,l2.a,true);
        const int label=labels[i];
        if(t<10)grad[t]=a3[t]-(t==label ? 1.0f : 0.0f);
        if(t==0) {
            float total=0;
            for(int k=0;k<10;++k) { float err=a3[k]-(k==label ? 1.0f : 0.0f); total+=0.5f*err*err; }
            losses[i]=total;
        }
        __syncthreads();
        fused_backward(l3,l2.a,grad,true);
        fused_backward(l2,l1.a,l3.grad_in,false);
        fused_backward(l1,input,l2.grad_in,false);
#ifdef BARENN_ORACLE
        if(traces && i<2) {
            Snapshot* s=&traces[i];
            snapshot_params(s,l1,l2,l3,false);
            if(t<32) { s->z[t]=z1[t]; s->a[t]=a1[t]; s->delta[t]=dlt1[t]; }
            if(t<16) { s->z[32+t]=z2[t]; s->a[32+t]=a2[t]; s->delta[32+t]=dlt2[t]; }
            if(t<10) { s->z[48+t]=z3[t]; s->a[48+t]=a3[t]; s->delta[48+t]=dlt3[t]; s->grad[t]=grad[t]; }
            if(t<64)s->gin[t]=gi1[t];
            if(t<32)s->gin[64+t]=gi2[t];
            if(t<16)s->gin[96+t]=gi3[t];
            __syncthreads();
        }
#endif
        // A última barreira do backward termina a atualização antes da próxima amostra.
    }
    for(int k=t;k<2048;k+=blockDim.x) d1.w[k]=w1[k];
    for(int k=t;k<512;k+=blockDim.x) d2.w[k]=w2[k];
    for(int k=t;k<160;k+=blockDim.x) d3.w[k]=w3[k];
    if(t<32)d1.b[t]=b1[t]; if(t<16)d2.b[t]=b2[t]; if(t<10)d3.b[t]=b3[t];
}

// Os buffers persistem na memória global para avaliação e entre épocas.
class LayerBuffers {
    int in, out;
    DeviceBuffer<float> w, b, z, a;
public:
    LayerBuffers(int inputs, int outputs)
        : in(inputs), out(outputs), w(inputs*outputs), b(outputs), z(outputs), a(outputs) {
        std::vector<float> weights(in*out), biases(out, 0.0f);
        std::mt19937 gen(1337);
        std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
        for (float& x : weights) x = dist(gen);
        CUDA(timed_copy(w.get(), weights.data(), weights.size()*sizeof(float), cudaMemcpyHostToDevice));
        CUDA(timed_copy(b.get(), biases.data(), biases.size()*sizeof(float), cudaMemcpyHostToDevice));
    }
    Layer view() const { return {in,out,w.get(),b.get(),z.get(),a.get(),nullptr,nullptr}; }
    void release() { w.release(); b.release(); z.release(); a.release(); }
};

// A captura copia os valores efetivamente calculados pela GPU.
#ifdef BARENN_ORACLE
static std::vector<float> host(const float* p, int n) {
    std::vector<float> v(n);
    CUDA(timed_copy(v.data(), p, n*sizeof(float), cudaMemcpyDeviceToHost));
    return v;
}
static void params(Layer l, const std::string& prefix, const std::string& name) {
    auto w = host(l.w, l.in*l.out), b = host(l.b, l.out);
    for (int i = 0; i < l.out; ++i)
        for (int j = 0; j < l.in; ++j)
            Barenn::Oracle::valor(prefix + "/" + name + "/peso/" + std::to_string(i) + "/" + std::to_string(j), w[i*l.in+j], std::numeric_limits<float>::max_digits10);
    Barenn::Oracle::vetor(prefix + "/" + name + "/vies", b);
}
static void snapshot_vector(const std::string& key, const float* p, int n) {
    Barenn::Oracle::vetor(key,std::vector<float>(p,p+n));
}
static void snapshot_params_host(const Snapshot& s, const std::string& prefix, const std::string& name,
                                 int in, int out, int woff, int boff, bool before) {
    const float* w=before ? s.before_w : s.after_w;
    const float* b=before ? s.before_b : s.after_b;
    for(int i=0;i<out;++i) for(int j=0;j<in;++j)
        Barenn::Oracle::valor(prefix+"/"+name+"/peso/"+std::to_string(i)+"/"+std::to_string(j),w[woff+i*in+j],std::numeric_limits<float>::max_digits10);
    snapshot_vector(prefix+"/"+name+"/vies",b+boff,out);
}
static void write_snapshot(const Snapshot& s, const std::vector<float>& input, int epoch, int index, float loss) {
    Barenn::Oracle::treino=true; Barenn::Oracle::epoca=epoch; Barenn::Oracle::amostra=index;
    const std::string root=Barenn::Oracle::prefixo();
    Barenn::Oracle::valor(root+"/perda",loss,std::numeric_limits<float>::max_digits10);
    snapshot_vector(root+"/grad_perda",s.grad,10);
    const char* names[]={"l1","l2","l3"};
    const int ins[]={64,32,16}, outs[]={32,16,10}, wo[]={0,2048,2560}, bo[]={0,32,48}, gi[]={0,64,96};
    const float* inputs[]={input.data(),s.a,s.a+32};
    const float* grads[]={s.gin+64,s.gin+96,s.grad};
    for(int l=0;l<3;++l) {
        const std::string f=root+"/"+names[l]+"/forward", b=root+"/"+names[l]+"/backward";
        snapshot_vector(f+"/entrada",inputs[l],ins[l]); snapshot_vector(f+"/z",s.z+bo[l],outs[l]);
        snapshot_vector(f+"/saida",s.a+bo[l],outs[l]);
        snapshot_vector(b+"/grad_saida",grads[l],outs[l]); snapshot_vector(b+"/delta",s.delta+bo[l],outs[l]);
        snapshot_vector(b+"/grad_entrada",s.gin+gi[l],ins[l]);
        snapshot_params_host(s,b+"/antes",names[l],ins[l],outs[l],wo[l],bo[l],true);
        snapshot_params_host(s,b+"/depois",names[l],ins[l],outs[l],wo[l],bo[l],false);
        for(int i=0;i<outs[l];++i) {
            float d=s.delta[bo[l]+i];
            Barenn::Oracle::valor(b+"/grad_vies/"+std::to_string(i),d,std::numeric_limits<float>::max_digits10);
            for(int j=0;j<ins[l];++j)
                Barenn::Oracle::valor(b+"/grad_peso/"+std::to_string(i)+"/"+std::to_string(j),d*inputs[l][j],std::numeric_limits<float>::max_digits10);
        }
    }
}

#endif

int main(int argc, char** argv) {
    try {
        if (argc != 1 && argc != 3 && argc != 5) {
            std::cerr << "Uso: treino_cuda [treino teste [numero_amostras epocas]]\n";
            return 1;
        }
        const std::string treino = argc >= 3 ? argv[1] : "optdigits.tra";
        const std::string teste = argc >= 3 ? argv[2] : "optdigits.tes";
        const auto data = Barenn::ler_dados(treino);
        const auto test = Barenn::ler_dados(teste);
        if (std::filesystem::equivalent(treino, teste))
            throw std::runtime_error("Use arquivos distintos para treino e teste.");
        if (data.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("Conjunto de treino grande demais");
        size_t n_chars=0, e_chars=0;
        const int n = argc == 5 ? std::stoi(argv[3], &n_chars) : static_cast<int>(data.size());
        const int epochs = argc == 5 ? std::stoi(argv[4], &e_chars) : 51;
        if (argc == 5 && (n_chars != std::string(argv[3]).size() || e_chars != std::string(argv[4]).size()))
            throw std::runtime_error("Número de amostras ou épocas inválido");
        if (n < 1 || static_cast<size_t>(n) > data.size() || epochs < 1)
            throw std::runtime_error("Número de amostras ou épocas inválido");
        auto total_start = std::chrono::steady_clock::now();
        std::vector<float> images(static_cast<size_t>(n)*Barenn::entradas);
        for (int i = 0; i < n; ++i)
            std::copy(data[i].imagem.begin(), data[i].imagem.end(), images.begin()+static_cast<size_t>(i)*Barenn::entradas);
        std::vector<int> labels(n);
        for(int i=0;i<n;++i) labels[i]=data[i].rotulo;
        DeviceBuffer<int> d_labels(n);
        CUDA(timed_copy(d_labels.get(),labels.data(),n*sizeof(int),cudaMemcpyHostToDevice));
        DeviceBuffer<float> d_images(images.size());
        CUDA(timed_copy(d_images.get(), images.data(), images.size()*sizeof(float), cudaMemcpyHostToDevice));
        LayerBuffers buffers1(64,32), buffers2(32,16), buffers3(16,10);
        Layer l1=buffers1.view(), l2=buffers2.view(), l3=buffers3.view();
        DeviceBuffer<float> losses(n);
        std::vector<float> host_losses(n);
        std::vector<double> epoch_losses(epochs);
#ifdef BARENN_ORACLE
        DeviceBuffer<Snapshot> d_traces(2);
        Barenn::Oracle::iniciar();
        if (Barenn::Oracle::ativa()) {
            params(l1,"inicial","l1"); params(l2,"inicial","l2"); params(l3,"inicial","l3");
        }
#endif
        auto start = std::chrono::steady_clock::now();
        for (int e = 0; e < epochs; ++e) {
#ifdef BARENN_ORACLE
          bool capture=Barenn::Oracle::ativa() && (e==0 || e==1 || e==50);
          Snapshot* current_traces=capture ? d_traces.get() : nullptr;
#else
          Snapshot* current_traces=nullptr;
#endif
          train_epoch<<<1,64>>>(l1,l2,l3,d_images.get(),d_labels.get(),n,losses.get(),current_traces);
          CUDA(cudaGetLastError());
          // Uma cópia por época; a soma em double segue a ordem da CPU.
          CUDA(timed_copy(host_losses.data(),losses.get(),n*sizeof(float),cudaMemcpyDeviceToHost));
          double total=0;
          for (float value : host_losses) total+=value;
          epoch_losses[e]=total/n;
#ifdef BARENN_ORACLE
          if(capture) {
              Snapshot traces[2];
              CUDA(timed_copy(traces,d_traces.get(),std::min(n,2)*sizeof(Snapshot),cudaMemcpyDeviceToHost));
              for(int i=0;i<std::min(n,2);++i) write_snapshot(traces[i],data[i].imagem,e+1,i,host_losses[i]);
          }

          if (Barenn::Oracle::ativa())
              Barenn::Oracle::valor("epoca/"+std::to_string(e+1)+"/perda",epoch_losses[e]);
#endif
        }
        CUDA(cudaDeviceSynchronize());
        std::chrono::duration<double> elapsed = std::chrono::steady_clock::now()-start;
        std::cout << std::setprecision(17);
        for (int e=0;e<epochs;++e) std::cout << "PERDA_EPOCA " << e+1 << " " << epoch_losses[e] << "\n";
        std::cout << "Tempo de treino (inclui lançamentos e sincronização): " << elapsed.count() << " s\n";
#ifdef BARENN_ORACLE
        if (Barenn::Oracle::ativa()) {
            params(l1,"final","l1"); params(l2,"final","l2"); params(l3,"final","l3");
        }
#endif
        // Avaliação sequencial por imagem; pesos permanecem fixos na GPU.
        auto evaluation_start = std::chrono::steady_clock::now();
        DeviceBuffer<float> d_test(64);
        std::array<std::array<size_t,10>,10> confusion{};
        size_t hits = 0;
        double test_loss = 0;
        for (size_t i=0; i<test.size(); ++i) {
            CUDA(timed_copy(d_test.get(),test[i].imagem.data(),64*sizeof(float),cudaMemcpyHostToDevice));
            forward<<<1,64>>>(l1,d_test.get(),false);
            forward<<<1,64>>>(l2,l1.a,false);
            forward<<<1,64>>>(l3,l2.a,true);
            CUDA(cudaGetLastError());
            std::vector<float> scores(10);
            CUDA(timed_copy(scores.data(),l3.a,10*sizeof(float),cudaMemcpyDeviceToHost));
            int predicted = static_cast<int>(std::max_element(scores.begin(),scores.end())-scores.begin());
            ++confusion[test[i].rotulo][predicted];
            hits += predicted == test[i].rotulo;
#ifdef BARENN_ORACLE
            Barenn::Oracle::treino=false; Barenn::Oracle::epoca=0; Barenn::Oracle::amostra=i;
#endif
            test_loss += Barenn::perda(scores,test[i].rotulo);
#ifdef BARENN_ORACLE
            if (Barenn::Oracle::ativa()) {
                const std::string key=Barenn::Oracle::prefixo();
                Barenn::Oracle::vetor(key+"/escore",scores);
                Barenn::Oracle::valor(key+"/rotulo",test[i].rotulo);
                Barenn::Oracle::valor(key+"/previsto",predicted);
                float second=-std::numeric_limits<float>::infinity();
                for (int j=0;j<10;++j) if (j!=predicted) second=std::max(second,scores[j]);
                Barenn::Oracle::valor(key+"/margem",scores[predicted]-second,std::numeric_limits<float>::max_digits10);
            }
#endif
        }
#ifdef BARENN_ORACLE
        if (Barenn::Oracle::ativa()) {
            for (int i=0;i<10;++i) for (int j=0;j<10;++j)
                Barenn::Oracle::valor("confusao/"+std::to_string(i)+"/"+std::to_string(j),confusion[i][j]);
            Barenn::Oracle::valor("teste/acertos",hits);
            Barenn::Oracle::terminar();
        }
        d_traces.release();
#endif
        losses.release();
        std::cout << "Avaliacao: " << hits << "/" << test.size() << " acertos; perda media " << test_loss/test.size() << "\n";
        std::cout << "Matriz de confusao (linhas: real; colunas: previsto):\n";
        for (const auto& row : confusion) {
            for (size_t j=0;j<row.size();++j) std::cout << (j ? " " : "") << row[j];
            std::cout << "\n";
        }
        d_test.release();
        double evaluation_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-evaluation_start).count();
        buffers1.release(); buffers2.release(); buffers3.release();
        d_images.release(); d_labels.release();
        std::chrono::duration<double> total_elapsed = std::chrono::steady_clock::now()-total_start;
        std::cout << "Tempo avaliacao (inclui transferencias): " << evaluation_seconds << " s\n";
        std::cout << "Tempo acumulado chamadas cudaMemcpy H2D: " << copy_h2d_seconds << " s; D2H: " << copy_d2h_seconds << " s\n";
        std::cout << "Tempo total (alocacao, transferencias, treino, avaliacao e liberacao): " << total_elapsed.count() << " s\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
