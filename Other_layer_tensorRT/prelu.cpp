/******************************/
/* prelu Layer with tensorRT api
/* author : chenzhi
/* date : 2018.1.31
/* warning: The implementation of the prelu layer is not verified to be correct.
/******************************/
//======================
// pluginimplement.cpp
//======================
PreluPlugin::PreluPlugin(const void* buffer, size_t size)
{
    const char* d = reinterpret_cast<const char*>(buffer), *a = d;
    input_c = read<int>(d);
    input_h = read<int>(d);
    input_w = read<int>(d);//input_h=input_w
    input_count = input_c * input_h * input_w;
    int weightCount = read<int>(d);
    mWeights = deserializeToDevice(d, weightCount);
    assert(d == a + size);
}

PreluPlugin::~PreluPlugin()
{
    cudaFree(const_cast<void*>(mWeights.values));
}
Dims PreluPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    assert(index == 0 && nbInputDims == 1 && inputs[0].nbDims == 3);
    return DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
}

int PreluPlugin::enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream)
{
    printf("start to enqueue Prelu");

    const float *bottom_data = reinterpret_cast<const float*>(inputs[0]);
    float *top_data = reinterpret_cast<float*>(outputs[0]);

    const int count = input_count;//input_c*input_h*input_w
    const int dim = input_h*input_w;
    const int channels = input_c;
    //slope data = prelu.mWeights
    const int div_factor = channel_shared_ ? channels : 1; //channel_shared_ default is false

    PReLUForward(count, channels, dim, bottom_data, top_data, reinterpret_cast<const float*>(mWeights.values), div_factor, stream);

    return 0;
}

size_t DepthwisePlugin::getSerializationSize()
{
    return 4*sizeof(int) + mWeights.count * sizeof(float);
}

void DepthwisePlugin::serialize(void* buffer)
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, dimsBottomData.c());
    write(d, dimsBottomData.h());
    write(d, dimsBottomData.w());
    write(d, (int)mWeights.count);
    serializeFromDevice(d, mWeights);
    assert(d == a + getSerializationSize());
}

void DepthwisePlugin::configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int)
{
    dimsBottomData = DimsCHW{inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]};
}

//======================
// mathFunctions.cu
//======================
template <typename Dtype>
__global__ void PReLU(const int nthreads, const int channels, const int dim, const float* const bottom_data,
		float* const top_data, const float* slope_data, const int div_factor) {
	CUDA_KERNEL_LOOP(index, nthreads) {
        int c = (index / dim) % channels / div_factor;
        top_data[index] = bottom_data[index] > 0 ? bottom_data[index] : bottom_data[index] * slope_data[c];
	}
}
cudaError_t PReLUForward(const int nthreads, const int channels, const int dim, const float* const bottom_data,
		float* const top_data, const float* slope_data, const int div_factor, cudaStream_t stream)
{
    PReLU<float><<<TENSORRT_GET_BLOCKS(nthreads), TENSORRT_CUDA_NUM_THREADS,0,stream>>>(nthreads, channels, dim, bottom_data, top_data, slope_data, div_factor, bias_term_);
    return cudaPeekAtLastError();
}