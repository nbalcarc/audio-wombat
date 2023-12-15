#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
 


 __global__ void vecmul(float *A, float* B, float *C, int size)
{
    // Row and Column indexes: 
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    // Are they bellow the maximum?
    if (col < size && row < size) {
       float result = 0;
       for(int ix=0;ix<size;ix++) {
          result += A[row*size+ix]*B[ix*size+col];
       }
       C[row*size+col] = result;
    }
}


__global__ void frame_image_rot(char* frames, char* image, uint32_t* output_raw, uint32_t image_size, uint32_t stride) {
    int id_frame = threadIdx.x; //the frame num
    int id_rot = threadIdx.y; //the rotation num
    uint32_t accum = 0; //i64

    int index;
    for (int i = 0; i < image_size; i++) { //for every index
        index = (i + id_rot * stride) % image_size;
        if (frames[id_frame * image_size + i] != 0) {
            accum += abs(frames[id_frame * image_size + i] - image[index]);
        }
    }

    output_raw[id_frame * (image_size / stride) + id_rot] = accum; //output
}


__global__ void min_outputs(uint32_t* output_raw, uint32_t* output, uint32_t rots) {
    int id = threadIdx.x;
    //int id = blockIdx.x*blockDim.x+threadIdx.x;
    //int row = blockIdx.y*blockDim.y+threadIdx.y;
    //int col = blockIdx.x*blockDim.x+threadIdx.x;
    //printf("aaaaaa: %d\n", col);
    //printf("aaaaab: %d\n", row);
    //printf("aaaaac: %d\n", id);

    for (int i = 0; i < rots; i++) {
        if (output_raw[id * rots + i] < output[id]) {
            output[id] = output_raw[id * rots + i];
        }
    }
}


// FRAMES = [a,b,c] 
// OG_IMAGE
// ROT()

// ROT_IMAGES = [ROT(OG_IMAGE, s) for s in image_size/stride]
// FOR EACH IMAGE IN ROT_IMAGES:
//     FOR EACH FRAME IN FRAMES:
//         audio wombat(IMAGE, FRAME)


extern "C" {

    void maxmul(float *A, float* B, float *C, int size) {

        int total = size*size;

        // Allocate device memory:
        float* gpu_A;
        float* gpu_B;
        float* gpu_C;
        int msize = total * sizeof(float);
        cudaMalloc((void**)&gpu_A, msize);
        cudaMemcpy(gpu_A,A,msize,cudaMemcpyHostToDevice);
        cudaMalloc((void**)&gpu_B, msize);
        cudaMemcpy(gpu_B,B,msize,cudaMemcpyHostToDevice);
        cudaMalloc((void**)&gpu_C,msize);

        // Blocks & grids:
        dim3 blocks(size,size);
        dim3 grid(1,1);

        // Call the kernel:
        vecmul<<<grid,blocks>>>(gpu_A,gpu_B,gpu_C,size);

        // Get the result Matrix:
        cudaMemcpy(C,gpu_C,msize,cudaMemcpyDeviceToHost);

        C[0] = 2.3;

        //Free device matrices
        cudaFree(gpu_A);
        cudaFree(gpu_B);
        cudaFree(gpu_C);
    }


    // Calculates the output of the given neural network arrays
    void audio_wombat_cuda(
        char* frames,
        char* image,
        uint32_t* output,

        uint32_t image_size,
        uint32_t stride,
        uint32_t frame_count
    ) {

        uint32_t* g_output_raw;

        // allocate arrays onto vram
        char* g_frames;
        char* g_image;
        uint32_t* g_output;

        cudaMalloc((void**)&g_frames, image_size * frame_count * sizeof(char));
        cudaMemcpy(g_frames, frames, image_size * frame_count * sizeof(char), cudaMemcpyHostToDevice);

        cudaMalloc((void**)&g_image, image_size * sizeof(char));
        cudaMemcpy(g_image, image, image_size * sizeof(char), cudaMemcpyHostToDevice);

        cudaMalloc((void**)&g_output, frame_count * sizeof(uint32_t));
        cudaMemcpy(g_output, output, frame_count * sizeof(uint32_t), cudaMemcpyHostToDevice);

        int total_rots = frame_count * image_size / stride;
        cudaMalloc((void**)&g_output_raw, total_rots * sizeof(uint32_t)); //one for each frame rotation

        dim3 grid(1, 1);
        dim3 blocks(frame_count, image_size / stride);

        // launch one thread per all rotations for all threads
        frame_image_rot<<<grid, blocks>>>(
            g_frames, //pointer to current frame
            g_image,
            g_output_raw,
            image_size,
            stride
        );

        // find the mins, one thread per rotation
        min_outputs<<<1, frame_count>>>(g_output_raw, g_output, image_size / stride);

        cudaMemcpy(output, g_output, frame_count * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        cudaFree(g_frames);
        cudaFree(g_image);
        cudaFree(g_output);
        cudaFree(g_output_raw);
    }
}

