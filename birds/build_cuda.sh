LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)
nvcc --ptxas-options=-v --compiler-options '-fPIC' -o cuda/libaudio_wombat.so --shared cuda/audio_wombat.cu
