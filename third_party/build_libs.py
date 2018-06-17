import subprocess
import os
import sys

curr_dir = os.path.dirname(os.path.abspath(__file__))


# build nms
def build_nms(arch):
    cuda_src = os.path.join(curr_dir, "nms", "src", "cuda", "nms_kernel.cu")
    cuda_out = os.path.join(curr_dir, "nms", "src", "cuda", "nms_kernel.cu.o")
    build_cuda = "/usr/local/cuda/bin/nvcc -c -o " + cuda_out + " " + cuda_src + " " \
                 + "-x cu -Xcompiler -fPIC {}".format(arch)
    print(build_cuda)
    build_ext = sys.executable + " " + os.path.join(curr_dir, "nms", "build.py")
    subprocess.call(build_cuda, shell=True)
    subprocess.call(build_ext, shell=True)


# build roi_align
def build_roi_align(arch):
    cuda_src = os.path.join(curr_dir, "roi_align", "src", "cuda", "crop_and_resize_kernel.cu")
    cuda_out = os.path.join(curr_dir, "roi_align", "src", "cuda", "crop_and_resize_kernel.cu.o")
    build_cuda = "/usr/local/cuda/bin/nvcc -c -o " + cuda_out + " " + cuda_src + " " \
                 + "-x cu -Xcompiler -fPIC {}".format(arch)
    print(build_cuda)
    build_ext = sys.executable + " " + os.path.join(curr_dir, "roi_align", "build.py")
    subprocess.call(build_cuda, shell=True)
    subprocess.call(build_ext, shell=True)


if __name__ == "__main__":
    CUDA8_ARCH = "--generate-code arch=compute_35,code=sm_35 " \
                 "--generate-code arch=compute_50,code=sm_50 " \
                 "--generate-code arch=compute_60,code=sm_60"

    CUDA9_ARCH = "--generate-code arch=compute_35,code=sm_35 " \
                 "--generate-code arch=compute_50,code=sm_50 " \
                 "--generate-code arch=compute_60,code=sm_60 " \
                 "--generate-code arch=compute_70,code=sm_70"

    cuda_version = sys.argv[1]
    assert cuda_version in ['cuda8', 'cuda9']
    build_arch = CUDA8_ARCH if cuda_version == 'cuda8' else CUDA9_ARCH

    build_nms(build_arch)
    build_roi_align(build_arch)
