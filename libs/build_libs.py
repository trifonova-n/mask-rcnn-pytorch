import subprocess
import os
import sys

curr_dir = os.path.dirname(os.path.abspath(__file__))


# build nms
def build_nms(arch):
    cuda_src = os.path.join(curr_dir, "nms", "src", "cuda", "nms_kernel.cu")
    cuda_out = os.path.join(curr_dir, "nms", "src", "cuda", "nms_kernel.cu.o")
    build_cuda = "/usr/local/cuda/bin/nvcc -c -o " + cuda_out + " " + cuda_src + " " \
                 + "-x cu -Xcompiler -fPIC -arch={}".format(arch)
    build_ext = sys.executable + " " + os.path.join(curr_dir, "nms", "build.py")
    subprocess.call(build_cuda, shell=True)
    subprocess.call(build_ext, shell=True)


# build roi_align
def build_roi_align(arch):
    cuda_src = os.path.join(curr_dir, "roi_align", "src", "cuda", "crop_and_resize_kernel.cu")
    cuda_out = os.path.join(curr_dir, "roi_align", "src", "cuda", "crop_and_resize_kernel.cu.o")
    build_cuda = "/usr/local/cuda/bin/nvcc -c -o " + cuda_out + " " + cuda_src + " " \
                 + "-x cu -Xcompiler -fPIC -arch={}".format(arch)
    build_ext = sys.executable + " " + os.path.join(curr_dir, "roi_align", "build.py")
    subprocess.call(build_cuda, shell=True)
    subprocess.call(build_ext, shell=True)


if __name__ == "__main__":
    build_nms(sys.argv[1])
    build_roi_align(sys.argv[1])
