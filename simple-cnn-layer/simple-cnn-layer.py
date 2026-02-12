import numpy as np

def conv2d(x, W, b):
    """
    Simple 2D convolution layer forward pass.
    Valid padding, stride=1.
    """
    # Write code here
    npX = np.array(x)
    npW = np.array(W)
    npB = np.array(b)

    (N, C_in, cin_w, cin_h)  =npX.shape
    (C_out, C_in, kernel_w, kernel_h) = npW.shape

    out_w = np.floor(cin_w - kernel_w).astype(np.int32) + 1
    out_h = np.floor(cin_h - kernel_h).astype(np.int32) + 1
    offset_x = np.ceil((cin_w - out_w)/2.0).astype(np.int32)
    offset_y =  np.ceil((cin_h - out_h)/2.0).astype(np.int32)
    output = np.empty((N, C_out, out_w,out_h),dtype = x.dtype)
    for n in range(N):
        for c in range(C_out):
            for i in range(out_w):
                for j in range(out_h):
                    source = npX[n,:, i:i + kernel_w,j :j + kernel_h]
                    output[n, c, i,j] = np.sum(source * npW[c]) + npB[c]
    return output

            