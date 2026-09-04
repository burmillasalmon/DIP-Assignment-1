"""P1 starter. Implement every function marked TODO.

Rules: NumPy array arithmetic only. scipy/cv2/skimage may be used to CHECK
your answers, never to produce them. numpy.fft is allowed in conv2d_fft.
"""
import numpy as np
import time
from scipy.signal import correlate2d
import tracemalloc
def kernel_bank(k=15):
    """Provided. Do not modify -- your rank table must match these kernels."""
    ax = np.arange(k) - (k - 1) / 2
    box = np.ones((k, k)) / (k * k)
    s = k / 6.0
    g1 = np.exp(-(ax**2) / (2 * s * s)); g1 /= g1.sum()
    gauss = np.outer(g1, g1)
    sobel = np.outer([1, 2, 1], [-1, 0, 1]).astype(float)
    xx, yy = np.meshgrid(ax, ax); r2 = xx**2 + yy**2
    log = (r2 - 2*s*s) / (s**4) * np.exp(-r2 / (2*s*s))
    motion0 = np.zeros((k, k)); motion0[k // 2, :] = 1.0 / k
    disk = (r2 <= (k/2.0)**2).astype(float); disk /= disk.sum()
    rand = np.random.default_rng(0).normal(size=(k, k)); rand /= np.abs(rand).sum()
    return {"box": box, "gaussian": gauss, "sobel3": sobel, "log": log,
            "log_dc_removed": log - log.mean(), "motion_0deg": motion0,
            "motion_45deg": np.eye(k) / k, "disk": disk, "random": rand}


def numeric_rank(K, tol=1e-10):
    """TODO 1.2: numerical rank from the singular values."""
    raise NotImplementedError

def conv2d_loops(img, K):
    x,y = img.shape
    kx,ky = K.shape
    pH, pW = kx//2, ky//2
    padded_img = np.pad(img, ((pH, pH), (pW, pW)), mode='constant', constant_values=0)
    final_image = np.zeros((x, y))
    for i in range(x):
        for j in range(y):
            sumn = 0.0
            for a in range(kx):
                for b in range(ky):
                    final_image[i,j] += padded_img[i + a, j + b] * K[kx - 1 - a, ky - 1 - b]
    return final_image

def conv2d_loops_timetest():
    img = np.asarray(Image.open(root/ "images/p1/base_2048.png")).astype(float)
    image_128 = img[:128, :128]
    image_2048 = img[:2048, :2048]
    K7 = kernel_bank(7)["gaussian"]
    K15 = kernel_bank(15)["gaussian"]
    timestart = time.time()
    temp = conv2d_loops(image_128, K7)
    time_end = time.time()
    print("Time for 128*128 K = 7 ->", time_end-timestart)

    timestart = time.time()
    temp = conv2d_loops(image_2048, K15)
    time_end = time.time()
    print("Time for 2048*2048 K = 15 ->", time_end-timestart)

def conv2d_taps(img, K):
    kx,ky = K.shape
    ph,pw = kx//2, ky//2
    x,y = img.shape
    padded_img = np.pad(img, ((ph,ph), (pw,pw)), mode='constant', constant_values=0)
    final_image = np.zeros((x,y))
    K = K[::-1, ::-1]
    for i in range(kx):
        for j in range(ky):
            curr = padded_img[i:i+x, j:j+y]
            #print(curr.shape)
            curr = curr*K[i][j]
            final_image += curr
    return final_image

def conv2d_im2col(img, K):
    kx,ky = K.shape
    ph,pw = kx//2, ky//2
    x,y = img.shape
    padded_img = np.pad(img, ((ph,ph), (pw,pw)), mode='constant', constant_values=0)
    
    patch = np.lib.stride_tricks.sliding_window_view(padded_img, (kx,ky))
    patch_matrix = patch.reshape(x*y, kx*ky)

    k_flat = K[::-1, ::-1].ravel()
    #print(patch_matrix.shape)
    final_image = patch_matrix @ k_flat
    return final_image.reshape(x,y)

def conv2d_im2col_memtest(img,K):
    tracemalloc.start()
    result = conv2d_im2col(img,K)
    current,peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print("Max usage of memory: ", peak / 1024 / 1024, "MB")
    return

def conv2d_fft(img, K):
    x, y = img.shape
    kx, ky = K.shape
    #K = K[::-1, ::-1]
    freq_img = np.fft.fft2(img, s=((x+kx-1),(y+ky-1)))
    frequency_k = np.fft.fft2(K, s=((x+kx-1),(y+ky-1)))

    index = (kx-1)//2
    index2 = (ky-1)//2
    ans = np.fft.ifft2(freq_img* frequency_k)[index:index+x, index2:index2+y]
    return ans
    


def conv2d_separable(img, K, tol=1e-10):
    """TODO 1.3: rank-1 only. Raise if K is not rank-1."""
    raise NotImplementedError


def conv2d_lowrank(img, K, r):
    """TODO 1.3: sum of r separable passes from the truncated SVD."""
    raise NotImplementedError


def psnr(a, b, peak=255.0):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64))**2)
    return float("inf") if mse == 0 else 10 * np.log10(peak * peak / mse)

def aymmetric_test(img):
    K = kernel_bank(7)["sobel3"]
    # scipy correlation to check the answer
    corr = correlate2d(img, K, mode="same", boundary="fill")

    conv_loops = conv2d_loops(img, K)
    print("Difference loops= ",np.abs(conv_loops-corr).max())
    conv_taps = conv2d_taps(img,K)
    print("Difference taps = ", np.abs(conv_taps-corr).max())
    conv_fft = conv2d_fft(img,K)
    print("Difference fft = ", np.abs(conv_fft - corr).max())
    conv_im2col = conv2d_im2col(img,K)
    print("Difference im2col = ", np.abs(corr - conv_im2col).max())

if __name__ == "__main__":
    from pathlib import Path
    from PIL import Image
    from scipy.signal import convolve2d          # checking only
    root = Path(__file__).resolve().parents[2]
    img = np.asarray(Image.open(root / "images/p1/base_2048.png")).astype(float)[:128, :128]
    K = kernel_bank(7)["gaussian"]
    ref = convolve2d(img, K, mode="same", boundary="fill")
    for fn in (conv2d_loops, conv2d_taps, conv2d_im2col, conv2d_fft):
        try:
            print(f"{fn.__name__:16s} max|err| = {np.abs(fn(img, K) - ref).max():.3e}")
        except NotImplementedError:
            print(f"{fn.__name__:16s} not implemented")

    #conv2d_loops_timetest()
    #aymmetric_test(img)
    #conv2d_im2col_memtest(img,K)