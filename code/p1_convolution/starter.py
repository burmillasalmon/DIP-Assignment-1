"""P1 starter. Implement every function marked TODO.

Rules: NumPy array arithmetic only. scipy/cv2/skimage may be used to CHECK
your answers, never to produce them. numpy.fft is allowed in conv2d_fft.
"""
import numpy as np
import time
from scipy.signal import correlate2d
import tracemalloc
import matplotlib.pyplot as plt
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


def numeric_rank(K, tol=1e-14):
    """TODO 1.2: numerical rank from the singular values."""
    S = np.linalg.svd(K, compute_uv=False)
    count = 0
    #print(max(S))
    for i in range(len(S)):
        if(S[i]>tol):
            count+=1
    return count;

def kernel_analysis():
    k = 15
    kernels = kernel_bank(k)
    for idx, (name,K) in enumerate(kernels.items()):
        rank = numeric_rank(K)
        print("name:", name, "rank:",rank)

        # plot
        S = np.linalg.svd(K, compute_uv = False)
        S_norm = S/S[0]

        plt.subplot(3,3, idx+1)
        plt.stem(S_norm, markerfmt='o', basefmt=" ")
        plt.ylim(1e-17, 1.1)
        plt.yscale('log')
        plt.title(name)
        plt.xlabel("Singualr Value Index")
        plt.ylabel("Normalized log(SV)")
    plt.tight_layout()
    plt.savefig("kernel_single_value_spectra.png", dpi=300)
    plt.show()

def conv2d_loops(img, K):
    x,y = img.shape
    kx,ky = K.shape
    pH, pW = kx//2, ky//2
    padded_img = np.pad(img, ((pH, pH), (pW, pW)), mode='constant', constant_values=0)
    final_image = np.zeros((x, y))
    for i in range(x):
        for j in range(y):
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
    return ans.real
    
def conv1d(img, kernel1d, axis):
    k = len(kernel1d)
    pad = k//2
    x,y = img.shape
    if(axis==1):
        padded = np.pad(img, ((0,0),(pad,pad)), mode='constant')
        answer = np.zeros((x,y))
        for i in range(k):
            answer = answer + padded[:, i:i+y] * kernel1d[k-1-i]
    else:
        padded = np.pad(img, ((pad,pad), (0,0)), mode='constant')
        answer = np.zeros((x,y))
        for i in range(k):
            answer = answer + padded[i: i+x, :] * kernel1d[k-1-i]
    return answer


def conv2d_separable(img, K):
    U, S, Vh = np.linalg.svd(K)
    u1 = U[:, 0]
    v1 = Vh[0,:]
    temp = conv1d(img, v1, axis=1)
    answer = conv1d(temp, u1, axis=0) * S[0]
    return answer;

def conv2d_lowrank(img, K, r):
    U, S, Vh = np.linalg.svd(K)
    r = min(r, len(S))
    X, Y = img.shape
    answer = np.zeros((X,Y))
    for i in range(r):
        ui = U[:, i]
        vi = Vh[i,:]
        temp = conv1d(img, vi, axis = 1)
        component = conv1d(temp, ui,axis=0)

        answer += S[i] * component
    return answer

def psnr(a, b, peak=255.0):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64))**2)
    return float("inf") if mse == 0 else 10 * np.log10(peak * peak / mse)

def psnr_vs_r():
    k = 21
    kernels = ['disk','log','motion_45deg','random']
    root = Path(__file__).resolve().parents[2]
    img = np.asarray(Image.open(root / "images/p1/base_2048.png")).astype(float)[:512, :512]    
    
    K_rand = kernel_bank(k)['random']
    U,S,Vh = np.linalg.svd(K_rand)
    print("Random kernel singular values", S[:5],"...",S[-5:])
    print("Sum of sv:",np.sum(S))
    ref = conv2d_im2col(img,K_rand)
    full_rank_approx = conv2d_lowrank(img,K_rand,k)
    print("PSNR at r = k:",psnr(ref, full_rank_approx))
    print("Max diff at r = k:", np.abs(ref - full_rank_approx).max())



    plt.figure(figsize=(10,6))

    for name in kernels:
        K = kernel_bank(k)[name]
        ref = conv2d_im2col(img,K)

        psnr_vals = []
        for r in range(1,k+1):
            approx = conv2d_lowrank(img,K,r)
            psnr_vals.append(psnr(ref,approx))
        plt.plot(range(1,k+1), psnr_vals, marker='o', label=name)
    plt.xlabel("Rank r")
    plt.ylabel("PSNR (db)")
    plt.legend()
    plt.ylim(0,80)
    plt.savefig("psnr_vs_rank.png", dpi=300)
    plt.close()


def aymmetric_test(img):
    K = kernel_bank(7)["sobel3"]
    corr = correlate2d(img, K, mode="same", boundary="fill")
    conv = convolve2d(img, K, mode="same", boundary="fill")
    conv_loops = conv2d_loops(img, K)
    print("Difference from true convolution",np.abs(conv_loops-conv).max())
    print("Difference loops= ",np.abs(conv_loops-corr).max())
    conv_taps = conv2d_taps(img,K)
    print("Difference from true convolution",np.abs(conv_taps-conv).max())
    print("Difference taps = ", np.abs(conv_taps-corr).max())
    conv_fft = conv2d_fft(img,K)
    print("Difference from true convolution",np.abs(conv_fft-conv).max())
    print("Difference fft = ", np.abs(conv_fft - corr).max())
    conv_im2col = conv2d_im2col(img,K)
    print("Difference from true convolution",np.abs(conv_im2col-conv).max())
    print("Difference im2col = ", np.abs(corr - conv_im2col).max())


def timetest(fn,img,K, times=3):
    times_res = []
    for i in range(times):
        start = time.time()
        fn(img,K)
        end = time.time()
        times_res.append(end-start)
    return np.median(times_res)

def runtime_k():
    root = Path(__file__).resolve().parents[2]
    img = np.asarray(Image.open(root / "images/p1/base_2048.png")).astype(float)[:512, :512]
    ks = np.array([3,7,11,15,21,31])
    times = []
    for k in ks:
        K = kernel_bank(k)["gaussian"]
        t = timetest(conv2d_taps, img, K)
        times.append(t)

    times = np.array(times)
    slope, intercept = np.polyfit(np.log(ks),np.log(times), 1)
    print("Slope:",slope)

    plt.figure()
    plt.loglog(ks, times, "o-", label="Tap loop")

    fitted = np.exp(intercept) * ks** slope
    plt.loglog(ks, fitted, "--", label="fitted slope")

    plt.xlabel("Kernel size")
    plt.ylabel("Time taken")
    plt.legend()
    plt.tight_layout()
    plt.savefig("runtime_vs_k.png", dpi=300)

def runtime_n():
    root = Path(__file__).resolve().parents[2]
    img = np.asarray(Image.open(root / "images/p1/base_2048.png")).astype(float)
    ns = np.array([128,256,512,1024,2048])
    times = []
    K = kernel_bank(15)["gaussian"]
    for n_curr in ns:
        img_curr = img[:n_curr, :n_curr]
        t = timetest(conv2d_taps, img_curr, K)
        times.append(t)
    times = np.array(times)
    slope, intercept = np.polyfit(np.log(ns),np.log(times), 1)
    print("Slope:",slope)

    plt.figure()
    plt.loglog(ns, times, "o-", label="Tap loop")

    fitted = np.exp(intercept) * ns** slope
    plt.loglog(ns, fitted, "--", label="fitted slope")

    plt.xlabel("Image size")
    plt.ylabel("Time taken")
    plt.legend()
    plt.tight_layout()
    plt.savefig("runtime_vs_n.png", dpi=300)

def im2col_memory_vs_N():
    root = Path(__file__).resolve().parents[2]
    img_full = np.asarray(Image.open(root / "images/p1/base_2048.png")).astype(float)
    ns = np.array([128,256,512,1024,2048])
    K = kernel_bank(15)["gaussian"]
    for N in ns:
        img = img_full[:N, :N]
        tracemalloc.start()
        try:
            conv2d_im2col(img, K)
            current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        print(f"N=",N, "peak=",peak)

def compare_taps_im2col():
    root = Path(__file__).resolve().parents[2]
    img_full = np.asarray(Image.open(root / "images/p1/base_2048.png")).astype(float)[:512,:512]
    ks = [3, 7, 11, 15, 21, 31]
    for k in ks:
        K = kernel_bank(k)["gaussian"]
        tap_time = timetest(conv2d_taps, img_full, K)
        im2col_time = timetest(conv2d_im2col, img_full, K)
        speedup = tap_time / im2col_time
        print("k=",k, "tap :", round(tap_time,4), "col :", round(im2col_time,4), "speedup:", round(speedup,4))


def separable_speedup():
    root = Path(__file__).resolve().parents[2]
    img_full = np.asarray(Image.open(root / "images/p1/base_2048.png")).astype(float)
    K = kernel_bank(15)["gaussian"]
    for N in [512, 2048]:
        img = img_full[:N, :N]
        tap_time = timetest( conv2d_taps, img, K)
        sep_time = timetest(conv2d_separable, img, K)
        speedup = tap_time / sep_time
        print(
            f"N={N}: "
            f"tap={tap_time:.4f}s "
            f"separable={sep_time:.4f}s "
            f"speedup={speedup:.4f}x"
        )

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
    #kernel_analysis()
    #psnr_vs_r()
    #runtime_k()
    #runtime_n()
    im2col_memory_vs_N()
    #compare_taps_im2col()
    #separable_speedup()

