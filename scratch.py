import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plt

A = image.imread("svd-image-compression-img.jpg")

R = A[:, :, 0] / 0xff
G = A[:, :, 1] / 0xff
B = A[:, :, 2] / 0xff
R_U, R_S, R_VT = np.linalg.svd(R)
G_U, G_S, G_VT = np.linalg.svd(G)
B_U, B_S, B_VT = np.linalg.svd(B)


def read_as_compressed(u, s, v, r):
    a = np.zeros((u.shape[0], v.shape[1]))
    for i in range(r):
        u_i = u[:, [i]]
        v_i = np.array([v[i]])
        a += s[i] * (u_i @ v_i)
    return a


RANKS = [0, 5, 25, 50, 100, 200]
for i in range(0, 6):
    R_compressed = read_as_compressed(R_U, R_S, R_VT, RANKS[i])
    G_compressed = read_as_compressed(G_U, G_S, G_VT, RANKS[i])
    B_compressed = read_as_compressed(B_U, B_S, B_VT, RANKS[i])
    compressed_float = np.dstack((R_compressed, G_compressed, B_compressed))
    compressed = (np.minimum(compressed_float, 1.0) * 0xff).astype(np.uint8)
    fig = plt.subplot(2, 3, i + 1)
    fig.imshow(compressed)
    fig.set_xticks([])
    fig.set_yticks([])
    fig.set_title(f'''rank {RANKS[i]}''')

plt.tight_layout()
plt.show()
