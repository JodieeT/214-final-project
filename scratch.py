import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def SVD(colorMatrix, rank):
    U, S, V_T = np.linalg.svd(colorMatrix)
    cM = np.zeros((colorMatrix.shape[0], colorMatrix.shape[1]))
    for k in range(rank):
        cM = cM + S[k] * np.dot(U[:, k].reshape(colorMatrix.shape[0], 1), V_T[k, :].reshape(1, colorMatrix.shape[1]))
    cM[cM < 0] = 0
    cM[cM > 255] = 255
    return np.rint(cM).astype("uint8")


img = Image.open(r'p1.jpg', 'r')
imgArr = np.array(img)

R = imgArr[:, :, 0]
G = imgArr[:, :, 1]
B = imgArr[:, :, 2]

rankArr = [5]
for i in range(len(rankArr)):
    cR = SVD(R, rankArr[i])
    cG = SVD(G, rankArr[i])
    cB = SVD(B, rankArr[i])
    compressedImg = np.stack((cR, cG, cB), 2)
    Image.fromarray(compressedImg).save("rank{}".format(rankArr[i]) + ".png")
    fig = plt.subplot(2, 3, i + 1)
    fig.imshow(compressedImg)
    fig.set_xticks([])
    fig.set_yticks([])
    fig.set_title(f'''Rank: {rankArr[i]}''')

plt.tight_layout()
plt.show()