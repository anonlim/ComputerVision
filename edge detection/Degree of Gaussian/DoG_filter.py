import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

class DoG:
    def __init__(self, file):
        #사진을 320*480사이즈로 저장한다.
        self.originalImage = cv2.cvtColor(cv2.imread(file),cv2.COLOR_BGR2RGB)
        self.originalImage=cv2.resize(self.originalImage,(320,480))
        #사진을 흑백으로 저장한다.
        self.src_gray = cv2.cvtColor(self.originalImage, cv2.COLOR_BGR2GRAY)

    #사진의 경계를 벗어나면, 다시 사진내부로 포인터를 넣어주는 함수이다.
    def replicate_boundary(self, r_curr, c_curr, row, col):
        '''현재의 포인터의 위치를 r(ow)_curr, c(olumn)_curr에 저장하고
            전체 사진의 사이즈를 row, col에 각각 저장한다.
        '''
        r_temp = r_curr
        c_temp = c_curr
        #계산 과정중 포인터가 사진의 크기를 벗어나면 경계값으로 다시 옮겨준다.
        if r_temp < 0:
            r_temp += 1
        elif r_temp >= row:
            r_temp -= 1
        if c_temp < 0:
            c_temp += 1
        elif c_temp >= col:
            c_temp -= 1
        return r_temp, c_temp

    #만든 가우시안 커널을 이용하여 이미지를 흐리게하는 함수이다.
    #커널을 파라미터로 받는다.
    def get_GaussianBlurred(self, gkernel):
        #사진의 크기값을 row, col에 각각 저장한다.
        row, col = self.src_gray.shape
        #G는 사진의 크기와 배열의 크기가 같은 0행렬이다.
        G = np.zeros((row, col))
        #사진의 크기에 맞게 커널을 일일이 곱하여 사진을 흐리게한다.
        for r in range(row):
            for c in range(col):
                for i in range(3):
                    for j in range(3):
                        r_temp, c_temp = self.replicate_boundary(r + i - 1, c + j - 1, row, col)
                        #G에 흑백사진과 가우시안 커널을 곱한 값을 저장하여 사진을 흐리게 만든다.
                        G[r, c] += self.src_gray[r_temp, c_temp] * gkernel[i, j]
        return G

    #2차원 커널을 만들기 위해 1차원 커널을 먼저 구한다.
    def get_gaussian_filter_1d(self,size, sigma):
        #filter size는 무조건 odd여야한다
        assert size%2==1
        #1차원 가우시안 배열을 저장한다.
        arr = np.arange(math.trunc(size/2)*(-1), math.ceil(size/2) ,1)
        #가우시안 필터 공식을 적용한다.
        kernel_raw = np.exp((-arr*arr)/(2*sigma*sigma))
        #정규화를 진행한다.
        kernel = kernel_raw/kernel_raw.sum()
        #1차원 커널을 곱하여 2차원커널을 만들어야하므로 (size,0)행렬을 만든다.
        kernel=np.array([kernel]).T
        return kernel

    def DoG_from_Scratches(self):
        #size 3, 시그마 0.5 짜리 1차원 가우시안필터를 만든다.
        xdir_gauss = self.get_gaussian_filter_1d(3,0.5)
        #만든 가우시안 필터를 곱하여 2차원 가우시안 필터로 만든다.
        gkernel1 = np.multiply(xdir_gauss.T, xdir_gauss)

        #size 3, 시그마 3 짜리 1차원 가우시안필터를 만든다.
        xdir_gauss = self.get_gaussian_filter_1d(3, 3)
        #만든 가우시안 필터를 곱하여 2차원 가우시안 필터로 만든다.
        gkernel2 = np.multiply(xdir_gauss.T, xdir_gauss)
        #첫번째 필터를 이용하여 원본 흑백사진을 Blur한다.
        self.G1_scratch = self.get_GaussianBlurred(gkernel1)
        #두번째 필터를 이용하여 원본 흑백사진을 Blur한다.
        self.G2_scratch = self.get_GaussianBlurred(gkernel2)
        #두 사진을 빼서 결과 사진을 self.DoG_scratch에 저장한다.
        self.DoG_scratch = self.G1_scratch - self.G2_scratch

#파일을 DoG 클래스의 attribute로 받아온다.
file = 'C:/Users/User/Desktop/pythonProject/6.jpg'
G = DoG(file)
#원본사진을 출력
plt.imshow(G.originalImage)

#DoG edge detector로 사진을 처리한다.
G.DoG_from_Scratches()

#DoG한 결과 사진을 흑백으로 출력(가우시안 필터1 결과사진-가우시안 필터2 결과사진)및 result.jpg파일로 저장한다.
plt.figure(figsize=(20,30))
plt.subplot(131), plt.imshow(G.DoG_scratch,cmap='gray'),plt.title('Resulting Image',fontsize=15)
plt.xticks([]), plt.yticks([])
plt.savefig('result.jpg',dpi=200,transparent=True,bbox_inches='tight')
#원본사진을 가우시안 필터1번을 사용하여 출력한다.
plt.subplot(132), plt.imshow(G.G1_scratch, cmap='gray'),plt.title('G1',fontsize=15)
plt.xticks([]), plt.yticks([])

#원본사진을 가우시안 필터2번을 사용하여 출력한다.
plt.subplot(133), plt.imshow(G.G2_scratch, cmap='gray'),plt.title('G2',fontsize=15)
plt.xticks([]), plt.yticks([])