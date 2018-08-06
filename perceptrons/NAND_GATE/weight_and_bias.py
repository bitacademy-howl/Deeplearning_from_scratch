# NAND 게이트 연산
# 가중치와 바이어스 도입...
# NAND gate 진리표
#####################################
#    x1    #    x2    #      y      #
#####################################
#     0    #     0    #      1      #
#     1    #     0    #      1      #
#     0    #     1    #      1      #
#     1    #     1    #      0      #
#####################################

# Definition : 입력값이 둘 다 1인 경우에만 0을 출력 NOT + AND


import numpy as np

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])

    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

print(NAND(0, 0))
print(NAND(0, 1))
print(NAND(1, 0))
print(NAND(1, 1))