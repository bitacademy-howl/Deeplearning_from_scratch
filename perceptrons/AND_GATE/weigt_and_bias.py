# AND 게이트 연산
# 가중치와 바이어스 도입...
# And gate 진리표
#####################################
#    x1    #    x2    #      y      #
#####################################
#     0    #     0    #      0      #
#     1    #     0    #      0      #
#     0    #     1    #      0      #
#     1    #     1    #      1      #
#####################################

# Definition : 입력값이 둘 다 1인 경우에만 1을 출력

import numpy as np

def AND(x1, x2):
    x = np.array([x1,x2])
    w = np.array([0.5, 0.5])

    b = -0.7
    tmp = np.sum(x*w) + b
    if tmp <= 0:
        return 0
    else:
        return 1

print(AND(0,0))
print(AND(0,1))
print(AND(1,0))
print(AND(1,1))