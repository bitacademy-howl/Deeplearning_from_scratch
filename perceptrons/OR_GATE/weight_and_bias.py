import numpy as np

# OR 게이트 연산
# 가중치와 바이어스 도입...
# OR gate 진리표
#####################################
#    x1    #    x2    #      y      #
#####################################
#     0    #     0    #      0      #
#     1    #     0    #      1      #
#     0    #     1    #      1      #
#     1    #     1    #      1      #
#####################################

# Definition : 입력값이 둘 중 하나만 1 이면 1 을 출력


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = - 0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

print(OR(0,0))
print(OR(0,1))
print(OR(1,0))
print(OR(1,1))