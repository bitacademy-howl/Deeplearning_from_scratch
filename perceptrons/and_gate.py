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

def AND(x1, x2):
    w1, w2 ,theta = 0.5, 0.5, 0.7
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1

# 가중치의 합이 theta를 넘어설때 1을 출력하므로 theta의 값은 0.5 초과의 수치이면 된다.

print(AND(0, 0))
print(AND(1, 0))
print(AND(0, 1))
print(AND(1, 1))

