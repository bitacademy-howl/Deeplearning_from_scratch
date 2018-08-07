def write01():
    # 텍스트 파일 쓰기

    f = open('test.txt', 'w', encoding='utf-8')
    size = f.write("Life is too short, You need Python")  # 내용을 쓰고 길이를 반환
    print("write size : {}".format(size))
    f.close()  # 열었으면 꼭 닫기


write01()


def read01():
    # 텍스트 파일 읽기
    f = open("test.txt", 'r', encoding="utf-8")
    text = f.read()
    print(text)
    f.close()


read01()


def write02():
    # 여러줄을 만들어 봅시다
    f = open("multilines.txt", "w", encoding="utf-8")
    for i in range(1, 10):
        f.write("Line %d\n" % i)
    f.close()


write02()


def read02():
    f = open("multilines.txt", "r", encoding="utf-8")
    text = f.read()
    print(text)
    f.close()


read02()


def read02_1():
    f = open("multilines.txt", "r", encoding="utf-8")
    # 루프를 돌면서 한줄씩 읽어와 봅시다
    while True:
        line = f.readline()
        if not line:
            break
        print(line)
    f.close()


read02_1()


def read02_2():
    f = open("multilines.txt", "r", encoding="utf-8")
    lines = f.readlines()
    print(lines)
    line = lines[5]
    print(line)

    f.close()


read02_2()

'''
sample 디렉토리 안에 sangbuk.csv 파일을 열고
각 줄을 dict로 만들고 리스트에 저장해 봅시다.
'''


def slamdunk():
    members = []

    # f = open("./sample/sangbuk.csv", "r")
    f = open("File_IO/sample/sangbuk.csv", "r")

    while True:
        line = f.readline()
        if not line:
            break
        line = line.strip().replace(" ", "")
        info = line.split(",")
        member = {"name": info[0], "number": info[1], "height": info[2], "position": info[3]}

        members.append(member)
    f.close()


# slamdunk()

# 바이너리 파일 다루기 모드 : b
def binfile():
    f_src = open("./sample/rose-flower.jpeg", "rb")
    data = f_src.read()
    f_src.close()

    f_dest = open("./sample/rose-flower-copy.jpeg", "wb")
    f_dest.write(data)
    f_dest.close()


binfile()


def safe_file():
    # 파일은 open하면 반드시 닫아줘야 하는데
    # with ~ as 를 이용하면 파이썬이 사용 후 닫아준다.

    with open("multilines.txt", "r") as f:
        for line in f.readlines():
            print(line, end="")

    print()
    # f가 닫혀있는지 확인
    print(f.closed)


safe_file()