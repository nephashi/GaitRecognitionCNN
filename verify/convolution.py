# this script can verify result of convolution operation

input = [
    [1,4,7,10,0],
    [13,16,19,22,0],
    [25,28,31,34,0],
    [37,40,43,46,0],
    [0,0,0,0,0]
]

kernel = [
    [0.007, 0.0126],
    [0.013, -0.0021]
]

input2 = [
    [2,5,8,11,0],
    [14,17,20,23,0],
    [26,29,32,35,0],
    [38,41,44,47,0],
    [0,0,0,0,0]
]

kernel2 = [
    [0.0397, -0.0062],
    [0.0316, -0.0415]
]

input3 = [
    [3,6,9,12,0],
    [15,18,21,24,0],
    [27,30,33,36,0],
    [39,42,45,48,0],
    [0,0,0,0,0]
]

kernel3 = [
    [0.0298, -0.0352],
    [-0.0444, -0.0028]
]

def conv(input, kernel):
    input_len = len(input)
    kernel_size = len(kernel)

    ans = []
    for i in range(input_len - kernel_size + 1):
        row = []
        for j in range(input_len - kernel_size + 1):
            sum = 0
            for conv_x in range(kernel_size):
                for conv_y in range(kernel_size):
                    sum += input[i + conv_x][j + conv_y] * kernel[conv_x][conv_y]
            row.append(sum)
        ans.append(row)
    return ans

print(conv(input3,kernel3))