


def MAE(vector):
    float sum = 0.0
    for i in range(len(vector)):
        
    return 

foutput_cuda = open("output_cuda.bin", "rb")
foutput_dpc = open("output_teste.bin", "rb")
print('Lendo arquivos')
output_cuda = foutput_cuda.read()
output_dpc = foutput_dpc.read()

dif = []
for i in range(len(output_cuda)):
    dif.append(output_cuda[i] - output_dpc[i])