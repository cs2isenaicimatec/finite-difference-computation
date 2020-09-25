




foutput_cuda = open("output_cuda.bin", "rb")
foutput_dpc = open("output_teste.bin", "rb")
print('Lendo arquivos')
output_cuda = foutput_cuda.read()
output_dpc = foutput_dpc.read()
i = 1321
while i < 1341:
    print(output_cuda[i])
    i+=1