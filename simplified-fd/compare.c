#include <stdio.h>
#include <math.h>

#define Null -99999

void compare(float *input1, float *input2, int len);
void rmse(float *dif, int len);
void stdev(float mean, float *dif, int len);


void main ()
{
    int mtxBufferLength = (315 + 2 * 50)*(195 + 2 * 50)*sizeof(float);
    float output_cuda[mtxBufferLength], output_dpc[mtxBufferLength];

    FILE *foutput_cuda, *foutput_dpc;
    foutput_cuda = fopen("output_cuda.bin", "rb");
    foutput_dpc = fopen("output_teste.bin", "rb");

    printf("Lendo arquivos...\n");
    fread(output_cuda, sizeof(output_cuda), 1, foutput_cuda);
    fread(output_dpc, sizeof(output_dpc), 1, foutput_dpc);
    fclose(foutput_cuda);
    fclose(foutput_dpc);

    printf("Comparando...\n");
    compare(output_cuda, output_dpc, mtxBufferLength);
}

void compare(float *input1, float *input2, int len)
{
    int i;
    float dif[len], mae, sum, acc;
    int count = 0, length = 0;
    for (i = 0; i < len; i++)
    {
        if (fabsf(input1[i]) < 1000000 && fabsf(input2[i]) < 1000000)
        {
            dif[i] = fabsf(input1[i] - input2[i]);
            sum += dif[i];
            if (input1[i] == input2[i])
            {
                count++;
            }
            length++;
        }
        else
        {
            dif[i] = Null;
        }
    }
    mae = sum / (float)length;
    acc = (float)count/(float)len*100;
    printf("Accuracy: %.5f%\n", acc);
    rmse(dif,len);
    printf("MAE: %.15f\n",mae);
    stdev(mae, dif, len);
}

void rmse(float *dif, int len)
{
    int i, length = 0;
    float sum = 0.0;
    for (i = 0; i < len; i++)
    {
        if (dif[i] != Null)
        {
            sum += pow(dif[i], 2);
            length++;
        }
    }
    printf("RMSE: %.15f\n", sqrt(sum/(float)length));
}

void stdev(float mean, float *dif, int len)
{
    int i, length = 0;
    float p = 0.0, sigma;
    for(i = 0; i < len; i++){
        if (dif[i] != Null)
        {    
            p = p + pow(dif[i] - mean, 2);
            length++;
        }
    }
    sigma = sqrt(p/((float)length-1));
    printf("Standard deviation: %.15f\n", sigma);
}
