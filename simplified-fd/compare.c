#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define Null -99999

void compare(float *input1, float *input2, int len);
void rmse(float *dif, int len);
void stdev(float mean, float *dif, int len);


void main ()
{
    int mtxBufferLength = (315 + 2 * 50)*(195 + 2 * 50)*sizeof(float);
    float *output_cuda;
    float *output_dpc;
    output_cuda = (float*)malloc(mtxBufferLength);
    output_dpc = (float*)malloc(mtxBufferLength);
    memset(output_cuda, 0, mtxBufferLength);
    memset(output_dpc, 0, mtxBufferLength);

    FILE *foutput_cuda, *foutput_dpc;
    foutput_cuda = fopen("output_cuda.bin", "rb");
    foutput_dpc = fopen("output_teste.bin", "rb");

    printf("Lendo arquivos...\n");
    fread(output_cuda, sizeof(float), (315 + 2 * 50)*(195 + 2 * 50), foutput_cuda);
    fread(output_dpc, sizeof(float), (315 + 2 * 50)*(195 + 2 * 50), foutput_dpc); 
    fclose(foutput_cuda);
    fclose(foutput_dpc);

    printf("Comparando...\n");
    compare(output_cuda, output_dpc, mtxBufferLength/sizeof(float));
    free(output_cuda);
    free(output_dpc);
}

void compare(float *input1, float *input2, int len)
{
    int i;
    float dif[len], mae, sum = 0.0, acc;
    int count = 0, length = 0;
    for (i = 0; i < len; i++)
    {
        if (fabsf(input1[i]) < 10 && fabsf(input2[i]) < 10)
        {
            dif[i] = fabsf(input1[i] - input2[i]);
            sum += dif[i];
            if (dif[i] == 0.0)
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
    printf("Accuracy: %.2f%\n", acc);
    rmse(dif,len);
    printf("MAE: %e\n",mae);
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
            sum += powf(dif[i], 2);
            length++;
        }
    }
    printf("RMSE: %e\n", sqrtf(sum/(float)length));
}

void stdev(float mean, float *dif, int len)
{
    int i, length = 0;
    float p = 0.0, sigma = 0.0;
    for(i = 0; i < len; i++){
        if (dif[i] != Null)
        {    
            p = p + powf(dif[i] - mean, 2);
            length++;
        }
    }
    sigma = sqrt(p/((float)length-1));
    printf("Standard deviation: %e\n", sigma);
}
