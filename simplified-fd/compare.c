#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "functions.h"

void compare(float *input1, float *input2, int len);
void rmse(float *dif, int len);
void stdev(float mean, float *dif, int len);
FILE* read_file(char *path);

void compare2D(int x, int z)
{
    int mtxBufferLength = x*z*sizeof(float);
    float **output_1;
    float **output_2;
    output_1 = alloc2float(x, z);
    output_2 = alloc2float(x, z);
    memset(*output_1, 0, mtxBufferLength);
    memset(*output_2, 0, mtxBufferLength);

    FILE *foutput_1, *foutput_2;
    foutput_1 = read_file("../host/rtm-cuda/output/dir.image");
    foutput_2 = read_file("./complete-code/output/dir.image");
    printf("Comparação de binários 2D:\n");
    printf("Lendo arquivos...\n");
    fread(*output_1, sizeof(float), x*z, foutput_1);
    fread(*output_2, sizeof(float), x*z, foutput_2); 
    fclose(foutput_1);
    fclose(foutput_2);

    printf("Comparando...\n");
    compare(*output_1, *output_2, x*z);
    free2float(output_1);
    free2float(output_2);
}

void compare1alloc(int len)
{
    int mtxBufferLength = len*sizeof(float);
    float *output_1;
    float *output_2;
    output_1 = alloc1float(len);
    output_2 = alloc1float(len);
    memset(output_1, 0, mtxBufferLength);
    memset(output_2, 0, mtxBufferLength);

    FILE *foutput_1, *foutput_2;
    foutput_1 = read_file("../host/rtm-cuda/file-teste");
    foutput_2 = read_file("./complete-code/file-teste");
    printf("Comparação de binários 1D:\n");
    printf("Lendo arquivos...\n");
    fread(output_1, sizeof(float), len, foutput_1);
    fread(output_2, sizeof(float), len, foutput_2); 
    fclose(foutput_1);
    fclose(foutput_2);

    printf("Comparando...\n");
    compare(output_1, output_2, len);
    free1float(output_1);
    free1float(output_2);
}

void main ()
{
    
    // compare3Dread(6,315,1700);
    printf("\n");
    compare1alloc(315*195);
}

void compare(float *input1, float *input2, int len)
{
    int i;
    float dif[len], mae, sum = 0.0, acc;
    int count = 0, length = 0;
    for (i = 0; i < len; i++)
    {
        dif[i] = fabsf(input1[i] - input2[i]);
        sum += dif[i];
        if (dif[i] == 0.0)
        {
            count++;
        }
        else
        {
            printf("%.30f\n", dif[i]);
        }
        
        length++;
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
        sum += powf(dif[i], 2);
        length++;
    }
    printf("RMSE: %e\n", sqrtf(sum/(float)length));
}

void stdev(float mean, float *dif, int len)
{
    int i, length = 0;
    float p = 0.0, sigma = 0.0;
    for(i = 0; i < len; i++){
        p = p + powf(dif[i] - mean, 2);
        length++;
    }
    sigma = sqrt(p/((float)length-1));
    printf("Standard deviation: %e\n", sigma);
}

FILE* read_file(char *path)
{
    FILE *file;
    file = fopen(path, "r");
    if(file == NULL)
    {
        perror("File 1: ");
        return;
    }
    return file;
}