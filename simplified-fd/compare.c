#include <stdio.h>
#include <math.h>

void compare(double *input1, double *input2, int len);
void rmse(double *dif, int len);
void stdev(double mean, double *dif, int len);


void main ()
{
    int mtxBufferLength = (315 + 2 * 50)*(195 + 2 * 50)*sizeof(double);
    double output_cuda[mtxBufferLength], output_dpc[mtxBufferLength];

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

void compare(double *input1, double *input2, int len)
{
    int i;
<<<<<<< HEAD
    float dif[len], acc;
    double sum, mae;
=======
    double dif[len], mae, sum, acc;
>>>>>>> 20e9fb8a499159f13e4f4d6bf4746941428a7f1b
    int count = 0;
    for (i = 0; i < len; i++)
    {
            dif[i] = fabsf(input1[i] - input2[i]);
            sum += dif[i];
            if (input1[i] == input2[i])
            {
                    count++;
            }
	    else
	    {
		    printf("%.35f\t-\t%.35f\n", input1[i],input2[i]);
	    }
    }
    mae = sum / (double)len;
<<<<<<< HEAD
    acc = (float)count/(float)len*100;
=======
    acc = (double)count/(double)len*100;
>>>>>>> 20e9fb8a499159f13e4f4d6bf4746941428a7f1b
    printf("Accuracy: %.5f%\n", acc);
    rmse(dif,len);
    printf("MAE: %.15f\n",mae);
    stdev(mae, dif, len);
}

void rmse(double *dif, int len)
{
    int i;
    double sum = 0.0;
    for (i = 0; i < len; i++)
    {
            sum += pow(dif[i], 2);
    }
    printf("RMSE: %.15f\n", sqrt(sum/(double)len));
}

void stdev(double mean, double *dif, int len)
{
    int i;
    double p = 0.0, sigma;
    for(i = 0; i < len; i++){
            p = p + pow(dif[i] - mean, 2);
    }
    sigma = sqrt(p/((double)len-1));
    printf("Standard deviation: %.15f\n", sigma);
}
