/*
 * test.c
 *
 *  Created on: Jun 18, 2018
 *      Author: root
 */

#include <stdio.h>
#include <stdlib.h>
#include  <sys/time.h>

#include "fd.h"
#include "fpga.h"
#include "arriadmactrl.h"

// new mod
//#define NX 415
//#define NZ 295

// pluto
#define NX 1201
#define NZ 6960

#define LENGTH (NX*NZ) //
#define NUMBER_OF_TESTS 1000//100000
#define TOTAL_SAMPLES (NUMBER_OF_TESTS*LENGTH*1.0)
float **p;
float **r;

/**
 * This function sends 500MB of data to
 * the FPGA's DDR, measuring the elapsed time
 * and calculating the thoughput
 */
int dmablock_test(int fpga_num) {

	struct timeval st, et;
	float transf_time = 0.;
	float avg_wr_time = 0., avg_rd_time = 0., avg_wr_thr = 0., avg_rd_thr = 0.;
	float throughput = 0.;
	unsigned int failed_samples_cnt = 0;
	float failing_perc = 0.0;

	int i, j, k;

	ssize_t fpga_fid;
	if (fpga_num==0){
		printf ("Opening Arria 10 on %s \n", LCORE_DEVICE_NODE_0);
		fpga_fid = fpga_open(LCORE_DEVICE_NODE_0);
	}else {
		printf ("Opening Arria 10 on %s \n", LCORE_DEVICE_NODE_1);
		fpga_fid = fpga_open(LCORE_DEVICE_NODE_1);
	}
	if (fpga_fid < 0) {
		printf("\nError. Could not open FPGA device at %s. Abort! \n",
		LCORE_DEVICE_DEFAULT_NODE);
		exit(-1);
	}

	r = alloc2float(NZ, NX);
	p = alloc2float(NZ, NX);

	printf("Running transfer tests... \n");
	for (k = 0; k < NUMBER_OF_TESTS; k++) {
		for (i = 0; i < NX; i++) {
			for (j = 0; j < NZ; j++) {
				p[i][j] = k*1.0;
				r[i][j] = 0.;
			}
		}
		arria_set_num_desc(fpga_fid, 1);
		gettimeofday(&st, NULL);
		fpga_write(fpga_fid, p, LENGTH, LCORE_DEFAULT_IMGSTART_ADDR);
		gettimeofday(&et, NULL);
		transf_time = (__elapsed_time(st, et) * 1.0);

		avg_wr_time += transf_time;

		if (transf_time > 0.0) {
			throughput = ((LENGTH * 4) / (transf_time / 1000000)) / 1000000;
		} else {
			throughput = 0.0;
		}
		avg_wr_thr += throughput;

		arria_set_num_desc(fpga_fid, 1);
		gettimeofday(&st, NULL);
		//fpga_read(fpga_fid, r, LENGTH, LCORE_DEFAULT_LAPLSTART_ADDR);
		fpga_read(fpga_fid, r, LENGTH, LCORE_DEFAULT_IMGSTART_ADDR);
		gettimeofday(&et, NULL);
		transf_time = (__elapsed_time(st, et) * 1.0);
		avg_rd_time += transf_time;

		if (transf_time > 0.0) {
			throughput = ((LENGTH * 4) / (transf_time / 1000000)) / 1000000;
		} else {
			throughput = 0.0;
		}
		avg_rd_thr += throughput;

		// COMPARING DATA...
		for (i = 0; i < NX; i++) {
			for (j = 0; j < NZ; j++) {
				if (p[i][j] != r[i][j]) {
					unsigned int a, b;
					a = *(unsigned int *) &p[i][j];
					b = *(unsigned int *) &r[i][j];

					printf("Failed! \n"
							"   k=%d \n"
							"   sent: %.8f (0x%x) \n"
							"   rcvd: %.8f (0x%x)  \n"
							"   diff: %.2f \n"
							"", i, p[i][j], a, r[i][j], b, r[i][j] - p[i][j]);
					failed_samples_cnt++;
				}
			}
		}
		printf ("\r... %d ", k);
	}
	printf ("\n", k);
	avg_wr_time = avg_wr_time / NUMBER_OF_TESTS;
	avg_rd_time = avg_rd_time / NUMBER_OF_TESTS;
	avg_wr_thr = avg_wr_thr / NUMBER_OF_TESTS;
	avg_rd_thr = avg_rd_thr / NUMBER_OF_TESTS;
	printf("-----------------------------------------------------\n");
	printf("Data comparison results: ");
	if (failed_samples_cnt == 0) {
		printf(" FULL BIT MATCH!\n");
	} else {
		failing_perc = ((failed_samples_cnt * 1.0) / (TOTAL_SAMPLES)) * 100.0;
		printf("%d/%d (%.2f %%) samples have failed!\n", failed_samples_cnt,
				TOTAL_SAMPLES, failing_perc);
	}
	printf("Direction: From HOST to FPGA \n");
	printf("Number of transfers:  %d \n", NUMBER_OF_TESTS * 2);
	printf("Bytes per transfers : %.2f  KB\n", ((LENGTH*sizeof(float))*1.0)/1024.0);
	printf("WRITE: \n");
	printf("  Total WRITES:       %d \n", NUMBER_OF_TESTS);
	printf("  AVG WR Time:        %.4f  (s)\n", avg_wr_time / 1000000);
	printf("  AVG WR Throughput:  %.2f  (MB/s)  \n", avg_wr_thr);

	printf("READ: \n");
	printf("  Total READS:        %d \n", NUMBER_OF_TESTS);
	printf("  AVG RD Time:        %.4f  (s)\n", avg_rd_time / 1000000);
	printf("  AVG RD Throughput:  %.2f  (MB/s)  \n", avg_rd_thr);
	printf("-----------------------------------------------------\n");
	close(fpga_fid);
	return 0;
}
