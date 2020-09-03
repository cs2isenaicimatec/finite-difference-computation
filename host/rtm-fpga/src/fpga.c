/*
 * fpga.c
 *
 *  Created on: May 14, 2018
 *      Author: alu
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>

#include "arriadmactrl.h"
#include "fpga.h"

ssize_t fpga_open(char * devnode_path) {
	ssize_t fid = open(devnode_path, O_RDWR);
	return fid;
}
void fpga_swreset(ssize_t fid) {
	unsigned int data = LCORE_SET;
	unsigned int data_size = sizeof(unsigned int);
	arria_set_block_size(fid, 1);
	arria_set_num_desc(fid, 1);
	arria_wr_buffer_to(fid, (char *) &data, data_size, LCORE_SW_RESET);
}

void fpga_run(ssize_t fid) {
	unsigned int data = LCORE_SET;
	unsigned int data_size = sizeof(unsigned int);
	//arria_set_block_size(fid, 1);
	//arria_set_num_desc(fid, 1);
	arria_wr_buffer_to(fid, (char *) &data, data_size, LCORE_RUN_STOP_REG);
}

unsigned char fpga_isbusy(ssize_t fid) {
	unsigned int data = LCORE_RESET;
	unsigned int data_size = sizeof(unsigned int);
	//arria_set_block_size(fid, 1);
	//arria_set_num_desc(fid, 1);
	arria_rd_buffer_from(fid, (char *) &data, data_size, LCORE_STATUS_REG);
	//printf ("data: %d  \n", data);
	return (data & LCORE_STATUS_BUSY_MASK) != 0;
}

void fpga_config(ssize_t fid, unsigned int nx, unsigned int nz,
		unsigned int dx2inv, unsigned int dz2inv, unsigned int img_start_addr,
		unsigned int lapl_start_addr) {

	unsigned int data;
	unsigned int data_size = sizeof(unsigned int);
	// set descriptor num to 1
//	arria_set_block_size(fid, 1);
	arria_set_num_desc(fid, 1);

//	arria_rd_buffer_from(fid, (char *) &data, data_size, LCORE_DX2INV_REG);
//	printf ("\n DATA: 0x%x  \n", data);
//	close(fid);
//	exit(0);
	// soft reset hardware
	data = LCORE_SET;
	arria_wr_buffer_to(fid, (char *) &data, data_size, LCORE_SW_RESET);
//	arria_wr_buffer_to(fid, (char *) &data, data_size, LCORE_SW_RESET);
	//usleep(1000);

	// set dx2inv
	data = dx2inv;
	arria_wr_buffer_to(fid, (char *) &data, data_size, LCORE_DX2INV_REG);

	// set dz2inv
	data = dz2inv;
	arria_wr_buffer_to(fid, (char *) &data, data_size, LCORE_DZ2INV_REG);

	// set img start addr
	data = img_start_addr;
	arria_wr_buffer_to(fid, (char *) &data, data_size, LCORE_IMG_STARTADDR_REG);

	// set img max x
	data = nx;
	arria_wr_buffer_to(fid, (char *) &data, data_size, LCORE_IMG_MAX_X_REG);

	// set img max z
	data = nz;
	arria_wr_buffer_to(fid, (char *) &data, data_size, LCORE_IMG_MAX_Z_REG);

	// set laplacian start addr
	data = lapl_start_addr;
	arria_wr_buffer_to(fid, (char *) &data, data_size,
			LCORE_LAPL_STARTADDR_REG);

	// set enable flag
//	data = LCORE_SET;
//	arria_wr_buffer_to(fid, (char *) &data, data_size, LCORE_ENABLE_REG);
	usleep(100);

}

void fpga_write(ssize_t fid, float ** buff, unsigned long length,
		unsigned int addr) {

//	arria_set_block_size(fid, ARRIA_DMA_NUM_DWORDS);
//	arria_set_num_desc(fid, ARRIA_DMA_DESCRIPTOR_NUM);

	if (arria_wr_buffer_to(fid, (char *) buff[0], length * sizeof(float),
			addr) == ARRIA_DRV_ERROR) {
		printf(
				"\nFatal Error! Driver not responding to WR operation... Abort.\n");
		fpga_close(fid);
		exit(1);
	}
//	int dataleft, size, idx=0;
//
//	float * ptr = (float *) &buff[0];
//	dataleft = length;
//	size = ARRIA_DMA_NUM_DWORDS;
//	while (dataleft > 0) {
//		if (dataleft > ARRIA_DMA_NUM_DWORDS) {
//			size = ARRIA_DMA_NUM_DWORDS;
//		} else {
//			size = dataleft;
//		}
//		dataleft -= size;
//		// writes to FPGA in chunks of size*4 bytes
//		if (arria_wr_buffer_to(fid, (char *) ptr, size*sizeof(float),
//				addr) == ARRIA_DRV_ERROR) {
//			printf(
//					"\nFatal Error! Driver not responding to WR operation... Abort.\n");
//			fpga_close(fid);
//			exit(1);
//		}
//		addr += (size * sizeof(float));
//		idx += size;
//		ptr = (float*)&buff[idx];
//		//printf (" FPGA_WRITE:  %d words written \n", size);
//	}

}

void fpga_read(ssize_t fid, float ** buff, unsigned long length,
		unsigned int addr) {

//	arria_set_block_size(fid, ARRIA_DMA_NUM_DWORDS);
//	arria_set_num_desc(fid, ARRIA_DMA_DESCRIPTOR_NUM);

	if (arria_rd_buffer_from(fid, (char *) buff[0], length * sizeof(float),
			addr) == ARRIA_DRV_ERROR) {
		printf(
				"\nFatal Error! Driver not responding to WR operation... Abort.\n");
		fpga_close(fid);
		exit(1);
	}

//	unsigned int dataleft, size, idx=0;
//
//	//float * ptr = (float *) &buff[0];
//	dataleft = length;
//	size = ARRIA_DMA_NUM_DWORDS;
//	while (dataleft > 0) {
//
//		if (dataleft > ARRIA_DMA_NUM_DWORDS) {
//			size = ARRIA_DMA_NUM_DWORDS;
//		} else {
//			size = dataleft;
//		}
//		// writes to FPGA in chunks of size*4 bytes
//		if (arria_rd_buffer_from(fid, (char *) &buff[idx], size * sizeof(float),
//				addr) == ARRIA_DRV_ERROR) {
//			printf(
//					"\nFatal Error! Driver not responding to WR operation... Abort.\n");
//			fpga_close(fid);
//			exit(1);
//		}
//		dataleft -= size;
//		addr += size * sizeof(float);
//		idx += size;
//		//ptr = (float*)&buff[idx];
//		//printf (" FPGA_WRITE:  %d words read \n", size);
//	}

}

void fpga_close(ssize_t fid) {
	close(fid);
}
