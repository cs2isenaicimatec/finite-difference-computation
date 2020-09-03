/*
 * arriadmactrl.h
 *
 *  Created on: 17 de out de 2017
 *      Author: macondo
 */

#ifndef ARRIADMACTRL_H_
#define ARRIADMACTRL_H_

#include <stdint.h>

#define ARRIA_DRV_ERROR					-1
#define ARRIA_DRV_SUCCESS				0

/**
*	This defines the number of DWORDS
*   transfer on each DMA transaction	
*	The max number of DWORD is 2^(18)-1 = 
*/
#define ARRIA_DMA_NUM_DWORDS           	2048//16384//2048
#define ARRIA_DMA_DESCRIPTOR_NUM 		1

/**
 * arria_run_cmd
 *
 * This function sends an Arria V DMA command via PCIe
 * driver The device file must have been previously opened
 * outside this function. This function does not check if
 * the file is properly opened.
 * All Arria V DMA commands update the cmd status field
 * of the dma_cmd structure. This field can be used
 * to check whether the operation was performed correctly
 * or not.
 *
 * @param fileId: device file descriptor
 * @param cmd: dma command struct to be executed
 * @return
 * 		0 - successful transfer
 * 		1 - unknown command
 */
int arria_run_cmd(ssize_t fileId, struct dma_cmd * cmd);


/**
 * arria_rd_buffer_from
 *
 * This function reads 'len' bytes from the DMA memory
 * region allocated by the driver. The DMA tranfers
 * is directed to read from the base address at 'baseaddr'
 *
 * @param fileId, 	device file descriptor.
 * @param buf, 		data buffer.
 * @param len, 		number of bytes to be read.
 * @param baseaddr,	avalon 64bit address to read from
 *
 * @return  0 - success
 * 		   -1 - fail
 */
int arria_rd_buffer_from(ssize_t fileId, char * buf, const uint32_t len,
		unsigned long long baseaddr);

/**
 * arria_wr_buffer_to
 *
 * This function writes 'len' bytes to the Arria V
 * FPGA at the Av. address specified by 'baseaddr'.
 *
 * @param fileId, 	device file descriptor.
 * @param buf, 		data buffer.
 * @param len, 		number of bytes to be read.
 * @param baseaddr,	avalon 64bit address to read from
 *
 * @return  0 - success
 * 		   -1 - fail
 */
int arria_wr_buffer_to(ssize_t fileId, char * buf, const uint32_t len,
		unsigned long long baseaddr);

/**
 * arria_rd_buffer
 *
 * This function reads 'len' bytes from the DMA memory
 * region allocated by the driver. The 'len' parameter
 * must be smaller than MAX_NUM_DWORDS*4, defined
 * in the 'arriapci.h' header file.
 *
 * @param fileId, device file descriptor.
 * @param buf, data buffer.
 * @param len, number of bytes to be read.
 *
 * @return  0 - success
 * 		   -1 - fail
 */
int arria_rd_buffer(ssize_t fileId, char * buf, const uint32_t len);

/**
 * arria_wr_buffer
 *
 * This function transfers an array of 'len' bytes to the
 * FPGA through the PICe driver. The desired base address
 * must have been correctly configured using the function
 * ''.
 * The 'len' parameter must be smaller than the current
 * NUM_WORDS defined for the DMA transfer.
 *
 * OBS: write and read operations are performed from the
 * point of view of the FPGA by the driver. Therefore
 * a transfer from the host to the device is seen as a
 * READ operation by the driver, while a WRITE one is
 * performed from the device to the host.
 *
 * @param fileId, device file descriptor.
 * @param buf, data buffer.
 * @param len, number of bytes to be transfered.
 *
 * @return  0 - success
 * 		   -1 - fail
 */
int arria_wr_buffer(ssize_t fileId, char * buf, const uint32_t len);

/**
 * arria_set_base_address
 *
 * This function sets the current base address
 * both for transfer operations. The base address
 * is a 64bit integer value within the Avalon
 * virtual address space. It must map into
 * a valid Avalon MM interface attached to the
 * bus on the FPGA device.
 *
 * @param fileId, device file descriptor.
 * @param buf, data buffer.
 * @param len, number of bytes to be transfered.
 *
 * @return  0 - success
 * 		   -1 - fail
 */
int arria_set_base_address(ssize_t fileId, unsigned long long baseaddr);


/**
 * arria_set_block_size
 *
 * Sets the default block size for DMA transfers
 *
 * @param fileId, device file descriptor.
 * @param block_size, block size in 32bit words
 *
 * @return  0 - success
 * 		   -1 - fail
 */
int arria_set_block_size(ssize_t fileId, unsigned int block_size);
/**
 * arria_set_num_desc
 *
 * Sets the default descriptors number for DMA transfers
 *
 * @param fileId, device file descriptor.
 * @param desc_num, block size in 32bit words
 *
 * @return  0 - success
 * 		   -1 - fail
 */
int arria_set_num_desc(ssize_t fileId, unsigned int desc_num);

int arria_print_driver_status(ssize_t fileId);

#endif /* ARRIADMACTRL_H_ */
