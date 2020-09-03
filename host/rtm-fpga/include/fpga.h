#ifndef FPGA_H
#define FPGA_H

#define LCORE_FPGA
#define LCORE_BASE_ADDR							0x10000000
#define LCORE_RUN_STOP_REG						(LCORE_BASE_ADDR + 0*4)
#define LCORE_ENABLE_REG						(LCORE_BASE_ADDR + 1*4)
#define LCORE_STATUS_REG						(LCORE_BASE_ADDR + 2*4)
#define LCORE_DX2INV_REG						(LCORE_BASE_ADDR + 3*4)
#define LCORE_DZ2INV_REG						(LCORE_BASE_ADDR + 4*4)
#define LCORE_IMG_STARTADDR_REG 				(LCORE_BASE_ADDR + 5*4)
#define LCORE_IMG_MAX_X_REG 					(LCORE_BASE_ADDR + 6*4)
#define LCORE_IMG_MAX_Z_REG 					(LCORE_BASE_ADDR + 7*4)
#define LCORE_LAPL_STARTADDR_REG 				(LCORE_BASE_ADDR + 8*4)
#define LCORE_LAPL_ENDADDR_REG 					(LCORE_BASE_ADDR + 9*4)
#define LCORE_RDCLK_WR_REG 						(LCORE_BASE_ADDR + 10*4)
#define LCORE_RDCLK_RD_REG 						(LCORE_BASE_ADDR + 11*4)
#define LCORE_RDCLK_BS_REG 						(LCORE_BASE_ADDR + 12*4)
#define LCORE_RDCLK_TT_REG 						(LCORE_BASE_ADDR + 13*4)
#define LCORE_SW_RESET							(LCORE_BASE_ADDR + 14*4)

#define LCORE_SET 						0x01
#define LCORE_RESET 					0x00

#define LCORE_DEVICE_NODE_0				"/dev/arriapci0"
#define LCORE_DEVICE_NODE_1				"/dev/arriapci1"

#define LCORE_DEVICE_DEFAULT_NODE		LCORE_DEVICE_NODE_0

#define LCORE_ERR 						(-1)

#define LCORE_DEFAULT_D2XINV 			0x3c23d70a
#define LCORE_DEFAULT_D2ZINV 			0x3c23d70a

#define LCORE_DEFAULT_IMGSTART_ADDR		0x00000000
#define LCORE_DEFAULT_LAPLSTART_ADDR	0x000FA000 // 1MB

#define LCORE_STATUS_BUSY_MASK			0x01
#define LCORE_DEFAULT_TIMEOUT			1000000

ssize_t fpga_open(char * devnode_path);
void fpga_config(ssize_t fid, unsigned int nx, unsigned int nz,
		unsigned int dx2inv, unsigned int dz2inv, unsigned int img_start_addr,
		unsigned int lapl_start_addr);
void fpga_write(ssize_t fid, float ** buff, unsigned long length,
		unsigned int addr);
void fpga_read(ssize_t fid, float ** buff, unsigned long length,
		unsigned int addr);
void fpga_run(ssize_t fid);
void fpga_swreset(ssize_t fid);
unsigned char fpga_isbusy(ssize_t fid);

void fpga_close(ssize_t fid);

static inline int __elapsed_time(struct timeval st, struct timeval et) {
	return ((et.tv_sec - st.tv_sec) * 1000000) + (et.tv_usec - st.tv_usec);
}

#endif
