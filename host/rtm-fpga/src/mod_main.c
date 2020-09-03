
/**
 * @author: Anderson
 */

#include <stdio.h>
#include <stdlib.h>

int rtm (int argc, char **argv);
int dmablock_test(int );

int main(int argc, char **argv) {


	//printf ("\nRTM co-design\n");
	int num = 0;
	if (argc >=2){
		num = atoi(argv[1]);
	}
	return rtm(argc, argv);
//	return dmablock_test(num);
}
