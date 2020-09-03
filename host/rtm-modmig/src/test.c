#ifdef USE_MPI
#include <mpi.h>
#endif
#include <stdio.h>

#define LENGTH 138375
char *sdoc[] = {	/* self documentation */
	" Seismic modeling using acoustic wave equation ",
	"				               ",
	NULL};


/* prototypes */
int main (int argc, char **argv){

    // Get the number of processes
    int numberOfProcesses;
    int processId;

	// Initialize the MPI environment
    MPI_Init(NULL, NULL);


    // Get the rank of the process
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);

    MPI_Status status;
    float vec[LENGTH];
    int i;
    int cnt=0;
    for (i=0; i< LENGTH/(processId+1); i++){
    	vec[i] = i;
    }
    if (processId==0){
    	while (cnt<(numberOfProcesses-1)*2){
    		printf ("P[%d] waiting... \n", processId);
	    	MPI_Probe(  MPI_ANY_SOURCE,
					    MPI_ANY_TAG,
					    MPI_COMM_WORLD,
					    &status);
	    	
	    	MPI_Recv(   vec,
					    LENGTH,
					    MPI_FLOAT,
					    status.MPI_SOURCE,
					    status.MPI_TAG,
					    MPI_COMM_WORLD,
					    &status);
	    	printf ("P[%d] received from %d  \n", processId, status.MPI_SOURCE);
	    	cnt++;
	    }
    }else{

    	printf ("P[%d] send \n", processId);
		MPI_Send(  vec,
				   LENGTH,
				   MPI_FLOAT,
				   0,
				   1,
				   MPI_COMM_WORLD);
		MPI_Send(  vec,
				   LENGTH,
				   MPI_FLOAT,
				   0,
				   1,
				   MPI_COMM_WORLD);
    }


	printf("Process[%d] finished\n", processId );
	MPI_Finalize();
	return 0;
	//return(CWP_Exit());
}
