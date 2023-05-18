__kernel void sgemm(__global float *A, __global float *B, __global float *C, int M, int N, int K) 
{
    int row = get_local_id(1);
    int col = get_local_id(0);
    int global_row = 32 * get_group_id(1) + row;
    int global_col = 32 * get_group_id(0) + col;

    __local float Asub[32][32];
    __local float Bsub[32][32];

    float sum = 0.0f;
    int num_tiles = (K + 31) / 32;
    for(int t=0; t < num_tiles; t++)
    {
        int t_row = 32 * t + row;
        int t_col = 32 * t + col;
        if((global_row < M) && (t*32 + col < K))
            Asub[row][col] = A[global_row*K + t_col];
        else
            Asub[row][col] = 0;
        if((global_col < N) && (t*32 + row < K))
            Bsub[row][col] = B[t_row*N + global_col];
        else
            Bsub[row][col] = 0;
    
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k=0; k < 32 ; k++)
        {
            sum += Asub[row][k] * Bsub[k][col];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(global_row < M && global_col < N)
        C[global_row*N + global_col] = sum;
   }