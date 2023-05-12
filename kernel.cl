
__kernel void sgemm(__global float *A, __global float *B, __global float *C, int M, int N, int K) 
{
   int i = get_global_id(1);
   int j = get_global_id(0);
   int k;
   float sum = 0.0f;

    if (i<M && j < N)
    {
        for(k=0; k<K; k++)
        {
            sum += A[i*K + k] * B[k*N + j];
        }
        C[i*N+j] = sum;
    }
}