#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define M 20000
#define N 20000
#define K 100

void init_matrix_vector(double **A, double *b) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = (double)rand() / RAND_MAX;
    for (int j = 0; j < N; j++)
        b[j] = (double)rand() / RAND_MAX;
}

void multiply_serial(double **A, double *b, double *c) {
    for (int i = 0; i < M; i++) {
        double sum = 0.0;
        for (int j = 0; j < N; j++)
            sum += A[i][j] * b[j];
        c[i] = sum;
    }
}

void multiply_row_striped(double **A, double *b, double *c) {
    #pragma omp parallel
    {
        int nth = omp_get_num_threads();
        int tid = omp_get_thread_num();
        int num_strips = (M + K - 1) / K;

        for (int strip = tid; strip < num_strips; strip += nth) {
            int i_begin = strip * K;
            int i_end = (i_begin + K < M) ? i_begin + K : M;

            for (int i = i_begin; i < i_end; i++) {
                double sum = 0.0;
                for (int j = 0; j < N; j++)
                    sum += A[i][j] * b[j];
                c[i] = sum;
            }
        }
    }
}

void multiply_col_striped(double **A, double *b, double *c) {
    for (int i = 0; i < M; i++) c[i] = 0.0;

    #pragma omp parallel
    {
        int nth = omp_get_num_threads();
        int tid = omp_get_thread_num();
        int num_strips = (N + K - 1) / K;
        double *c_local = (double*)calloc(M, sizeof(double));

        for (int strip = tid; strip < num_strips; strip += nth) {
            int j_begin = strip * K;
            int j_end = (j_begin + K < N) ? j_begin + K : N;

            for (int i = 0; i < M; i++) {
                double sum = 0.0;
                for (int j = j_begin; j < j_end; j++)
                    sum += A[i][j] * b[j];
                c_local[i] += sum;
            }
        }

        #pragma omp critical
        {
            for (int i = 0; i < M; i++)
                c[i] += c_local[i];
        }
        free(c_local);
    }
}

void multiply_blocked(double **A, double *b, double *c) {
    for (int i = 0; i < M; i++) c[i] = 0.0;

    #pragma omp parallel
    {
        int nth = omp_get_num_threads();
        int tid = omp_get_thread_num();
        int blocks_i = (M + K - 1) / K;
        int blocks_j = (N + K - 1) / K;
        int total_blocks = blocks_i * blocks_j;
        double *c_local = (double*)calloc(M, sizeof(double));

        for (int blk = tid; blk < total_blocks; blk += nth) {
            int bi = blk / blocks_j;
            int bj = blk % blocks_j;
            int i_begin = bi * K;
            int i_end = (i_begin + K < M) ? i_begin + K : M;
            int j_begin = bj * K;
            int j_end = (j_begin + K < N) ? j_begin + K : N;

            for (int i = i_begin; i < i_end; i++) {
                double sum = 0.0;
                for (int j = j_begin; j < j_end; j++)
                    sum += A[i][j] * b[j];
                c_local[i] += sum;
            }
        }

        #pragma omp critical
        {
            for (int i = 0; i < M; i++)
                c[i] += c_local[i];
        }
        free(c_local);
    }
}

int main() {
    double **A = (double**)malloc(M * sizeof(double*));
    for (int i = 0; i < M; i++) A[i] = (double*)malloc(N * sizeof(double));
    double *b = (double*)malloc(N * sizeof(double));
    double *c_serial = (double*)malloc(M * sizeof(double));
    double *c1 = (double*)malloc(M * sizeof(double));
    double *c2 = (double*)malloc(M * sizeof(double));
    double *c3 = (double*)malloc(M * sizeof(double));

    init_matrix_vector(A, b);

    omp_set_num_threads(1);
    double t0 = omp_get_wtime();
    multiply_serial(A, b, c_serial);
    double t1 = omp_get_wtime();

    int threads[] = {1, 2, 4, 8, 16, 32, 64};
    int num = sizeof(threads) / sizeof(threads[0]);

    FILE *fp = fopen("results.csv", "w");
    if (!fp) return 1;
    fprintf(fp, "Потоки,По строкам,По столбцам,Блочное\n");

    printf("| Потоки | По строкам | По столбцам |  Блочное  |\n");
    printf("|--------|------------|-------------|-----------|\n");

    for (int k = 0; k < num; k++) {
        int p = threads[k];
        omp_set_num_threads(p);

        multiply_row_striped(A, b, c1);
        multiply_col_striped(A, b, c2);
        multiply_blocked(A, b, c3);

        double t_row = 0.0, t_col = 0.0, t_blk = 0.0;
        int reps = 3;
        for (int rep = 0; rep < reps; rep++) {
            double ts = omp_get_wtime();
            multiply_row_striped(A, b, c1);
            t_row += omp_get_wtime() - ts;

            ts = omp_get_wtime();
            multiply_col_striped(A, b, c2);
            t_col += omp_get_wtime() - ts;

            ts = omp_get_wtime();
            multiply_blocked(A, b, c3);
            t_blk += omp_get_wtime() - ts;
        }
        t_row /= reps;
        t_col /= reps;
        t_blk /= reps;

        printf("| %6d | %10.6f | %11.6f | %9.6f |\n", p, t_row, t_col, t_blk);
        fprintf(fp, "%d,%.8f,%.8f,%.8f\n", p, t_row, t_col, t_blk);
    }
    fclose(fp);

    FILE *gp = popen("gnuplot -persistent", "w");
    if (gp) {
        fprintf(gp, "set encoding utf8\n");
        fprintf(gp, "set title 'Умножение матрицы на вектор + OpenMP'\n");
        fprintf(gp, "set xlabel 'Потоки'\n");
        fprintf(gp, "set ylabel 'Время (с)'\n");
        fprintf(gp, "set grid\n");
        fprintf(gp, "set logscale x 2\n");
        fprintf(gp, "set datafile separator ','\n");
        fprintf(gp, "plot 'results.csv' using 1:2 with linespoints lw 2 pt 7 title 'По строкам', \\\n");
        fprintf(gp, "     'results.csv' using 1:3 with linespoints lw 2 pt 5 title 'По столбцам', \\\n");
        fprintf(gp, "     'results.csv' using 1:4 with linespoints lw 2 pt 9 title 'Блочное'\n");
        fflush(gp);
        pclose(gp);
    }

    for (int i = 0; i < M; i++) free(A[i]);
    free(A); free(b); free(c_serial); free(c1); free(c2); free(c3);
    return 0;
}