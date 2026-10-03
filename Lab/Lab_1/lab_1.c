#include <stdio.h>
#include <math.h>
#include <omp.h>

double f(double x) {
    double s = sin(2.0 * x);
    return x / (s * s * s);
}

double simpson(double a, double b, int n) {
    double h = (b - a) / n;
    double sum_odd = 0.0, sum_even = 0.0;
    int i;

    #pragma omp parallel for reduction(+:sum_odd,sum_even) 
    for (i = 1; i < n; i++) {
        double x = a + i * h;
        if (i % 2 == 1)
            sum_odd += f(x);
        else
            sum_even += f(x);
    }

    double y0 = f(a);
    double yn = f(b);
    return (h / 3.0) * (y0 + yn + 4.0 * sum_odd + 2.0 * sum_even);
}

int main() {
    double a = 0.1, b = 0.5;
    double eps = 1e-6;
    int n = 10000000;
    double I_h, I_h2, error;

    omp_set_num_threads(1);
    I_h = simpson(a, b, n);

    while (1) {
        n *= 2;
        I_h2 = simpson(a, b, n);
        error = fabs(I_h2 - I_h) / 15.0;
        if (error < eps) break;
        I_h = I_h2;
    }

    int threads[] = {1, 2, 4, 8, 16, 32, 64};
    int num = sizeof(threads) / sizeof(threads[0]);
    double t1 = 0.0;

    FILE *fp = fopen("results.csv", "w");
    if (!fp) return 1;
    fprintf(fp, "Threads,Time\n");

    printf("| Threads | Items/Thread | Time (s) |\n");
    printf("|---------|--------------|----------|\n");

    for (int k = 0; k < num; k++) {
        int p = threads[k];
        int items_per_thread = n / p;

        omp_set_num_threads(p);
        simpson(a, b, n);

        double t_start = omp_get_wtime();
        for (int rep = 0; rep < 3; rep++) {
            simpson(a, b, n);
        }
        double tp = (omp_get_wtime() - t_start) / 3.0;

        printf("| %7d | %12d | %8.6f |\n", p, items_per_thread, tp);
        fprintf(fp, "%d,%.8f\n", p, tp);
    }
    fclose(fp);

    FILE *gp = popen("gnuplot -persistent", "w");
    if (gp) {
        fprintf(gp, "set title 'Симпсон + OpenMP'\n");
        fprintf(gp, "set xlabel 'Потоки'\n");
        fprintf(gp, "set ylabel 'Время (s)'\n");
        fprintf(gp, "set grid\n");
        fprintf(gp, "set logscale x 2\n");
        fprintf(gp, "set datafile separator ','\n");
        fprintf(gp, "plot 'results.csv' using 1:2 with linespoints lw 2 pt 7 title 'Time'\n");
        fflush(gp);
        pclose(gp);
    }

    return 0;
}