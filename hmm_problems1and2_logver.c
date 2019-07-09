#include <stdint.h>
#include <math.h>

#define LZERO (-1.0e10) // ~log(0)
#define LSMALL (-0.5e10) // log values < LSMALL art set to LZERO
#define minLogExp -log(-LZERO) // ~=-23

#define N 3 // state 
#define M 3 // observation
#define SIZE 7 // obserbation sequence length

double LogAdd(double, double);

double forward(int*, int);
double backward(int*, int);
double decode(int*, int, int*); // return max prob & sotre state sequence: *q

//Initial Probability, pi
double pi[N] = {0.5, 0.2, 0.3};
	
// Trabsition Probability, A
double a[N][N] = {
	{0.6, 0.2, 0.4},
	{0.5, 0.3, 0.2},
	{0.4, 0.1, 0.5}
};

// Observation Probability, B
double b[N][M] = {
	{0.7, 0.1, 0.2},
	{0.1, 0.6, 0.3},
	{0.3, 0.3, 0.4}
};

double alpha[SIZE][N] = {LZERO};
double beta[SIZE][N] = {LZERO};
double delta[SIZE][N] = {LZERO};
double psi[SIZE][N] = {LZERO};

int main() {

	int o[SIZE] = {0,0,2,1,2,1,0};
	int q[SIZE] = {0};

	int i, j, t;
	///////take log and print it/////////
	printf("\nInitial Probability, pi\n");
	for (i = 0; i < N; i++) {
		pi[i] = log(pi[i]);
		printf("%lf ", pi[i]);
	}
	printf("\n");

	printf("\nTrabsition Probability, A\n");
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			a[i][j] = log(a[i][j]);
			printf("%lf ", a[i][j]);
		}
		printf("\n");
	}

	printf("\nObservation Probability, B\n");
	for (i = 0; i < N; i++) {
		for (j = 0; j < M; j++) {
			b[i][j] = log(b[i][j]);
			printf("%lf ", b[i][j]);
		}
		printf("\n");
	}
	//////////////////////


	///////print result///////////
	printf("\nresult:\n");
	printf("forward:  %lf\n", forward(o,SIZE));
	printf("backward: %lf\n", backward(o,SIZE));
	printf("decode:   %lf\n", decode(o,SIZE,q));
	printf("state sequence: ");
	for (i = 0; i < SIZE; i++) {
		printf("%d ", q[i]);
	}
	printf("\n");
	
	//////////////////////
	
	system("pause");

	return 0;
}

///ok
double forward(int *o, int T) {
	
	int t, i, j;
	
	for (t = 0; t < T; t++) {
		for (j = 0; j < N; j++) {
			if (t == 0)
				alpha[t][j] = pi[j] + b[j][o[t]];
			else {
				double p = LZERO;
				for (i = 0; i < N; i++) {
					double temp = alpha[t-1][i] + a[i][j];
					p = LogAdd(p, temp);
				}
				alpha[t][j] = p + b[j][o[t]];
			}
		}
	}

	double p = LZERO;
	for (i = 0; i < N; i++)
		p = LogAdd(p, alpha[T-1][i]);
	return p;
}

///ok
double backward(int *o, int T) {
	
	int t, i, j;
	
	for (t = T-1; t >= 0; t--) {
		for (i = 0; i < N; i++) {
			if (t == T-1)
				beta[t][i] = log(1.0);
			else {
				double p = LZERO;
				for (j = 0; j < N; j++) {
					double temp = a[i][j] + b[j][o[t+1]] + beta[t+1][j];
					p = LogAdd(p, temp);
				}
				beta[t][i] = p;
			}
		}
	}

	double p = LZERO;
	for (j = 0; j < N; j++) {
		double temp = pi[j] + b[j][o[0]] + beta[0][j];
		p = LogAdd(p, temp);
	}
	return p;
}

///ok
double decode(int *o, int T, int *q) {

	int t, i, j;
	
	for (t = 0; t < T; t++) {
		for (j = 0; j < N; j++) {
			if (t == 0)
				delta[t][j] = pi[j] + b[j][o[t]];
			else {
				double p = LSMALL;
				for (i = 0; i < N; i++) {
					double w = delta[t-1][i] + a[i][j];
					if (w > p) {
						p = w;
						psi[t][j] = i;
					}
					delta[t][j] = p + b[j][o[t]];
				}
			}
		}
	}

	double p = LSMALL;
	for (j = 0; j < N; j++) {
		if (delta[T-1][j] > p) {
			p = delta[T-1][j];
			q[T-1] = j;
		}
	}
	for (t = T-1; t > 0; t--) {
		q[t-1] = psi[t][q[t]];
	}

	return p;
}

///ok
double LogAdd(double x, double y) {
	
	double temp, diff, z;
	
	if (x < y) {
		temp = x;
		x = y;
		y = temp;
	}

	diff = y - x; // notice that diff <= 0
	if (diff < minLogExp) // if y' is far smaller that x'
		return (x < LSMALL) ? LZERO : x;
	else {
		z = exp(diff);
		return x + log(1.0+z);
	}
}
