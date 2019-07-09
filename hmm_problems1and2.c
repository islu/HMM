#include <stdio.h>

#define N 3 // states
#define M 3 // observations
#define T 7 // obserbation sequence length

double forward(int*, int);
double backward(int*, int);

int main() {

	char obserbations[M][20] = {"up", "down", "unchanged"};

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

	double alpha[T][N] = {0.0};
	double beta[T][N] = {0.0};
	double delta[T][N] = {0.0};
	double psi[T][N] = {0.0};
	
	int q[T] = {0}; // store best probability index
	int o[T] = {0,0,2,1,2,1,0};
	int i, j, t;
	
	// 1. evaluation problem
	// find the probability 
	// P(up, up, unchanged, down, unchanged, down, up | this model)

	for (t = 0; t < T; t++) {
		for (j = 0; j < N; j++) {
			if (t == 0) {
				alpha[t][j] = pi[j] * b[j][o[t]];
			}
			else {
				double p = 0.0;
				for (i = 0; i < N; i++) {
					p += alpha[t-1][i] * a[i][j];
				}
				alpha[t][j] = p * b[j][o[t]];
			}
		}
	}

	double p = 0;
	for (i = 0; i < N; i++) 
		p += alpha[T-1][i];

	printf("probability: %.32lf\n", p);

	// 2. decoding problem

	for (t = 0; t < T; t++) {
		for (j = 0; j < N; j++) {
			if (t == 0) 
				delta[t][j] = pi[j] * b[j][o[t]];
			else {
				double p  = -1e9;
				for (i = 0; i < N; i++) {
					double w = delta[t-1][i] * a[i][j];
					if (w > p) {
						p = w;
						psi[t][j] = i;
					}
					delta[t][j] = p * b[j][o[t]];
				}
			}
		}
	}

	
	p = -1e9;
	for (j = 0; j < N; j++){
		if (delta[T-1][j] > p) {
			p = delta[T-1][j];
			q[T-1] = j;
		}
	}

	for (t = T-1; t > 0; t--) {
		q[t-1] = psi[t][q[t]];
	}

	printf("max probability: %.32lf\n", p);

	printf("beast state sequence: ");
	for (i = 0; i < T; i++) {
		printf("%d ", q[i]);
	}
	printf("\n");



	return 0;
}

