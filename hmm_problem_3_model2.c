#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define LZERO (-1.0e10) // ~log(0)
#define LSMALL (-0.5e10) // log values < LSMALL art set to LZERO
#define minLogExp -log(-LZERO) // ~=-23

#define N 3 // states
#define M 3 // observations
#define L 50 // train cycle
#define DN 9
#define SIZE 20 // obserbation sequence max length

double LogAdd(double, double);

double forward(int*, int);
double backward(int*, int);
double decode(int*, int, int*); // return max prob & sotre state sequence: *q
void SingleLearn(int*, int); 
void SingleUpdate(int*, int);
void MultipleLearn(int*, int);
void MultipleUpdate(int*, int);

void reset();

void printMenu();
void printResult(int*, int, int*);

double pi[N] = {0.34,0.33,0.033}; // initial probability
double a[N][N] = { {0.34,0.33,0.33}, {0.33,0.34,0.33}, {0.33,0.33,0.34} }; // trabsition probability, A
double b[N][N] = { {0.34,0.33,0.33}, {0.33,0.34,0.33}, {0.33,0.33,0.34} }; // observation probability, B

double alpha[SIZE][N] = {LZERO};
double beta[SIZE][N] = {LZERO};
double delta[SIZE][N] = {LZERO};
double psi[SIZE][N] = {LZERO};
double gamma[SIZE][N] = {LZERO};
double xi[SIZE][N][N] = {LZERO};

double pi2[N] = {LZERO};
double p2A = LZERO;
double p1A = LZERO;

int main() 
{
	int i, j, t;
	//int o[SIZE] = {0,0,2,1,2,1,0};
	int o[SIZE] = {0};
	int q[SIZE] = {0};

	char TrainSet1[DN][SIZE] = 
	{
		"ABBCABCAABC", "ABCABC", "ABCAABC", "BBABCAB",
		"BCAABCCAB", "CACCABCA", "CABCABCA", "CABCA", "CABCA"
	};
	char TrainSet2[DN][SIZE] = 
	{
		"BBBCCBC", "CCBABB", "AACCBBB", "BBABBAC",
		"CCAABBAB", "BBBCCBAA", "ABBBBABA", "CCCCC", "BBAAA"
	};
	///////take log and print it (pi, A, B)/////////
	//printf("\nInitial Probability, pi\n");
	for (i = 0; i < N; i++) {
		pi[i] = log(pi[i]);
		//printf("%lf ", pi[i]);
	}
	//printf("\n");
	//printf("\nTrabsition Probability, A\n");
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			a[i][j] = log(a[i][j]);
			//printf("%lf ", a[i][j]);
		}
		//printf("\n");
	}
	//printf("\nObservation Probability, B\n");
	for (i = 0; i < N; i++) {
		for (j = 0; j < M; j++) {
			b[i][j] = log(b[i][j]);
			//printf("%lf ", b[i][j]);
		}
		//printf("\n");
	}

	printf("\n///////////////////////////////////////////////////////////\n");

	for (t = 0; t < L; t++) {
		for (i = 0; i < DN; i++) {
			int len = strlen(TrainSet2[i]);
			for (j = 0; j < len; j++) {
				o[j] = TrainSet2[i][j] - 'A';
				//printf("%d ", o[j]);
				SingleLearn(o,len);
				SingleUpdate(o,len);
			}
			//printf("\n");
		}
		if (t == 0)  {
			printf("/////////////////  first //////////////////////\n\n");
			printMenu();
		}
		if (t == 49) {
			printf("/////////////////  50th //////////////////////\n\n");
			printMenu();
		}
		//printResult(o,len,q);
	}

	printf("\n///////////////////////////////////////////////////////////\n");


	int P3_1[SIZE] = {0,1,2,0,1,2,2,0,1};
	int P3_2[SIZE] = {0,0,1,0,1,2,2,2,2,1,1,1};


	printf("use TrainSet2 to train ///model2///\n");

	for (i = 0; i < DN; i++) {
		int len = strlen(TrainSet1[i]);
		for (j = 0; j < len; j++)
			o[j] = TrainSet1[i][j] - 'A';
		printf("input TrainSet1 %d decode: %lf\n", i+1, decode(o,len,q));
	}
	printf("\n");
	for (i = 0; i < DN; i++) {
		int len = strlen(TrainSet2[i]);
		for (j = 0; j < len; j++)
			o[j] = TrainSet2[i][j] - 'A';
		printf("input TrainSet2 %d decode: %lf\n", i+1, decode(o,len,q));
	}
	printf("\n");
	printf("P3.1 decode: %lf\n", decode(P3_1,9,q));
	printf("P3.2 decode: %lf\n", decode(P3_2,12,q));

	return 0;
}

//ok
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

//ok
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

//ok
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

//ok
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

//single train utterance
void SingleLearn(int *o, int T) {
	int t, i, j, k;

	forward(o, T);
	backward(o, T);

	for (t = 0; t < T; t++) {
		double p = LZERO;
		for (i = 0; i < N; i++) {
			double temp = alpha[t][i] + beta[t][i];
			p = LogAdd(p, temp);
		}
		//assert(p != LZERO);
		
		for (i = 0; i < N; i++) {
			gamma[t][i] = alpha[t][i] + beta[t][i] - p;
		}
	}

	for (t = 0; t < T-1; t++) {
		double p = LZERO;
		for (i = 0; i < N; i++) {
			for (j = 0; j < N; j++) {
				double temp = alpha[t][i] + a[i][j] + b[j][o[t+1]] + beta[t+1][j];
				p = LogAdd(p, temp);
				//assert(p != LZERO);
			}
		}

		for (i = 0; i < N; i++) {
			for (j = 0; j < N; j++) {
				xi[t][i][j] = alpha[t][i] + a[i][j] + b[j][o[t+1]] + beta[t+1][j] - p;
			}
		}
	}
}

//single train utterance
void SingleUpdate(int *o, int T) {
	
	int i, j, t, k;
	//update pi
	for (i = 0; i < N; i++) {
		pi[i] = gamma[0][i];
	}

	//update A
	for (i = 0; i < N; i++) {
		double p2 = LZERO;
		for (t = 0; t < T-1; t++) {
			p2 = LogAdd(p2, gamma[t][i]);
		}
		//assert(p2 != LZERO);

		for (j = 0; j < N; j++) {
			double p1 = LZERO;
			for (t = 0; t < T-1; t++) {
				p1 = LogAdd(p1, xi[t][i][j]);
			}
			a[i][j] = p1 - p2;
		}
	}
	//update B
	for (i = 0; i < N; i++) {
		double p[M] = {LZERO};
		double p2 = LZERO;
		for (t = 0; t < T; t++) {
			p[o[t]] = LogAdd(p[o[t]], gamma[t][i]);
			p2 = LogAdd(p2, gamma[t][i]);
		}
		//assert(p2 != LZERO);

		for (k = 0; k < M; k++) {
			b[i][k] = p[k] - p2;
		}
	}
}

void MultipleLearn(int *o, int T) {
	int i, j, t, k;

	//pi
	for (i = 0; i < N; i++)
		pi2[i] = LogAdd(pi2[i], gamma[0][i]);

	//A
	for (i = 0; i < N; i++) {
		for (t = 0; t < T-1; t++)
			p2A = LogAdd(p2A, gamma[t][i]);
		//assert(p2 != LZERO);
		for (j = 0; j < N; j++) {
			for (t = 0; t < T-1; t++) 
				p1A = LogAdd(p1A, xi[t][i][j]);
			
			a[i][j] = p1A - p2A;
		}
	}
	//B
	for (i = 0; i < N; i++) {
		double p[M] = {LZERO};
		double p2 = LZERO;
		for (t = 0; t < T; t++) {
			p[o[t]] = LogAdd(p[o[t]], gamma[t][i]);
			p2 = LogAdd(p2, gamma[t][i]);
		}
		//assert(p2 != LZERO);

		for (k = 0; k < M; k++) 
			b[i][k] = p[k] - p2;
	}
}

void MultipleUpdate(int *o, int T) {
	int i, j, k, t;
	for (i = 0; i < N; i++)
		pi[N] = pi2[N] - DN;

	//for
}

void reset() {
	
}


//ok
void printMenu() {
	int i, j;
	printf("Initial Probability, pi\n");
	for (i = 0; i < N; i++)
		printf("%lf ", pi[i]);
	printf("\n\n");

	printf("Trabsition Probability, A\n");
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++)
			printf("%lf ", a[i][j]);
		printf("\n");
	}
	printf("\n");

	printf("Observation Probability, B\n");
	for (i = 0; i < N; i++) {
		for (j = 0; j < M; j++)
			printf("%lf ", b[i][j]);
		printf("\n");
	}
	printf("\n");
}

//ok
void printResult(int *o, int T, int *q) {
	int i;
	printf("\nresult:\n");
	printf("forward:  %lf\n", forward(o,T));
	printf("backward: %lf\n", backward(o,T));
	printf("decode:   %lf\n", decode(o,T,q));
	printf("state sequence: ");
	for (i = 0; i < T; i++) {
		printf("%d ", q[i]);
	}
	printf("\n");
}
