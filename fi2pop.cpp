/*  fi2pop.cpp  –  minimal complete FI‑2Pop baseline  */
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

/* ------------ problem parameters (demo) ------------ */
#define GENOME_LEN   10
#define MIN_SUM_BITS 5 
#define POP_SIZE     20
#define MAX_GENS     30

/* ------------ individual representation ---------------- */
typedef struct {
    int bits[GENOME_LEN];
} Individual;

/* ------------ helpers ---------------------------------- */
static int rand_bit(void)       { return rand() & 1;            }
static int rand_int(int n)      { return rand() % n;            }
static double rand_unit(void)   { return rand() / (double)RAND_MAX; }


/* ------------ GA primitives ---------------------------- */
void random_genome(Individual *g) {
    for (int i = 0; i < GENOME_LEN; ++i) g->bits[i] = rand_bit();
}

bool check_constraints(const Individual *ind, int *violation) {
    int sum = 0;
    for (int i = 0; i < GENOME_LEN; ++i) sum += ind->bits[i];
    if (sum >= MIN_SUM_BITS) { *violation = 0; return true; }
    *violation = MIN_SUM_BITS - sum;
    return false;
}

int feasible_fitness (const Individual *ind)             {
    int sum = 0; for (int i = 0; i < GENOME_LEN; ++i) sum += ind->bits[i];
    return sum;
}
int infeasible_fitness(const Individual *ind,int viol)   { return -viol; } 
void mutate(Individual *ind, double rate) {
    for (int i = 0; i < GENOME_LEN; ++i)
        if (rand_unit() < rate) ind->bits[i] ^= 1;
}

void crossover(const Individual *p1,const Individual *p2,Individual *c1,Individual *c2){
    int cut = 1 + rand_int(GENOME_LEN-1);
    for (int i = 0; i < GENOME_LEN; ++i) {
        c1->bits[i] = (i < cut) ? p1->bits[i] : p2->bits[i];
        c2->bits[i] = (i < cut) ? p2->bits[i] : p1->bits[i];
    }
}

void tournament_select(Individual src[],int fit[],int n,int k,int out_n,Individual dst[]){
    for (int i = 0; i < out_n; ++i) {
        int best = rand_int(n);
        for (int j = 1; j < k; ++j) {
            int cand = rand_int(n);
            if (fit[cand] > fit[best]) best = cand;
        }
        dst[i] = src[best];
    }
}

void trim_by_fitness(Individual pop[],int fit[],int *size){
    /* simple bubble‑like partial sort: keep top POP_SIZE */
    for(int i=0;i<*size-1 && i<POP_SIZE;i++){
        for(int j=i+1;j<*size;j++){
            if(fit[j] > fit[i]){
                Individual tmp = pop[i]; pop[i]=pop[j]; pop[j]=tmp;
                int tmpf = fit[i]; fit[i]=fit[j]; fit[j]=tmpf;
            }
        }
    }
    if(*size > POP_SIZE) *size = POP_SIZE;
}

/* ------------ FI‑2Pop GA -------------------------------- */
void fi2popGA(int max_gens){
    Individual feas[POP_SIZE*2], infeas[POP_SIZE*2];
    int fSize=0, iSize=0;

    /* initial random pool */
    for (int i = 0; i < POP_SIZE*2; ++i) {
        Individual ind; 
        random_genome(&ind);
    
        int viol = 0;
        bool feasb = check_constraints(&ind, &viol);
    
        if (feasb) {
            feas[fSize++]   = ind;    /* store into feasible array */
        } else {
            infeas[iSize++] = ind;    /* store into infeasible array */
        }
    }

    int fitF[POP_SIZE*2], fitI[POP_SIZE*2];
    Individual parents[POP_SIZE], childBuf[POP_SIZE*2];

    for(int g=0; g<max_gens; g++){
        /* -------- evaluate feasible -------- */
        for(int i=0;i<fSize;i++) fitF[i]=feasible_fitness(&feas[i]);

        /* -------- evaluate infeasible & migrate -------- */
        int newF = 0, newI = 0;
        for(int i=0;i<iSize;i++){
            int viol=0; bool feasb = check_constraints(&infeas[i],&viol);
            if(feasb) { feas[fSize + newF++] = infeas[i]; }
            else {
                infeas[newI] = infeas[i];
                fitI[newI]   = infeasible_fitness(&infeas[i],viol);
                newI++;
            }
        }
        fSize += newF; iSize = newI;

        /* -------- breed feasible pop -------- */
        int parN = (fSize>0)?POP_SIZE:0;
        if(parN>0){
            tournament_select(feas,fitF,fSize,3,parN,parents);
            int childN=0;
            for(int i=0;i<parN;i+=2){
                Individual c1,c2;
                crossover(&parents[i],&parents[(i+1)%parN],&c1,&c2);
                mutate(&c1, 0.02);mutate(&c2, 0.02);
                childBuf[childN++] = c1; childBuf[childN++] = c2;
            }
            /* classify offspring */
            for(int i=0;i<childN;i++){
                int viol=0; bool feasb = check_constraints(&childBuf[i],&viol);
                if(feasb && fSize<POP_SIZE*2) feas[fSize++] = childBuf[i];
                else if(!feasb && iSize<POP_SIZE*2){ infeas[iSize] = childBuf[i];
                                                     fitI[iSize++] = infeasible_fitness(&childBuf[i],viol);}
            }
        }

        /* -------- breed infeasible pop -------- */
        parN = (iSize>0)?POP_SIZE:0;
        if(parN>0){
            tournament_select(infeas,fitI,iSize,3,parN,parents);
            int childN=0;
            for(int i=0;i<parN;i+=2){
                Individual c1,c2;
                crossover(&parents[i],&parents[(i+1)%parN],&c1,&c2);
                mutate(&c1,0.02); mutate(&c2,0.02);
                childBuf[childN++] = c1; childBuf[childN++] = c2;
            }
            for(int i=0;i<childN;i++){
                int viol=0; bool feasb = check_constraints(&childBuf[i],&viol);
                if(feasb && fSize<POP_SIZE*2) feas[fSize++] = childBuf[i];
                else if(!feasb && iSize<POP_SIZE*2){ infeas[iSize] = childBuf[i];
                                                     fitI[iSize++] = infeasible_fitness(&childBuf[i],viol);}
            }
        }

        /* -------- trim populations to POP_SIZE -------- */
        trim_by_fitness(feas,   fitF, &fSize);
        trim_by_fitness(infeas, fitI, &iSize);

        int bestFit = -1;
        if (fSize > 0) {
            bestFit = feasible_fitness(&feas[0]);
        }
                
        /* -------- log -------- */
        printf("Gen %2d | Feasible %2d | Infeasible %2d | Best Feasible = %d\n",
            g, fSize, iSize, bestFit);
    }
}

/* ------------ entry point ------------------------------ */
int main(void){
    srand((unsigned)time(NULL));
    fi2popGA(MAX_GENS);
    return 0;
}
