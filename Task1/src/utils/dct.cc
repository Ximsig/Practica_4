#include "dct.h"
#include "image.h"
#include <math.h>
#include <omp.h>

// =====================================================================
// dct::direct y dct::inverse:
//
// IMPORTANTE: estas dos funciones se llaman desde un bucle EXTERNO
// sobre bloques (en main.cc, compute_dct) que YA está paralelizado
// con #pragma omp parallel for. Si paralelizásemos también los
// bucles internos aquí dentro, tendríamos paralelismo anidado y
// produciríamos oversubscription de hilos.
//
// Por defecto OpenMP no activa el paralelismo anidado, así que un
// #pragma omp parallel for aquí dentro se ignoraría cuando se
// invoca desde una región paralela. Aun así, dejamos la directiva
// comentada para documentar la decisión y poder probar la versión
// alternativa (paralelizar internamente) si se desactiva el bucle
// externo.
//
// La granularidad gruesa (un bloque entero por hilo, ~7000 bloques
// para una imagen 680x680) da mejor rendimiento que paralelizar
// los 64 elementos de cada bloque 8x8.
// =====================================================================

void dct::direct(float **dct, const Block<float> &matrix, int channel)
{
    int m = matrix.size;
    int n = m;

    // #pragma omp parallel for collapse(2)  // activar SOLO si se desactiva el bucle de bloques en compute_dct
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            // ci y cj se declaran DENTRO del bucle para que sean
            // automáticamente privadas en cada iteración (evita
            // condiciones de carrera si se paraleliza este bucle).
            float ci, cj;
            if (i == 0)
                ci = 1 / sqrt(m);
            else
                ci = sqrt(2) / sqrt(m);
            if (j == 0)
                cj = 1 / sqrt(n);
            else
                cj = sqrt(2) / sqrt(n);

            float sum = 0;
            for (int k = 0; k < m; k++) {
                for (int l = 0; l < n; l++) {
                    sum += matrix.get_pixel(k, l, channel) * 
                           cos((2 * k + 1) * i * M_PI / (2 * m)) * 
                           cos((2 * l + 1) * j * M_PI / (2 * n));
                }
            }
            dct[i][j] = ci * cj * sum;
        }
    }
}

void dct::inverse(Block<float> &idctMatrix, float **dctMatrix, int channel, float min, float max){

    int size = idctMatrix.size;
                   
    // #pragma omp parallel for collapse(2)  // activar SOLO si se desactiva el bucle de bloques en compute_dct
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) { 
            float Cu, Cv;
            float sum = 0.0;
            for (int u = 0; u < size; u++) {
                for (int v = 0; v < size; v++) {
                    if (u == 0)
                        Cu = 1./sqrt(2.0);
                    else
                        Cu = 1.0;

                    if (v == 0)
                        Cv = 1./sqrt(2.0);
                    else
                        Cv = (1.0);  

                    sum += (dctMatrix[u][v] * cos((2 * i + 1) * u * M_PI / (size*2)) *
                            cos((2 * j + 1) * v * M_PI / (size*2)));
                }               
            }
            idctMatrix.set_pixel(i, j, channel, (float)(0.25 * Cu * Cv * sum));        
        }
    }    
 }
 
// =====================================================================
// dct::normalize: misma estrategia que Image::normalized().
// Reducción min/max en el primer bucle, escritura paralela en el segundo.
// =====================================================================
void dct::normalize(float **DCTMatrix, int size){
    float max_v=-99999999.0, min_v=999999999.0;
    #pragma omp parallel for collapse(2) reduction(min:min_v) reduction(max:max_v)
    for (int i=0;i<size;i++){
        for (int j=0;j<size;j++){
            if (DCTMatrix[i][j] < min_v) min_v=DCTMatrix[i][j];
            if (DCTMatrix[i][j] > max_v) max_v=DCTMatrix[i][j];
        }
    }
    #pragma omp parallel for collapse(2)
    for (int i=0;i<size;i++){
        for (int j=0;j<size;j++){
            DCTMatrix[i][j] = 255.0 * (DCTMatrix[i][j] -min_v)/ (max_v - min_v);
        }
    }
}

void dct::assign(float **DCTMatrix, Block<float> &block, int channel){
    #pragma omp parallel for collapse(2)
    for (int i=0;i<block.size;i++){
        for (int j=0;j<block.size;j++){
            block.set_pixel(i, j, channel, (float)DCTMatrix[i][j]);
        }
    }
}

float **dct::create_matrix(int x_size, int y_size){
    float **m = new float*[x_size]; //(float**)calloc(dimX, sizeof(float*));
    float *p = new float[x_size*y_size];//(float*)calloc(dimX*dimY, sizeof(float));
    for(int i=0; i<x_size;i++){
        m[i] = &p[i*y_size];
    }
    return m;
}

void dct::delete_matrix(float **m){
    delete [] m[0];
    delete [] m;
}
