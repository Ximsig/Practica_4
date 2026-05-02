#include "dct.h"
#include "image.h"
#include <math.h>
#include <omp.h>


// DCT directa: pasa el bloque del dominio espacial al de frecuencias.
void dct::direct(float **dct, const Block<float> &matrix, int channel)
{
    int m = matrix.size;
    int n = m;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            // Declaramos ci y cj DENTRO del bucle para que sean privadas
            // por iteración (por si en algún momento se paraleliza).
            float ci, cj;
            if (i == 0) ci = 1 / sqrt(m);
            else        ci = sqrt(2) / sqrt(m);
            if (j == 0) cj = 1 / sqrt(n);
            else        cj = sqrt(2) / sqrt(n);

            // Doble sumatorio que define la DCT.
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


// DCT inversa: del dominio de frecuencias al espacial.
// Mismo esquema que la directa pero con los coeficientes Cu, Cv.
void dct::inverse(Block<float> &idctMatrix, float **dctMatrix, int channel, float min, float max){

    int size = idctMatrix.size;
                   
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) { 
            float Cu, Cv;
            float sum = 0.0;
            for (int u = 0; u < size; u++) {
                for (int v = 0; v < size; v++) {
                    if (u == 0) Cu = 1./sqrt(2.0);
                    else        Cu = 1.0;
                    if (v == 0) Cv = 1./sqrt(2.0);
                    else        Cv = 1.0;

                    sum += (dctMatrix[u][v] * cos((2 * i + 1) * u * M_PI / (size*2)) *
                            cos((2 * j + 1) * v * M_PI / (size*2)));
                }               
            }
            idctMatrix.set_pixel(i, j, channel, (float)(0.25 * Cu * Cv * sum));        
        }
    }    
 }
 

// Normaliza la matriz al rango 0..255 buscando el min y el max.
// Usamos reduction para que cada hilo tenga su copia y luego se
// combinen sin condiciones de carrera.
void dct::normalize(float **DCTMatrix, int size){
    float max_v = -99999999.0, min_v = 999999999.0;

    #pragma omp parallel for collapse(2) reduction(min:min_v) reduction(max:max_v)
    for (int i=0;i<size;i++){
        for (int j=0;j<size;j++){
            if (DCTMatrix[i][j] < min_v) min_v = DCTMatrix[i][j];
            if (DCTMatrix[i][j] > max_v) max_v = DCTMatrix[i][j];
        }
    }

    // Segundo bucle: cada hilo escribe en su propia celda, sin conflictos.
    #pragma omp parallel for collapse(2)
    for (int i=0;i<size;i++){
        for (int j=0;j<size;j++){
            DCTMatrix[i][j] = 255.0 * (DCTMatrix[i][j] - min_v) / (max_v - min_v);
        }
    }
}


// Copia una matriz DCT al bloque de la imagen original.
void dct::assign(float **DCTMatrix, Block<float> &block, int channel){
    #pragma omp parallel for collapse(2)
    for (int i=0;i<block.size;i++){
        for (int j=0;j<block.size;j++){
            block.set_pixel(i, j, channel, (float)DCTMatrix[i][j]);
        }
    }
}


// Reserva una matriz 2D contigua en memoria (más rápida que un new doble).
float **dct::create_matrix(int x_size, int y_size){
    float **m = new float*[x_size];
    float *p = new float[x_size*y_size];
    for(int i=0; i<x_size; i++){
        m[i] = &p[i*y_size];
    }
    return m;
}

void dct::delete_matrix(float **m){
    delete [] m[0];
    delete [] m;
}