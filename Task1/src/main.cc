// main.cc - Práctica 4 CAR
// Análisis forense de imágenes con OpenMP + std::async
// Joaquín y Linxi

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "png.h"
#include <vector>
#include <assert.h>
#include <iostream>
#include <memory>
#include "utils/image.h"
#include "utils/dct.h"
#include <string>
#include <chrono>
#include <omp.h>     // para #pragma omp y omp_get_num_threads
#include <future>    // para std::async y std::future


// Kernel SRM 3x3. Es un filtro fijo (los valores no cambian)
// que sirve para resaltar el ruido de alta frecuencia de la imagen.
Image<float> get_srm_3x3() {
    Image<float> kernel(3, 3, 1);
    kernel.set(0, 0, 0, -1); kernel.set(0, 1, 0, 2); kernel.set(0, 2, 0, -1);
    kernel.set(1, 0, 0, 2); kernel.set(1, 1, 0, -4); kernel.set(1, 2, 0, 2);
    kernel.set(2, 0, 0, -1); kernel.set(2, 1, 0, 2); kernel.set(2, 2, 0, -1);
    return kernel;
}

// Versión 5x5 del mismo filtro. Coge una vecindad más grande
// así que tarda más pero detecta patrones más amplios.
Image<float> get_srm_5x5() {
    Image<float> kernel(5, 5, 1);
    kernel.set(0, 0, 0, -1); kernel.set(0, 1, 0, 2); kernel.set(0, 2, 0, -2); kernel.set(0, 3, 0, 2); kernel.set(0, 4, 0, -1);
    kernel.set(1, 0, 0, 2); kernel.set(1, 1, 0, -6); kernel.set(1, 2, 0, 8); kernel.set(1, 3, 0, -6); kernel.set(1, 4, 0, 2);
    kernel.set(2, 0, 0, -2); kernel.set(2, 1, 0, 8); kernel.set(2, 2, 0, -12); kernel.set(2, 3, 0, 8); kernel.set(2, 4, 0, -2);
    kernel.set(3, 0, 0, 2); kernel.set(3, 1, 0, -6); kernel.set(3, 2, 0, 8); kernel.set(3, 3, 0, -6); kernel.set(3, 4, 0, 2);
    kernel.set(4, 0, 0, -1); kernel.set(4, 1, 0, 2); kernel.set(4, 2, 0, -2); kernel.set(4, 3, 0, 2); kernel.set(4, 4, 0, -1);
    return kernel;
}

// Selector. Solo aceptamos kernel de 3 o 5, el assert evita que
// alguien pase otro tamaño por error y luego haya un return raro.
Image<float> get_srm_kernel(int size) {
    assert(size == 3 || size == 5);
    switch(size){
        case 3:
            return get_srm_3x3();
        case 5:
            return get_srm_5x5();
    }
    return get_srm_3x3();   // por si acaso, aunque el assert ya lo cubre
}


// SRM = aplicar el filtro de ruido a la imagen.
// Pasos: gris -> float -> convolución con el kernel -> abs -> normalizar -> 0..255 -> uchar.
// La convolución de dentro ya está paralelizada con OpenMP en image.h.
Image<unsigned char> compute_srm(const Image<unsigned char> &image, int kernel_size) {
    auto begin = std::chrono::steady_clock::now();
    std::cout<<"Computing SRM "<<kernel_size<<"x"<<kernel_size<<"..."<<std::endl;

    Image<float> srm = image.to_grayscale().convert<float>();
    srm = srm.convolution(get_srm_kernel(kernel_size));
    srm = srm.abs().normalized();
    srm = srm * 255;
    Image<unsigned char> result = srm.convert<unsigned char>();
    
    auto end = std::chrono::steady_clock::now();
    std::cout<<"SRM elapsed time: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"ms"<<std::endl;
    return result;
}


// DCT por bloques. Aquí es donde más ganamos paralelizando porque
// los bloques son independientes entre sí (cada uno escribe en su
// propia región de la imagen).
Image<unsigned char> compute_dct(const Image<unsigned char> &image, int block_size, bool invert) {
    auto begin = std::chrono::steady_clock::now();
    std::cout<<"Computing"; 
    if (invert) std::cout<<" inverse";
    else std::cout<<" direct";
    std::cout<<" DCT "<<block_size<<"x"<<block_size<<"..."<<std::endl;

    // Pasamos a gris en float y troceamos en bloques de 8x8.
    Image<float> grayscale = image.convert<float>().to_grayscale();
    std::vector<Block<float>> blocks = grayscale.get_blocks(block_size);

    // Aquí está el reparto entre hilos. Usamos schedule(dynamic) porque
    // cuando invert=true los bloques tienen un poco más de trabajo
    // (hay que anular las bajas frecuencias y reconstruir), así no
    // se queda ningún hilo parado esperando a otro más lento.
    #pragma omp parallel for schedule(dynamic)
    for(int i=0;i<(int)blocks.size();i++){
        // Cada hilo crea su propia matriz auxiliar -> nada compartido.
        float **dctBlock = dct::create_matrix(block_size, block_size);
        dct::direct(dctBlock, blocks[i], 0);

        if (invert) {
          // Ponemos a 0 las bajas frecuencias (esquina superior izq)
          // y reconstruimos la imagen sin ellas.
          for(int k=0;k<blocks[i].size/2;k++)
            for(int l=0;l<blocks[i].size/2;l++)
              dctBlock[k][l] = 0.0;
          dct::inverse(blocks[i], dctBlock, 0, 0.0, 255.);
        } else {
            // En la versión directa simplemente volcamos los coefs DCT.
            dct::assign(dctBlock, blocks[i], 0);
        }
        dct::delete_matrix(dctBlock);
    }

    Image<unsigned char> result = grayscale.convert<unsigned char>();
    auto end = std::chrono::steady_clock::now();
    std::cout<<"DCT elapsed time: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"ms"<<std::endl;
    return result;
}


// ELA = Error Level Analysis. Idea: comprimir la imagen como JPEG y
// restarla a la original. Las zonas que vienen de otra imagen
// recomprimida cambian de forma distinta -> se ven como "manchas".
//
// OJO: aquí hay I/O al disco (guardar y leer _temp.jpg), eso es serie
// y no se puede paralelizar bien. Por eso este proceso es el que peor
// escala con más hilos.
Image<unsigned char> compute_ela(const Image<unsigned char> &image, int quality){
    std::cout<<"Computing ELA..."<<std::endl;
    auto begin = std::chrono::steady_clock::now();

    Image<unsigned char> grayscale = image.to_grayscale();
    save_to_file("_temp.jpg", grayscale, quality);   // I/O serie
    Image<float> compressed = load_from_file("_temp.jpg").convert<float>();   // I/O serie

    // Resta original - recomprimida y normalizamos a 0..255.
    compressed = compressed + (grayscale.convert<float>()*(-1));
    compressed = compressed.abs().normalized() * 255;
    Image<unsigned char> result = compressed.convert<unsigned char>();

    auto end = std::chrono::steady_clock::now();
    std::cout<<"ELA elapsed time: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"ms"<<std::endl;
    return result;
}


int main(int argc, char **argv) {
    // Hay que pasarle una imagen por argumento.
    if(argc == 1) {
        std::cerr<<"Image filename missing from arguments. Usage ./detect <filename>"<<std::endl;
        exit(1);
    }
    
    // Pequeño print para saber con cuántos hilos OMP estamos corriendo.
    // Útil cuando cambiamos OMP_NUM_THREADS para los benchmarks.
    #pragma omp parallel
    {
        #pragma omp single
        std::cout<<"Running with "<<omp_get_num_threads()<<" OpenMP threads"<<std::endl;
    }
    
    int block_size = 8;   // tamaño de bloque DCT estándar (como en JPEG)
    
    auto total_begin = std::chrono::steady_clock::now();
    
    // Cargamos la imagen una sola vez. Luego se la pasamos por
    // referencia a las 5 tareas, no queremos copiarla 5 veces.
    Image<unsigned char> image = load_from_file(argv[1]);


    // -----------------------------------------------------------------
    // Aquí va lo importante de la práctica: lanzamos los 5 procesos
    // como tareas asíncronas con std::async. Como no dependen unos de
    // otros, pueden correr todos en paralelo.
    //
    // std::ref(image) -> evita que async haga copia profunda de la
    // imagen para cada tarea (las imágenes pueden ser grandes y se
    // notaría mucho). Como solo leemos de ella, es seguro compartirla.
    //
    // NOTA: para la versión COMBINADA (async + OpenMP) hay que limitar
    // los hilos OMP por proceso, si no se nos van de las manos. Con
    // 5 procesos y 16 hilos cada uno serían 80 hilos para 16 cores
    // lógicos. Descomentar la siguiente línea cuando se quiera probar:
    // omp_set_num_threads(4);

    std::future<Image<unsigned char>> f_srm3    = std::async(std::launch::async, compute_srm, std::ref(image), 3);
    std::future<Image<unsigned char>> f_srm5    = std::async(std::launch::async, compute_srm, std::ref(image), 5);
    std::future<Image<unsigned char>> f_ela     = std::async(std::launch::async, compute_ela, std::ref(image), 90);
    std::future<Image<unsigned char>> f_dct_inv = std::async(std::launch::async, compute_dct, std::ref(image), block_size, true);
    std::future<Image<unsigned char>> f_dct_dir = std::async(std::launch::async, compute_dct, std::ref(image), block_size, false);

    // .get() espera a que la tarea termine y nos devuelve el resultado.
    // Lo guardamos en disco según vamos recibiendo cada futuro.
    // Los nombres de archivo coinciden con los que pide el enunciado.
    save_to_file("srm_kernel_3x3.png", f_srm3.get());
    save_to_file("srm_kernel_5x5.png", f_srm5.get());
    save_to_file("ela.png",            f_ela.get());
    save_to_file("dct_invert.png",     f_dct_inv.get());
    save_to_file("dct_direct.png",     f_dct_dir.get());
    
    auto total_end = std::chrono::steady_clock::now();
    std::cout<<"TOTAL elapsed time: "<<std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_begin).count()<<"ms"<<std::endl;

    return 0;
}