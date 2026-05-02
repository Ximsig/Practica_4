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
#include <omp.h>
#include <future>

Image<float> get_srm_3x3() {
    Image<float> kernel(3, 3, 1);
    kernel.set(0, 0, 0, -1); kernel.set(0, 1, 0, 2); kernel.set(0, 2, 0, -1);
    kernel.set(1, 0, 0, 2); kernel.set(1, 1, 0, -4); kernel.set(1, 2, 0, 2);
    kernel.set(2, 0, 0, -1); kernel.set(2, 1, 0, 2); kernel.set(2, 2, 0, -1);
    return kernel;
}

Image<float> get_srm_5x5() {
    Image<float> kernel(5, 5, 1);
    kernel.set(0, 0, 0, -1); kernel.set(0, 1, 0, 2); kernel.set(0, 2, 0, -2); kernel.set(0, 3, 0, 2); kernel.set(0, 4, 0, -1);
    kernel.set(1, 0, 0, 2); kernel.set(1, 1, 0, -6); kernel.set(1, 2, 0, 8); kernel.set(1, 3, 0, -6); kernel.set(1, 4, 0, 2);
    kernel.set(2, 0, 0, -2); kernel.set(2, 1, 0, 8); kernel.set(2, 2, 0, -12); kernel.set(2, 3, 0, 8); kernel.set(2, 4, 0, -2);
    kernel.set(3, 0, 0, 2); kernel.set(3, 1, 0, -6); kernel.set(3, 2, 0, 8); kernel.set(3, 3, 0, -6); kernel.set(3, 4, 0, 2);
    kernel.set(4, 0, 0, -1); kernel.set(4, 1, 0, 2); kernel.set(4, 2, 0, -2); kernel.set(4, 3, 0, 2); kernel.set(4, 4, 0, -1);
    return kernel;
}

Image<float> get_srm_kernel(int size) {
    assert(size == 3 || size == 5);
    switch(size){
        case 3:
            return get_srm_3x3();
        case 5:
            return get_srm_5x5();
    }
    return get_srm_3x3();
}


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

// =====================================================================
// compute_dct: el bucle sobre los bloques se paraleliza con OpenMP.
//
// Cada bloque se procesa de forma totalmente independiente:
//   - create_matrix asigna memoria local al hilo.
//   - dct::direct/inverse trabajan sobre dctBlock (local) y blocks[i]
//     (cada bloque escribe en una región distinta de la imagen).
//   - delete_matrix libera la memoria local.
//
// schedule(dynamic) porque, aunque cada bloque tiene el mismo coste
// teórico, la rama if (invert) hace que algunos bloques sean más
// caros que otros. Con dynamic los hilos rápidos siguen cogiendo
// trabajo mientras los lentos terminan.
// =====================================================================
Image<unsigned char> compute_dct(const Image<unsigned char> &image, int block_size, bool invert) {
    auto begin = std::chrono::steady_clock::now();
    std::cout<<"Computing"; 
    if (invert) std::cout<<" inverse";
    else std::cout<<" direct";
    std::cout<<" DCT "<<block_size<<"x"<<block_size<<"..."<<std::endl;
    Image<float> grayscale = image.convert<float>().to_grayscale();
    std::vector<Block<float>> blocks = grayscale.get_blocks(block_size);

    #pragma omp parallel for schedule(dynamic)
    for(int i=0;i<(int)blocks.size();i++){
        float **dctBlock = dct::create_matrix(block_size, block_size);
        dct::direct(dctBlock, blocks[i], 0);
        if (invert) {
          for(int k=0;k<blocks[i].size/2;k++)
            for(int l=0;l<blocks[i].size/2;l++)
              dctBlock[k][l] = 0.0;
          dct::inverse(blocks[i], dctBlock, 0, 0.0, 255.);
        }else dct::assign(dctBlock, blocks[i], 0);
        dct::delete_matrix(dctBlock);
    }
    Image<unsigned char> result = grayscale.convert<unsigned char>();
    auto end = std::chrono::steady_clock::now();
    std::cout<<"DCT elapsed time: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"ms"<<std::endl;
    return result;
}

// =====================================================================
// compute_ela: parte importante NO paralelizable (I/O del JPEG).
// Las operaciones aritméticas (+, *, abs, normalized) usan los
// operadores ya paralelizados en image.h, así que aprovechan OpenMP
// indirectamente.
// =====================================================================
Image<unsigned char> compute_ela(const Image<unsigned char> &image, int quality){
    std::cout<<"Computing ELA..."<<std::endl;
    auto begin = std::chrono::steady_clock::now();
    Image<unsigned char> grayscale = image.to_grayscale();
    save_to_file("_temp.jpg", grayscale, quality);
    Image<float> compressed = load_from_file("_temp.jpg").convert<float>();
    compressed = compressed + (grayscale.convert<float>()*(-1));
    compressed = compressed.abs().normalized() * 255;
    Image<unsigned char> result = compressed.convert<unsigned char>();
    auto end = std::chrono::steady_clock::now();
    std::cout<<"ELA elapsed time: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"ms"<<std::endl;
    return result;
}

int main(int argc, char **argv) {
    if(argc == 1) {
        std::cerr<<"Image filename missing from arguments. Usage ./detect <filename>"<<std::endl;
        exit(1);
    }
    
    // Información de hilos para los logs
    #pragma omp parallel
    {
        #pragma omp single
        std::cout<<"Running with "<<omp_get_num_threads()<<" OpenMP threads"<<std::endl;
    }
    
    int block_size=8;
    
    auto total_begin = std::chrono::steady_clock::now();
    
    Image<unsigned char> image = load_from_file(argv[1]);

    // =================================================================
    // VERSIÓN PARALELA ASÍNCRONA (std::async)
    // Usamos std::ref(image) para evitar que std::async haga una copia 
    // profunda de la imagen para cada hilo, pasándola por referencia.
    // =================================================================
    
    // IMPORTANTE: Si vamos a hacer la versión COMBINADA (OpenMP + Async) 
    // hay que descomentar la siguiente línea para 
    // evitar oversubscription (ej. repartir 16 hilos hardware entre 4-5 procesos)
    // omp_set_num_threads(4); 

    std::future<Image<unsigned char>> f_srm3 = std::async(std::launch::async, compute_srm, std::ref(image), 3);
    std::future<Image<unsigned char>> f_srm5 = std::async(std::launch::async, compute_srm, std::ref(image), 5);
    std::future<Image<unsigned char>> f_ela  = std::async(std::launch::async, compute_ela, std::ref(image), 90);
    std::future<Image<unsigned char>> f_dct_inv = std::async(std::launch::async, compute_dct, std::ref(image), block_size, true);
    std::future<Image<unsigned char>> f_dct_dir = std::async(std::launch::async, compute_dct, std::ref(image), block_size, false);

    // .get() bloquea el hilo principal hasta que la tarea termina, 
    // recupera el Image<unsigned char> resultante, y se guarda en disco.
    save_to_file("srm_kernel_3x3.png", f_srm3.get());
    save_to_file("srm_kernel_5x5.png", f_srm5.get());
    save_to_file("ela.png", f_ela.get());
    save_to_file("dct_invert.png", f_dct_inv.get());
    save_to_file("dct_direct.png", f_dct_dir.get());
    
    auto total_end = std::chrono::steady_clock::now();
    std::cout<<"TOTAL elapsed time: "<<std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_begin).count()<<"ms"<<std::endl;

    return 0;
}
