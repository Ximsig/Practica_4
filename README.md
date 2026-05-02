# Práctica 4 — Computación de Alto Rendimiento

Paralelización mediante OpenMP y programación asíncrona (`std::async`)
del análisis forense de manipulación de imágenes digitales.

**Grado en Ingeniería en Inteligencia Artificial — Universidad de Alicante**

## Autores

- Joaquín Sigüenza Chilar
- Linxi Jiang

## Estructura del repositorio

```
.
├── Task0/                   Ejercicios previos
│   ├── async_functions.cpp      Ejemplo de std::async
│   ├── initialize_vectors.cpp   Comparativa push_back vs reserva
│   └── suma_vectores_float.cpp  Suma de vectores con OpenMP
│
├── Task1/                   Análisis forense paralelizado
│   ├── CMakeLists.txt
│   ├── face_swap_enhanced.png   Imagen de prueba (manipulada)
│   ├── compressions.png         Imagen de referencia
│   └── src/
│       ├── main.cc              Programa principal (OpenMP + std::async)
│       ├── CMakeLists.txt
│       └── utils/
│           ├── image.h          Operaciones de imagen paralelizadas
│           ├── image.cc         Lectura/escritura PNG y JPEG
│           ├── dct.h
│           ├── dct.cc           Transformada DCT directa e inversa
│           └── CMakeLists.txt
│
└── diagramas/               Diagramas de la memoria
    ├── dependencias_procesos.png
    ├── flujo_srm.png
    ├── flujo_ela.png
    └── flujo_dct.png
```

## Compilación

Requisitos: `cmake`, `g++` con soporte de OpenMP, `libpng-dev`, `libjpeg-dev`.

```bash
cd Task1
cmake -S . -B build
cd build
make
```

## Ejecución

```bash
# Con número de hilos por defecto del sistema
./detect ../face_swap_enhanced.png

# Especificando el número de hilos OpenMP
OMP_NUM_THREADS=8 ./detect ../face_swap_enhanced.png
```

El programa genera cinco ficheros PNG de salida:

- `srm_kernel_3x3.png` — análisis SRM con kernel 3×3
- `srm_kernel_5x5.png` — análisis SRM con kernel 5×5
- `ela.png` — Error Level Analysis
- `dct_invert.png` — DCT inversa (anula bajas frecuencias)
- `dct_direct.png` — DCT directa

## Estrategia de paralelización

- **OpenMP** (paralelismo de datos): bucles internos sobre píxeles e
  iteración por bloques DCT en `compute_dct`.
- **std::async** (paralelismo funcional): los cinco procesos de análisis
  se lanzan como tareas asíncronas independientes en `main.cc`.
- **Versión combinada**: ambos niveles activos. Se recomienda fijar
  `OMP_NUM_THREADS` entre 4 y 8 para evitar oversubscription en máquinas
  de 16 hilos lógicos.
