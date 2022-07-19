/*
 * Some compilers do not support OpenMP directives for memory allocation (#pragma omp allocate(...)).
 * To ensure correct compilation in these cases, this flag can be set to 'true'. The benchmark will then be compiled
 * without methods relying on these directives.
 */
#ifndef NO_MEM_DIRECTIVES
#define NO_MEM_DIRECTIVES false
#endif

/*
 * Some compilers do not support OpenMP 'loop' directives. To ensure correct compilation in these cases, this flag can
 * be set to 'true'. The benchmark will then be compiled without methods relying on this directive.
 */
#ifndef NO_LOOP_DIRECTIVES
#define NO_LOOP_DIRECTIVES false
#endif

/*
 * NVC++ cannot compile code with a '#pragma omp parallel for collapse(2)' inside an existing
 * '#pragma omp target teams distribute for ...'. Methods using this can be deactivated with this setting.
 */
#ifndef NO_NESTED_PARALLEL_FOR
#define NO_NESTED_PARALLEL_FOR false
#endif

/*
 * The matrix size to be used. This is a compile-time definition due to tiling and shared memory allocation.
 */
#ifndef MATRIX_SIZE
#define MATRIX_SIZE 8192
#endif

//#if MATRIX_SIZE>8192
//#ifndef __warn_matrix_size
//#warning Large matrices require a lot of RAM. There needs to be at least enough RAM to fit 3 matrices.
//#define __warn_matrix_size
//#endif
//#endif

/*
 * The tile size for tiled matrix multiplication. This is a compile-time definition due to shared memory allocation.
 * This value needs to evenly divide into MATRIX_SIZE as there is no routine for checking the bounds.
 */
#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

//#if TILE_SIZE>32
//#ifndef __warn_tile
//#warning A TILE_SIZE > 32 will often not fit into the shared memory. Be aware, that a total size of TILE_SIZE x TILE_SIZE x sizeof(DATA_TYPE) x 2 is needed.
//#define __warn_tile
//#endif
//#endif

/*
 * The block size to use for the K-blocked method.
 */
#ifndef K_BLOCK_SIZE
#define K_BLOCK_SIZE 1024
#endif

/*
 * The amount of tiles per axis that can be fitted into the matrix. This field is not user-defined.
 */
#define TILE_AXIS_SIZE (MATRIX_SIZE / TILE_SIZE)

/*
 * The data type that should be used for the matrices.
 */
#ifndef DATA_TYPE
#define DATA_TYPE double
#endif

/*
 * The upper and lower values that the matrices should be populated with. Depending on the data type a negative value
 * in VALUE_RANGE_LOWER could lead to bad results.
 */
#ifndef VALUE_RANGE_LOWER
#define VALUE_RANGE_LOWER -1
#endif
#ifndef VALUE_RANGE_UPPER
#define VALUE_RANGE_UPPER 1
#endif

// shortcuts
#define DT DATA_TYPE
#define SIZE MATRIX_SIZE
#define TAS TILE_AXIS_SIZE
