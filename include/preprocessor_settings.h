#pragma once

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
 * The matrix size to be used. This is a compile-time definition due to tiling and shared memory allocation.
 */
#ifndef MATRIX_SIZE
#define MATRIX_SIZE 8192
#endif

/*
 * The tile size for tiled matrix multiplication (A & B).
 * This is a compile-time definition due to shared memory allocation.
 */
#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

/*
 * The amount of tiles per axis that can be fitted into the matrix. This field is not user-defined.
 */
#ifndef TILE_AXIS_SIZE
#define TILE_AXIS_SIZE (MATRIX_SIZE / TILE_SIZE)
#endif

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

#ifndef A_THREAD_LIMIT
#define A_THREAD_LIMIT -1
#endif

#ifndef A_TEAMS
#define A_TEAMS -1
#endif

#ifndef A_BLOCK_SIZE
#define A_BLOCK_SIZE 1024
#endif

/*
 * Should the compiler defaults for num_teams and num_threads be overwritten
 */
#ifndef OVERWRITE_THREAD_LIMIT
#define OVERWRITE_THREAD_LIMIT false
#endif

// shortcuts
#define DT DATA_TYPE
#define SIZE MATRIX_SIZE
#define TAS TILE_AXIS_SIZE
