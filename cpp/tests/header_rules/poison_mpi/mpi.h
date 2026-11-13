/* A poisoned mpi.h, first on the include path of the header check: any pantr core header that
 * includes mpi.h, directly or through another header, fails to compile here. */
#error "poisoned mpi.h reached: a pantr core header must not include mpi.h"
