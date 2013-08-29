//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is an OpenMP C version of the NPB BT code. This OpenMP  //
//  C version is developed by the Center for Manycore Programming at Seoul //
//  National University and derived from the OpenMP Fortran versions in    //
//  "NPB3.3-OMP" developed by NAS.                                         //
//                                                                         //
//  Permission to use, copy, distribute and modify this software for any   //
//  purpose with or without fee is hereby granted. This software is        //
//  provided "as is" without express or implied warranty.                  //
//                                                                         //
//  Information on NPB 3.3, including the technical report, the original   //
//  specifications, source code, results and information on how to submit  //
//  new results, is available at:                                          //
//                                                                         //
//           http://www.nas.nasa.gov/Software/NPB/                         //
//                                                                         //
//  Send comments or suggestions for this OpenMP C version to              //
//  cmp@aces.snu.ac.kr                                                     //
//                                                                         //
//          Center for Manycore Programming                                //
//          School of Computer Science and Engineering                     //
//          Seoul National University                                      //
//          Seoul 151-744, Korea                                           //
//                                                                         //
//          E-mail:  cmp@aces.snu.ac.kr                                    //
//                                                                         //
//-------------------------------------------------------------------------//

//-------------------------------------------------------------------------//
// Authors: Sangmin Seo, Jungwon Kim, Jun Lee, Jeongho Nah, Gangwon Jo,    //
//          and Jaejin Lee                                                 //
//-------------------------------------------------------------------------//

#include "header.h"

void adi(int rank, int size, int root, int kstart, int kend, int kdelta, int *kdeltas, int *disps)
{

    double ts, te;

    // MPI_Barrier(MPI_COMM_WORLD);
    ts = MPI_Wtime();

    compute_rhs(rank,size,root, kstart, kend, kdelta,kdeltas,disps);
    // MPI_Barrier(MPI_COMM_WORLD);
    te = MPI_Wtime();
    tcrhs += te - ts;

    // MPI_Barrier(MPI_COMM_WORLD);
    ts = MPI_Wtime();

    x_solve(rank,size,root, kstart, kend, kdelta);

    // MPI_Barrier(MPI_COMM_WORLD);
    te = MPI_Wtime();
    tx += te - ts;

    // MPI_Barrier(MPI_COMM_WORLD);
    ts = MPI_Wtime();

    y_solve(rank,size,root, kstart, kend, kdelta, kdeltas, disps);

    // MPI_Barrier(MPI_COMM_WORLD);
    te = MPI_Wtime();
    ty += te - ts;

    // MPI_Barrier(MPI_COMM_WORLD);
    ts = MPI_Wtime();

    z_solve(rank,size,root, kstart, kend, kdelta,kdeltas,disps);

    // MPI_Barrier(MPI_COMM_WORLD);
    te = MPI_Wtime();
    tz += te - ts;

    // MPI_Barrier(MPI_COMM_WORLD);
    ts = MPI_Wtime();

    add(rank, size, root, kstart, kend, kdelta,kdeltas,disps);

    // MPI_Barrier(MPI_COMM_WORLD);
    te = MPI_Wtime();
    tadd += te - ts;

    // if(rank == root) {
    //     printf("%d -> compute_rhs took: %f secs\n",rank,tcomputerhs);
    //     printf("%d -> x_solve took: %f secs\n",rank,tx);
    //     printf("%d -> y_solve took: %f secs\n",rank,ty);
    //     printf("%d -> z_solve took: %f secs\n",rank,tz);
    //     printf("%d -> add took: %f secs\n",rank,tadd);
    // }

}
