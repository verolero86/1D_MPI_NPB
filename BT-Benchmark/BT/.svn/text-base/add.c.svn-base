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
#include "timers.h"

//---------------------------------------------------------------------
// addition of update to the vector u
//---------------------------------------------------------------------
void add(int rank, int size, int root, int kstart, int kend, int kdelta, int *kdeltas, int *disps)
{
    int i, j, k, m;
    // int kstart, kend, kdelta;
    int kstartp1, kendm1;
    int scount, rcount;
    int rcounts[size];
    int errcode;

    int rankm1, rankp1;
    MPI_Status status;
    int tright, tleft;
    int p;

    // kdelta = (int) grid_points[2]/size;
    // kstart = rank*kdelta;
    // kend = kstart + kdelta;
    // if(rank == size-1){
    //     kend = grid_points[2] - 1;
    // }

    rankm1 = rank - 1;
    rankp1 = rank + 1;

    kstartp1 = kstart;
    kendm1 = kend;
    if(rank == 0) {
        kstartp1 = kstart + 1;
    } else if(rank == size - 1) {
        kendm1 = kend - 1;
    }

    if(rank == root) {
        if (timeron) timer_start(t_add);
    }

// #pragma omp parallel for default(shared) private(i,j,k,m)
    // for (k = 1; k <= grid_points[2]-2; k++) {
    for (k = kstartp1; k < kendm1; k++) {
        for (j = 1; j <= grid_points[1]-2; j++) {
            for (i = 1; i <= grid_points[0]-2; i++) {
                for (m = 0; m < 5; m++) {
                    u[k][j][i][m] = u[k][j][i][m] + rhs[k][j][i][m];
                }
            }
        }
    }


    // scount = kdelta*(JMAXP+1)*(IMAXP+1)*5;
    // rcount = scount;
    // errcode = MPI_Allgather(&u[kstart][0][0][0],scount,MPI_DOUBLE,&u[0][0][0][0],rcount,MPI_DOUBLE,MPI_COMM_WORLD);

    disps[0] = 0;
    rcounts[0] = kdeltas[0]*(JMAXP+1)*(IMAXP+1)*5;
    for(p=1; p<size; p++) {
        rcounts[p] = kdeltas[p]*(JMAXP+1)*(IMAXP+1)*5;
        disps[p] = disps[p-1] + rcounts[p-1];
    }
    
    errcode = MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,&u[0][0][0][0],rcounts,disps,MPI_DOUBLE,MPI_COMM_WORLD);
    //
    // THIS SENDS ONLY BOUNDARIES OF u BETWEEN NEIGHBORS
    // scount = (JMAXP+1)*(IMAXP+1)*5;
    // rcount = scount;

    // tright = 0;
    // if(rank == 0) {
    //     MPI_Send(&u[kend-1][0][0],scount,MPI_DOUBLE,rankp1,tright,MPI_COMM_WORLD);
    // } else {
    //     MPI_Recv(&u[kstart-1][0][0],rcount,MPI_DOUBLE,rankm1,tright,MPI_COMM_WORLD,&status);
    //     if(rankp1 < size) {
    //         MPI_Send(&u[kend-1][0][0],scount,MPI_DOUBLE,rankp1,tright,MPI_COMM_WORLD);
    //     }
    // }

    // tleft = 1;
    // if(rank == size-1) {
    //     MPI_Send(&u[kstart][0][0],scount,MPI_DOUBLE,rankm1,tleft,MPI_COMM_WORLD);
    // } else {
    //     MPI_Recv(&u[kend][0][0],rcount,MPI_DOUBLE,rankp1,tleft,MPI_COMM_WORLD,&status);
    //     if(rankm1 >= 0) {
    //         MPI_Send(&u[kstart][0][0],scount,MPI_DOUBLE,rankm1,tleft,MPI_COMM_WORLD);
    //     }
    // }

    if(rank == root) {
        if (timeron) timer_stop(t_add);
    }
}
