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

#include <math.h>
#include "header.h"

//---------------------------------------------------------------------
// this function computes the norm of the difference between the
// computed solution and the exact solution
//---------------------------------------------------------------------
void error_norm(double rms[5],int rank,int size,int root, int kstart, int kend, int kdelta)
{
    int i, j, k, m, d;
    double xi, eta, zeta, u_exact[5], add;
    double rms_local[5];

    // int kstart, kdelta, kend;
    int scount, rcount;
    int errcode;

    // kdelta = (int) grid_points[2]/size;
    // kstart = rank*kdelta;
    // kend = kstart + kdelta;
    // if(rank == size-1){
    //     kend = grid_points[2] - 1;
    // }

    for (m = 0; m < 5; m++) {
        rms[m] = 0.0;
    }

    //#pragma omp parallel default(shared) \
    //        private(i,j,k,m,zeta,eta,xi,add,u_exact,rms_local) shared(rms)
    // open parallel 
    for (m = 0; m < 5; m++) {
        rms_local[m] = 0.0;
    }
    //#pragma omp for nowait
    // for (k = 0; k <= grid_points[2]-1; k++) 
    for (k = kstart; k < kend; k++) {
        zeta = (double)(k) * dnzm1;
        for (j = 0; j <= grid_points[1]-1; j++) {
            eta = (double)(j) * dnym1;
            for (i = 0; i <= grid_points[0]-1; i++) {
                xi = (double)(i) * dnxm1;
                exact_solution(xi, eta, zeta, u_exact);

                for (m = 0; m < 5; m++) {
                    add = u[k][j][i][m]-u_exact[m];
                    rms_local[m] = rms_local[m] + add*add;
                }
            }
        }
    }

    /*
       for (m = 0; m < 5; m++) {
// #pragma omp atomic
rms[m] += rms_local[m];
}
//end parallel
*/

    MPI_Allreduce(&rms_local[0],&rms[0],5,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

    for (m = 0; m < 5; m++) {
        for (d = 0; d < 3; d++) {
            rms[m] = rms[m] / (double)(grid_points[d]-2);
        }
        rms[m] = sqrt(rms[m]);
    }
    // for(m=0; m<5; m++) {
    //     printf("%d -> error_norm: rms[%d] = %f\n",rank,m,rms[m]);
    // }
}

void rhs_norm(double rms[5],int rank, int size, int root, int kstart, int kend, int kdelta)
{
    int i, j, k, d, m;
    double add;
    double rms_local[5];

    // int kstart, kdelta, kend;
    int kstartp1, kendm1;
    int scount, rcount;
    int errcode;

    // kdelta = (int) grid_points[2]/size;
    // kstart = rank*kdelta;
    // kend = kstart + kdelta;
    // if(rank == size-1){
    //     kend = grid_points[2] - 1;
    // }

    kstartp1 = kstart;
    kendm1 = kend;
    if(rank == 0) {
        kstartp1 = kstart + 1;
    } else if(rank == size - 1) {
        kendm1 = kend - 1;
    }

    for (m = 0; m < 5; m++) {
        rms[m] = 0.0;
    } 

    //#pragma omp parallel default(shared) private(i,j,k,m,add,rms_local) \
    //    shared(rms)
    // open parallel
    for (m = 0; m < 5; m++) {
        rms_local[m] = 0.0;
    }
    // #pragma omp for nowait
    for (k = kstartp1; k < kendm1; k++) {
        for (j = 1; j <= grid_points[1]-2; j++) {
            for (i = 1; i <= grid_points[0]-2; i++) {
                for (m = 0; m < 5; m++) {
                    add = rhs[k][j][i][m];
                    rms_local[m] = rms_local[m] + add*add;
                    // rms[m] = rms[m] + add*add;
                } 
            } 
        } 
    } 
    /*
       for (m = 0; m < 5; m++) {
        // #pragma omp atomic
        rms[m] += rms_local[m];
        }
    */  
    //end parallel

    MPI_Allreduce(&rms_local[0],&rms[0],5,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

    for (m = 0; m < 5; m++) {
        for (d = 0; d < 3; d++) {
            rms[m] = rms[m] / (double)(grid_points[d]-2);
        } 
        rms[m] = sqrt(rms[m]);
    } 

}
