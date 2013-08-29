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

//---------------------------------------------------------------------
// This subroutine initializes the field variable u using 
// tri-linear transfinite interpolation of the boundary values     
//---------------------------------------------------------------------
void initialize(int rank, int size, int root, int kstart, int kend, int kdelta, int *kdeltas, int *disps)
{
    int i, j, k, m, ix, iy, iz;
    double xi, eta, zeta, Pface[2][3][5], Pxi, Peta, Pzeta, temp[5];

    // int kstart, kdelta, kend;
    int scount, rcount;
    int rcounts[size];
    int errcode;
    int p;

    // kdelta = (int) grid_points[2]/size;
    // kstart = rank*kdelta;
    // kend = kstart + kdelta;
    // if(rank == size-1){
    //     kend = grid_points[2] - 1;
    // }

    // #pragma omp parallel default(shared) \
    //             private(i,j,k,m,zeta,eta,xi,ix,iy,iz,Pface,Pxi,Peta,Pzeta,temp)
    // {
    //---------------------------------------------------------------------
    // Later (in compute_rhs) we compute 1/u for every element. A few of 
    // the corner elements are not used, but it convenient (and faster) 
    // to compute the whole thing with a simple loop. Make sure those 
    // values are nonzero by initializing the whole thing here. 
    //---------------------------------------------------------------------
    // #pragma omp for schedule(static)
    // for (k = 0; k <= grid_points[2]-1; k++)
    for (k = kstart; k < kend; k++) {
        for (j = 0; j <= grid_points[1]-1; j++) {
            for (i = 0; i <= grid_points[0]-1; i++) {
                for (m = 0; m < 5; m++) {
                    u[k][j][i][m] = 1.0; //*1000000000*k+1000000*j+1000*i+m;
                }
            }
        }
    }

    //scount = kdelta*(grid_points[1]+1)*(grid_points[0]+1)*5;
    scount = kdelta*(JMAXP+1)*(IMAXP+1)*5;
    rcount = scount;

    for(p=0; p<size; p++) {
        rcounts[p] = kdeltas[p]*(JMAXP+1)*(IMAXP+1)*5;
        disps[p] = (JMAXP+1)*(IMAXP+1)*5*disps[p];
    }
    // printf("%d -> scount = %d, rcount = %d\n",rank,scount,rcount);
    // errcode = MPI_Allgather(&u[kstart][0][0][0],scount,MPI_DOUBLE,&u[0][0][0][0],rcount,MPI_DOUBLE,MPI_COMM_WORLD);
    // errcode = MPI_Allgatherv(&u[kstart][0][0][0],scount,MPI_DOUBLE,&u[0][0][0][0],rcount,disps,MPI_DOUBLE,MPI_COMM_WORLD);
    // errcode = MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,&u[0][0][0][0],rcounts,disps,MPI_DOUBLE,MPI_COMM_WORLD);
    // printf("%d -> errcode = %d\n",rank,errcode);

    //---------------------------------------------------------------------
    // first store the "interpolated" values everywhere on the grid    
    //---------------------------------------------------------------------
    // #pragma omp for schedule(static) nowait
    // for (k = 0; k <= grid_points[2]-1; k++) 
    for (k = kstart; k < kend ; k++) {
        zeta = (double)(k) * dnzm1;
        for (j = 0; j <= grid_points[1]-1; j++) {
            eta = (double)(j) * dnym1;
            for (i = 0; i <= grid_points[0]-1; i++) {
                xi = (double)(i) * dnxm1;

                for (ix = 0; ix < 2; ix++) {
                    exact_solution((double)ix, eta, zeta, &Pface[ix][0][0]);
                }

                for (iy = 0; iy < 2; iy++) {
                    exact_solution(xi, (double)iy , zeta, &Pface[iy][1][0]);
                }

                for (iz = 0; iz < 2; iz++) {
                    exact_solution(xi, eta, (double)iz, &Pface[iz][2][0]);
                }

                for (m = 0; m < 5; m++) {
                    Pxi   = xi   * Pface[1][0][m] + (1.0-xi)   * Pface[0][0][m];
                    Peta  = eta  * Pface[1][1][m] + (1.0-eta)  * Pface[0][1][m];
                    Pzeta = zeta * Pface[1][2][m] + (1.0-zeta) * Pface[0][2][m];

                    u[k][j][i][m] = Pxi + Peta + Pzeta - 
                        Pxi*Peta - Pxi*Pzeta - Peta*Pzeta + 
                        Pxi*Peta*Pzeta;
                }
            }
        }
    }

    //---------------------------------------------------------------------
    // now store the exact values on the boundaries        
    //---------------------------------------------------------------------

    //---------------------------------------------------------------------
    // west face                                                  
    //---------------------------------------------------------------------
    i = 0;
    xi = 0.0;
    // #pragma omp for schedule(static) nowait
    // for (k = 0; k <= grid_points[2]-1; k++) {
    for (k = kstart; k < kend; k++) {
        zeta = (double)(k) * dnzm1;
        for (j = 0; j <= grid_points[1]-1; j++) {
            eta = (double)(j) * dnym1;
            exact_solution(xi, eta, zeta, temp);
            for (m = 0; m < 5; m++) {
                u[k][j][i][m] = temp[m];
            }
        }
    }

    //---------------------------------------------------------------------
    // east face                                                      
    //---------------------------------------------------------------------
    i = grid_points[0]-1;
    xi = 1.0;
    // #pragma omp for schedule(static) nowait
    for (k = kstart; k < kend; k++) {
        zeta = (double)(k) * dnzm1;
        for (j = 0; j <= grid_points[1]-1; j++) {
            eta = (double)(j) * dnym1;
            exact_solution(xi, eta, zeta, temp);
            for (m = 0; m < 5; m++) {
                u[k][j][i][m] = temp[m];
            }
        }
    }

    //---------------------------------------------------------------------
    // south face                                                 
    //---------------------------------------------------------------------
    j = 0;
    eta = 0.0;
    // #pragma omp for schedule(static) nowait
    for (k = kstart; k < kend; k++) {
        zeta = (double)(k) * dnzm1;
        for (i = 0; i <= grid_points[0]-1; i++) {
            xi = (double)(i) * dnxm1;
            exact_solution(xi, eta, zeta, temp);
            for (m = 0; m < 5; m++) {
                u[k][j][i][m] = temp[m];
            }
        }
    }

    //---------------------------------------------------------------------
    // north face                                    
    //---------------------------------------------------------------------
    j = grid_points[1]-1;
    eta = 1.0;
    // #pragma omp for schedule(static)
    // for (k = kstart; k <= grid_points[2]-1; k++)
    for (k = kstart; k < kend; k++) {
        zeta = (double)(k) * dnzm1;
        for (i = 0; i <= grid_points[0]-1; i++) {
            xi = (double)(i) * dnxm1;
            exact_solution(xi, eta, zeta, temp);
            for (m = 0; m < 5; m++) {
                u[k][j][i][m] = temp[m];
            }
        }
    }

    scount = kdelta*(JMAXP+1)*(IMAXP+1)*5;
    rcount = scount;
    // printf("%d -> scount = %d, rcount = %d\n",rank,scount,rcount);
    // errcode = MPI_Allgather(&u[kstart][0][0][0],scount,MPI_DOUBLE,&u[0][0][0][0],rcount,MPI_DOUBLE,MPI_COMM_WORLD);
    // errcode = MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,&u[0][0][0][0],rcounts,disps,MPI_DOUBLE,MPI_COMM_WORLD);
    // printf("%d -> errcode = %d\n",rank,errcode);

    //---------------------------------------------------------------------
    // bottom face                                       
    //---------------------------------------------------------------------
    k = 0;
    zeta = 0.0;
    if(rank == (int) k/kdelta) {
        // #pragma omp for schedule(static) nowait
        for (j = 0; j <= grid_points[1]-1; j++) {
            eta = (double)(j) * dnym1;
            for (i =0; i <= grid_points[0]-1; i++) {
                xi = (double)(i) *dnxm1;
                exact_solution(xi, eta, zeta, temp);
                for (m = 0; m < 5; m++) {
                    u[k][j][i][m] = temp[m];
                }
            }
        }
    }

    //---------------------------------------------------------------------
    // top face     
    //---------------------------------------------------------------------
    k = grid_points[2]-1;
    zeta = 1.0;
    if(rank == (int) k/kdelta) {
        // #pragma omp for schedule(static) nowait
        for (j = 0; j <= grid_points[1]-1; j++) {
            eta = (double)(j) * dnym1;
            for (i = 0; i <= grid_points[0]-1; i++) {
                xi = (double)(i) * dnxm1;
                exact_solution(xi, eta, zeta, temp);
                for (m = 0; m < 5; m++) {
                    u[k][j][i][m] = temp[m];
                }
            }
        }
    }
    // end parallel

    // Because the array u is padded, we need to send extra elements in j and i
    // scount = kdelta*(JMAXP+1)*(IMAXP+1)*5;
    // rcount = scount;
    // printf("%d -> scount = %d, rcount = %d\n",rank,scount,rcount);
    // printf("%d -> start here, rcounts = %d, disps = %d\n",rank,rcounts[rank],disps[rank]);
    // errcode = MPI_Allgather(&u[kstart][0][0][0],scount,MPI_DOUBLE,&u[0][0][0][0],rcount,MPI_DOUBLE,MPI_COMM_WORLD);
    errcode = MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,&u[0][0][0][0],rcounts,disps,MPI_DOUBLE,MPI_COMM_WORLD);
    // printf("%d -> errcode = %d\n",rank,errcode);


    // for (k = 0; k <= grid_points[2]-1; k++) {
    //     for (j = 0; j <= grid_points[1]-1; j++) {
    //         for (i = 0; i <= grid_points[0]-1; i++) {
    //             for (m = 0; m < 5; m++) {
    //                 printf("%d -> u[%d][%d][%d][%d] = %f\n",rank,k,j,i,m,u[k][j][i][m]);
    //                 // printf("%d -> allu[%d][%d][%d][%d] = %f\n",rank,k,j,i,m,allu[k][j][i][m]);
    //             }
    //         }
    //     }
    // }
}


void lhsinit(double lhs[][3][5][5], int ni)
{
    int i, m, n;

    //---------------------------------------------------------------------
    // zero the whole left hand side for starters
    // set all diagonal values to 1. This is overkill, but convenient
    //---------------------------------------------------------------------
    i = 0;
    for (n = 0; n < 5; n++) {
        for (m = 0; m < 5; m++) {
            lhs[i][0][n][m] = 0.0;
            lhs[i][1][n][m] = 0.0;
            lhs[i][2][n][m] = 0.0;
        }
        lhs[i][1][n][n] = 1.0;
    }
    i = ni;
    for (n = 0; n < 5; n++) {
        for (m = 0; m < 5; m++) {
            lhs[i][0][n][m] = 0.0;
            lhs[i][1][n][m] = 0.0;
            lhs[i][2][n][m] = 0.0;
        }
        lhs[i][1][n][n] = 1.0;
    }
}
