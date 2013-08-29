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
#include "work_lhs.h"
#include "timers.h"

//---------------------------------------------------------------------
// Performs line solves in Z direction by first factoring
// the block-tridiagonal matrix into an upper triangular matrix, 
// and then performing back substitution to solve for the unknow
// vectors of each line.  
// 
// Make sure we treat elements zero to cell_size in the direction
// of the sweep.
//---------------------------------------------------------------------
void z_solve(int rank, int size, int root, int kstart, int kend, int kdelta, int *kdeltas, int *disps)
{
    int i, j, k, m, n, ksize;
    int kstartp1, kendm1;
    int rcounts[size];
    int p;
    int errcode;

    int jstart, jend, jdelta;
    int jstartp1, jendm1;

    int rankm1, rankp1;

    rankm1 = rank - 1;
    rankp1 = rank + 1;
    
    kstartp1 = kstart;
    kendm1 = kend;

    jdelta = (int) grid_points[1]/size;
    jstart = rank*jdelta;
    jend = jstart + jdelta;

    if(rank == size-1){
        jend = grid_points[1];
        jdelta = jend - jstart;
    }

    jstartp1 = jstart;
    jendm1 = jend;

    if(rank == 0) {
        kstartp1 = kstart + 1;
        jstartp1 = jstart + 1;
    } else if(rank == size - 1) {
        kendm1 = kend - 1;
        jendm1 = jend - 1;
    }

    //---------------------------------------------------------------------
    //---------------------------------------------------------------------

    if(rank == root) {
        if (timeron) timer_start(t_zsolve);
    }

    //---------------------------------------------------------------------
    //---------------------------------------------------------------------
    
    //---------------------------------------------------------------------
    // This function computes the left hand side for the three z-factors   
    //---------------------------------------------------------------------

    ksize = grid_points[2]-1;

    //---------------------------------------------------------------------
    // Compute the indices for storing the block-diagonal matrix;
    // determine c (labeled f) and s jacobians
    //---------------------------------------------------------------------
//#pragma omp parallel for default(shared) shared(ksize) private(i,j,k,m,n)
    // for (j = 1; j <= grid_points[1]-2; j++)
    for (j = jstartp1; j < jendm1; j++) {
        for (i = 1; i <= grid_points[0]-2; i++) {
            for (k = 0; k <= ksize; k++) {
            // for (k = kstart; k < kend; k++) 
                tmp1 = 1.0 / u[k][j][i][0];
                tmp2 = tmp1 * tmp1;
                tmp3 = tmp1 * tmp2;

                fjac[k][0][0] = 0.0;
                fjac[k][1][0] = 0.0;
                fjac[k][2][0] = 0.0;
                fjac[k][3][0] = 1.0;
                fjac[k][4][0] = 0.0;

                fjac[k][0][1] = - ( u[k][j][i][1]*u[k][j][i][3] ) * tmp2;
                fjac[k][1][1] = u[k][j][i][3] * tmp1;
                fjac[k][2][1] = 0.0;
                fjac[k][3][1] = u[k][j][i][1] * tmp1;
                fjac[k][4][1] = 0.0;

                fjac[k][0][2] = - ( u[k][j][i][2]*u[k][j][i][3] ) * tmp2;
                fjac[k][1][2] = 0.0;
                fjac[k][2][2] = u[k][j][i][3] * tmp1;
                fjac[k][3][2] = u[k][j][i][2] * tmp1;
                fjac[k][4][2] = 0.0;

                fjac[k][0][3] = - (u[k][j][i][3]*u[k][j][i][3] * tmp2 ) 
                    + c2 * qs[k][j][i];
                fjac[k][1][3] = - c2 *  u[k][j][i][1] * tmp1;
                fjac[k][2][3] = - c2 *  u[k][j][i][2] * tmp1;
                fjac[k][3][3] = ( 2.0 - c2 ) *  u[k][j][i][3] * tmp1;
                fjac[k][4][3] = c2;

                fjac[k][0][4] = ( c2 * 2.0 * square[k][j][i] - c1 * u[k][j][i][4] )
                    * u[k][j][i][3] * tmp2;
                fjac[k][1][4] = - c2 * ( u[k][j][i][1]*u[k][j][i][3] ) * tmp2;
                fjac[k][2][4] = - c2 * ( u[k][j][i][2]*u[k][j][i][3] ) * tmp2;
                fjac[k][3][4] = c1 * ( u[k][j][i][4] * tmp1 )
                    - c2 * ( qs[k][j][i] + u[k][j][i][3]*u[k][j][i][3] * tmp2 );
                fjac[k][4][4] = c1 * u[k][j][i][3] * tmp1;

                njac[k][0][0] = 0.0;
                njac[k][1][0] = 0.0;
                njac[k][2][0] = 0.0;
                njac[k][3][0] = 0.0;
                njac[k][4][0] = 0.0;

                njac[k][0][1] = - c3c4 * tmp2 * u[k][j][i][1];
                njac[k][1][1] =   c3c4 * tmp1;
                njac[k][2][1] =   0.0;
                njac[k][3][1] =   0.0;
                njac[k][4][1] =   0.0;

                njac[k][0][2] = - c3c4 * tmp2 * u[k][j][i][2];
                njac[k][1][2] =   0.0;
                njac[k][2][2] =   c3c4 * tmp1;
                njac[k][3][2] =   0.0;
                njac[k][4][2] =   0.0;

                njac[k][0][3] = - con43 * c3c4 * tmp2 * u[k][j][i][3];
                njac[k][1][3] =   0.0;
                njac[k][2][3] =   0.0;
                njac[k][3][3] =   con43 * c3 * c4 * tmp1;
                njac[k][4][3] =   0.0;

                njac[k][0][4] = - (  c3c4
                        - c1345 ) * tmp3 * (u[k][j][i][1]*u[k][j][i][1])
                    - ( c3c4 - c1345 ) * tmp3 * (u[k][j][i][2]*u[k][j][i][2])
                    - ( con43 * c3c4
                            - c1345 ) * tmp3 * (u[k][j][i][3]*u[k][j][i][3])
                    - c1345 * tmp2 * u[k][j][i][4];

                njac[k][1][4] = (  c3c4 - c1345 ) * tmp2 * u[k][j][i][1];
                njac[k][2][4] = (  c3c4 - c1345 ) * tmp2 * u[k][j][i][2];
                njac[k][3][4] = ( con43 * c3c4
                        - c1345 ) * tmp2 * u[k][j][i][3];
                njac[k][4][4] = ( c1345 )* tmp1;
            }

            
            //---------------------------------------------------------------------
            // now jacobians set, so form left hand side in z direction
            //---------------------------------------------------------------------
            lhsinit(lhs, ksize);
            for (k = 1; k <= ksize-1; k++) {
            // for (k = kstartp1; k < kendm1; k++)
                tmp1 = dt * tz1;
                tmp2 = dt * tz2;

                lhs[k][AA][0][0] = - tmp2 * fjac[k-1][0][0]
                    - tmp1 * njac[k-1][0][0]
                    - tmp1 * dz1; 
                lhs[k][AA][1][0] = - tmp2 * fjac[k-1][1][0]
                    - tmp1 * njac[k-1][1][0];
                lhs[k][AA][2][0] = - tmp2 * fjac[k-1][2][0]
                    - tmp1 * njac[k-1][2][0];
                lhs[k][AA][3][0] = - tmp2 * fjac[k-1][3][0]
                    - tmp1 * njac[k-1][3][0];
                lhs[k][AA][4][0] = - tmp2 * fjac[k-1][4][0]
                    - tmp1 * njac[k-1][4][0];

                lhs[k][AA][0][1] = - tmp2 * fjac[k-1][0][1]
                    - tmp1 * njac[k-1][0][1];
                lhs[k][AA][1][1] = - tmp2 * fjac[k-1][1][1]
                    - tmp1 * njac[k-1][1][1]
                    - tmp1 * dz2;
                lhs[k][AA][2][1] = - tmp2 * fjac[k-1][2][1]
                    - tmp1 * njac[k-1][2][1];
                lhs[k][AA][3][1] = - tmp2 * fjac[k-1][3][1]
                    - tmp1 * njac[k-1][3][1];
                lhs[k][AA][4][1] = - tmp2 * fjac[k-1][4][1]
                    - tmp1 * njac[k-1][4][1];

                lhs[k][AA][0][2] = - tmp2 * fjac[k-1][0][2]
                    - tmp1 * njac[k-1][0][2];
                lhs[k][AA][1][2] = - tmp2 * fjac[k-1][1][2]
                    - tmp1 * njac[k-1][1][2];
                lhs[k][AA][2][2] = - tmp2 * fjac[k-1][2][2]
                    - tmp1 * njac[k-1][2][2]
                    - tmp1 * dz3;
                lhs[k][AA][3][2] = - tmp2 * fjac[k-1][3][2]
                    - tmp1 * njac[k-1][3][2];
                lhs[k][AA][4][2] = - tmp2 * fjac[k-1][4][2]
                    - tmp1 * njac[k-1][4][2];

                lhs[k][AA][0][3] = - tmp2 * fjac[k-1][0][3]
                    - tmp1 * njac[k-1][0][3];
                lhs[k][AA][1][3] = - tmp2 * fjac[k-1][1][3]
                    - tmp1 * njac[k-1][1][3];
                lhs[k][AA][2][3] = - tmp2 * fjac[k-1][2][3]
                    - tmp1 * njac[k-1][2][3];
                lhs[k][AA][3][3] = - tmp2 * fjac[k-1][3][3]
                    - tmp1 * njac[k-1][3][3]
                    - tmp1 * dz4;
                lhs[k][AA][4][3] = - tmp2 * fjac[k-1][4][3]
                    - tmp1 * njac[k-1][4][3];

                lhs[k][AA][0][4] = - tmp2 * fjac[k-1][0][4]
                    - tmp1 * njac[k-1][0][4];
                lhs[k][AA][1][4] = - tmp2 * fjac[k-1][1][4]
                    - tmp1 * njac[k-1][1][4];
                lhs[k][AA][2][4] = - tmp2 * fjac[k-1][2][4]
                    - tmp1 * njac[k-1][2][4];
                lhs[k][AA][3][4] = - tmp2 * fjac[k-1][3][4]
                    - tmp1 * njac[k-1][3][4];
                lhs[k][AA][4][4] = - tmp2 * fjac[k-1][4][4]
                    - tmp1 * njac[k-1][4][4]
                    - tmp1 * dz5;

                lhs[k][BB][0][0] = 1.0
                    + tmp1 * 2.0 * njac[k][0][0]
                    + tmp1 * 2.0 * dz1;
                lhs[k][BB][1][0] = tmp1 * 2.0 * njac[k][1][0];
                lhs[k][BB][2][0] = tmp1 * 2.0 * njac[k][2][0];
                lhs[k][BB][3][0] = tmp1 * 2.0 * njac[k][3][0];
                lhs[k][BB][4][0] = tmp1 * 2.0 * njac[k][4][0];

                lhs[k][BB][0][1] = tmp1 * 2.0 * njac[k][0][1];
                lhs[k][BB][1][1] = 1.0
                    + tmp1 * 2.0 * njac[k][1][1]
                    + tmp1 * 2.0 * dz2;
                lhs[k][BB][2][1] = tmp1 * 2.0 * njac[k][2][1];
                lhs[k][BB][3][1] = tmp1 * 2.0 * njac[k][3][1];
                lhs[k][BB][4][1] = tmp1 * 2.0 * njac[k][4][1];

                lhs[k][BB][0][2] = tmp1 * 2.0 * njac[k][0][2];
                lhs[k][BB][1][2] = tmp1 * 2.0 * njac[k][1][2];
                lhs[k][BB][2][2] = 1.0
                    + tmp1 * 2.0 * njac[k][2][2]
                    + tmp1 * 2.0 * dz3;
                lhs[k][BB][3][2] = tmp1 * 2.0 * njac[k][3][2];
                lhs[k][BB][4][2] = tmp1 * 2.0 * njac[k][4][2];

                lhs[k][BB][0][3] = tmp1 * 2.0 * njac[k][0][3];
                lhs[k][BB][1][3] = tmp1 * 2.0 * njac[k][1][3];
                lhs[k][BB][2][3] = tmp1 * 2.0 * njac[k][2][3];
                lhs[k][BB][3][3] = 1.0
                    + tmp1 * 2.0 * njac[k][3][3]
                    + tmp1 * 2.0 * dz4;
                lhs[k][BB][4][3] = tmp1 * 2.0 * njac[k][4][3];

                lhs[k][BB][0][4] = tmp1 * 2.0 * njac[k][0][4];
                lhs[k][BB][1][4] = tmp1 * 2.0 * njac[k][1][4];
                lhs[k][BB][2][4] = tmp1 * 2.0 * njac[k][2][4];
                lhs[k][BB][3][4] = tmp1 * 2.0 * njac[k][3][4];
                lhs[k][BB][4][4] = 1.0
                    + tmp1 * 2.0 * njac[k][4][4] 
                    + tmp1 * 2.0 * dz5;

                lhs[k][CC][0][0] =  tmp2 * fjac[k+1][0][0]
                    - tmp1 * njac[k+1][0][0]
                    - tmp1 * dz1;
                lhs[k][CC][1][0] =  tmp2 * fjac[k+1][1][0]
                    - tmp1 * njac[k+1][1][0];
                lhs[k][CC][2][0] =  tmp2 * fjac[k+1][2][0]
                    - tmp1 * njac[k+1][2][0];
                lhs[k][CC][3][0] =  tmp2 * fjac[k+1][3][0]
                    - tmp1 * njac[k+1][3][0];
                lhs[k][CC][4][0] =  tmp2 * fjac[k+1][4][0]
                    - tmp1 * njac[k+1][4][0];

                lhs[k][CC][0][1] =  tmp2 * fjac[k+1][0][1]
                    - tmp1 * njac[k+1][0][1];
                lhs[k][CC][1][1] =  tmp2 * fjac[k+1][1][1]
                    - tmp1 * njac[k+1][1][1]
                    - tmp1 * dz2;
                lhs[k][CC][2][1] =  tmp2 * fjac[k+1][2][1]
                    - tmp1 * njac[k+1][2][1];
                lhs[k][CC][3][1] =  tmp2 * fjac[k+1][3][1]
                    - tmp1 * njac[k+1][3][1];
                lhs[k][CC][4][1] =  tmp2 * fjac[k+1][4][1]
                    - tmp1 * njac[k+1][4][1];

                lhs[k][CC][0][2] =  tmp2 * fjac[k+1][0][2]
                    - tmp1 * njac[k+1][0][2];
                lhs[k][CC][1][2] =  tmp2 * fjac[k+1][1][2]
                    - tmp1 * njac[k+1][1][2];
                lhs[k][CC][2][2] =  tmp2 * fjac[k+1][2][2]
                    - tmp1 * njac[k+1][2][2]
                    - tmp1 * dz3;
                lhs[k][CC][3][2] =  tmp2 * fjac[k+1][3][2]
                    - tmp1 * njac[k+1][3][2];
                lhs[k][CC][4][2] =  tmp2 * fjac[k+1][4][2]
                    - tmp1 * njac[k+1][4][2];

                lhs[k][CC][0][3] =  tmp2 * fjac[k+1][0][3]
                    - tmp1 * njac[k+1][0][3];
                lhs[k][CC][1][3] =  tmp2 * fjac[k+1][1][3]
                    - tmp1 * njac[k+1][1][3];
                lhs[k][CC][2][3] =  tmp2 * fjac[k+1][2][3]
                    - tmp1 * njac[k+1][2][3];
                lhs[k][CC][3][3] =  tmp2 * fjac[k+1][3][3]
                    - tmp1 * njac[k+1][3][3]
                    - tmp1 * dz4;
                lhs[k][CC][4][3] =  tmp2 * fjac[k+1][4][3]
                    - tmp1 * njac[k+1][4][3];

                lhs[k][CC][0][4] =  tmp2 * fjac[k+1][0][4]
                    - tmp1 * njac[k+1][0][4];
                lhs[k][CC][1][4] =  tmp2 * fjac[k+1][1][4]
                    - tmp1 * njac[k+1][1][4];
                lhs[k][CC][2][4] =  tmp2 * fjac[k+1][2][4]
                    - tmp1 * njac[k+1][2][4];
                lhs[k][CC][3][4] =  tmp2 * fjac[k+1][3][4]
                    - tmp1 * njac[k+1][3][4];
                lhs[k][CC][4][4] =  tmp2 * fjac[k+1][4][4]
                    - tmp1 * njac[k+1][4][4]
                    - tmp1 * dz5;
            }

            //---------------------------------------------------------------------
            //---------------------------------------------------------------------

            //---------------------------------------------------------------------
            // performs guaussian elimination on this cell.
            // 
            // assumes that unpacking routines for non-first cells 
            // preload C' and rhs' from previous cell.
            // 
            // assumed send happens outside this routine, but that
            // c'(KMAX) and rhs'(KMAX) will be sent to next cell.
            //---------------------------------------------------------------------

            //---------------------------------------------------------------------
            // outer most do loops - sweeping in i direction
            //---------------------------------------------------------------------

            //---------------------------------------------------------------------
            // multiply c[0][j][i] by b_inverse and copy back to c
            // multiply rhs(0) by b_inverse(0) and copy to rhs
            //---------------------------------------------------------------------
            binvcrhs( lhs[0][BB], lhs[0][CC], rhs[0][j][i] );

            //---------------------------------------------------------------------
            // begin inner most do loop
            // do all the elements of the cell unless last 
            //---------------------------------------------------------------------
            for (k = 1; k <= ksize-1; k++) {
            // for (k = kstartp1; k < kendm1; k++)
                //-------------------------------------------------------------------
                // subtract A*lhs_vector(k-1) from lhs_vector(k)
                // 
                // rhs(k) = rhs(k) - A*rhs(k-1)
                //-------------------------------------------------------------------
                matvec_sub(lhs[k][AA], rhs[k-1][j][i], rhs[k][j][i]);

                //-------------------------------------------------------------------
                // B(k) = B(k) - C(k-1)*A(k)
                // matmul_sub(AA,i,j,k,c,CC,i,j,k-1,c,BB,i,j,k)
                //-------------------------------------------------------------------
                matmul_sub(lhs[k][AA], lhs[k-1][CC], lhs[k][BB]);

                //-------------------------------------------------------------------
                // multiply c[k][j][i] by b_inverse and copy back to c
                // multiply rhs[0][j][i] by b_inverse[0][j][i] and copy to rhs
                //-------------------------------------------------------------------
                binvcrhs( lhs[k][BB], lhs[k][CC], rhs[k][j][i] );
            }

            //---------------------------------------------------------------------
            // Now finish up special cases for last cell
            //---------------------------------------------------------------------

            //---------------------------------------------------------------------
            // rhs(ksize) = rhs(ksize) - A*rhs(ksize-1)
            //---------------------------------------------------------------------
            matvec_sub(lhs[ksize][AA], rhs[ksize-1][j][i], rhs[ksize][j][i]);

            //---------------------------------------------------------------------
            // B(ksize) = B(ksize) - C(ksize-1)*A(ksize)
            // matmul_sub(AA,i,j,ksize,c,
            // $              CC,i,j,ksize-1,c,BB,i,j,ksize)
            //---------------------------------------------------------------------
            matmul_sub(lhs[ksize][AA], lhs[ksize-1][CC], lhs[ksize][BB]);

            //---------------------------------------------------------------------
            // multiply rhs(ksize) by b_inverse(ksize) and copy to rhs
            //---------------------------------------------------------------------
            binvrhs( lhs[ksize][BB], rhs[ksize][j][i] );

            //---------------------------------------------------------------------
            //---------------------------------------------------------------------

            //---------------------------------------------------------------------
            // back solve: if last cell, then generate U(ksize)=rhs(ksize)
            // else assume U(ksize) is loaded in un pack backsub_info
            // so just use it
            // after u(kstart) will be sent to next cell
            //---------------------------------------------------------------------

            for (k = ksize-1; k >= 0; k--) {
            // for (k = kendm1-1; k >= kstart; k--)
                for (m = 0; m < BLOCK_SIZE; m++) {
                    for (n = 0; n < BLOCK_SIZE; n++) {
                        rhs[k][j][i][m] = rhs[k][j][i][m] 
                            - lhs[k][CC][n][m]*rhs[k+1][j][i][n];
                    }
                }
            }
        }

    }

    // MPI_Barrier(MPI_COMM_WORLD);
    tflipstart = MPI_Wtime();
    // for(j=0; j<grid_points[1]; j++)
    for(j=jstart; j<jend; j++) {
        for(i=0; i<=grid_points[0]-1; i++) {
            for(k=0; k<=ksize; k++) {
                for(m=0; m<BLOCK_SIZE; m++) {
                    rhsp[j][k][i][m] = rhs[k][j][i][m];
                }
            }
        }
    }

    // MPI_Barrier(MPI_COMM_WORLD);
    tflipend = MPI_Wtime();

    tflip += tflipend - tflipstart;

    disps[0] = 0;
    rcounts[0] = KMAX*(kdeltas[0])*(IMAXP+1)*5;
    for(p=1; p<size; p++) {
        rcounts[p] = KMAX*(kdeltas[p])*(IMAXP+1)*5;
        disps[p] = disps[p-1] + rcounts[p-1];
    }
    
    // MPI_Barrier(MPI_COMM_WORLD);
    tflipstart = MPI_Wtime();

    errcode = MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,&rhsp[0][0][0][0],rcounts,disps,MPI_DOUBLE,MPI_COMM_WORLD);

    for(j=0; j<grid_points[1]; j++) {
        for(i=0; i<=grid_points[0]-1; i++) {
            for(k=0; k<=ksize; k++) {
                for(m=0; m<BLOCK_SIZE; m++) {
                    rhs[k][j][i][m] = rhsp[j][k][i][m];
                }
            }
        }
        
    }

    // MPI_Barrier(MPI_COMM_WORLD);
    tflipend = MPI_Wtime();
    tflip += tflipend - tflipstart;

    if(rank == root) {
        if (timeron) timer_stop(t_zsolve);
    }
}
