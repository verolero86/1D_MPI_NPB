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

void compute_rhs(int rank,int size,int root, int kstart, int kend, int kdelta, int *kdeltas, int *disps)
{
    int i, j, k, m;
    double rho_inv, uijk, up1, um1, vijk, vp1, vm1, wijk, wp1, wm1;

    // int kstart, kend, kdelta;
    int kstartp1, kendm1;
    int kstartbdy, kendbdy;
    int scount, rcount;
    int rcounts[size];
    int p;
    int errcode;

    int rankm1, rankp1;
    MPI_Status status;
    int tright, tleft;

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
        if (timeron) timer_start(t_rhs);
    }
    // #pragma omp parallel default(shared) private(i,j,k,m,rho_inv,uijk,up1,um1,
    //                                     vijk,vp1,vm1,wijk,wp1,wm1)
    // open parallel bracket
    //---------------------------------------------------------------------
    // compute the reciprocal of density, and the kinetic energy, 
    // and the speed of sound.
    //---------------------------------------------------------------------
    // #pragma omp for schedule(static) nowait
    // for (k = 0; k <= grid_points[2]-1; k++)
    for(k=kstart; k<kend; k++) {
        for (j = 0; j <= grid_points[1]-1; j++) {
            for (i = 0; i <= grid_points[0]-1; i++) {
                rho_inv = 1.0/u[k][j][i][0];
                rho_i[k][j][i] = rho_inv;
                us[k][j][i] = u[k][j][i][1] * rho_inv;
                vs[k][j][i] = u[k][j][i][2] * rho_inv;
                ws[k][j][i] = u[k][j][i][3] * rho_inv;
                square[k][j][i] = 0.5* (
                        u[k][j][i][1]*u[k][j][i][1] + 
                        u[k][j][i][2]*u[k][j][i][2] +
                        u[k][j][i][3]*u[k][j][i][3] ) * rho_inv;
                qs[k][j][i] = square[k][j][i] * rho_inv;
            }
        }
    }

    scount = kdelta*(JMAXP+1)*(IMAXP+1);
    scount = (JMAXP+1)*(IMAXP+1);
    rcount = scount;

    if(size > 1) {
        tright = 0;
        if(rank == 0) {
            MPI_Send(&rho_i[kend-1][0][0],scount,MPI_DOUBLE,rankp1,tright,MPI_COMM_WORLD);
        } else {
            MPI_Recv(&rho_i[kstart-1][0][0],rcount,MPI_DOUBLE,rankm1,tright,MPI_COMM_WORLD,&status);
            if(rankp1 < size) {
                MPI_Send(&rho_i[kend-1][0][0],scount,MPI_DOUBLE,rankp1,tright,MPI_COMM_WORLD);
            }
        }

        tleft = 1;
        if(rank == size-1) {
            MPI_Send(&rho_i[kstart][0][0],scount,MPI_DOUBLE,rankm1,tleft,MPI_COMM_WORLD);
        } else {
            MPI_Recv(&rho_i[kend][0][0],rcount,MPI_DOUBLE,rankp1,tleft,MPI_COMM_WORLD,&status);
            if(rankm1 >= 0) {
                MPI_Send(&rho_i[kstart][0][0],scount,MPI_DOUBLE,rankm1,tleft,MPI_COMM_WORLD);
            }
        }

        tright = 2;
        if(rank == 0) {
            MPI_Send(&us[kend-1][0][0],scount,MPI_DOUBLE,rankp1,tright,MPI_COMM_WORLD);
        } else {
            MPI_Recv(&us[kstart-1][0][0],rcount,MPI_DOUBLE,rankm1,tright,MPI_COMM_WORLD,&status);
            if(rankp1 < size) {
                MPI_Send(&us[kend-1][0][0],scount,MPI_DOUBLE,rankp1,tright,MPI_COMM_WORLD);
            }
        }

        tleft = 3;
        if(rank == size-1) {
            MPI_Send(&us[kstart][0][0],scount,MPI_DOUBLE,rankm1,tleft,MPI_COMM_WORLD);
        } else {
            MPI_Recv(&us[kend][0][0],rcount,MPI_DOUBLE,rankp1,tleft,MPI_COMM_WORLD,&status);
            if(rankm1 >= 0) {
                MPI_Send(&us[kstart][0][0],scount,MPI_DOUBLE,rankm1,tleft,MPI_COMM_WORLD);
            }
        }

        // vs boundary exchange of width k=1
        tright = 4;
        if(rank == 0) {
            MPI_Send(&vs[kend-1][0][0],scount,MPI_DOUBLE,rankp1,tright,MPI_COMM_WORLD);
        } else {
            MPI_Recv(&vs[kstart-1][0][0],rcount,MPI_DOUBLE,rankm1,tright,MPI_COMM_WORLD,&status);
            if(rankp1 < size) {
                MPI_Send(&vs[kend-1][0][0],scount,MPI_DOUBLE,rankp1,tright,MPI_COMM_WORLD);
            }
        }

        tleft = 5;
        if(rank == size-1) {
            MPI_Send(&vs[kstart][0][0],scount,MPI_DOUBLE,rankm1,tleft,MPI_COMM_WORLD);
        } else {
            MPI_Recv(&vs[kend][0][0],rcount,MPI_DOUBLE,rankp1,tleft,MPI_COMM_WORLD,&status);
            if(rankm1 >= 0) {
                MPI_Send(&vs[kstart][0][0],scount,MPI_DOUBLE,rankm1,tleft,MPI_COMM_WORLD);
            }
        }

        // ws boundary exchange of width k=1
        tright = 6;
        if(rank == 0) {
            MPI_Send(&ws[kend-1][0][0],scount,MPI_DOUBLE,rankp1,tright,MPI_COMM_WORLD);
        } else {
            MPI_Recv(&ws[kstart-1][0][0],rcount,MPI_DOUBLE,rankm1,tright,MPI_COMM_WORLD,&status);
            if(rankp1 < size) {
                MPI_Send(&ws[kend-1][0][0],scount,MPI_DOUBLE,rankp1,tright,MPI_COMM_WORLD);
            }
        }

        tleft = 7;
        if(rank == size-1) {
            MPI_Send(&ws[kstart][0][0],scount,MPI_DOUBLE,rankm1,tleft,MPI_COMM_WORLD);
        } else {
            MPI_Recv(&ws[kend][0][0],rcount,MPI_DOUBLE,rankp1,tleft,MPI_COMM_WORLD,&status);
            if(rankm1 >= 0) {
                MPI_Send(&ws[kstart][0][0],scount,MPI_DOUBLE,rankm1,tleft,MPI_COMM_WORLD);
            }
        }
    }

    // square boundary exchange of width k=1
    tright = 8;
    if(rank == 0) {
        MPI_Send(&square[kend-1][0][0],scount,MPI_DOUBLE,rankp1,tright,MPI_COMM_WORLD);
    } else {
        MPI_Recv(&square[kstart-1][0][0],rcount,MPI_DOUBLE,rankm1,tright,MPI_COMM_WORLD,&status);
        if(rankp1 < size) {
            MPI_Send(&square[kend-1][0][0],scount,MPI_DOUBLE,rankp1,tright,MPI_COMM_WORLD);
        }
    }

    tleft = 9;
    if(rank == size-1) {
        MPI_Send(&square[kstart][0][0],scount,MPI_DOUBLE,rankm1,tleft,MPI_COMM_WORLD);
    } else {
        MPI_Recv(&square[kend][0][0],rcount,MPI_DOUBLE,rankp1,tleft,MPI_COMM_WORLD,&status);
        if(rankm1 >= 0) {
            MPI_Send(&square[kstart][0][0],scount,MPI_DOUBLE,rankm1,tleft,MPI_COMM_WORLD);
        }
    }


    // qs boundary exchange of width k=1
    tright = 10;
    if(rank == 0) {
        MPI_Send(&qs[kend-1][0][0],scount,MPI_DOUBLE,rankp1,tright,MPI_COMM_WORLD);
    } else {
        MPI_Recv(&qs[kstart-1][0][0],rcount,MPI_DOUBLE,rankm1,tright,MPI_COMM_WORLD,&status);
        if(rankp1 < size) {
            MPI_Send(&qs[kend-1][0][0],scount,MPI_DOUBLE,rankp1,tright,MPI_COMM_WORLD);
        }
    }

    tleft = 11;
    if(rank == size-1) {
        MPI_Send(&qs[kstart][0][0],scount,MPI_DOUBLE,rankm1,tleft,MPI_COMM_WORLD);
    } else {
        MPI_Recv(&qs[kend][0][0],rcount,MPI_DOUBLE,rankp1,tleft,MPI_COMM_WORLD,&status);
        if(rankm1 >= 0) {
            MPI_Send(&qs[kstart][0][0],scount,MPI_DOUBLE,rankm1,tleft,MPI_COMM_WORLD);
        }
    }

    disps[0] = 0;
    rcounts[0] = kdeltas[0]*(JMAXP+1)*(IMAXP+1);
    for(p=1; p<size; p++) {
        rcounts[p] = kdeltas[p]*(JMAXP+1)*(IMAXP+1);
        disps[p] = disps[p-1] + rcounts[p-1];
    }

    // Removed during transformation to point-to-point
    // errcode = MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,&rho_i[0][0][0],rcounts,disps,MPI_DOUBLE,MPI_COMM_WORLD);
    // errcode = MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,&us[0][0][0],rcounts,disps,MPI_DOUBLE,MPI_COMM_WORLD);
    // errcode = MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,&vs[0][0][0],rcounts,disps,MPI_DOUBLE,MPI_COMM_WORLD);
    // errcode = MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,&ws[0][0][0],rcounts,disps,MPI_DOUBLE,MPI_COMM_WORLD);

    errcode = MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,&square[0][0][0],rcounts,disps,MPI_DOUBLE,MPI_COMM_WORLD);
    errcode = MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,&qs[0][0][0],rcounts,disps,MPI_DOUBLE,MPI_COMM_WORLD);

    //---------------------------------------------------------------------
    // copy the exact forcing term to the right hand side;  because 
    // this forcing term is known, we can store it on the whole grid
    // including the boundary                   
    //---------------------------------------------------------------------
    // #pragma omp for schedule(static)
    // for (k = 0; k <= grid_points[2]-1; k++) 
    for (k = kstart; k < kend; k++) {
        for (j = 0; j <= grid_points[1]-1; j++) {
            for (i = 0; i <= grid_points[0]-1; i++) {
                for (m = 0; m < 5; m++) {
                    rhs[k][j][i][m] = forcing[k][j][i][m];
                }
            }
        }
    }

    // #pragma omp master
    if (rank == root) {
        if (timeron) timer_start(t_rhsx);
    }
    //---------------------------------------------------------------------
    // compute xi-direction fluxes 
    //---------------------------------------------------------------------
    // for (k = 1; k <= grid_points[2]-2; k++)
    for (k = kstartp1; k < kendm1; k++) {
        for (j = 1; j <= grid_points[1]-2; j++) {
            for (i = 1; i <= grid_points[0]-2; i++) {
                uijk = us[k][j][i];
                up1  = us[k][j][i+1];
                um1  = us[k][j][i-1];

                rhs[k][j][i][0] = rhs[k][j][i][0] + dx1tx1 * 
                    (u[k][j][i+1][0] - 2.0*u[k][j][i][0] + 
                     u[k][j][i-1][0]) -
                    tx2 * (u[k][j][i+1][1] - u[k][j][i-1][1]);

                rhs[k][j][i][1] = rhs[k][j][i][1] + dx2tx1 * 
                    (u[k][j][i+1][1] - 2.0*u[k][j][i][1] + 
                     u[k][j][i-1][1]) +
                    xxcon2*con43 * (up1 - 2.0*uijk + um1) -
                    tx2 * (u[k][j][i+1][1]*up1 - 
                            u[k][j][i-1][1]*um1 +
                            (u[k][j][i+1][4]- square[k][j][i+1]-
                             u[k][j][i-1][4]+ square[k][j][i-1])*
                            c2);

                rhs[k][j][i][2] = rhs[k][j][i][2] + dx3tx1 * 
                    (u[k][j][i+1][2] - 2.0*u[k][j][i][2] +
                     u[k][j][i-1][2]) +
                    xxcon2 * (vs[k][j][i+1] - 2.0*vs[k][j][i] +
                            vs[k][j][i-1]) -
                    tx2 * (u[k][j][i+1][2]*up1 - 
                            u[k][j][i-1][2]*um1);

                rhs[k][j][i][3] = rhs[k][j][i][3] + dx4tx1 * 
                    (u[k][j][i+1][3] - 2.0*u[k][j][i][3] +
                     u[k][j][i-1][3]) +
                    xxcon2 * (ws[k][j][i+1] - 2.0*ws[k][j][i] +
                            ws[k][j][i-1]) -
                    tx2 * (u[k][j][i+1][3]*up1 - 
                            u[k][j][i-1][3]*um1);

                rhs[k][j][i][4] = rhs[k][j][i][4] + dx5tx1 * 
                    (u[k][j][i+1][4] - 2.0*u[k][j][i][4] +
                     u[k][j][i-1][4]) +
                    xxcon3 * (qs[k][j][i+1] - 2.0*qs[k][j][i] +
                            qs[k][j][i-1]) +
                    xxcon4 * (up1*up1 -       2.0*uijk*uijk + 
                            um1*um1) +
                    xxcon5 * (u[k][j][i+1][4]*rho_i[k][j][i+1] - 
                            2.0*u[k][j][i][4]*rho_i[k][j][i] +
                            u[k][j][i-1][4]*rho_i[k][j][i-1]) -
                    tx2 * ( (c1*u[k][j][i+1][4] - 
                                c2*square[k][j][i+1])*up1 -
                            (c1*u[k][j][i-1][4] - 
                             c2*square[k][j][i-1])*um1 );
            }
        }

        //---------------------------------------------------------------------
        // add fourth order xi-direction dissipation               
        //---------------------------------------------------------------------
        for (j = 1; j <= grid_points[1]-2; j++) {
            i = 1;
            for (m = 0; m < 5; m++) {
                rhs[k][j][i][m] = rhs[k][j][i][m]- dssp * 
                    ( 5.0*u[k][j][i][m] - 4.0*u[k][j][i+1][m] +
                      u[k][j][i+2][m]);
            }

            i = 2;
            for (m = 0; m < 5; m++) {
                rhs[k][j][i][m] = rhs[k][j][i][m] - dssp * 
                    (-4.0*u[k][j][i-1][m] + 6.0*u[k][j][i][m] -
                     4.0*u[k][j][i+1][m] + u[k][j][i+2][m]);
            }
        }

        for (j = 1; j <= grid_points[1]-2; j++) {
            for (i = 3; i <= grid_points[0]-4; i++) {
                for (m = 0; m < 5; m++) {
                    rhs[k][j][i][m] = rhs[k][j][i][m] - dssp * 
                        (  u[k][j][i-2][m] - 4.0*u[k][j][i-1][m] + 
                           6.0*u[k][j][i][m] - 4.0*u[k][j][i+1][m] + 
                           u[k][j][i+2][m] );
                }
            }
        }

        for (j = 1; j <= grid_points[1]-2; j++) {
            i = grid_points[0]-3;
            for (m = 0; m < 5; m++) {
                rhs[k][j][i][m] = rhs[k][j][i][m] - dssp *
                    ( u[k][j][i-2][m] - 4.0*u[k][j][i-1][m] + 
                      6.0*u[k][j][i][m] - 4.0*u[k][j][i+1][m] );
            }

            i = grid_points[0]-2;
            for (m = 0; m < 5; m++) {
                rhs[k][j][i][m] = rhs[k][j][i][m] - dssp *
                    ( u[k][j][i-2][m] - 4.*u[k][j][i-1][m] +
                      5.*u[k][j][i][m] );
            }
        }
    }
    // #pragma omp master
    if(rank == root) {
        if (timeron) timer_stop(t_rhsx);
        if (timeron) timer_start(t_rhsy);
    }

    //---------------------------------------------------------------------
    // compute eta-direction fluxes 
    //---------------------------------------------------------------------
    // #pragma omp for schedule(static)
    // for (k = 1; k <= grid_points[2]-2; k++)
    for (k = kstartp1; k < kendm1; k++) {
        for (j = 1; j <= grid_points[1]-2; j++) {
            for (i = 1; i <= grid_points[0]-2; i++) {
                vijk = vs[k][j][i];
                vp1  = vs[k][j+1][i];
                vm1  = vs[k][j-1][i];
                rhs[k][j][i][0] = rhs[k][j][i][0] + dy1ty1 * 
                    (u[k][j+1][i][0] - 2.0*u[k][j][i][0] + 
                     u[k][j-1][i][0]) -
                    ty2 * (u[k][j+1][i][2] - u[k][j-1][i][2]);
                rhs[k][j][i][1] = rhs[k][j][i][1] + dy2ty1 * 
                    (u[k][j+1][i][1] - 2.0*u[k][j][i][1] + 
                     u[k][j-1][i][1]) +
                    yycon2 * (us[k][j+1][i] - 2.0*us[k][j][i] + 
                            us[k][j-1][i]) -
                    ty2 * (u[k][j+1][i][1]*vp1 - 
                            u[k][j-1][i][1]*vm1);
                rhs[k][j][i][2] = rhs[k][j][i][2] + dy3ty1 * 
                    (u[k][j+1][i][2] - 2.0*u[k][j][i][2] + 
                     u[k][j-1][i][2]) +
                    yycon2*con43 * (vp1 - 2.0*vijk + vm1) -
                    ty2 * (u[k][j+1][i][2]*vp1 - 
                            u[k][j-1][i][2]*vm1 +
                            (u[k][j+1][i][4] - square[k][j+1][i] - 
                             u[k][j-1][i][4] + square[k][j-1][i])
                            *c2);
                rhs[k][j][i][3] = rhs[k][j][i][3] + dy4ty1 * 
                    (u[k][j+1][i][3] - 2.0*u[k][j][i][3] + 
                     u[k][j-1][i][3]) +
                    yycon2 * (ws[k][j+1][i] - 2.0*ws[k][j][i] + 
                            ws[k][j-1][i]) -
                    ty2 * (u[k][j+1][i][3]*vp1 - 
                            u[k][j-1][i][3]*vm1);
                rhs[k][j][i][4] = rhs[k][j][i][4] + dy5ty1 * 
                    (u[k][j+1][i][4] - 2.0*u[k][j][i][4] + 
                     u[k][j-1][i][4]) +
                    yycon3 * (qs[k][j+1][i] - 2.0*qs[k][j][i] + 
                            qs[k][j-1][i]) +
                    yycon4 * (vp1*vp1       - 2.0*vijk*vijk + 
                            vm1*vm1) +
                    yycon5 * (u[k][j+1][i][4]*rho_i[k][j+1][i] - 
                            2.0*u[k][j][i][4]*rho_i[k][j][i] +
                            u[k][j-1][i][4]*rho_i[k][j-1][i]) -
                    ty2 * ((c1*u[k][j+1][i][4] - 
                                c2*square[k][j+1][i]) * vp1 -
                            (c1*u[k][j-1][i][4] - 
                             c2*square[k][j-1][i]) * vm1);
            }
        }

        //---------------------------------------------------------------------
        // add fourth order eta-direction dissipation         
        //---------------------------------------------------------------------
        j = 1;
        for (i = 1; i <= grid_points[0]-2; i++) {
            for (m = 0; m < 5; m++) {
                rhs[k][j][i][m] = rhs[k][j][i][m]- dssp * 
                    ( 5.0*u[k][j][i][m] - 4.0*u[k][j+1][i][m] +
                      u[k][j+2][i][m]);
            }
        }

        j = 2;
        for (i = 1; i <= grid_points[0]-2; i++) {
            for (m = 0; m < 5; m++) {
                rhs[k][j][i][m] = rhs[k][j][i][m] - dssp * 
                    (-4.0*u[k][j-1][i][m] + 6.0*u[k][j][i][m] -
                     4.0*u[k][j+1][i][m] + u[k][j+2][i][m]);
            }
        }

        for (j = 3; j <= grid_points[1]-4; j++) {
            for (i = 1; i <= grid_points[0]-2; i++) {
                for (m = 0; m < 5; m++) {
                    rhs[k][j][i][m] = rhs[k][j][i][m] - dssp * 
                        (  u[k][j-2][i][m] - 4.0*u[k][j-1][i][m] + 
                           6.0*u[k][j][i][m] - 4.0*u[k][j+1][i][m] + 
                           u[k][j+2][i][m] );
                }
            }
        }

        j = grid_points[1]-3;
        for (i = 1; i <= grid_points[0]-2; i++) {
            for (m = 0; m < 5; m++) {
                rhs[k][j][i][m] = rhs[k][j][i][m] - dssp *
                    ( u[k][j-2][i][m] - 4.0*u[k][j-1][i][m] + 
                      6.0*u[k][j][i][m] - 4.0*u[k][j+1][i][m] );
            }
        }

        j = grid_points[1]-2;
        for (i = 1; i <= grid_points[0]-2; i++) {
            for (m = 0; m < 5; m++) {
                rhs[k][j][i][m] = rhs[k][j][i][m] - dssp *
                    ( u[k][j-2][i][m] - 4.*u[k][j-1][i][m] +
                      5.*u[k][j][i][m] );
            }
        }
    }
    // #pragma omp master
    if(rank == root) {
        if (timeron) timer_stop(t_rhsy);
        if (timeron) timer_start(t_rhsz);
    }
    //---------------------------------------------------------------------
    // compute zeta-direction fluxes 
    //---------------------------------------------------------------------
    // #pragma omp for schedule(static)
    // for (k = 1; k <= grid_points[2]-2; k++)
    for (k = kstartp1; k < kendm1; k++) {
        for (j = 1; j <= grid_points[1]-2; j++) {
            for (i = 1; i <= grid_points[0]-2; i++) {
                wijk = ws[k][j][i];
                wp1  = ws[k+1][j][i];
                wm1  = ws[k-1][j][i];

                rhs[k][j][i][0] = rhs[k][j][i][0] + dz1tz1 * 
                    (u[k+1][j][i][0] - 2.0*u[k][j][i][0] + 
                     u[k-1][j][i][0]) -
                    tz2 * (u[k+1][j][i][3] - u[k-1][j][i][3]);
                rhs[k][j][i][1] = rhs[k][j][i][1] + dz2tz1 * 
                    (u[k+1][j][i][1] - 2.0*u[k][j][i][1] + 
                     u[k-1][j][i][1]) +
                    zzcon2 * (us[k+1][j][i] - 2.0*us[k][j][i] + 
                            us[k-1][j][i]) -
                    tz2 * (u[k+1][j][i][1]*wp1 - 
                            u[k-1][j][i][1]*wm1);
                rhs[k][j][i][2] = rhs[k][j][i][2] + dz3tz1 * 
                    (u[k+1][j][i][2] - 2.0*u[k][j][i][2] + 
                     u[k-1][j][i][2]) +
                    zzcon2 * (vs[k+1][j][i] - 2.0*vs[k][j][i] + 
                            vs[k-1][j][i]) -
                    tz2 * (u[k+1][j][i][2]*wp1 - 
                            u[k-1][j][i][2]*wm1);
                rhs[k][j][i][3] = rhs[k][j][i][3] + dz4tz1 * 
                    (u[k+1][j][i][3] - 2.0*u[k][j][i][3] + 
                     u[k-1][j][i][3]) +
                    zzcon2*con43 * (wp1 - 2.0*wijk + wm1) -
                    tz2 * (u[k+1][j][i][3]*wp1 - 
                            u[k-1][j][i][3]*wm1 +
                            (u[k+1][j][i][4] - square[k+1][j][i] - 
                             u[k-1][j][i][4] + square[k-1][j][i])
                            *c2);
                rhs[k][j][i][4] = rhs[k][j][i][4] + dz5tz1 * 
                    (u[k+1][j][i][4] - 2.0*u[k][j][i][4] + 
                     u[k-1][j][i][4]) +
                    zzcon3 * (qs[k+1][j][i] - 2.0*qs[k][j][i] + 
                            qs[k-1][j][i]) +
                    zzcon4 * (wp1*wp1 - 2.0*wijk*wijk + 
                            wm1*wm1) +
                    zzcon5 * (u[k+1][j][i][4]*rho_i[k+1][j][i] - 
                            2.0*u[k][j][i][4]*rho_i[k][j][i] +
                            u[k-1][j][i][4]*rho_i[k-1][j][i]) -
                    tz2 * ( (c1*u[k+1][j][i][4] - 
                                c2*square[k+1][j][i])*wp1 -
                            (c1*u[k-1][j][i][4] - 
                             c2*square[k-1][j][i])*wm1);
            }
        }
    }

    //---------------------------------------------------------------------
    // add fourth order zeta-direction dissipation                
    //---------------------------------------------------------------------
    k = 1;
    if(rank == (int) k/kdelta) {
        // #pragma omp for schedule(static) nowait
        for (j = 1; j <= grid_points[1]-2; j++) {
            for (i = 1; i <= grid_points[0]-2; i++) {
                for (m = 0; m < 5; m++) {
                    rhs[k][j][i][m] = rhs[k][j][i][m]- dssp * 
                        ( 5.0*u[k][j][i][m] - 4.0*u[k+1][j][i][m] +
                          u[k+2][j][i][m]);
                }
            }
        }
    }

    k = 2;
    if(rank == (int) k/kdelta) {
        // #pragma omp for schedule(static) nowait
        for (j = 1; j <= grid_points[1]-2; j++) {
            for (i = 1; i <= grid_points[0]-2; i++) {
                for (m = 0; m < 5; m++) {
                    rhs[k][j][i][m] = rhs[k][j][i][m] - dssp * 
                        (-4.0*u[k-1][j][i][m] + 6.0*u[k][j][i][m] -
                         4.0*u[k+1][j][i][m] + u[k+2][j][i][m]);
                }
            }
        }
    }

    // #pragma omp for schedule(static) nowait
    //
    // Since we don't know who gets k=3 or k=KMAX-4, we need to 
    // recheck the k boundaries

    kstartbdy = kstart;
    kendbdy = kend;
    if(kstartbdy < 3) {
        kstartbdy = 3;
    }
    if(kendbdy > grid_points[2] - 3){
        kendbdy = grid_points[2] - 3;
    }


    // for (k = 3; k <= grid_points[2]-4; k++)
    for (k = kstartbdy; k < kendbdy; k++) {
        for (j = 1; j <= grid_points[1]-2; j++) {
            for (i = 1; i <= grid_points[0]-2; i++) {
                for (m = 0; m < 5; m++) {
                    rhs[k][j][i][m] = rhs[k][j][i][m] - dssp * 
                        (  u[k-2][j][i][m] - 4.0*u[k-1][j][i][m] + 
                           6.0*u[k][j][i][m] - 4.0*u[k+1][j][i][m] + 
                           u[k+2][j][i][m] );
                }
            }
        }
    }

    k = grid_points[2]-3;
    if(rank == (int) k/kdelta) {
        // #pragma omp for schedule(static) nowait
        for (j = 1; j <= grid_points[1]-2; j++) {
            for (i = 1; i <= grid_points[0]-2; i++) {
                for (m = 0; m < 5; m++) {
                    rhs[k][j][i][m] = rhs[k][j][i][m] - dssp *
                        ( u[k-2][j][i][m] - 4.0*u[k-1][j][i][m] + 
                          6.0*u[k][j][i][m] - 4.0*u[k+1][j][i][m] );
                }
            }
        }
    }

    k = grid_points[2]-2;
    if(rank == (int) k/kdelta) {
        // #pragma omp for schedule(static)
        for (j = 1; j <= grid_points[1]-2; j++) {
            for (i = 1; i <= grid_points[0]-2; i++) {
                for (m = 0; m < 5; m++) {
                    rhs[k][j][i][m] = rhs[k][j][i][m] - dssp *
                        ( u[k-2][j][i][m] - 4.*u[k-1][j][i][m] +
                          5.*u[k][j][i][m] );
                }
            }
        }
    }

    // #pragma omp master
    if(rank == root) {
        if (timeron) timer_stop(t_rhsz);
    }

    // #pragma omp for schedule(static) nowait
    for (k = kstartp1; k < kendm1; k++) {
        for (j = 1; j <= grid_points[1]-2; j++) {
            for (i = 1; i <= grid_points[0]-2; i++) {
                for (m = 0; m < 5; m++) {
                    rhs[k][j][i][m] = rhs[k][j][i][m] * dt;
                }
            }
        }
    }
    // close curly bracket end parallel
    if(rank == root) {
        if (timeron) timer_stop(t_rhs);
    }

}
