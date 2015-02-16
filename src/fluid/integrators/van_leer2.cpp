//======================================================================================
// Athena++ astrophysical MHD code
// Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
//
// This program is free software: you can redistribute and/or modify it under the terms
// of the GNU General Public License (GPL) as published by the Free Software Foundation,
// either version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
// PARTICULAR PURPOSE.  See the GNU General Public License for more details.
//
// You should have received a copy of GNU GPL in the file LICENSE included in the code
// distribution.  If not see <http://www.gnu.org/licenses/>.
//======================================================================================

// C++ headers
#include <algorithm>   // min,max

// Primary header
#include "fluid_integrator.hpp"

// Athena headers
#include "../../athena.hpp"                  // enums, macros, Real
#include "../../athena_arrays.hpp"           // AthenaArray
#include "../../coordinates/coordinates.hpp" // Coordinates
#include "../fluid.hpp"                      // Fluid
#include "../../field/field.hpp"             // Fields
#include "../../mesh.hpp"                    // MeshBlock
#include "../srcterms/srcterms.hpp"          // PhysicalSourceTerms()

// OpenMP header
#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

//======================================================================================
//! \file van_leer2.cpp
//  \brief van-Leer (MUSCL-Hancock) second-order integrator
//======================================================================================

//--------------------------------------------------------------------------------------
//! \fn  void FluidIntegrator::Predict
//  \brief predictor step for 2nd order VL integrator

void FluidIntegrator::OneStep(MeshBlock *pmb,AthenaArray<Real> &u, AthenaArray<Real> &w,
 InterfaceField &b, AthenaArray<Real> &bcc, const int step)
{
  int tid=0;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
  int max_nthreads = pmb->pmy_mesh->nthreads_mesh;
 
  AthenaArray<Real> b1,b2,b3,ei_x1f,ei_x2f,ei_x3f,w_x1f,w_x2f,w_x3f;
  b1.InitWithShallowCopy(b.x1f);
  b2.InitWithShallowCopy(b.x2f);
  b3.InitWithShallowCopy(b.x3f);
  ei_x1f.InitWithShallowCopy(pmb->pfield->ei.x1f);
  ei_x2f.InitWithShallowCopy(pmb->pfield->ei.x2f);
  ei_x3f.InitWithShallowCopy(pmb->pfield->ei.x3f);
  w_x1f.InitWithShallowCopy(pmb->pfield->wght.x1f);
  w_x2f.InitWithShallowCopy(pmb->pfield->wght.x2f);
  w_x3f.InitWithShallowCopy(pmb->pfield->wght.x3f);

  Real dt;
  if (step == 1) {
    dt = 0.5*(pmb->pmy_mesh->dt);
  } else {
    dt = (pmb->pmy_mesh->dt);
  }

#pragma omp parallel default(shared) private(tid) num_threads(max_nthreads)
{
#ifdef OPENMP_PARALLEL
  tid=omp_get_thread_num();
#endif
  AthenaArray<Real> wl, wr, flx, flx_jm1, flx_km1, area, area_m1, vol;
  wl.InitWithShallowSlice(wl_,3,tid,1);
  wr.InitWithShallowSlice(wr_,3,tid,1);
  flx.InitWithShallowSlice(flx_,3,tid,1);
  flx_jm1.InitWithShallowSlice(jflx_,3,tid,1);
  flx_km1.InitWithShallowSlice(kflx_,4,tid,1);
  area.InitWithShallowSlice(face_area_,2,tid,1);
  area_m1.InitWithShallowSlice(face_area_m1_,2,tid,1);
  vol.InitWithShallowSlice(cell_volume_,2,tid,1);

//--------------------------------------------------------------------------------------
// i-direction

  for (int k=ks; k<=ke; ++k){ 
#pragma omp for schedule(static)
    for (int j=js; j<=je; ++j){

      // reconstruct L/R states
      if (step == 1) {
        DonorCellX1(k,j,w,bcc,wl,wr);
      } else {
        PiecewiseLinearX1(k,j,w,bcc,wl,wr);
      }

      // compute fluxes
      RiemannSolver(k,j,is,ie+1,IVX,b1,wl,wr,flx);

      // update conserved fluid variables
      pmb->pcoord->Face1Area(k,j,is,ie+1,area);
      pmb->pcoord->CellVolume(k,j,is,ie,vol);
      for (int n=0; n<NFLUID; ++n){
#pragma simd
        for (int i=is; i<=ie; ++i){
          Real& flx_i   = flx(n,i  );
          Real& flx_ip1 = flx(n,i+1);
          Real& area_i   = area(i);
          Real& area_ip1 = area(i+1);
          Real& dvol = vol(i);
          u(n,k,j,i) -= dt*(area_ip1*flx_ip1 - area_i*flx_i)/dvol;
        }
      }

      // add coordinate (geometric) source terms
      pmb->pcoord->CoordSrcTermsX1(k,j,dt,flx,w,bcc,u);

      // store electric fields, compute weights for GS07 CT algorithm
      if (MAGNETIC_FIELDS_ENABLED) {
#pragma simd
        for (int i=is; i<=ie+1; ++i){
          ei_x1f(X1E3,k,j,i) = -flx(IBY,i); // flx(IBY) = (v1*b2 - v2*b1) = -EMFZ
          ei_x1f(X1E2,k,j,i) =  flx(IBZ,i); // flx(IBZ) = (v1*b3 - v3*b1) =  EMFY
          const Real& dx = pmb->pcoord->CenterWidth1(k,j,i);
          Real v_over_c = (1024)*dt*flx(IDN,i)/(dx*(wl(IDN,i) + wr(IDN,i)));
          Real tmp_min = std::min(0.5,v_over_c);
          w_x1f(k,j,i) = 0.5 + std::max(-0.5,tmp_min);
        }
      }

    }
  }

//--------------------------------------------------------------------------------------
// j-direction

  if (pmb->block_size.nx2 > 1) {

    for (int k=ks; k<=ke; ++k){

      // reconstruct L/R states at j=js
      if (step == 1) {
        DonorCellX2(k,js,w,bcc,wl,wr);
      } else {
        PiecewiseLinearX2(k,js,w,bcc,wl,wr);
      }

      // compute and store fluxes at j=js
      RiemannSolver(k,js,is,ie,IVY,b2,wl,wr,flx_jm1); 
      
      // store electric fields, compute weights for GS07 CT algorithm at j=js
      if (MAGNETIC_FIELDS_ENABLED) {
#pragma simd
        for (int i=is; i<=ie; ++i){
          ei_x2f(X2E1,k,js,i) = -flx_jm1(IBY,i); // flx(IBY) = (v2*b3 - v3*b2) = -EMFX
          ei_x2f(X2E3,k,js,i) =  flx_jm1(IBZ,i); // flx(IBZ) = (v2*b1 - v1*b2) =  EMFZ
          const Real& dx = pmb->pcoord->CenterWidth2(k,js,i);
          Real v_over_c = (1024)*dt*flx_jm1(IDN,i)/(dx*(wl(IDN,i) + wr(IDN,i)));
          Real tmp_min = std::min(0.5,v_over_c);
          w_x2f(k,js,i) = 0.5 + std::max(-0.5,tmp_min);
        }
      }

#pragma omp for schedule(static)
      for (int j=js+1; j<=je+1; ++j){

        // reconstruct L/R states
        if (step == 1) {
          DonorCellX2(k,j,w,bcc,wl,wr);
        } else {
          PiecewiseLinearX2(k,j,w,bcc,wl,wr);
        }

        // compute fluxes
        RiemannSolver(k,j,is,ie,IVY,b2,wl,wr,flx); 

        // update conserved fluid variables
        pmb->pcoord->Face2Area(k,j-1,is,ie,area_m1);
        pmb->pcoord->Face2Area(k,j  ,is,ie,area   );
        pmb->pcoord->CellVolume(k,j-1,is,ie,vol);
        for (int n=0; n<NFLUID; ++n){
#pragma simd
          for (int i=is; i<=ie; ++i){
            Real& flx_ijm1 = flx_jm1(n,i);
            Real& flx_ij   = flx(n,i);
            Real& area_j   = area(i);
            Real& area_jm1 = area_m1(i);
            Real& dvol = vol(i);
            u(n,k,j-1,i) -= dt*(area_j*flx_ij - area_jm1*flx_ijm1)/dvol;
          }
        }

        // add coordinate (geometric) source terms
        pmb->pcoord->CoordSrcTermsX2(k,j,dt,flx,flx_jm1,w,bcc,u);

        // store electric fields, compute weights for GS07 CT algorithm
        if (MAGNETIC_FIELDS_ENABLED) {
#pragma simd
          for (int i=is; i<=ie; ++i){
            ei_x2f(X2E1,k,j,i) = -flx(IBY,i); // flx(IBY) = (v2*b3 - v3*b2) = -EMFX
            ei_x2f(X2E3,k,j,i) =  flx(IBZ,i); // flx(IBZ) = (v2*b1 - v1*b2) =  EMFZ
            const Real& dx = pmb->pcoord->CenterWidth2(k,j,i);
            Real v_over_c = (1024)*dt*flx(IDN,i)/(dx*(wl(IDN,i) + wr(IDN,i)));
            Real tmp_min = std::min(0.5,v_over_c);
            w_x2f(k,j,i) = 0.5 + std::max(-0.5,tmp_min);
          }
        }

        // store fluxes for j=j-1 in next iteration
        flx_jm1 = flx; 

      }
    }
  }

//--------------------------------------------------------------------------------------
// k-direction 

  if (pmb->block_size.nx3 > 1) {

    for (int j=js; j<=je; ++j){

      // reconstruct L/R states at k=ks
      if (step == 1) {
        DonorCellX3(ks,j,w,bcc,wl,wr);
      } else {
        PiecewiseLinearX3(ks,j,w,bcc,wl,wr);
      }

      // compute and store fluxes at k=ks
      RiemannSolver(ks,j,is,ie,IVZ,b3,wl,wr,flx);

      // store electric fields, compute weights for GS07 CT algorithm at k=ks
      if (MAGNETIC_FIELDS_ENABLED) {
#pragma simd
        for (int i=is; i<=ie; ++i){
          ei_x3f(X3E2,ks,j,i) = -flx(IBY,i); // flx(IBY) = (v3*b1 - v1*b3) = -EMFY
          ei_x3f(X3E1,ks,j,i) =  flx(IBZ,i); // flx(IBZ) = (v3*b2 - v2*b3) =  EMFX
          const Real& dx = pmb->pcoord->CenterWidth3(ks,j,i);
          Real v_over_c = (1024)*dt*flx(IDN,i)/(dx*(wl(IDN,i) + wr(IDN,i)));
          Real tmp_min = std::min(0.5,v_over_c);
          w_x3f(ks,j,i) = 0.5 + std::max(-0.5,tmp_min);
        }
      }

      // store fluxes at k=ks over all i,j
      for (int n=0; n<NFLUID; ++n){
#pragma simd
        for (int i=is; i<=ie; ++i){
          flx_km1(n,j,i) = flx(n,i);
        }
      }
    }

    for (int k=ks+1; k<=ke+1; ++k){
#pragma omp for schedule(static)
      for (int j=js; j<=je; ++j){

        // reconstruct L/R states
        if (step == 1) {
          DonorCellX3(k,j,w,bcc,wl,wr);
        } else {
          PiecewiseLinearX3(k,j,w,bcc,wl,wr);
        }

        // compute fluxes
        RiemannSolver(k,j,is,ie,IVZ,b3,wl,wr,flx);

        // update conserved fluid variables
        pmb->pcoord->Face3Area(k-1,j,is,ie,area_m1);
        pmb->pcoord->Face3Area(k  ,j,is,ie,area   );
        pmb->pcoord->CellVolume(k-1,j,is,ie,vol);
        for (int n=0; n<NFLUID; ++n){
#pragma simd
          for (int i=is; i<=ie; ++i){
            Real& flx_ikm1 = flx_km1(n,j,i);
            Real& flx_ik   = flx(n,i);
            Real& area_km1 = area_m1(i);
            Real& area_k   = area(i);
            Real& dvol = vol(i);
            u(n,k-1,j,i) -= dt*(area_k*flx_ik - area_km1*flx_ikm1)/dvol;
          }
        }

        // add coordinate (geometric) source terms
        pmb->pcoord->CoordSrcTermsX3(k,j,dt,flx,flx_km1,w,bcc,u);

        // store electric fields, compute weights for GS07 CT algorithm
        if (MAGNETIC_FIELDS_ENABLED) {
#pragma simd
          for (int i=is; i<=ie; ++i){
            ei_x3f(X3E2,k,j,i) = -flx(IBY,i); // flx(IBY) = (v3*b1 - v1*b3) = -EMFY
            ei_x3f(X3E1,k,j,i) =  flx(IBZ,i); // flx(IBZ) = (v3*b2 - v2*b3) =  EMFX
            const Real& dx = pmb->pcoord->CenterWidth3(k,j,i);
            Real v_over_c = (1024)*dt*flx(IDN,i)/(dx*(wl(IDN,i) + wr(IDN,i)));
            Real tmp_min = std::min(0.5,v_over_c);
            w_x3f(k,j,i) = 0.5 + std::max(-0.5,tmp_min);
          }
        }

        // store fluxes for k=k-1 in next iteration
        for (int n=0; n<NFLUID; ++n){
#pragma simd
          for (int i=is; i<=ie; ++i){
            flx_km1(n,j,i) = flx(n,i);
          }
        }

      }
    }
  }

} // end of omp parallel region

//--------------------------------------------------------------------------------------
//  Add source terms for half a timestep

//  pmb->pcoord->CoordinateSourceTerms(dt,w,u);
  pmb->pfluid->pf_srcterms->PhysicalSourceTerms(pmb->pmy_mesh->time,dt,w,u);
  if (pmb->pfluid->pf_srcterms->UserSourceTerm != NULL)
    pmb->pfluid->pf_srcterms->UserSourceTerm(pmb->pmy_mesh->time,dt,w,u);

  return;
}
