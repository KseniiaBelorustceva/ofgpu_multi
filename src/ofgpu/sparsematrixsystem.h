/***************************************************************************
    begin                : Fri Apr 1 2011
    copyright            : (C) 2011 Symscape
    website              : www.symscape.com
***************************************************************************/

#ifndef sparsematrixsystem_h
#define sparsematrixsystem_h

#pragma once

#include <cusp/ell_matrix.h>
#include <cusp/precond/ainv.h>
#include <cusp/precond/diagonal.h>
#include <cusp/precond/aggregation/smoothed_aggregation.h>

#include "ofgpu/index.h"
#include "ofgpu/real.h"
#include "ofgpu/solvertelemetry.h"
#include "ofgpu/sparsematrixargs.h"
#include "ofgpu/the.h"


namespace ofgpu
{
   typedef cusp::device_memory MemorySpace;
   typedef cusp::array1d<index, MemorySpace> IndexArray;
   typedef cusp::array1d<real, MemorySpace> RealArray;

   /** Sparse matrix (ELLPACK/ITPACK format) on device */
   typedef cusp::ell_matrix<index, real, MemorySpace> Matrix;


  /**@brief GPGPU sparse matrix system for linear solver
   *
   *@author Rich Smith
   */
  class SparseMatrixSystem
  {
  public: 
    /** Access as singleton the<SparseMatrixSystem>() */
    static SparseMatrixSystem & getSingleton();

    /** Set CUDA device */
    void setDevice(int const requestedDevice);

    /** Prints devices info */
    void printDevicesInfo();

     /** Set multiple CUDA devices */
    void setMultipleDevices();

    /** Setting requested cuda device */ 
    void* routine( int deviceID); 

    /** Provides a cache of previous call to avoid reallocation of device memory */
    template<class T_Solver>
    void solve(SparseMatrixArgs const & args);

  private:
    SparseMatrixSystem();
    ~SparseMatrixSystem();

    /** Initialize ell_matrix : ELLPACK/ITPACK matrix format indices (mesh) */
    void initialize(SparseMatrixArgs const & args);

    /** Assign matrix entries */
    void assignMatrix(SparseMatrixArgs const & args);

    /** Update prior to solve */
    void update(SparseMatrixArgs const & args);

  private:
    /** Sparse matrix */
    Matrix m_A;

    /** Lower LDU faces start */
    IndexArray m_lowFacesStart;

    /** Lower LDU faces start to actual id */
    IndexArray m_lowFace;

    /** Upper LDU faces start */
    IndexArray m_upFacesStart;

    /** Max number of non-zero entries per row in matrix */
    index m_maxEntriesPerRow;

    /** CUDA device id */
    int m_device;
  };


  inline
  static bool isSame(char const * const a, char const * const b)
  {
    return (0 == std::strcmp(a, b));
  }


  template<class T_Solver>
  inline
  void     
  SparseMatrixSystem::solve(SparseMatrixArgs const & args)
  {
    update(args);
 
    // allocate storage for solution (x) and right hand side (b)
    RealArray x(args.xSolution, args.xSolution + args.nCells);
    RealArray b(args.bSource, args.bSource + args.nCells);
    
    // set stopping criteria:
    SolverTelemetry monitor(args.maxIterations, args.relativeTolerance, args.absoluteTolerance);
    monitor.setNormScale(m_A, x, b);

    // solve the linear system A * x = b with preconditioner M
    if (isSame("diagonal", args.preconditionerName)) {
      cusp::precond::diagonal<real, MemorySpace> M(m_A);
      T_Solver::perform(m_A, x, b, monitor, M);      
    }
    else if (isSame("smoothed_aggregation", args.preconditionerName)) {
      cusp::precond::aggregation::smoothed_aggregation<index, real, MemorySpace> M(m_A);
      T_Solver::perform(m_A, x, b, monitor, M);      
    }
    else if (isSame("scaled_bridson_ainv", args.preconditionerName)) {
      cusp::precond::scaled_bridson_ainv<real, MemorySpace> M(m_A);
      T_Solver::perform(m_A, x, b, monitor, M);      
    }
    else if (isSame("bridson_ainv", args.preconditionerName)) {
      cusp::precond::bridson_ainv<real, MemorySpace> M(m_A);
      T_Solver::perform(m_A, x, b, monitor, M);      
    }
    else if (isSame("nonsym_bridson_ainv", args.preconditionerName)) {
      cusp::precond::nonsym_bridson_ainv<real, MemorySpace> M(m_A);
      T_Solver::perform(m_A, x, b, monitor, M);      
    }
    else {
     if (!isSame("no", args.preconditionerName)) {
      std::cout << "Unsupported gpu preconditioner: " << args.preconditionerName << ", using: no" << std::endl;
     }

      cusp::identity_operator<real, MemorySpace> M(m_A.num_rows, m_A.num_rows);
      T_Solver::perform(m_A, x, b, monitor, M);      
    }

    thrust::copy(x.begin(), x.end(), args.xSolution);

    args.initialResidual = monitor.initialNorm();
    args.finalResidual = monitor.currentNorm();
    args.iterationsPerformed = monitor.iterationCount();
    args.converged = monitor.converged();
  }
}

#endif
