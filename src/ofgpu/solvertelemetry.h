/***************************************************************************
    begin                : Mon Apr 4 2011
    copyright            : (C) 2011 Symscape
    website              : www.symscape.com
***************************************************************************/
/*
    This file is part of ofgpu.

    ofgpu is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ofgpu is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with ofgpu.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef solvertelemetry_h
#define solvertelemetry_h

#pragma once

#include <cusp/multiply.h>

#include "ofgpu/real.h"


namespace ofgpu
{
  /**@brief Monitor to match OF lduMatrix::solverPerformance
   *
   *@author Rich Smith
   */
  class SolverTelemetry
  {
    public:
      SolverTelemetry(size_t const iterationLimit = 500, 
		      real const relativeTolerance = 1e-5,
		      real const absoluteTolerance = 1e-10)
      : m_norm(std::numeric_limits<real>::max())
      , m_initialNorm(-1)
      , m_normScale(1)
      , m_small(1e-20)
      , m_relativeTol(relativeTolerance)
      , m_absoluteTol(absoluteTolerance)
      , m_iterationLimit(iterationLimit)
      , m_iterationCount(0)
    {}
   
    struct FabsFn
    {
      __host__ __device__
      real operator()(real const & x) const 
      { 
	      return fabs(x);
      }
    }; // struct FabsFn

    struct MagNormFn
    {
      enum {AX_i = 0, TMP_i, B_i};

      template <typename Tuple>
      __host__ __device__
      void operator() (Tuple t)
      {
	      // Mimics fabs(Ax - tmp) + fabs(b - tmp)
        thrust::get<TMP_i>(t) = 
	        fabs(thrust::get<AX_i>(t) - thrust::get<TMP_i>(t)) + 
	        fabs(thrust::get<B_i>(t) - thrust::get<TMP_i>(t));
      }
    }; // struct MagNormFn

    /** Set normScale according to OF definition */
    template <typename Matrix, typename Vector>
    void setNormScale(Matrix const & A, Vector const & x, Vector const & b)
    {
      index const N = x.size();

      real const xRef = thrust::reduce(x.begin(), x.end()) / N;

      Vector xRefVector(N, xRef);
      Vector tmp(N);
      Vector Ax(N);

      // Would prefer to use and assign to tmp instead of using xRefVector, but corrupts tmp
      cusp::multiply(A, xRefVector, tmp);
      cusp::multiply(A, x, Ax);

      // Mag norm per cell
      thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(Ax.begin(), tmp.begin(), b.begin())),
      		             thrust::make_zip_iterator(thrust::make_tuple(Ax.end(), tmp.end(), b.end())),
		                   MagNormFn());

      m_normScale = thrust::reduce(tmp.begin(), tmp.end()) + m_small;
    }

    /** Increment iteration count */
    void operator++(void) 
    {  
      ++m_iterationCount; 
    }

    /** Converged? */
    bool converged() const
    {
      bool const isConverged = ((m_norm < (m_relativeTol * m_initialNorm)) ||
				                        (m_norm < (m_absoluteTol * m_normScale)));
      return isConverged;
    }
 
    /** Capture initial residual */
    template<typename Vector>
    bool finished(const Vector& r)
    {
      FabsFn unary_op;
      thrust::plus<real> binary_op;
      real init = 0;

      // sum of magnitudes, not really a norm which would be sqrt(sum of squares).
      // Mimics OF definition
      m_norm = thrust::transform_reduce(r.begin(), r.end(), unary_op, init, binary_op);

      if (0. > m_initialNorm) {
	      m_initialNorm = m_norm;
      }

      bool const isFinished = (converged() ||
			                         (m_iterationCount >= m_iterationLimit));
      return isFinished;
    }

    /** Access initial residual norm */
    real initialNorm() const
    {
      return m_initialNorm / m_normScale;
    }

    /** Access current residual norm */
    real currentNorm() const 
    { 
      return m_norm / m_normScale; 
    }

    /** Access current iteration count */
    index iterationCount() const 
    { 
      return m_iterationCount; 
    }

  private:
    real m_norm;
    real m_initialNorm;
    real m_normScale;
    real m_small;

    real m_relativeTol;
    real m_absoluteTol;

    index m_iterationLimit;
    index m_iterationCount;
  };
}

#endif
