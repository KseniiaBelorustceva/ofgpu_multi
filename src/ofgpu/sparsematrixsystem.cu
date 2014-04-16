/***************************************************************************
    begin                : Fri Apr 1 2011
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

#include <iostream>

#include "ofgpu/sparsematrixsystem.h"


namespace ofgpu
{
  SparseMatrixSystem::SparseMatrixSystem()
    : m_maxEntriesPerRow(0)
    , m_device(-1)
  {}
  
    
  SparseMatrixSystem::~SparseMatrixSystem()
  {}


  SparseMatrixSystem &
  SparseMatrixSystem::getSingleton()
  {
    // Skip cleaning up to avoid conflicts on Windows
    static SparseMatrixSystem* singleton = new SparseMatrixSystem;
    return *singleton;
  }


/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 *
 * Applies to _ConvertSMVer2Cores only
 */
// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
    {
        { 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
        { 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
        { 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
        { 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
        { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
        { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
        { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
        { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
        {   -1, -1 }
    };

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one to run properly
	std::cout << "MapSMtoCores for SM " << major << "." << minor << " is undefined. Default to use " << nGpuArchCoresPerSM[7].Cores << " Cores/SM" << std::endl;

    return nGpuArchCoresPerSM[7].Cores;
}
// end of GPU Architecture definitions


  void
  SparseMatrixSystem::setDevice(int const requestedDevice)
  {
    std::cout << "ofgpu CUDA info begin" << std::endl;

    if (0 <= m_device) {
      std::cout << "Warning: CUDA device already set as: " << m_device << std::endl;
      return;
    }

	int deviceCount = 0, driverVersion = 0, runtimeVersion = 0;
	cudaGetDeviceCount(&deviceCount);

	std::cout << "Available CUDA devices" << std::endl;

	for (int i = 0; deviceCount > i; ++i) {
	  cudaSetDevice(i);
	  cudaDeviceProp prop;
	  cudaGetDeviceProperties(&prop, i);
      cudaDriverGetVersion(&driverVersion);
      cudaRuntimeGetVersion(&runtimeVersion);

	  std::cout << "  Device: " << i << std::endl
	            << "    Name: \"" << prop.name << "\"" << std::endl
				<< "    Driver Version: " << driverVersion/1000 << "." << (driverVersion%100)/10 << std::endl
				<< "    Runtime Version: " << runtimeVersion/1000 << "." << (runtimeVersion%100)/10 << std::endl
	            << "    Capability: " << prop.major << "." << prop.minor << std::endl
	            << "    Processors: " << prop.multiProcessorCount << std::endl
	            << "    Cores: " << prop.multiProcessorCount * _ConvertSMVer2Cores(prop.major, prop.minor) << std::endl
	            << "    Memory (MBytes): " << (float)prop.totalGlobalMem/1048576.0f << std::endl
				<< "    GPU Clock rate (GHz): " << prop.clockRate * 1e-6f << std::endl;
	}

    m_device = requestedDevice;
    cudaSetDevice(m_device);
    cudaGetDevice(&m_device);

    if (requestedDevice != m_device) {
      std::cout << "Warning: Could not find requested CUDA device = " << requestedDevice << std::endl;
    }

	std::cout << "Selected CUDA device: " << m_device << std::endl
	          << "ofgpu CUDA info end" << std::endl
              << std::endl;
  }


  struct RowEntryFn
  {
    enum {LOW_FACE_START_i = 0, LOW_FACE_START_i_plus_1, UP_FACE_START_i, UP_FACE_START_i_plus_1, ROW_ENTRY_i};

    template <typename Tuple>
    __host__ __device__
    void operator() (Tuple t)
    {
        // rowEntry[i] = lowFacesStart[i + 1] - lowFacesStart[i] + 1 + upFacesStart[i + 1] - upFacesStart[i]
        thrust::get<ROW_ENTRY_i>(t) = thrust::get<LOW_FACE_START_i_plus_1>(t) - thrust::get<LOW_FACE_START_i>(t) + 1 + 
	                              thrust::get<UP_FACE_START_i_plus_1>(t)  - thrust::get<UP_FACE_START_i>(t);
    }
  };


  struct AssignIndicesFn
  {
    index const indicesRows;
    index const * const lowFaceToCell; 
    index const * const upFaceToCell;
    
    // Can't access overloaded operators in thrust, so need to get raw data for A
    index* const column_indices;

    index const X;
      
    AssignIndicesFn(Matrix & A,
		    IndexArray const & lowFaceToCell, 
		    IndexArray const & upFaceToCell)
      : indicesRows(A.column_indices.num_rows)
      , lowFaceToCell(lowFaceToCell.data().get())
      , upFaceToCell(upFaceToCell.data().get())
      , column_indices(A.column_indices.values.data().get())
      , X(Matrix::invalid_index)
    {}


    __host__ __device__
    index colIndex(index const i, index const j) const
    {
      // Mimics cusp::detail::index_of<index>(i, j, A.column_indices.num_rows, A.column_indices.num_cols, cusp::column_major())
      return i + j * indicesRows;
    }


    __host__ __device__
    void setLower(index const i, index const j, index const f)
    {
      column_indices[colIndex(i,j)] = lowFaceToCell[f];
    }


    __host__ __device__
    void setDiagonal(index const i, index const j)
    {
      column_indices[colIndex(i, j)] = i;
    }


    __host__ __device__
    void setUpper(index const i, index const j, index const f)
    {
      column_indices[colIndex(i,j)] = upFaceToCell[f];
    }


    __host__ __device__
    void setPadding(index const i, index const j)
    {
      column_indices[colIndex(i,j)] = X;
    }
  }; // struct AssignIndicesFn


  struct AssignValuesFn
  {
    index const valuesRows;
    real  const * const lowCellValueFromFace;
    real  const * const diagCellValue;
    real  const * const upCellValueFromFace;

    // Can't access overloaded operators in thrust, so need to get raw data for A
    real* const values;
 

    AssignValuesFn(Matrix & A,
		   RealArray const & lowCellValueFromFace,
		   RealArray const & diagCellValue,
		   RealArray const & upCellValueFromFace)
      : valuesRows(A.values.num_rows)
      , lowCellValueFromFace(lowCellValueFromFace.data().get())
      , diagCellValue(diagCellValue.data().get())
      , upCellValueFromFace(upCellValueFromFace.data().get())
      , values(A.values.values.data().get())
    {}


    __host__ __device__
    index valueIndex(index const i, index const j) const
    {
      //  Mimics cusp::detail::index_of<index>(i, j, A.values.num_rows, A.values.num_cols, cusp::column_major())
      return i + j * valuesRows;
    }


    __host__ __device__
    void setLower(index const i, index const j, index const f)
    {
      values[valueIndex(i, j)] = lowCellValueFromFace[f];
    }


    __host__ __device__
    void setDiagonal(index const i, index const j)
    {
      values[valueIndex(i, j)] = diagCellValue[i];
    }


    __host__ __device__
    void setUpper(index const i, index const j, index const f)
    {
       values[valueIndex(i, j)] = upCellValueFromFace[f];
    }


    __host__ __device__
    void setPadding(index const i, index const j)
    {
       values[valueIndex(i, j)] = 0.;
    }
  }; // struct AssignValuesFn


  template<class T_Assigner>
  struct MatrixAssignFn
  {
    T_Assigner assigner;
    index const maxEntriesPerRow;
    index const * const lowFace;
      
    enum {LOW_FACE_START_i = 0, LOW_FACE_START_i_plus_1, UP_FACE_START_i, UP_FACE_START_i_plus_1, ROW_i};


    MatrixAssignFn(T_Assigner const & assigner,
		   index const maxEntriesPerRow,
		   IndexArray const & lowFace)
      : assigner(assigner)
      , maxEntriesPerRow(maxEntriesPerRow)
      , lowFace(lowFace.data().get())
    {}


    template <typename Tuple>
    __host__ __device__
    void operator() (Tuple t)
    {
      index const rowI = thrust::get<ROW_i>(t);
      index entryJ     = 0;
      index face;

      index fBegin = thrust::get<LOW_FACE_START_i>(t);
      index fEnd   = thrust::get<LOW_FACE_START_i_plus_1>(t);

      for (index f = fBegin; f < fEnd; ++f) {
		face = lowFace[f];
		assigner.setLower(rowI, entryJ, face);
		++entryJ;
      }

      assigner.setDiagonal(rowI, entryJ);
      ++entryJ;

      fBegin = thrust::get<UP_FACE_START_i>(t);
      fEnd   = thrust::get<UP_FACE_START_i_plus_1>(t);

      for (index f = fBegin; f < fEnd; ++f) {
		assigner.setUpper(rowI, entryJ, f);
		++entryJ;
      }
     
      for (; entryJ < maxEntriesPerRow; ++entryJ) {
		assigner.setPadding(rowI, entryJ);
      }
    }
  };



  void     
  SparseMatrixSystem::initialize(SparseMatrixArgs const & args)
  {   
    m_lowFacesStart.assign(args.lowFacesStart, args.lowFacesStart + args.nCells + 1);
    m_upFacesStart.assign(args.upFacesStart, args.upFacesStart + args.nCells + 1);

    {
      IndexArray dRowEntries(args.nCells);
      
      // Count entries per row (cell)
      thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(m_lowFacesStart.begin(), ++m_lowFacesStart.begin(), 
								    m_upFacesStart.begin(), ++m_upFacesStart.begin(), 
								    dRowEntries.begin())),
		       thrust::make_zip_iterator(thrust::make_tuple(--m_lowFacesStart.end(), m_lowFacesStart.end(), 
								    --m_upFacesStart.end(), m_upFacesStart.end(), 
								    dRowEntries.end())),
		       RowEntryFn());
      
      // Total entry count
      index entryCount = thrust::reduce(dRowEntries.begin(), dRowEntries.end());
      
      // Max entries per row
      m_maxEntriesPerRow = *(thrust::max_element(dRowEntries.begin(), dRowEntries.end()));

      m_A.resize(args.nCells, args.nCells, entryCount, m_maxEntriesPerRow);
    }

    m_lowFace.assign(args.lowFace, args.lowFace + args.nFaces);

    IndexArray dLowFaceToCell(args.lowFaceToCell, args.lowFaceToCell + args.nFaces);
    IndexArray dUpFaceToCell(args.upFaceToCell, args.upFaceToCell + args.nFaces);
    thrust::counting_iterator<int> first(0);
    thrust::counting_iterator<int> last = first + args.nCells;
    
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(m_lowFacesStart.begin(), ++m_lowFacesStart.begin(), 
								  m_upFacesStart.begin() , ++m_upFacesStart.begin(),
								  first)),
                     thrust::make_zip_iterator(thrust::make_tuple(--m_lowFacesStart.end(), m_lowFacesStart.end(), 
								  --m_upFacesStart.end() , m_upFacesStart.end(),
								  last)),
                     MatrixAssignFn<AssignIndicesFn>(AssignIndicesFn(m_A, dLowFaceToCell, dUpFaceToCell),
						     m_maxEntriesPerRow,
						     m_lowFace));
  }


  void
  SparseMatrixSystem::assignMatrix(SparseMatrixArgs const & args)
  {
    RealArray dLowCellValueFromFace(args.lowCellValueFromFace, args.lowCellValueFromFace + args.nFaces);
    RealArray dDiagCellValue(args.diagCellValue, args.diagCellValue + args.nCells);
    RealArray dUpCellValueFromFace(args.upCellValueFromFace, args.upCellValueFromFace + args.nFaces);
    thrust::counting_iterator<int> first(0);
    thrust::counting_iterator<int> last = first + args.nCells;

    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(m_lowFacesStart.begin(), ++m_lowFacesStart.begin(), 
								  m_upFacesStart.begin() , ++m_upFacesStart.begin(),
								  first)),
                     thrust::make_zip_iterator(thrust::make_tuple(--m_lowFacesStart.end(), m_lowFacesStart.end(), 
								  --m_upFacesStart.end() , m_upFacesStart.end(),
								  last)),
                     MatrixAssignFn<AssignValuesFn>(AssignValuesFn(m_A, dLowCellValueFromFace, dDiagCellValue, dUpCellValueFromFace),
						    m_maxEntriesPerRow,
						    m_lowFace));
  }


  void
  SparseMatrixSystem::update(SparseMatrixArgs const & args)
  {
    if (m_A.num_rows != args.nCells) {
      initialize(args);
    }
    
    assignMatrix(args);
  }
}
