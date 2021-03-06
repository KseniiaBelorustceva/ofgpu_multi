/***************************************************************************
    begin                : Thu Aug 4 2011
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


#include "ofgpu/ofgpuconfig.h"
#include "ofgpu/sparsematrixsystem.h"


extern "C"
OFGPU_EXPORT void ofgpuConfig(int const cudaDevice)
{
  using namespace ofgpu;

  //the<SparseMatrixSystem>().setDevice(cudaDevice);
  // Attempt to load all available devices
    the<SparseMatrixSystem>().setMultipleDevices();
}
