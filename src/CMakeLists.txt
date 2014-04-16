#/***************************************************************************
#    begin                : Thu Mar 24 2011
#    copyright            : (C) 2011 Symscape
#    website              : www.symscape.com
# ***************************************************************************/
#
#    This file is part of ofgpu.
#
#    ofgpu is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    ofgpu is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with ofgpu.  If not, see <http://www.gnu.org/licenses/>.

project(ofgpu)
cmake_minimum_required(VERSION 2.8)

# Rid ourselves of matching args between if, else and endif
set(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS true)

if(WIN32)
 set(CMAKE_BUILD_TYPE "release" CACHE STRING
    "Build type from release|debug")
	
 # Avoid warnings for:
 # 4244 4267 4305: conversions possible loss of data
 set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4244 /wd4267 /wd4305")
endif()

find_package(CUDA)

cuda_include_directories(${ofgpu_SOURCE_DIR})
include_directories(${ofgpu_SOURCE_DIR})
set(OFGPU_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/${CMAKE_BUILD_TYPE}/lib)

add_subdirectory(ofgpu)
