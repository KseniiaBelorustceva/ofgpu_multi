/***************************************************************************
    begin                : Fri Apr 22 2011
    copyright            : (C) 2011 Symscape
    website              : www.symscape.com
 ***************************************************************************/

#ifndef ofgpuexport_h
#define ofgpuexport_h


#if defined(WIN32)
 #if defined(ofgpu_EXPORTS)
  #define OFGPU_EXPORT __declspec( dllexport ) 
 #else
  #define OFGPU_EXPORT __declspec( dllimport ) 
 #endif
#else
  #define OFGPU_EXPORT
#endif


#endif
