/***************************************************************************
    begin                : Mon Apr 4 2011
    copyright            : (C) 2011 Symscape
    website              : www.symscape.com
***************************************************************************/

#ifndef sparsematrixargs_h
#define sparsematrixargs_h

#pragma once

#include <string>

#include "ofgpu/index.h"
#include "ofgpu/real.h"


namespace ofgpu
{
  /**@brief Arguments destined for SparseMatrixSystem
   *
   *@author Rich Smith
   */
  struct SparseMatrixArgs {
    char const * const preconditionerName;
    index const nCells;
    index const nFaces;
    index const * const lowFacesStart;
    index const * const lowFace;
    index const * const lowFaceToCell;
    real const * const lowCellValueFromFace; 
    real const * const diagCellValue;
    index const * const upFacesStart;
    index const * const upFaceToCell;
    real const * const upCellValueFromFace;
    real* const xSolution;
    real const * const bSource;
    index const maxIterations;
    real const absoluteTolerance;
    real const relativeTolerance;
    real & initialResidual;
    real & finalResidual;
    index & iterationsPerformed;
    bool & converged;

    SparseMatrixArgs(char const * const preconditionerName,
		     index const nCells, index const nFaces, 
		     index const * const lowFacesStart, index const * const lowFace, index const * const lowFaceToCell, real const * const lowCellValueFromFace, 
		     real const * const diagCellValue, 
		     index const * const upFacesStart, index const * const upFaceToCell,  real const * const upCellValueFromFace,
		     real* const xSolution,
		     real const * const bSource,
		     index const maxIterations, real const absoluteTolerance, real const relativeTolerance,
		     real & initialResidual, real & finalResidual, index & iterationsPerformed, bool & converged)
      : preconditionerName(preconditionerName)
      , nCells(nCells)
      , nFaces(nFaces)
      , lowFacesStart(lowFacesStart)
      , lowFace(lowFace)
      , lowFaceToCell(lowFaceToCell)
      , lowCellValueFromFace(lowCellValueFromFace)
      , diagCellValue(diagCellValue)
      , upFacesStart(upFacesStart)
      , upFaceToCell(upFaceToCell)
      , upCellValueFromFace(upCellValueFromFace)
      , xSolution(xSolution)
      , bSource(bSource)
      , maxIterations(maxIterations)
      , absoluteTolerance(absoluteTolerance)
      , relativeTolerance(relativeTolerance)
      , initialResidual(initialResidual)
      , finalResidual(finalResidual)
      , iterationsPerformed(iterationsPerformed)
      , converged(converged)
    {}
  };
}

#endif
