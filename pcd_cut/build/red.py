import sys

import numpy
import vtk

reader = vtk.vtkPolyDataReader()
reader.SetFileName(sys.argv[1])
reader.Update()

polydata = reader.GetOutput()

for i in range(polydata.GetNumberOfCells()):
   pts = polydata.GetCell(i).GetPoints()    
   np_pts = numpy.array([pts.GetPoint(i) for i in range(pts.GetNumberOfPoints())])
   print (np_pts)
