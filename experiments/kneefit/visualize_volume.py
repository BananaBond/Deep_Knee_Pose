import pyvista as pv
import numpy as np
import nibabel as nib
from mayavi import mlab

# Assuming `ct_scan` is your 3D numpy array
# Convert to a PyVista mesh
ct_file = r"D:\kneefit_model\SUBN_02_Femur_RE_Volume.nii"
nii_data = nib.load(ct_file)
ct_scan = nii_data.get_fdata()

# grid = pv.UniformGrid()
# grid.dimensions = ct_scan.shape
# grid.spacing = (1, 1, 1)  # Define the voxel spacing
# grid.point_data["values"] = ct_scan.flatten(order="F")  # Flatten the array in Fortran order

# # Visualization
# p = pv.Plotter()
# p.add_volume(grid, cmap="gray")
# p.show()

# x, y, z = np.mgrid[-50:50, -50:50, -50:50]

# # Create a PyVista structured grid
# grid = pv.StructuredGrid()
# grid.points = np.c_[x.ravel(order="F"), y.ravel(order="F"), z.ravel(order="F")]
# grid.point_arrays["CT_scan_data"] = ct_scan.ravel(order="F")
# grid.dimensions = ct_scan.shape

# # Plot using PyVista
# plotter = pv.Plotter()
# plotter.add_volume(grid, cmap="bone", opacity="linear")
# plotter.show()

# Assuming `ct_scan` is your 3D numpy array
mlab.volume_slice(ct_scan, plane_orientation='x_axes', slice_index=50)
mlab.volume_slice(ct_scan, plane_orientation='y_axes', slice_index=50)
mlab.volume_slice(ct_scan, plane_orientation='z_axes', slice_index=50)
mlab.show()