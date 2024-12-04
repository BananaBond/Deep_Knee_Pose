import trimesh
import numpy as np
import argparse
import numpy as np
from stl import mesh
import ipdb
import open3d as o3d
import cupy as cp
from trimesh import voxel

def read_stl_file(file_path):
    # Load the STL file
    your_mesh = mesh.Mesh.from_file(file_path)
    # your_mesh = trimesh.load(file_path)
    your_mesh = trimesh.load_mesh(file_path, force="mesh", enable_post_processing=True, solid=True)

    # Get vertices and faces
    # vertices = mesh.vertices
    # faces = mesh.faces

    # Extract vertices and faces (fv)
    fv = {
        'vertices': your_mesh.vertices.reshape((-1, 3)),  # Flatten to (N, 3) array
        'faces': your_mesh.faces
        # 'faces': your_mesh.vectors.shape[0]  # Number of faces
    }

    # Optionally, extract points for testing
    # Here, we'll use a simple grid of points within the bounding box of the mesh
    min_coords = np.min(your_mesh.vertices, axis=0)
    max_coords = np.max(your_mesh.vertices, axis=0)
    grid_spacing = 1.0  # Adjust as needed
    # ipdb.set_trace()
    x = np.arange(min_coords[0], max_coords[0], grid_spacing)
    y = np.arange(min_coords[1], max_coords[1], grid_spacing)
    z = np.arange(min_coords[2], max_coords[2], grid_spacing)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    points = np.vstack((xx.flatten(), yy.flatten(), zz.flatten())).T

    print(points.shape)

    return fv, points

def show(chair_mesh, chair_voxels, colors=(1, 1, 1, 0.3)):
    # scene = trimesh.scene()
    scene = chair_mesh.scene()
    scene.add_geometry(chair_voxels.as_boxes(colors=colors))
    scene.show()

def inmesh(fv, points):
    # Load the mesh from the input data (fv.vertices and fv.faces)
    mesh = trimesh.Trimesh(vertices=fv["vertices"], faces=fv["faces"])
    
    # ideally pitch should be the value below because our voxel spacing is not uniform for z-axis
    # pitch = (0.390625, 0.390625, 0.799988)

    #decide best pitch value that accurately represents the surface and also does not generate to many dimensions
    array = mesh.voxelized(pitch=0.1)

    # UNCOMMENT TO SEE VOXEL
    # show(mesh, array, colors=(1, 0, 0, 0.3))


    ipdb.set_trace()

    # print("mesh shape", mesh.shape)

    # Use trimesh's contains_points method to check if points are inside the mesh\
    # OUT OF MEMORY ISSUE WITHOUT GPU
    # inside = mesh.contains(points)

    # # Convert boolean array to integers (1 for inside, 0 for outside)
    # in_mesh = inside.astype(np.int)

    # Convert data to CuPy arrays
    vertices_cp = cp.asarray(fv["vertices"])
    faces_cp = cp.asarray(fv["faces"])
    points_cp = cp.asarray(points)

    # Define the CUDA kernel for point-in-triangle check
    point_in_triangle_kernel = cp.ElementwiseKernel(
        in_params='raw float64 vertices, raw int64 faces, raw float64 points',
        out_params='int32 inside',
        operation='''
        int i = i0;
        float3 v0 = make_float3(vertices[faces[i * 3] * 3], vertices[faces[i * 3] * 3 + 1], vertices[faces[i * 3] * 3 + 2]);
        float3 v1 = make_float3(vertices[faces[i * 3 + 1] * 3], vertices[faces[i * 3 + 1] * 3 + 1], vertices[faces[i * 3 + 1] * 3 + 2]);
        float3 v2 = make_float3(vertices[faces[i * 3 + 2] * 3], vertices[faces[i * 3 + 2] * 3 + 1], vertices[faces[i * 3 + 2] * 3 + 2]);
        float3 p = make_float3(points[i * 3], points[i * 3 + 1], points[i * 3 + 2]);

        // Vector operations to check if point p is inside triangle v0, v1, v2
        float3 u = v1 - v0;
        float3 v = v2 - v0;
        float3 w = p - v0;

        float uu = dot(u, u);
        float uv = dot(u, v);
        float vv = dot(v, v);
        float wu = dot(w, u);
        float wv = dot(w, v);

        float denominator = uv * uv - uu * vv;
        float s = (uv * wv - vv * wu) / denominator;
        float t = (uv * wu - uu * wv) / denominator;

        if (s >= 0 && t >= 0 && (s + t) <= 1) {
            inside = 1;
        } else {
            inside = 0;
        }
        ''',
        name='point_in_triangle_kernel'
    )

    # Prepare an output array
    inside_cp = cp.zeros(points_cp.shape[0], dtype=cp.int32)

    # Run the kernel
    point_in_triangle_kernel(vertices_cp, faces_cp, points_cp, inside_cp)

    # Convert result back to NumPy
    inside_np = cp.asnumpy(inside_cp)

    print("Points inside the mesh:", inside_np)

    ipdb.set_trace()

    # return in_mesh

def visualize_stl(args):
    mesh = o3d.io.read_triangle_mesh(args.input)
    mesh = mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], window_name="STL", left=1000, top=200, width=800, height=650)


def main(args):

    if args.visualize:
        visualize_stl(args)
    
    fv, points = read_stl_file(args.input)

    # Now you can use fv and points as needed in your application
    print(f"Number of vertices: {len(fv['vertices'])}")
    print(f"Number of faces: {len(fv['faces'])}")
    print(f"Number of points: {len(points)}")

    inmesh(fv, points)

    # print(in_mesh.shape)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--input', type=str, default=r"C:\Users\Public Admin\Desktop\JTA_test_real\SUBN_02_Femur_RE_Surface.stl",
                        help='an integer for the accumulator')
    parser.add_argument('--output', type=str, default=r"C:\Users\Public Admin\Desktop\Gitlab\kneedeeppose\diffpose\preprocess\file_conversion\output.txt",
                        help='an integer for the accumulator')
    parser.add_argument('--visualize', action='store_true', help='Visualize the mesh')
    
    args = parser.parse_args()
    
    main(args)