#include <igl/readOFF.h>
#include <igl/writeOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
/*** insert any necessary libigl headers here ***/
#include <igl/per_face_normals.h>
#include <igl/copyleft/marching_cubes.h>
#include <igl/floor.h>
#include <igl/slice.h>
#include <igl\bounding_box_diagonal.h>

using namespace std;
using Viewer = igl::opengl::glfw::Viewer;

// Input: imported points, #P x3
Eigen::MatrixXd P;

// Input: imported normals, #P x3
Eigen::MatrixXd N;

// Intermediate result: constrained points, #C x3
Eigen::MatrixXd constrained_points;

// Intermediate result: implicit function values at constrained points, #C x1
Eigen::VectorXd constrained_values;

// Parameter: degree of the polynomial
int polyDegree = 0;

// Parameter: Wendland weight function radius (make this relative to the size of the mesh)
double wendlandRadius = 0.1;

// Parameter: grid resolution
int resolution = 40;

// Intermediate result: grid points, at which the imlicit function will be evaluated, #G x3
Eigen::MatrixXd grid_points;

// Bounding box diagonal length
double boundingBoxDiagonal;

double spatialRate = 0.1;

// Spatial index length of each dimension
int x_len, y_len, z_len;

// Index of constrained points list at each space grid
std::vector<std::vector<int> > spatial_points;

// Intermediate result: implicit function values at the grid points, #G x1
Eigen::VectorXd grid_values;

// Intermediate result: grid point colors, for display, #G x3
Eigen::MatrixXd grid_colors;

// Intermediate result: grid lines, for display, #L x6 (each row contains
// starting and ending point of line segment)
Eigen::MatrixXd grid_lines;

// Output: vertex array, #V x3
Eigen::MatrixXd V;

// Output: face array, #F x3
Eigen::MatrixXi F;

// Output: face normals of the reconstructed mesh, #F x3
Eigen::MatrixXd FN;

// Functions
void createGrid();
void evaluateImplicitFunc();
void getLines();
bool callback_key_down(Viewer& viewer, unsigned char key, int modifiers);
void get_close_points(Eigen::RowVector3d point, double h, std::vector<int>& result_vec, std::vector<double>& d_vec);

// Creates a grid_points array for the simple sphere example. The points are
// stacked into a single matrix, ordered first in the x, then in the y and
// then in the z direction. If you find it necessary, replace this with your own
// function for creating the grid.
void createGrid() {
    grid_points.resize(0, 3);
    grid_colors.resize(0, 3);
    grid_lines. resize(0, 6);
    grid_values.resize(0);
    V. resize(0, 3);
    F. resize(0, 3);
    FN.resize(0, 3);

    // Grid bounds: axis-aligned bounding box
    Eigen::RowVector3d bb_min, bb_max;
    bb_min = P.colwise().minCoeff();
    bb_max = P.colwise().maxCoeff();

    // Bounding box dimensions
    Eigen::RowVector3d dim = bb_max - bb_min;

    // Grid spacing
    const double dx = dim[0] / (double)(resolution - 1);
    const double dy = dim[1] / (double)(resolution - 1);
    const double dz = dim[2] / (double)(resolution - 1);
    // 3D positions of the grid points -- see slides or marching_cubes.h for ordering
    grid_points.resize(resolution * resolution * resolution, 3);
    // Create each gridpoint
    for (unsigned int x = 0; x < resolution; ++x) {
        for (unsigned int y = 0; y < resolution; ++y) {
            for (unsigned int z = 0; z < resolution; ++z) {
                // Linear index of the point at (x,y,z)
                int index = x + resolution * (y + resolution * z);
                // 3D point at (x,y,z)
                grid_points.row(index) = bb_min + Eigen::RowVector3d(x * dx, y * dy, z * dz);
            }
        }
    }
}

// Function for explicitly evaluating the implicit function for a sphere of
// radius r centered at c : f(p) = ||p-c|| - r, where p = (x,y,z).
// This will NOT produce valid results for any mesh other than the given
// sphere.
// Replace this with your own function for evaluating the implicit function
// values at the grid points using MLS
/*void evaluateImplicitFunc() {
    // Sphere center
    auto bb_min = grid_points.colwise().minCoeff().eval();
    auto bb_max = grid_points.colwise().maxCoeff().eval();
    Eigen::RowVector3d center = 0.5 * (bb_min + bb_max);

    double radius = 0.5 * (bb_max - bb_min).minCoeff();

    // Scalar values of the grid points (the implicit function values)
    grid_values.resize(resolution * resolution * resolution);

    // Evaluate sphere's signed distance function at each gridpoint.
    for (unsigned int x = 0; x < resolution; ++x) {
        for (unsigned int y = 0; y < resolution; ++y) {
            for (unsigned int z = 0; z < resolution; ++z) {
                // Linear index of the point at (x,y,z)
                int index = x + resolution * (y + resolution * z);

                // Value at (x,y,z) = implicit function for the sphere
                grid_values[index] = (grid_points.row(index) - center).norm() - radius;
            }
        }
    }
}*/


void get_close_points(Eigen::RowVector3d point, double h, std::vector<int>& result_vec, std::vector<double>& d_vec) {
    result_vec.clear();
    d_vec.clear();
    double unit_length = spatialRate * boundingBoxDiagonal;
    Eigen::RowVector3d m = P.colwise().minCoeff();
    Eigen::RowVector3d p_dim = (point - m) / unit_length;
    int px = floor(p_dim[0]);
    int py = floor(p_dim[1]);
    int pz = floor(p_dim[2]);

    int num_cells = ceil(h / unit_length);
    for (int i = max(0, px - num_cells); i < min(x_len, px + num_cells + 1); i++) {
        for (int j = max(0, py - num_cells); j < min(y_len, py + num_cells + 1); j++) {
            for (int k = max(0, pz - num_cells); k < min(z_len, pz + num_cells + 1); k++) {
                for (int it = 0; it < spatial_points[i + x_len * (j + y_len * k)].size(); it++) {
                    int it_idx = spatial_points[i + x_len * (j + y_len * k)][it];
                    double distance = (constrained_points.row(it_idx) - point).norm();
                    if (distance <= h) {
                        result_vec.push_back(it_idx);
                        d_vec.push_back(distance);
                    }
                    distance = (constrained_points.row(P.rows() + it_idx) - point).norm();
                    if (distance <= h) {
                        result_vec.push_back(P.rows() + it_idx);
                        d_vec.push_back(distance);
                    }
                    distance = (constrained_points.row(P.rows() * 2 + it_idx) - point).norm();
                    if (distance <= h) {
                        result_vec.push_back(P.rows() * 2 + it_idx);
                        d_vec.push_back(distance);
                    }
                }
            }
        }
    }
}
void evaluateImplicitFunc() {
    double newWendlandRadius = wendlandRadius * boundingBoxDiagonal;
    grid_values.resize(resolution * resolution * resolution);

    for (unsigned int x = 0; x < resolution; x++) {
        for (unsigned int y = 0; y < resolution; y++) {
            for (unsigned int z = 0; z < resolution; z++) {
                std::vector<int> result_vec;
                std::vector<double> d_vec;
                int index = x + resolution * (y + resolution * z);
                double px = grid_points(index, 0);
                double py = grid_points(index, 1);
                double pz = grid_points(index, 2);
                get_close_points(grid_points.row(index), newWendlandRadius, result_vec, d_vec);
                if (result_vec.size() == 0) {
                    grid_values[index] = 100000000000;
                }
                else {
                    Eigen::VectorXd r_vec = Eigen::VectorXd::Map(d_vec.data(), d_vec.size());
                    Eigen::VectorXd weight = (1 - (r_vec.array() / newWendlandRadius)).pow(4) * (4 * r_vec.array() / newWendlandRadius + 1);
                    Eigen::VectorXi p_idx = Eigen::VectorXi::Map(result_vec.data(), result_vec.size());
                    Eigen::MatrixXd A, nearby_points;
                    Eigen::VectorXd bx, fi;
                    Eigen::VectorXi col_array(3);
                    col_array << 0, 1, 2;
                    igl::slice(constrained_points, p_idx, col_array, nearby_points);
                    col_array.resize(1);
                    col_array << 0;
                    igl::slice(constrained_values, p_idx, col_array, fi);

                    Eigen::MatrixXd M_squares = nearby_points.cwiseProduct(nearby_points);
                    Eigen::MatrixXd nearby_points_r(nearby_points.rows(), 3);
                    nearby_points_r << nearby_points.col(1), nearby_points.col(2), nearby_points.col(0);
                    Eigen::MatrixXd M_products = nearby_points.cwiseProduct(nearby_points_r);

                    switch (polyDegree)
                    {
                    case 0:
                    {
                        A = Eigen::MatrixXd::Ones(nearby_points.rows(), 1);
                        bx.resize(1);
                        bx << 1;
                        break;
                    }
                    case 1:
                    {
                        A.resize(nearby_points.rows(), 4);
                        A << Eigen::MatrixXd::Ones(nearby_points.rows(), 1), nearby_points;
                        bx.resize(4);
                        bx << 1, px, py, pz;
                        break;
                    }
                    default:
                        A.resize(nearby_points.rows(), 10);
                        A << Eigen::MatrixXd::Ones(nearby_points.rows(), 1), nearby_points, M_squares, M_products;
                        bx.resize(10);
                        bx << 1, px, py, pz, pow(px, 2), pow(py, 2), pow(pz, 2), px* py, py* pz, pz* px;
                        break;
                    }

                    Eigen::VectorXd c = (A.transpose() * weight.asDiagonal() * A).ldlt().solve(A.transpose() * weight.asDiagonal() * fi);
                    grid_values[index] = bx.dot(c);
                }
            }
        }
    }
}


// Code to display the grid lines given a grid structure of the given form.
// Assumes grid_points have been correctly assigned
// Replace with your own code for displaying lines if need be.
void getLines() {
    int nnodes = grid_points.rows();
    grid_lines.resize(3 * nnodes, 6);
    int numLines = 0;

    for (unsigned int x = 0; x<resolution; ++x) {
        for (unsigned int y = 0; y < resolution; ++y) {
            for (unsigned int z = 0; z < resolution; ++z) {
                int index = x + resolution * (y + resolution * z);
                if (x < resolution - 1) {
                    int index1 = (x + 1) + y * resolution + z * resolution * resolution;
                    grid_lines.row(numLines++) << grid_points.row(index), grid_points.row(index1);
                }
                if (y < resolution - 1) {
                    int index1 = x + (y + 1) * resolution + z * resolution * resolution;
                    grid_lines.row(numLines++) << grid_points.row(index), grid_points.row(index1);
                }
                if (z < resolution - 1) {
                    int index1 = x + y * resolution + (z + 1) * resolution * resolution;
                    grid_lines.row(numLines++) << grid_points.row(index), grid_points.row(index1);
                }
            }
        }
    }

    grid_lines.conservativeResize(numLines, Eigen::NoChange);
}







int getNearestPoint(Eigen::RowVectorXd temp) {
    double diag;
    double minDiagonal = boundingBoxDiagonal;
    int min_index = -1;
    double unit_length = spatialRate * boundingBoxDiagonal;
    Eigen::RowVector3d m = P.colwise().minCoeff();
    Eigen::RowVector3d p_dim = (temp - m) / unit_length;

    int px = floor(p_dim[0]);
    int py = floor(p_dim[1]);
    int pz = floor(p_dim[2]);

    for (int i = max(0, px - 1); i < min(x_len, px + 2); i++) {
        for (int j = max(0, py - 1); j < min(y_len, py + 2); j++) {
            for (int k = max(0, pz - 1); k < min(z_len, pz + 2); k++) {
                for (int it = 0; it < spatial_points[i + x_len * (j + y_len * k)].size(); it++) {
                    int it_idx = spatial_points[i + x_len * (j + y_len * k)][it];
                    diag = (P.row(it_idx) - temp).norm();
                    if (diag < minDiagonal) {
                        minDiagonal = diag;
                        min_index = it_idx;
                    }
                }
            }
        }
    }
    return min_index;
}

bool callback_key_down(Viewer &viewer, unsigned char key, int modifiers) {
    if (key == '1') {
        // Show imported points
        boundingBoxDiagonal = igl::bounding_box_diagonal(P);
        viewer.data().clear();
        viewer.core.align_camera_center(P);
        viewer.data().point_size = 11;
        viewer.data().add_points(P, Eigen::RowVector3d(0,0,0));
    }

    if (key == '2') {
        // Show all constraints
        viewer.data().clear();
        viewer.core.align_camera_center(P);
        spatial_points.clear();

        Eigen::RowVector3d m = P.colwise().minCoeff();
        Eigen::RowVector3d M = P.colwise().maxCoeff();

        int min_index = -1;
        double Px, Py, Pz, diag , minDiagonal;
        double unit_length = spatialRate * boundingBoxDiagonal;
        double eps;
        Eigen::MatrixXi P_index;
        Eigen::RowVector3d dim_length = (M - m) / unit_length;

        x_len = floor(dim_length[0]) + 1;
        y_len = floor(dim_length[1]) + 1;
        z_len = floor(dim_length[2]) + 1;

        minDiagonal = boundingBoxDiagonal;
        
        Eigen::MatrixXd P_temp = (P - m.replicate(P.rows(), 1)) / unit_length;

        igl::floor(P_temp, P_index);

        spatial_points.resize(x_len * y_len * z_len);
        for (int i = 0; i < P_index.rows(); i++) {
            int spatial_idx = P_index(i, 0) + x_len * (P_index(i, 1) + y_len * P_index(i, 2));
            spatial_points[spatial_idx].push_back(i);
        }

        constrained_points.resize(P.rows() * 4, 3);
        constrained_values.setZero(P.rows() * 4);

        
        Eigen::RowVectorXd temp1(3);
        Eigen::RowVectorXd temp2(3);
        Eigen::RowVectorXd temp3(3);
        for (int i = 0; i < P.rows(); i++) {
            eps = 0.02 * boundingBoxDiagonal;
            temp1 = P.row(i) + eps * N.row(i).normalized();
            minDiagonal = boundingBoxDiagonal;
            min_index = -1;
            unit_length = spatialRate * boundingBoxDiagonal;
            Eigen::RowVector3d m = P.colwise().minCoeff();
            Eigen::RowVector3d p_dim = (temp1 - m) / unit_length;
            int px = floor(p_dim[0]);
            int py = floor(p_dim[1]);
            int pz = floor(p_dim[2]);
            for (int i = max(0, px - 1); i < min(x_len, px + 2); i++) {
                for (int j = max(0, py - 1); j < min(y_len, py + 2); j++) {
                    for (int k = max(0, pz - 1); k < min(z_len, pz + 2); k++) {
                        for (int it = 0; it < spatial_points[i + x_len * (j + y_len * k)].size(); it++) {
                            int it_idx = spatial_points[i + x_len * (j + y_len * k)][it];
                            diag = (P.row(it_idx) - temp1).norm();
                            if (diag < minDiagonal) {
                                minDiagonal = diag;
                                min_index = it_idx;
                            }
                        }
                    }
                }
            }
            while (min_index != i) {
                eps = eps / 2;
                temp1 = P.row(i) + eps * N.row(i).normalized();
                minDiagonal = boundingBoxDiagonal;
                min_index = -1;
                unit_length = spatialRate * boundingBoxDiagonal;
                m = P.colwise().minCoeff();
                p_dim = (temp1 - m) / unit_length;
                px = floor(p_dim[0]);
                py = floor(p_dim[1]);
                pz = floor(p_dim[2]);
                for (int i = max(0, px - 1); i < min(x_len, px + 2); i++) {
                    for (int j = max(0, py - 1); j < min(y_len, py + 2); j++) {
                        for (int k = max(0, pz - 1); k < min(z_len, pz + 2); k++) {
                            for (int it = 0; it < spatial_points[i + x_len * (j + y_len * k)].size(); it++) {
                                int it_idx = spatial_points[i + x_len * (j + y_len * k)][it];
                                diag = (P.row(it_idx) - temp1).norm();
                                if (diag < minDiagonal) {
                                    minDiagonal = diag;
                                    min_index = it_idx;
                                }
                            }
                        }
                    }
                }
            }
            constrained_points.row(i + P.rows()) = temp1;
            constrained_values(i + P.rows()) = eps;

            eps = 0.01 * boundingBoxDiagonal;
            temp2 = P.row(i) - eps * N.row(i).normalized();
            while (getNearestPoint(temp2) != i) {
                eps = eps / 2;
                temp2 = P.row(i) - eps * N.row(i).normalized();
            }
            constrained_points.row(i + 2 * P.rows()) = temp2;
            constrained_values(i + 2 * P.rows()) = (-1) * eps;

            temp3 = P.row(i);
            while (getNearestPoint(temp3) != i) {
                eps = eps / 2;
                temp3 = P.row(i);
            }
            constrained_points.row(i + 3 * P.rows()) = temp3;
        }

        viewer.data().point_size = 8;
        viewer.data().add_points(constrained_points.block(P.rows(), 0, P.rows(), 3), Eigen::RowVector3d(0, 0, 1));
        viewer.data().add_points(constrained_points.block(P.rows()*2, 0, P.rows(), 3), Eigen::RowVector3d(1, 0, 0));
        viewer.data().add_points(constrained_points.block(P.rows() * 3, 0, P.rows(), 3), Eigen::RowVector3d(0, 1, 0));
    }

    if (key == '3') {
        // Show grid points with colored nodes and connected with lines
        viewer.data().clear();
        viewer.core.align_camera_center(P);
        // Add code for creating a grid
        // Add your code for evaluating the implicit function at the grid points
        // Add code for displaying points and lines
        // You can use the following example:

        /*** begin: sphere example, replace (at least partially) with your code ***/
        // Make grid
        createGrid();

        // Evaluate implicit function
        evaluateImplicitFunc();

        // get grid lines
        getLines();

        // Code for coloring and displaying the grid points and lines
        // Assumes that grid_values and grid_points have been correctly assigned.
        grid_colors.setZero(grid_points.rows(), 3);

        // Build color map
        for (int i = 0; i < grid_points.rows(); ++i) {
            double value = grid_values(i);
            if (value < 0) {
                grid_colors(i, 1) = 1;
            }
            else {
                if (value > 0)
                    grid_colors(i, 0) = 1;
            }
        }

        // Draw lines and points
        viewer.data().point_size = 8;
        viewer.data().add_points(grid_points, grid_colors);
        viewer.data().add_edges(grid_lines.block(0, 0, grid_lines.rows(), 3),
                              grid_lines.block(0, 3, grid_lines.rows(), 3),
                              Eigen::RowVector3d(0.8, 0.8, 0.8));
        /*** end: sphere example ***/
    }

    if (key == '4') {
        // Show reconstructed mesh
        viewer.data().clear();
        // Code for computing the mesh (V,F) from grid_points and grid_values
        if ((grid_points.rows() == 0) || (grid_values.rows() == 0)) {
            cerr << "Not enough data for Marching Cubes !" << endl;
            return true;
        }
        // Run marching cubes
        igl::copyleft::marching_cubes(grid_values, grid_points, resolution, resolution, resolution, V, F);
        if (V.rows() == 0) {
            cerr << "Marching Cubes failed!" << endl;
            return true;
        }

        igl::per_face_normals(V, F, FN);
        viewer.data().set_mesh(V, F);
        viewer.data().show_lines = true;
        viewer.data().show_faces = true;
        viewer.data().set_normals(FN);
    }

    return true;
}

bool callback_load_mesh(Viewer& viewer,string filename)
{
  igl::readOFF(filename,P,F,N);
  callback_key_down(viewer,'1',0);
  return true;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
      cout << "Usage ex2_bin <mesh.off>" << endl;
      igl::readOFF("C:/Users/orash/Desktop/Computer Science/Geometric Processing/Project/geometryprocessing2021-Orash94/assignment2/data/luigi.off",P,F,N);
    }
	  else
	  {
		  // Read points and normals
		  igl::readOFF(argv[1],P,F,N);
	  }

    Viewer viewer;
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);

    viewer.callback_key_down = callback_key_down;

    menu.callback_draw_viewer_menu = [&]()
    {
      // Draw parent menu content
      menu.draw_viewer_menu();

      // Add new group
      if (ImGui::CollapsingHeader("Reconstruction Options", ImGuiTreeNodeFlags_DefaultOpen))
      {
        // Expose variable directly ...
        ImGui::InputInt("Resolution", &resolution, 0, 0);
        ImGui::InputInt("Polynomial Degree", &polyDegree, 0, 0);
        ImGui::InputDouble("Wedland Radius", &wendlandRadius, 0, 0);

        if (ImGui::Button("Reset Grid", ImVec2(-1,0)))
        {
          std::cout << "ResetGrid\n";
          // Recreate the grid
          createGrid();
          // Switch view to show the grid
          callback_key_down(viewer,'3',0);
        }

        if (ImGui::Button("Export Mesh"))
        {
            std::string f = igl::file_dialog_save();
            igl::writeOFF(f, V, F);
        }

        // TODO: Add more parameters to tweak here...
      }

    };

    callback_key_down(viewer, '1', 0);

    viewer.launch();
}
