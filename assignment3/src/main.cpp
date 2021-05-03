#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <igl/local_basis.h>
#include <igl/grad.h>
#include <igl/min_quad_with_fixed.h>
#include <igl/cotmatrix.h>


/*** insert any necessary libigl headers here ***/
#include <igl/boundary_loop.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/harmonic.h>
#include <igl/lscm.h>
#include <igl/adjacency_matrix.h>
#include <igl/sum.h>
#include <igl/diag.h>
#include <igl/speye.h>
#include <igl/repdiag.h>
#include <igl/cat.h>
#include <igl/dijkstra.h>

using namespace std;
using namespace Eigen;
using Viewer = igl::opengl::glfw::Viewer;

Viewer viewer;

// vertex array, #V x3
Eigen::MatrixXd V;

// face array, #F x3
Eigen::MatrixXi F;

// UV coordinates, #V x2
Eigen::MatrixXd UV;


Eigen::VectorXd distortion; // Vector that represents the distortion size for each face, #F x 1

Eigen::MatrixXd distortionColors; // matrix for the distortion colors, #F x 3

bool showingUV = false;
bool freeBoundary = false;
double TextureResolution = 10;
int distortionType = -1;
igl::opengl::ViewerCore temp3D;
igl::opengl::ViewerCore temp2D;
bool angle = false;
bool edge = false;


void Redraw()
{
	viewer.data().clear();

	if (!showingUV)
	{
		viewer.data().set_mesh(V, F);
		viewer.data().set_face_based(false);

    if(UV.size() != 0)
    {
      viewer.data().set_uv(TextureResolution*UV);
      viewer.data().show_texture = true;
    }
	}
	else
	{
		viewer.data().show_texture = false;
		viewer.data().set_mesh(UV, F);
	}
	viewer.data().set_colors(distortionColors);
}

bool callback_mouse_move(Viewer &viewer, int mouse_x, int mouse_y)
{
	if (showingUV)
		viewer.mouse_mode = igl::opengl::glfw::Viewer::MouseMode::Translation;
	return false;
}

static void computeSurfaceGradientMatrix(SparseMatrix<double> & D1, SparseMatrix<double> & D2)
{
	MatrixXd F1, F2, F3;
	SparseMatrix<double> DD, Dx, Dy, Dz;

	igl::local_basis(V, F, F1, F2, F3);
	igl::grad(V, F, DD);
	Dx = DD.topLeftCorner(F.rows(), V.rows());
	Dy = DD.block(F.rows(), 0, F.rows(), V.rows());
	Dz = DD.bottomRightCorner(F.rows(), V.rows());
	D1 = F1.col(0).asDiagonal()*Dx + F1.col(1).asDiagonal()*Dy + F1.col(2).asDiagonal()*Dz;
	D2 = F2.col(0).asDiagonal()*Dx + F2.col(1).asDiagonal()*Dy + F2.col(2).asDiagonal()*Dz;
}


void evaluateDistortion(int type) {

	Eigen::SparseMatrix<double> Dx, Dy;
	Eigen::MatrixXd J1, J2, J3, J4, J, D;

	distortionColors.conservativeResize(F.rows(), 3);
	J.conservativeResize(2, 2);
	distortion.conservativeResize(F.rows());
	computeSurfaceGradientMatrix(Dx, Dy);

	J1 = Dx * UV.col(0);
	J2 = Dx * UV.col(1);
	J3 = Dy * UV.col(0);
	J4 = Dy * UV.col(1);


	switch (type) {
	case 0:
	{
		Eigen::MatrixXd I;
		I = Eigen::MatrixXd::Identity(2, 2);

		for (int i = 0; i < F.rows(); i++) {
			J(0, 0) = J1(i, 0);
			J(0, 1) = J2(i, 0);
			J(1, 0) = J3(i, 0);
			J(1, 1) = J4(i, 0);
			D = J + J.transpose() - J.trace() * I;
			distortion[i] = pow(D.norm(), 2);
		}
		break;
	}
	case 1:
	{
		Eigen::MatrixXd U, V, R, UVT;
		R.conservativeResize(2, 2);

		for (int i = 0; i < F.rows(); i++) {
			J(0, 0) = J1(i, 0);
			J(0, 1) = J2(i, 0);
			J(1, 0) = J3(i, 0);
			J(1, 1) = J4(i, 0);
			JacobiSVD<MatrixXd> svd(J, ComputeThinU | ComputeThinV);
			U = svd.matrixU();
			V = svd.matrixV();
			UVT = U * V.transpose();
			R(0, 0) = 1;
			R(0, 1) = 0;
			R(1, 0) = 0;
			if (UVT.norm() == 0)
				R(1, 1) = 0;
			else
			{
				R(1, 1) = 1;
			}
			R = U * R * V.transpose();
			D = J - R;
			distortion[i] = pow(D.norm(), 2);
		}
		break;
	}
	default:
	{
		for (int i = 0; i < F.rows(); i++) {
			J(0, 0) = J1(i, 0);
			J(0, 1) = J2(i, 0);
			J(1, 0) = J3(i, 0);
			J(1, 1) = J4(i, 0);

			distortion[i] = pow(J(0, 0) * J(1, 1) - J(1, 0) * J(0, 1) - 1, 2);
		}
	}

	}
}

static inline void SSVD2x2(const Eigen::Matrix2d& J, Eigen::Matrix2d& U, Eigen::Matrix2d& S, Eigen::Matrix2d& V)
{
	double e = (J(0) + J(3))*0.5;
	double f = (J(0) - J(3))*0.5;
	double g = (J(1) + J(2))*0.5;
	double h = (J(1) - J(2))*0.5;
	double q = sqrt((e*e) + (h*h));
	double r = sqrt((f*f) + (g*g));
	double a1 = atan2(g, f);
	double a2 = atan2(h, e);
	double rho = (a2 - a1)*0.5;
	double phi = (a2 + a1)*0.5;

	S(0) = q + r;
	S(1) = 0;
	S(2) = 0;
	S(3) = q - r;

	double c = cos(phi);
	double s = sin(phi);
	U(0) = c;
	U(1) = s;
	U(2) = -s;
	U(3) = c;

	c = cos(rho);
	s = sin(rho);
	V(0) = c;
	V(1) = -s;
	V(2) = s;
	V(3) = c;
}



void ConvertConstraintsToMatrixForm(VectorXi indices, MatrixXd positions, Eigen::SparseMatrix<double> &C, VectorXd &d)
{
	// Convert the list of fixed indices and their fixed positions to a linear system
	// Hint: The matrix C should contain only one non-zero element per row and d should contain the positions in the correct order.
	Eigen::SparseMatrix<double> c;
	c.resize(indices.rows(), V.rows());
	C.resize(2 * indices.rows(), 2 * V.rows());
	d.resize(2 * indices.rows(), 1);
	d = (Map<VectorXd>(positions.data(), positions.cols() * positions.rows()));
	for (int i = 0; i < indices.rows(); i++) {
		int index = indices(i, 0);
		c.insert(i, index) = 1;
	}

	igl::repdiag(c, 2, C);
}

void computeParameterization(int type)
{
	VectorXi fixed_UV_indices;
	MatrixXd fixed_UV_positions;

	VectorXd b, d;
	Eigen::SparseMatrix<double>A, C;
	// Find the indices of the boundary vertices of the mesh and put them in fixed_UV_indices
	if (!freeBoundary)
	{
		// The boundary vertices should be fixed to positions on the unit disc. Find these position and
		// save them in the #V x 2 matrix fixed_UV_position.
		igl::boundary_loop(F, fixed_UV_indices);
		igl::map_vertices_to_circle(V, fixed_UV_indices, fixed_UV_positions);

	}
	else
	{
		// Fix two UV vertices. This should be done in an intelligent way. Hint: The two fixed vertices should be the two most distant one on the mesh.
		vector<vector<int>> VV;
		igl::adjacency_list(F, VV);

		double max_distance = numeric_limits<double>::min();
		int firstIndex, secondIndex;
		firstIndex = -1;
		secondIndex = -1;
		for (int i = 0; i < V.rows(); i++) {
			Eigen::VectorXd minDistance;
			Eigen::VectorXi prev;
			igl::dijkstra(i, {}, VV, minDistance, prev);
			for (int j = 0; j < minDistance.rows(); j++) {
				if (minDistance(j, 0) > max_distance) {
					max_distance = minDistance(j, 0);
					firstIndex = i;
					secondIndex = j;
				}
			}
		}

		fixed_UV_indices.resize(2, 1);
		fixed_UV_indices(0, 0) = firstIndex;
		fixed_UV_indices(1, 0) = secondIndex;

		igl::map_vertices_to_circle(V, fixed_UV_indices, fixed_UV_positions);

	}

	ConvertConstraintsToMatrixForm(fixed_UV_indices, fixed_UV_positions, C, d);

	// Find the linear system for the parameterization (1- Tutte, 2- Harmonic, 3- LSCM, 4- ARAP)
	// and put it in the matrix A.
	// The dimensions of A should be 2#V x 2#V.

	b.resize(2 * V.rows(), 1);
	b.setZero();

	if (type == '1') {
		// Add your code for computing uniform Laplacian for Tutte parameterization
		// Hint: use the adjacency matrix of the mesh
		Eigen::SparseMatrix<double> adjacents, adjcentsDiag, U;
		Eigen::SparseVector<double> adjacentsSum;
		igl::adjacency_matrix(F, adjacents);
		igl::sum(adjacents, 1, adjacentsSum);
		igl::diag(adjacentsSum, adjcentsDiag);
		U = adjacents - adjcentsDiag;

		igl::repdiag(U, 2, A);
	}

	if (type == '2') {
		// Add your code for computing cotangent Laplacian for Harmonic parameterization
		// Use can use a function "cotmatrix" from libIGL, but ~~~~***READ THE DOCUMENTATION***~~~~
		Eigen::SparseMatrix<double> L;
		igl::cotmatrix(V, F, L);
		igl::repdiag(L, 2, A);
	}

	if (type == '3') {
		// Add your code for computing the system for LSCM parameterization
		// Note that the libIGL implementation is different than what taught in the tutorial! Do not rely on it!!
		Eigen::SparseMatrix<double> Dx, Dy, Dxt, Dyt, Jr, Jc, J, J1, J2, J3, J4;
		Eigen::VectorXd Ta;

		computeSurfaceGradientMatrix(Dx, Dy);
		igl::doublearea(V, F, Ta);
		Ta /= 2;
		auto At = Ta.asDiagonal();

		J1 = Dx.transpose() * At * Dx + Dy.transpose() * At * Dy;
		J2 = Dy.transpose() * At * Dx - Dx.transpose() * At * Dy;
		J3 = Dx.transpose() * At * Dy - Dy.transpose() * At * Dx;
		J4 = Dx.transpose() * At * Dx + Dy.transpose() * At * Dy;
		igl::cat(2, J1, J2, Jr);
		igl::cat(2, J3, J4, Jc);
		igl::cat(1, Jr, Jc, J);

		A = J;
	}

	if (type == '4') {
		// Add your code for computing ARAP system and right-hand side
		// Implement a function that computes the local step first
		// Then construct the matrix with the given rotation matrices
	}

	// Solve the linear system.
	// Construct the system as discussed in class and the assignment sheet
	// Use igl::cat to concatenate matrices
	// Use Eigen::SparseLU to solve the system. Refer to tutorial 3 for more detail
	Eigen::SparseMatrix<double> N, M, Ct, lhs;
	VectorXd rhs;
	Ct = C.transpose();
	Eigen::SparseMatrix<double> zeros(Ct.cols(), C.rows());
	igl::cat(1, A, C, N);
	igl::cat(1, Ct, zeros, M);
	igl::cat(2, N, M, lhs);

	rhs.resize(b.rows() + d.rows(), 1);
	rhs << b, d;

	Eigen::SparseLU<SparseMatrix<double>> solver;
	lhs.makeCompressed();
	solver.analyzePattern(lhs);
	solver.factorize(lhs);
	Eigen::VectorXd res = solver.solve(rhs);

	// The solver will output a vector
	UV.resize(V.rows(), 2);
	//UV.col(0) =
	//UV.col(1) =
	for (int i = 0; i < V.rows(); i++) {
		UV(i, 0) = res(i, 0);
		UV(i, 1) = res(i + V.rows(), 0);
	}
}

bool callback_key_pressed(Viewer &viewer, unsigned char key, int modifiers) {
	switch (key) {
	case '1':
	case '2':
	case '3':
	case '4':
		computeParameterization(key);
		break;
	case '5':
	{
		if (angle == true) {
			distortionType = 0;
		}
		else if (edge == true) {
			distortionType = 1;
		}
		else
		{
			distortionType = -1;
		}
		evaluateDistortion(distortionType);

		distortion = distortion / distortion.maxCoeff();
		for (int i = 0; i < F.rows(); i++) {
			distortionColors(i, 0) = 1;
			distortionColors(i, 1) = 1 - distortion[i];
			distortionColors(i, 2) = 1 - distortion[i];
		}
		break;
	}
	case '+':
		TextureResolution /= 2;
		break;
	case '-':
		TextureResolution *= 2;
		break;
	case ' ': // space bar -  switches view between mesh and parameterization
    if(showingUV)
    {
      temp2D = viewer.core;
      viewer.core = temp3D;
      showingUV = false;
    }
    else
    {
      if(UV.rows() > 0)
      {
        temp3D = viewer.core;
        viewer.core = temp2D;
        showingUV = true;
      }
      else { std::cout << "ERROR ! No valid parameterization\n"; }
    }
    break;
	}
	Redraw();
	return true;
}

bool load_mesh(string filename)
{
	igl::read_triangle_mesh(filename,V,F);
	Redraw();
	viewer.core.align_camera_center(V);
	showingUV = false;

	return true;
}

bool callback_init(Viewer &viewer)
{
	temp3D = viewer.core;
	temp2D = viewer.core;
	temp2D.orthographic = true;

	return false;
}

int main(int argc,char *argv[]) {
	if(argc != 2) {
		cout << "Usage ex3_bin <mesh.off/obj>" << endl;
		load_mesh("C:/Users/orash/Desktop/Computer Science/Geometric Processing/Project/geometryprocessing2021-Orash94/assignment3/data/bunny.off");
	}
	else
	{
		// Read points and normals
		load_mesh(argv[1]);
	}

	igl::opengl::glfw::imgui::ImGuiMenu menu;
	viewer.plugins.push_back(&menu);

	menu.callback_draw_viewer_menu = [&]()
	{
		// Draw parent menu content
		menu.draw_viewer_menu();

		// Add new group
		if (ImGui::CollapsingHeader("Parmaterization", ImGuiTreeNodeFlags_DefaultOpen))
		{
			// Expose variable directly ...
			ImGui::Checkbox("Free boundary", &freeBoundary);
			if (ImGui::Button("Angle Preservation")) {
				angle = true;
				edge = false;
			}
			if (ImGui::Button("Edge Preservation")) {
				angle = false;
				edge = true;
			}


			// TODO: Add more parameters to tweak here...
		}
	};

	viewer.callback_key_pressed = callback_key_pressed;
	viewer.callback_mouse_move = callback_mouse_move;
	viewer.callback_init = callback_init;

	viewer.launch();
}
