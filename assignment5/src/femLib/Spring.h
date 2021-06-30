#pragma once

#include "Element.h"

/**
	This class implements the interface for an elementary energy unit. As a function of deformed, undeformed,
	and other parameters, such as boundary conditions, each class that extends this one will define a potential energy.
	The deformed energy depends on a number of nodes.
*/
class Spring : public Element {

public:
	Spring(const std::array<int, 2> &nodeIndices, const VectorXd &X)
		: nodeIndices(nodeIndices) {
	}
	virtual ~Spring() {}

	// Returns the number of nodes this unit depends on
	virtual int getNumNodes() const {
		return 2;
	}
	// Returns the global index of node `i`
	virtual int getNodeIndex(int i) const {
		return nodeIndices[i];
	}

	// Returns the element's mass
	virtual double getMass() const {
		return 0;
	}

	// Returns the energy value given deformed `x` and undeformed `X` state
	virtual double getEnergy(const VectorXd& x, const VectorXd& X) {

		// Ex 1.2
		// Task: Given `x` and `X`, return the spring energy.

		// Some notes:
		// `x` and `X` contain the current and rest positions of all
		// nodes. You can extract the position of e.g. node 0 like this:
		// Vector2d x1 = getVertex(0, x);
		// or to get the rest position of node 0:
		// Vector X1 = getVertex(0, X);
		// The spring stiffness is stored in the variable `k`.

		Vector2d x0 = getNodePos(0, x);
		Vector2d X0 = getNodePos(0, X);

		Vector2d x1 = getNodePos(1, x);
		Vector2d X1 = getNodePos(1, X);

		double l = (x1 - x0).norm();
		double L = (X1 - X0).norm();

		double epsilon = l / L - 1;

		return 0.5 * k * pow(epsilon, 2) * L;
	}

	// Adds the gradient to `grad` given deformed `x` and undeformed `X` state
	virtual void addEnergyGradientTo(const VectorXd& x, const VectorXd& X, VectorXd& grad) {

		// Ex 1.2
		// Task: Given `x` and `X`, add the gradient of the spring energy to `grad`.

		// Again, you can extract the position of e.g. node 0 like this:
		// Vector2d x1 = getVertex(0, x);
		// and the spring stiffness is stored in `k`.

		// Remember that `grad` is a vector of size 2*N, where N is the total
		// number of nodes in the system. Make sure you are writing to the
		// correct location in `grad`. To get the global index of node 0 of
		// this spring, use this function:
		// int globalIndex0 = getNodeIndex(0);
		// or for node 1
		// int globalIndex1 = getNodeIndex(1);

		Vector2d x0 = getNodePos(0, x);
		Vector2d x1 = getNodePos(1, x);
		Vector2d X0 = getNodePos(0, X);
		Vector2d X1 = getNodePos(1, X);
		int i0 = getNodeIndex(0);
		int i1 = getNodeIndex(1);

		double l0 = (x1 - x0).norm();
		double L1 = (X1 - X0).norm();
		double eps = l0 / L1 - 1;

		Vector2d f0 = -k * eps * (x1 - x0) / l0;
		Vector2d f1 = -f0;

		grad(i0 * 2) += f0(0);
		grad(i0 * 2 + 1) += f0(1);
		grad(i1 * 2) += f1(0);
		grad(i1 * 2 + 1) += f1(1);
	}

	// Adds the hessian entries to `hesEntries` given deformed `x` and undeformed `X` state
	virtual void addEnergyHessianTo(const VectorXd& x, const VectorXd& X, std::vector<Tripletd>& hesEntries) {

		// Ex 1.4
		// Task: Given `x` and `X`, add the hessian of the spring energy to `hesEntries`.

		Vector2d x0 = getNodePos(0, x);
		Vector2d x1 = getNodePos(1, x);
		Vector2d X0 = getNodePos(0, X);
		Vector2d X1 = getNodePos(1, X);
		int i0 = getNodeIndex(0);
		int i1 = getNodeIndex(1);
		Vector2d v = x1 - x0;
		double l0 = (x1 - x0).norm();
		double L1 = (X1 - X0).norm();
		double eps = l0 / L1 - 1;
		MatrixXd H;
		H.resize(2, 2);


		H = -k * (1 / L1 * (v * v.transpose()) / (v.transpose() * v) + eps / l0 * (MatrixXd::Identity(2, 2) - (v * v.transpose()) / (v.transpose() * v)));

		hesEntries.push_back(Tripletd(i0 * 2 + 0, i0 * 2 + 0, -H(0, 0)));
		hesEntries.push_back(Tripletd(i0 * 2 + 0, i0 * 2 + 1, -H(0, 1)));
		hesEntries.push_back(Tripletd(i0 * 2 + 1, i0 * 2 + 0, -H(1, 0)));
		hesEntries.push_back(Tripletd(i0 * 2 + 1, i0 * 2 + 1, -H(1, 1)));
		hesEntries.push_back(Tripletd(i0 * 2 + 0, i1 * 2 + 0, H(0, 0)));
		hesEntries.push_back(Tripletd(i0 * 2 + 0, i1 * 2 + 1, H(0, 1)));
		hesEntries.push_back(Tripletd(i0 * 2 + 1, i1 * 2 + 0, H(1, 0)));
		hesEntries.push_back(Tripletd(i0 * 2 + 1, i1 * 2 + 1, H(1, 1)));
		hesEntries.push_back(Tripletd(i1 * 2 + 0, i0 * 2 + 0, H(0, 0)));
		hesEntries.push_back(Tripletd(i1 * 2 + 0, i0 * 2 + 1, H(0, 1)));
		hesEntries.push_back(Tripletd(i1 * 2 + 1, i0 * 2 + 0, H(1, 0)));
		hesEntries.push_back(Tripletd(i1 * 2 + 1, i0 * 2 + 1, H(1, 1)));
		hesEntries.push_back(Tripletd(i1 * 2 + 0, i1 * 2 + 0, -H(0, 0)));
		hesEntries.push_back(Tripletd(i1 * 2 + 0, i1 * 2 + 1, -H(0, 1)));
		hesEntries.push_back(Tripletd(i1 * 2 + 1, i1 * 2 + 0, -H(1, 0)));
		hesEntries.push_back(Tripletd(i1 * 2 + 1, i1 * 2 + 1, -H(1, 1)));
	}

protected:
	// the collection of nodes that define the triangle element
	std::array<int, 2> nodeIndices;
	// spring stiffness
	double k = 20.0;
};
