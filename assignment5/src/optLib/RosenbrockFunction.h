#pragma once

#include "ObjectiveFunction.h"

class RosenbrockFunction : public ObjectiveFunction {
public:

    RosenbrockFunction() {
		a = 1; b = 100;
    }

    virtual double computeValue(const VectorXd& x) {

		// Ex 1.1
		// return f(x)
		return pow((a - x(0, 0)), 2) + b * pow((x(1, 0) - pow(x(0, 0), 2)), 2);
	}

    virtual void addGradientTo(VectorXd& grad, const VectorXd& x) {

		// Ex 1.1
		// write df/dx in `grad`
		grad(0, 0) = 2 * (x(0, 0) - a + 2 * b * pow(x(0, 0), 3) - 2 * b * x(0, 0) * x(1, 0));
		grad(1, 0) = 2 * b * (x(1, 0) - pow(x(0, 0), 2));
    }

	virtual void addHessianEntriesTo(std::vector<Tripletd>& hessianEntries, const VectorXd& x) {

		// Ex 1.2
		// write d^2f/dx^2 in `hessianEntries`

		hessianEntries.push_back(Tripletd(0, 0, 12 * b * pow(x(0), 2) - 4 * b * x(1) + 2));
		hessianEntries.push_back(Tripletd(0, 1, -4 * b * x(0)));
		hessianEntries.push_back(Tripletd(1, 0, -4 * b * x(0)));
		hessianEntries.push_back(Tripletd(1, 1, 2 * b));
    }

    double a, b;
};
