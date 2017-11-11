#pragma once
#include <vector>

class Matrix
{
public:
	struct Shape
	{
		size_t rows;
		size_t columns;
	};

	Matrix(const Shape & shape);
	Matrix(size_t rows, size_t columns);
	Matrix(const std::vector<std::vector<double>> & data);
	Matrix(std::vector<std::vector<double>> && data);
	Matrix(const std::vector<double> & data);
	Matrix(const Matrix & other);
	Matrix & operator=(const Matrix & other);
	Matrix & operator=(Matrix && other);

	double & operator()(size_t x, size_t y);
	const double & operator()(size_t x, size_t y) const;

	Matrix & operator+=(const Matrix & other);
	Matrix & operator-=(const Matrix & other);
	Matrix & operator*=(double scalar);
	Matrix & operator/=(double scalar);

	Matrix & operator+(const Matrix & other);
	Matrix & operator-(const Matrix & other);
	Matrix & operator*(double scalar);
	Matrix & operator/(double scalar);
	Matrix  operator*(const Matrix & other) const;

	Matrix transpose() const;
	Matrix hadamardProduct(const Matrix & other) const;

	friend std::ostream & operator<<(std::ostream & stream, const Matrix & matrix);
	friend std::istream & operator>>(std::istream & stream, Matrix & matrix);

	size_t GetNumberOfRows() const;
	size_t GetNumberOfColumns() const;
	Shape GetShape() const;

	void Reset(double value);

private:
	std::vector<std::vector<double>> m_data;
};
