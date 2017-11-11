#include "Matrix.h"
#include <ostream>
#include <istream>
#include <stdexcept>
#include <iomanip>

Matrix::Matrix(size_t rows, size_t columns)
{
	if (rows == 0 || columns == 0)
		throw std::runtime_error("invalid dimensions");

	for (size_t i = 0; i < rows; ++i)
		m_data.push_back(std::vector<double>(columns, 0.0));
}

Matrix::Matrix(const Shape & shape)
	: Matrix(shape.rows, shape.columns)
{

}

Matrix::Matrix(const std::vector<std::vector<double>> & data)
	: m_data(data)
{
	if (m_data.size() == 0 || m_data[0].size() == 0)
		throw std::runtime_error("invalid dimensions");
}

Matrix::Matrix(std::vector<std::vector<double>> && data)
	: m_data(std::move(data))
{
	if (m_data.size() == 0 || m_data[0].size() == 0)
		throw std::runtime_error("invalid dimensions");

}

Matrix::Matrix(const std::vector<double> & data)
{
	for (size_t i = 0; i < data.size(); ++i)
		m_data.push_back({data[i]});
}

Matrix::Matrix(const Matrix & other)
	: m_data(other.m_data)
{
}

Matrix & Matrix::operator=(const Matrix & other)
{
	m_data = other.m_data;

	return *this;
}

Matrix & Matrix::operator=(Matrix && other)
{
	m_data = std::move(other.m_data);

	return *this;
}

double& Matrix::operator()(size_t x, size_t y)
{
	return m_data[x][y];
}

const double & Matrix::operator()(size_t x, size_t y) const
{
	return m_data[x][y];
}

std::ostream & operator<<(std::ostream & stream, const Matrix & matrix)
{
	for (size_t i = 0; i < matrix.m_data.size(); ++i)
	{
		for (size_t j = 0; j < matrix.m_data[i].size(); ++j)
		{
			stream << std::fixed << std::setprecision(2);
			stream << matrix.m_data[i][j] << " ";
		}
		stream << std::endl;
	}

	return stream;
}

std::istream & operator>>(std::istream & stream, Matrix & matrix)
{
	for (size_t i = 0; i < matrix.m_data.size(); ++i)
		for (size_t j = 0; j < matrix.m_data[i].size(); ++j)
			stream >> matrix.m_data[i][j];

	return stream;
}

Matrix & Matrix::operator+=(const Matrix & other)
{
	if (m_data.size() != other.m_data.size() || m_data[0].size() != other.m_data[0].size())
		throw std::runtime_error("invalid dimensions");

	for (size_t i = 0; i < other.m_data.size(); ++i)
		for (size_t j = 0; j < other.m_data[i].size(); ++j)
			m_data[i][j] += other.m_data[i][j];

	return *this;
}

Matrix & Matrix::operator-=(const Matrix & other)
{
	if (m_data.size() != other.m_data.size() || m_data[0].size() != other.m_data[0].size())
		throw std::runtime_error("invalid dimensions");

	for (size_t i = 0; i < other.m_data.size(); ++i)
		for (size_t j = 0; j < other.m_data[i].size(); ++j)
			m_data[i][j] -= other.m_data[i][j];

	return *this;
}

Matrix & Matrix::operator*=(double scalar)
{
	for (size_t i = 0; i < m_data.size(); ++i)
		for (size_t j = 0; j < m_data[i].size(); ++j)
			m_data[i][j] *= scalar;

	return *this;
}

Matrix & Matrix::operator/=(double scalar)
{
	for (size_t i = 0; i < m_data.size(); ++i)
		for (size_t j = 0; j < m_data[i].size(); ++j)
			m_data[i][j] /= scalar;

	return *this;
}

Matrix Matrix::transpose()
{
	Matrix ret(m_data[0].size(), m_data.size());
	for (size_t i = 0; i < m_data.size(); ++i)
		for (size_t j = 0; j < m_data[i].size(); ++j)
			ret.m_data[j][i] = m_data[i][j];

	return ret;
}

Matrix & Matrix::operator+(const Matrix & other)
{
	if (m_data.size() != other.m_data.size() || m_data[0].size() != other.m_data[0].size())
		throw std::runtime_error("invalid dimensions");

	for (size_t i = 0; i < other.m_data.size(); ++i)
		for (size_t j = 0; j < other.m_data[i].size(); ++j)
			m_data[i][j] += other.m_data[i][j];

	return *this;
}

Matrix & Matrix::operator-(const Matrix & other)
{
	if (m_data.size() != other.m_data.size() || m_data[0].size() != other.m_data[0].size())
		throw std::runtime_error("invalid dimensions");

	for (size_t i = 0; i < other.m_data.size(); ++i)
		for (size_t j = 0; j < other.m_data[i].size(); ++j)
			m_data[i][j] -= other.m_data[i][j];

	return *this;
}

Matrix & Matrix::operator*(double scalar)
{
	for (size_t i = 0; i < m_data.size(); ++i)
		for (size_t j = 0; j < m_data[i].size(); ++j)
			m_data[i][j] *= scalar;

	return *this;
}

Matrix & Matrix::operator/(double scalar)
{
	for (size_t i = 0; i < m_data.size(); ++i)
		for (size_t j = 0; j < m_data[i].size(); ++j)
			m_data[i][j] /= scalar;

	return *this;
}

Matrix Matrix::operator*(const Matrix & other)
{
	if (m_data[0].size() != other.m_data.size())
		throw std::runtime_error("invalid dimensions");

	Matrix ret(m_data.size(), other.m_data[0].size());

	for (size_t row = 0; row < m_data.size(); ++row)
	{
		for (size_t column = 0; column < other.m_data[0].size(); ++column)
		{
			// row of this and column of other should have same number of elements -> C
			for (size_t i = 0; i < other.m_data.size(); ++i)
			{
				ret(row, column) += m_data[row][i] * other(i, column);
			}
		}
	}

	return ret;
}

Matrix Matrix::hadamardProduct(const Matrix & other)
{
	if (m_data.size() != other.m_data.size() || m_data[0].size() != other.m_data[0].size())
		throw std::runtime_error("invalid dimensions");

	Matrix ret(m_data);

	for (size_t i = 0; i < m_data.size(); ++i)
		for (size_t j = 0; j < m_data[i].size(); ++j)
			ret.m_data[i][j] *= other.m_data[i][j];

	return ret;
}

size_t Matrix::GetNumberOfRows() const
{
	return m_data.size();
}

size_t Matrix::GetNumberOfColumns() const
{
	return m_data[0].size();
}

void Matrix::Reset(double value)
{
	for (size_t i = 0; i < m_data.size(); ++i)
		for (size_t j = 0; j < m_data[i].size(); ++j)
			m_data[i][j] = value;
}

Matrix::Shape  Matrix::GetShape() const
{
	return { m_data.size(), m_data[0].size() };
}
