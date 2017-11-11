#pragma once
#include <vector>
#include <cassert>
#include <istream>
#include <ostream>

template <size_t R, size_t C>
class MatrixT
{
public:
	MatrixT();
	MatrixT(const std::vector<std::vector<double>> & data);
	MatrixT(const MatrixT & other);
	MatrixT & operator=(const MatrixT & other);

	double & operator()(size_t x, size_t y);
	const double & operator()(size_t x, size_t y) const;

	MatrixT & operator+=(const MatrixT & other);
	MatrixT & operator-=(const MatrixT & other);
	MatrixT & operator*=(double scalar);
	MatrixT & operator/=(double scalar);

	MatrixT & operator+(const MatrixT & other);
	MatrixT & operator-(const MatrixT & other);
	MatrixT & operator*(double scalar);
	MatrixT & operator/(double scalar);
	template<size_t T>
	MatrixT<R, T> operator*(const MatrixT<C, T> & other);

	MatrixT<C, R> transpose();

	friend std::ostream & operator<<(std::ostream & stream, const MatrixT<R, C> & matrix);
	friend std::istream & operator>>(std::istream & stream, MatrixT<R, C> & matrix);

private:
	double m_data[R][C];
};

///////////////////////////////////////////////////////////////////////////////
// implementation
///////////////////////////////////////////////////////////////////////////////

template <size_t R, size_t C>
MatrixT<R, C>::MatrixT()
{
	for (size_t i = 0; i < R; ++i)
		for (size_t j = 0; j < C; ++j)
			m_data[i][j] = 0.0;
}

template <size_t R, size_t C>
MatrixT<R, C>::MatrixT(const std::vector<std::vector<double>> & data)
{
	assert(data.size() == R);
	assert(data[0].size() == C);

	for (size_t row = 0; row < data.size(); ++row)
	{
		for (size_t column = 0; column < data[row].size(); ++column)
			m_data[row][column] = data[row][column];
	}
}

template <size_t R, size_t C>
MatrixT<R, C>::MatrixT(const MatrixT<R, C> & other)
{
	for (size_t i = 0; i < R; ++i)
		for (size_t j = 0; j < C; ++j)
			m_data[i][j] = other.m_data[i][j];
}

template <size_t R, size_t C>
MatrixT<R, C> & MatrixT<R, C>::operator=(const MatrixT<R, C> & other)
{
	for (size_t i = 0; i < R; ++i)
		for (size_t j = 0; j < C; ++j)
			m_data[i][j] = other.m_data[i][j];
	return *this;
}

template <size_t R, size_t C>
double& MatrixT<R, C>::operator()(size_t x, size_t y)
{ 
	return m_data[x][y]; 
}

template <size_t R, size_t C>
const double & MatrixT<R, C>::operator()(size_t x, size_t y) const
{
	return m_data[x][y];
}

template <size_t R, size_t C>
std::ostream & operator<<(std::ostream & stream, const MatrixT<R, C> & matrix)
{
	for (size_t i = 0; i < R; ++i)
	{
		for (size_t j = 0; j < C; ++j)
			stream << matrix.m_data[i][j] << " ";
		stream << std::endl;
	}

	return stream;
}

template <size_t R, size_t C>
std::istream & operator>>(std::istream & stream, MatrixT<R, C> & matrix)
{
	for (size_t i = 0; i < R; ++i)
		for (size_t j = 0; j < C; ++j)
			stream >> matrix.m_data[i][j];

	return stream;
}

template <size_t R, size_t C>
MatrixT<R, C> & MatrixT<R, C>::operator+=(const MatrixT<R, C> & other)
{
	for (size_t i = 0; i < R; ++i)
		for (size_t j = 0; j < C; ++j)
			m_data[i][j] += other.m_data[i][j];

	return *this;
}

template <size_t R, size_t C>
MatrixT<R, C> & MatrixT<R, C>::operator-=(const MatrixT<R, C> & other)
{
	for (size_t i = 0; i < R; ++i)
		for (size_t j = 0; j < C; ++j)
			m_data[i][j] -= other.m_data[i][j];

	return *this;
}

template <size_t R, size_t C>
MatrixT<R, C> & MatrixT<R, C>::operator*=(double scalar)
{
	for (size_t i = 0; i < R; ++i)
		for (size_t j = 0; j < C; ++j)
			m_data[i][j] *= scalar;

	return *this;
}

template <size_t R, size_t C>
MatrixT<R, C> & MatrixT<R, C>::operator/=(double scalar)
{
	for (size_t i = 0; i < R; ++i)
		for (size_t j = 0; j < C; ++j)
			m_data[i][j] /= scalar;

	return *this;
}

template <size_t R, size_t C>
MatrixT<C, R> MatrixT<R, C>::transpose()
{
	MatrixT<C, R> ret;
	for (size_t i = 0; i < R; ++i)
		for (size_t j = 0; j < C; ++j)
			ret.m_data[j][i] = m_data[i][j];

	return ret;
}

template <size_t R, size_t C>
MatrixT<R, C> & MatrixT<R, C>::operator+(const MatrixT<R, C> & other)
{
	for (size_t i = 0; i < R; ++i)
		for (size_t j = 0; j < C; ++j)
			m_data[i][j] += other[i][j];

	return *this;
}

template <size_t R, size_t C>
MatrixT<R, C> & MatrixT<R, C>::operator-(const MatrixT<R, C> & other)
{
	for (size_t i = 0; i < R; ++i)
		for (size_t j = 0; j < C; ++j)
			m_data[i][j] -= other[i][j];

	return *this;
}

template <size_t R, size_t C>
MatrixT<R, C> & MatrixT<R, C>::operator*(double scalar)
{
	for (size_t i = 0; i < R; ++i)
		for (size_t j = 0; j < C; ++j)
			m_data[i][j] *= scalar;

	return *this;
}

template <size_t R, size_t C>
MatrixT<R, C> & MatrixT<R, C>::operator/(double scalar)
{
	for (size_t i = 0; i < R; ++i)
		for (size_t j = 0; j < C; ++j)
			m_data[i][j] /= scalar;

	return *this;
}

template<size_t R, size_t C>
template<size_t T>
inline MatrixT<R, T> MatrixT<R, C>::operator*(const MatrixT<C, T>& other)
{
	MatrixT<R, T> ret;

	for (size_t row = 0; row < R; ++row)
	{
		for (size_t column = 0; column < T; ++column)
		{
			// row of this and column of other should have same number of elements -> C
			for (size_t i = 0; i < C; ++i)
			{
				ret(row, column) += m_data[row][i] * other(i, column);
			}
		}
	}

	return ret;
}
