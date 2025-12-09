
#include <SciFilter.h>
#include <math.h>
#include <cassert>
using namespace std;
namespace SciFilter
{
    static MatrixXd reshape(
        const MatrixXd &A,
        Index rows,
        Index cols)
    {
        Index total = A.rows() * A.cols();
        assert(rows * cols == total);

        // Expand by line number
        vector<double> data;
        data.reserve(total);
        for (int i = 0; i < A.rows(); ++i)
        {
            for (int j = 0; j < A.cols(); ++j)
            {
                data.push_back(A(i, j));
            }
        }

        // Fill
        MatrixXd B(rows, cols);
        int idx = 0;
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                B(i, j) = data[idx++];
            }
        }

        return B;
    }

    static MatrixXd companion(
        const VectorXd &a)
    {
        Index n = a.size();
        assert(n >= 2 && a(0) != 0);
        // Row 1：-a[1:] / a[0]
        RowVectorXd first_row = -a.tail(n - 1).transpose() / a(0);
        MatrixXd c = MatrixXd::Zero(n - 1, n - 1);
        c.row(0) = first_row;
        for (int i = 1; i < n - 1; ++i)
        {
            c(i, i - 1) = 1.0;
        }
        return c;
    }

    static VectorXd lfilter_zi(
        VectorXd b,
        VectorXd a)
    {
        while (a.rows() > 1 && a[0] == 0.0)
        {
            a = a.segment(1, a.rows() - 1);
        }
        assert(a.rows() > 0);
        if (a[0] != 1.0)
        {
            b = b / a[0];
            a = a / a[0];
        }
        Index n = max(a.rows(), b.rows());
        if (a.rows() < n)
        {
            Index pd = n - a.rows();
            a.conservativeResize(n);
            a.tail(pd).setConstant(0);
        }
        else if (b.rows() < n)
        {
            Index pd = n - b.rows();
            b.conservativeResize(n);
            b.tail(pd).setConstant(0);
        }
        MatrixXd IminusA = MatrixXd::Identity(n - 1, n - 1) - companion(a).transpose();
        VectorXd B = b.segment(1, n - 1) - a.segment(1, n - 1) * b[0];
        VectorXd zi = IminusA.partialPivLu().solve(B);
        return zi;
    }

    static MatrixXd sosfilt_zi(const MatrixXd &sos)
    {
        size_t n_sections = sos.rows();
        MatrixXd zi(n_sections, 2);
        double scale = 1.0;
        int mid = 3;
        for (int i = 0; i < n_sections; ++i)
        {
            VectorXd b(mid);
            for (size_t j = 0; j < mid; j++)
            {
                b[j] = sos(i, j);
            }
            VectorXd a;
            a.resize(sos.cols() - mid);
            for (size_t j = 0; j < a.rows(); j++)
            {
                a[j] = sos(i, mid + j);
            }
            zi.row(i) = scale * lfilter_zi(b, a);
            scale *= b.sum() / a.sum();
        }
        return zi;
    }

    static VectorXd axis_slice(
        const VectorXd &a,
        int start,
        int stop,
        int step)
    {
        Index size = a.rows();
        int ss = abs(step);
        if (start < 0)
        {
            start += size;
        }
        if (stop < 0)
        {
            stop += size;
        }
        if (stop < start)
        {
            swap(start, stop);
            start++;
            stop++;
        }
        VectorXd b(stop - start);
        size_t j = start;
        for (size_t i = 0; i < b.rows(); ++i)
        {
            b[i] = a[j];
            j += ss;
        }
        if (step < 0)
        {
            b.reverseInPlace();
        }

        return b;
    }

    static VectorXd odd_ext(
        const VectorXd &x,
        const int n)
    {
        if (n < 1)
            return x;
        VectorXd left_end = axis_slice(x, 0, 1, 1);
        VectorXd left_ext = axis_slice(x, n, 0, -1);
        VectorXd right_end = axis_slice(x, -1, -2, -1);
        VectorXd right_ext = axis_slice(x, -2, -(n + 2), -1);
        VectorXd ext(left_ext.rows() + x.rows() + right_ext.rows());
        VectorXd left = 2 * VectorXd::Constant(left_ext.rows(), left_end[0]) - left_ext;
        int offset = 0;
        for (size_t i = 0; i < left.rows(); i++)
        {
            ext[offset++] = left[i];
        }
        for (size_t i = 0; i < x.rows(); i++)
        {
            ext[offset++] = x[i];
        }
        VectorXd right = 2 * VectorXd::Constant(right_ext.rows(), right_end[0]) - right_ext;
        for (size_t i = 0; i < right.rows(); i++)
        {
            ext[offset++] = right[i];
        }
        return ext;
    }

    static VectorXd _validate_pad(
        const string &padtype,
        const size_t padlen,
        const VectorXd &x,
        const int ntaps,
        int &_edge)
    {
        if (padlen == -1)
        {
            _edge = ntaps * 3;
        }
        else
        {
            _edge = padlen;
        }
        assert(padtype == "odd");
        return odd_ext(x, _edge);
    }

    static void sosfilt(
        const Ref<const MatrixXd> &sos, // (n_sections, 6)
        Ref<VectorXd> x,                // (n_signals, n_samples), modified in-place
        MatrixXd &zi                    // (n_signals, n_sections, 2), modified in-place
    )
    {
        const Index n_samples = x.rows();
        const Index n_sections = sos.rows();

        const double const_1 = 1.0;

        for (Index n = 0; n < n_samples; ++n)
        {
            // Ensure copy
            double x_cur = const_1 * x(n);

            for (Index s = 0; s < n_sections; ++s)
            {
                double x_new = sos(s, 0) * x_cur + zi(s, 0);
                zi(s, 0) = sos(s, 1) * x_cur - sos(s, 4) * x_new + zi(s, 1);
                zi(s, 1) = sos(s, 2) * x_cur - sos(s, 5) * x_new;
                x_cur = x_new;
            }

            x(n) = x_cur;
            ;
        }
    }

    static void lfilter(VectorXd b, VectorXd a, Ref<VectorXd> x, VectorXd &Z)
    {
        double a0 = a[0];
        for (Index n = 0; n < b.rows(); ++n)
        {
            b[n] /= a0;
            a[n] /= a0;
        }
        for (size_t k = 0; k < x.rows(); ++k)
        {
            double xn = x[k], y;

            if (b.rows() > 1)
            {
                y = Z[0] + b[0] * xn;
                for (size_t n = 0; n < b.rows() - 2; ++n)
                {
                    Z[n] = Z[n + 1] + xn * b[n + 1] - y * a[n + 1];
                }
                Z[b.rows() - 2] = xn * b[b.rows() - 1] - y * a[b.rows() - 1];
            }
            else
            {
                y = xn * b[0];
            }

            x[k] = y;
        }
    }

    VectorXd filtfilt(
        const VectorXd &b,
        const VectorXd &a,
        const VectorXd &x)
    {
        int edge, ntaps = max(a.rows(), b.rows());
        VectorXd ext = _validate_pad("odd", -1, x, ntaps, edge);
        VectorXd zi = lfilter_zi(b, a);
        double x_0 = ext[0];
        VectorXd zf = zi * x_0;
        lfilter(b, a, ext, zf);
        double y_0 = ext.tail(1)[0];
        zf = zi * y_0;
        VectorXd y = ext.reverse();
        lfilter(b, a, y, zf);
        y.reverseInPlace();
        if (edge > 0)
        {
            y = axis_slice(y, edge, -edge, 1);
        }
        return y;
    }

    VectorXd sosfiltfilt(
        const MatrixXd &sos,
        const VectorXd &x)
    {
        Index n_sections = sos.rows();
        int ntaps = 2 * n_sections + 1;
        ntaps -= min((sos.col(2).array() == 0).count(), (sos.col(5).array() == 0).count());
        int edge;
        VectorXd ext = _validate_pad("odd", -1, x, ntaps, edge);
        MatrixXd zi = sosfilt_zi(sos);
        zi = reshape(zi, n_sections, 2);
        double x_0 = ext[0];
        MatrixXd zf = zi * x_0;
        sosfilt(sos, ext, zf);
        double y_0 = ext.tail(1)[0];
        zf = zi * y_0;
        VectorXd y = ext.reverse();
        sosfilt(sos, y, zf);
        y.reverseInPlace();
        if (edge > 0)
        {
            y = axis_slice(y, edge, -edge, 1);
        }
        return y;
    }
}