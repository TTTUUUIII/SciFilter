#include <Eigen5/Eigen/Eigen>
using namespace Eigen;
namespace DSPFilter
{
    VectorXd sosfiltfilt(const MatrixXd &sos, const VectorXd &x);
}