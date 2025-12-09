#include <Eigen5/Eigen/Eigen>
using namespace Eigen;
namespace SciFilter
{
    VectorXd sosfiltfilt(const MatrixXd &sos, const VectorXd &x);
}