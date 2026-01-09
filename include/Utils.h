#ifndef UTILS_H_
#define UTILS_H_

#include <Eigen/Core>
#include <cmath>

namespace ScanReconstruction {

#ifndef MIN
#define MIN(a, b) ((a < b) ? a : b)
#endif

#ifndef MAX
#define MAX(a, b) ((a < b) ? b : a)
#endif

#ifndef ABS
#define ABS(a) ((a < 0) ? -a : a)
#endif

#ifndef CLAMP
#define CLAMP(x, a, b) MAX((a), MIN((b), (x)))
#endif

inline Eigen::Matrix3f skew(const Eigen::Vector3f v) {
    Eigen::Matrix3f m;
    m << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
    return m;
}

inline float rho(float r, float huber_b) {
    float tmp = std::fabs(r) - huber_b;
    tmp = std::max(tmp, 0.0f);
    return r * r - tmp * tmp;
}

inline float rho_deriv(float r, float huber_b) { return 2.0f * CLAMP(r, -huber_b, huber_b); }

inline float rho_deriv2(float r, float huber_b) { return fabs(r) < huber_b ? 2.0f : 0.0f; }

}  // namespace ScanReconstruction
#endif
