// Exact interim coarse-correlated-equilibrium gap for two finite followers.
// Python supplies the fixed-commitment payoff table and the empirical mixture
// of product-strategy snapshots. The loop order mirrors the reference Python
// implementation so the native kernel changes execution cost, not semantics.
#include <algorithm>
#include <cmath>
#include <vector>

extern "C" double exact_bcce_gap_two_player(
    int t0_count, int t1_count, int a0_count, int a1_count,
    int snapshots, const double* strategy0, const double* strategy1,
    const double* type_probability0, const double* type_probability1,
    const double* utilities) {
  auto p0 = [=](int s, int t, int a) {
    return strategy0[(s * t0_count + t) * a0_count + a];
  };
  auto p1 = [=](int s, int t, int a) {
    return strategy1[(s * t1_count + t) * a1_count + a];
  };
  auto utility = [=](int t0, int t1, int a0, int a1, int player) {
    return utilities[
        ((((t0 * t1_count + t1) * a0_count + a0) * a1_count + a1)
         * 2) + player
    ];
  };

  const double snapshot_weight = 1.0 / static_cast<double>(snapshots);
  double max_gap = 0.0;

  // Follower 0: own type, snapshot, other type, other action, own action.
  for (int t0 = 0; t0 < t0_count; ++t0) {
    double current = 0.0;
    std::vector<double> deviations(a0_count, 0.0);
    for (int s = 0; s < snapshots; ++s) {
      for (int t1 = 0; t1 < t1_count; ++t1) {
        const double type_weight = type_probability1[t1];
        for (int a1 = 0; a1 < a1_count; ++a1) {
          const double other_action_probability = p1(s, t1, a1);
          for (int a0 = 0; a0 < a0_count; ++a0) {
            current += utility(t0, t1, a0, a1, 0)
                * other_action_probability * p0(s, t0, a0)
                * type_weight * snapshot_weight;
          }
          for (int deviation = 0; deviation < a0_count; ++deviation) {
            deviations[deviation] += utility(t0, t1, deviation, a1, 0)
                * other_action_probability * type_weight * snapshot_weight;
          }
        }
      }
    }
    max_gap = std::max(
        max_gap,
        *std::max_element(deviations.begin(), deviations.end()) - current
    );
  }

  // Follower 1: own type, snapshot, other type, other action, own action.
  for (int t1 = 0; t1 < t1_count; ++t1) {
    double current = 0.0;
    std::vector<double> deviations(a1_count, 0.0);
    for (int s = 0; s < snapshots; ++s) {
      for (int t0 = 0; t0 < t0_count; ++t0) {
        const double type_weight = type_probability0[t0];
        for (int a0 = 0; a0 < a0_count; ++a0) {
          const double other_action_probability = p0(s, t0, a0);
          for (int a1 = 0; a1 < a1_count; ++a1) {
            current += utility(t0, t1, a0, a1, 1)
                * other_action_probability * p1(s, t1, a1)
                * type_weight * snapshot_weight;
          }
          for (int deviation = 0; deviation < a1_count; ++deviation) {
            deviations[deviation] += utility(t0, t1, a0, deviation, 1)
                * other_action_probability * type_weight * snapshot_weight;
          }
        }
      }
    }
    max_gap = std::max(
        max_gap,
        *std::max_element(deviations.begin(), deviations.end()) - current
    );
  }

  return std::max(0.0, max_gap);
}
