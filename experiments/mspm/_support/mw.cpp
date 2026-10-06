// MSPM follower computation after the queried payoff table is complete.
// Python retains query scheduling and its exact random.Random type stream.
// No fast-math: candidate order, tie breaking and update order are scientific.
#include <algorithm>
#include <cmath>
#include <vector>

extern "C" double exact_bcce_gap_two_player(
    int, int, int, int, int, const double*, const double*,
    const double*, const double*, const double*);

namespace {
struct Response {
  int types, messages, updates = 1, selected = -1, checked = 0;
  double epsilon, tolerance;
  std::vector<double> weights, utilities, probability0, probability1;
  std::vector<double> empirical, deviations, history0, history1, gaps;

  Response(int t, int m, const double* w, const double* u,
           const double* p0, const double* p1, const double* e,
           const double* d, double rate, double threshold)
      : types(t), messages(m), epsilon(rate), tolerance(threshold),
        weights(w, w + 2*t*m), utilities(u, u + t*t*m*m*2),
        probability0(p0, p0+t), probability1(p1, p1+t),
        empirical(e, e + t*t*m*m), deviations(d, d+2*m),
        history0(w, w+t*m), history1(w+t*m, w+2*t*m), gaps(3) {}

  int row(int player, int type) const {
    return (player*types + type)*messages;
  }

  int profile(int t0, int t1, int a0, int a1) const {
    return ((t0*types + t1)*messages + a0)*messages + a1;
  }

  std::vector<double> joint(const std::vector<double>& strategy) const {
    std::vector<double> result(empirical.size());
    for (int t0=0; t0<types; ++t0)
      for (int t1=0; t1<types; ++t1)
        for (int a0=0; a0<messages; ++a0)
          for (int a1=0; a1<messages; ++a1)
            result[profile(t0,t1,a0,a1)] =
                strategy[row(0,t0)+a0] * strategy[row(1,t1)+a1];
    return result;
  }

  std::vector<double> deterministic() const {
    std::vector<double> result(weights.size(), 0.0);
    for (int player=0; player<2; ++player)
      for (int type=0; type<types; ++type) {
        const int begin = row(player,type);
        const auto first = weights.begin()+begin;
        const int best = std::max_element(first, first+messages)-first;
        result[begin+best] = 1.0;  // first argmax, including ties
      }
    return result;
  }

  double gap(const double* first, const double* second, int snapshots) const {
    return exact_bcce_gap_two_player(
        types, types, messages, messages, snapshots, first, second,
        probability0.data(), probability1.data(), utilities.data());
  }

  int certify() {
    checked = 1;
    gaps[0] = gap(weights.data(), weights.data()+types*messages, 1);
    if (gaps[0] <= tolerance) return selected = 0;
    const auto pure = deterministic();
    checked = 2;
    gaps[1] = gap(pure.data(), pure.data()+types*messages, 1);
    if (gaps[1] <= tolerance) return selected = 1;
    checked = 3;
    gaps[2] = gap(history0.data(), history1.data(), updates);
    if (gaps[2] <= tolerance) return selected = 2;
    return selected = -1;
  }

  int advance(int type0, int type1) {
    if (type0<0 || type0>=types || type1<0 || type1>=types) return -3;
    std::fill(deviations.begin(), deviations.end(), 0.0);
    // Both counterfactual expectations use the pre-update strategy.
    for (int a0=0; a0<messages; ++a0)
      for (int a1=0; a1<messages; ++a1) {
        const int entry = profile(type0,type1,a0,a1)*2;
        deviations[a0] += utilities[entry] * weights[row(1,type1)+a1];
        deviations[messages+a1] += utilities[entry+1] * weights[row(0,type0)+a0];
      }
    const int sampled[2] = {type0, type1};
    for (int player=0; player<2; ++player) {
      const int begin = row(player,sampled[player]);
      double total = 0.0;
      for (int action=0; action<messages; ++action) {
        weights[begin+action] *= std::pow(1.0+epsilon, deviations[player*messages+action]);
        total += weights[begin+action];
      }
      if (!std::isfinite(total) || total<=0.0) return -3;
      for (int action=0; action<messages; ++action) weights[begin+action] /= total;
    }
    ++updates;
    const auto mixed = joint(weights);
    for (size_t i=0; i<empirical.size(); ++i)
      empirical[i] += (mixed[i]-empirical[i])/updates;
    history0.insert(history0.end(), weights.begin(), weights.begin()+types*messages);
    history1.insert(history1.end(), weights.begin()+types*messages, weights.end());
    return certify();
  }
};
}  // namespace

extern "C" void* mspm_mw_create(
    int types, int messages, const double* weights, const double* utilities,
    const double* p0, const double* p1, const double* empirical,
    const double* deviations, double epsilon, double tolerance) {
  try {
    return new Response(types, messages, weights, utilities, p0, p1,
                        empirical, deviations, epsilon, tolerance);
  } catch (...) { return nullptr; }
}

extern "C" int mspm_mw_certify(void* handle) {
  try { return static_cast<Response*>(handle)->certify(); }
  catch (...) { return -2; }
}

extern "C" int mspm_mw_advance(void* handle, int type0, int type1) {
  try { return static_cast<Response*>(handle)->advance(type0,type1); }
  catch (...) { return -2; }
}

extern "C" int mspm_mw_export(
    void* handle, double* weights, double* empirical, double* deviations,
    double* selected_joint, double* gaps) {
  try {
    const auto& state = *static_cast<Response*>(handle);
    std::copy(state.weights.begin(),state.weights.end(),weights);
    std::copy(state.empirical.begin(),state.empirical.end(),empirical);
    std::copy(state.deviations.begin(),state.deviations.end(),deviations);
    std::copy(state.gaps.begin(),state.gaps.begin()+state.checked,gaps);
    if (state.selected>=0) {
      const auto joint = (state.selected==2 ? state.empirical :
          state.joint(state.selected==1 ? state.deterministic() : state.weights));
      std::copy(joint.begin(),joint.end(),selected_joint);
    }
    return state.checked;
  } catch (...) { return -2; }
}

extern "C" void mspm_mw_destroy(void* handle) {
  delete static_cast<Response*>(handle);
}
