// One seller Q-learning loop for fixed rules and on-demand policy queries.
// Pausing keeps the chosen prices, RNG, Q tables and stopping clock intact.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace {
struct SellerLearning {
  static constexpr int window = 10000;
  int m, states, size, limit, stability_required;
  double alpha, delta, beta;
  std::vector<double> q;
  std::vector<int> greedy;
  std::vector<int64_t> visits;
  std::vector<int> recent;
  std::mt19937_64 rng;
  int rounds = 0, joint, next_joint = -1, actions[2] = {0, 0};
  int64_t last_change = 0, changes = 0, explored[2] = {0, 0};
  double cumulative = 0., tail_cs = 0., tail_u[2] = {0., 0.};

  SellerLearning(int prices, int memory, int steps, uint64_t seed,
                 double learning_rate, double discount, double decay,
                 const double* initial_q, int initial_joint, int stable)
      : m(prices), states(memory ? m*m : 1), size(2*states*m), limit(steps),
        stability_required(stable), alpha(learning_rate), delta(discount), beta(decay),
        q(initial_q, initial_q+size), greedy(2*states), visits(size, 0),
        recent(window, -1), rng(seed), joint(initial_joint) {
    for (int i=0; i<2; ++i)
      for (int s=0; s<states; ++s) greedy[i*states+s]=argmax(q.data()+offset(i,s));
  }
  int offset(int player, int state) const { return (player*states+state)*m; }
  int argmax(const double* row) const {
    int best=0;
    for (int a=1; a<m; ++a) if (row[a]>row[best]) best=a;
    return best;
  }
  double uniform() { return static_cast<double>(rng()>>11)*(1./9007199254740992.); }
  bool done() const {
    return rounds>=limit || (stability_required>0 && rounds-last_change>=stability_required);
  }
  // Return the first unknown realized price pair; -1 means the round limit was reached.
  int advance(int until, const double* utility, const double* surplus,
              const uint8_t* known=nullptr, const double* tape=nullptr,
              int trace_length=0, double* trace=nullptr) {
    while (rounds<until && !done()) {
      const int state=states>1 ? joint : 0;
      const double epsilon=std::exp(-beta*rounds);
      if (next_joint<0) {
        for (int i=0; i<2; ++i) {
          const double decision=tape ? tape[4*rounds+2*i] : uniform();
          const double choice=tape ? tape[4*rounds+2*i+1] : uniform();
          const bool explore=decision<epsilon;
          actions[i]=explore ? std::min(m-1,static_cast<int>(choice*m)) : greedy[i*states+state];
          explored[i]+=explore;
        }
        next_joint=actions[0]*m+actions[1];
      }
      if (known && !known[next_joint]) return next_joint;
      const int next_state=states>1 ? next_joint : 0;
      bool changed=false;
      double updated[2];
      for (int i=0; i<2; ++i) {
        const int idx=offset(i,state)+actions[i];
        const double next_max=q[offset(i,next_state)+greedy[i*states+next_state]];
        q[idx]=(1.-alpha)*q[idx]+alpha*(utility[2*next_joint+i]+delta*next_max);
        ++visits[idx];
        const int best=argmax(q.data()+offset(i,state));
        changed=changed || best!=greedy[i*states+state];
        greedy[i*states+state]=best;
        updated[i]=q[idx];
      }
      if (changed) { last_change=rounds+1; ++changes; }
      cumulative+=surplus[next_joint];
      const int old=recent[rounds%window];
      if (old>=0) {
        tail_cs-=surplus[old];
        for (int i=0; i<2; ++i) tail_u[i]-=utility[2*old+i];
      }
      recent[rounds%window]=next_joint;
      tail_cs+=surplus[next_joint];
      for (int i=0; i<2; ++i) tail_u[i]+=utility[2*next_joint+i];
      if (rounds<trace_length) {
        double* row=trace+rounds*8;
        row[0]=state; row[1]=actions[0]; row[2]=actions[1]; row[3]=next_state;
        row[4]=epsilon; row[5]=updated[0]; row[6]=updated[1]; row[7]=surplus[next_joint];
      }
      joint=next_joint;
      next_joint=-1;
      ++rounds;
    }
    return -1;
  }
  void save(double* saved_q, int* saved_joint, double* row, int64_t* saved_visits) const {
    std::copy(q.begin(),q.end(),saved_q);
    std::copy(visits.begin(),visits.end(),saved_visits);
    *saved_joint=joint;
    const int tail_n=std::min(rounds,window);
    row[0]=rounds ? cumulative/rounds : 0.; row[1]=tail_n ? tail_cs/tail_n : 0.;
    row[2]=tail_n ? tail_u[0]/tail_n : 0.; row[3]=tail_n ? tail_u[1]/tail_n : 0.;
    row[4]=rounds-last_change; row[5]=changes; row[6]=std::exp(-beta*rounds);
    row[7]=static_cast<double>(std::count_if(visits.begin(),visits.end(),[](int64_t n){return n>0;}))/size;
    row[8]=explored[0]; row[9]=explored[1];
    row[10]=*std::min_element(visits.begin(),visits.end()); row[11]=tail_n;
  }
};
}

// Fixed-table evaluations and baselines use the same loop without any pauses.
extern "C" void pricing_q_response(
    int m, int memory, int steps, uint64_t seed, double alpha, double delta,
    double beta, const double* utility, const double* surplus,
    const double* initial_q, int initial_joint, int n_checkpoints,
    const int* checkpoints, const double* random_tape, int replay,
    int trace_length, double* saved_q, int* saved_joint, double* stats,
    int64_t* saved_visits, double* trace, int stability_required, int* actual_steps) {
  SellerLearning learning(m,memory,steps,seed,alpha,delta,beta,initial_q,initial_joint,stability_required);
  for (int cp=0; cp<n_checkpoints; ++cp) {
    learning.advance(checkpoints[cp],utility,surplus,nullptr,replay ? random_tape : nullptr,trace_length,trace);
    learning.save(saved_q+cp*learning.size,saved_joint+cp,stats+cp*12,saved_visits+cp*learning.size);
  }
  *actual_steps=learning.rounds;
}
extern "C" void* pricing_q_create(int m, int steps, uint64_t seed,
    double alpha, double delta, double beta, const double* initial_q,
    int initial_joint, int stability_required) {
  return new SellerLearning(m,1,steps,seed,alpha,delta,beta,initial_q,initial_joint,stability_required);
}
extern "C" int pricing_q_advance(void* handle, const double* utility,
    const double* surplus, const uint8_t* known) {
  auto* learning=static_cast<SellerLearning*>(handle);
  return learning->advance(learning->limit,utility,surplus,known);
}
extern "C" int pricing_q_export(void* handle, double* q, int* joint,
    double* stats, int64_t* visits) {
  auto* learning=static_cast<SellerLearning*>(handle);
  learning->save(q,joint,stats,visits);
  return learning->rounds;
}
extern "C" void pricing_q_destroy(void* handle) {
  delete static_cast<SellerLearning*>(handle);
}
