#ifndef kokkos_solver_bfgs_h
#define kokkos_solver_bfgs_h
 
#include <deal.II/base/config.h>
 
#include <deal.II/lac/solver.h>
 
#include <deal.II/numerics/history.h>
 
#include <deal.II/optimization/line_minimization.h>
 
#include <limits>
 
#include <Kokkos_Core.hpp>

double l2norm(
  const Kokkos::View<double*> g)
{
  double g_l2_norm = 0.0;
  Kokkos::parallel_reduce(
    "compute_l2_norm", g.extent(0),
    KOKKOS_LAMBDA(const int i, double &local_sum) {
      local_sum += g(i) * g(i);
    },
    g_l2_norm);
  return std::sqrt(g_l2_norm);
}

double l1norm(
  const Kokkos::View<double*> g)
{
  double g_l1_norm = 0.0;
  Kokkos::parallel_reduce(
    "compute_l1_norm", g.extent(0),
    KOKKOS_LAMBDA(const int i, double &local_sum) {
      local_sum += std::abs(g(i));
    },
    g_l1_norm);
  return g_l1_norm;
}

void init(
  Kokkos::View<double*> src,
  const Kokkos::View<double*> trg)
{
  Kokkos::parallel_for(
    "initialize", trg.extent(0), KOKKOS_LAMBDA(const int i) {
      src(i) = trg(i);
    });
}

double dot(
  const Kokkos::View<double*> a,
  const Kokkos::View<double*> b)
{
  double result = 0.0;
  Kokkos::parallel_reduce(
    "dot_product", a.extent(0),
    KOKKOS_LAMBDA(const int i, double &local_sum) {
      local_sum += a(i) * b(i);
    },
    result);
  return result;
}

void add(
  Kokkos::View<double*> src,
  const double               &c,
  const Kokkos::View<double*> trg)
{
  Kokkos::parallel_for(
    "add", trg.extent(0), KOKKOS_LAMBDA(const int i) {
      src(i) += c * trg(i);
    });
}

void sadd(
  Kokkos::View<double*> src,
  const Kokkos::View<double*> trg,
  const double               &c1 = 1.,
  const double               &c2 = 1.)
{
  Kokkos::parallel_for(
    "sadd", trg.extent(0), KOKKOS_LAMBDA(const int i) {
      src(i) += c2 * trg(i) + c1 * src(i);
    });
}

void negate(
  Kokkos::View<double*> src)
{
  Kokkos::parallel_for(
    "negate", src.extent(0), KOKKOS_LAMBDA(const int i) {
      src(i) *= -1.;
    });
}

class SolverKokkosBFGS
{
public: 
 
  struct AdditionalData
  {
    explicit AdditionalData(const unsigned int max_history_size = 5,
                            const bool         debug_output     = false);
 
    unsigned int max_history_size;
    bool debug_output;
  };

  struct BFGSControl
  {
    enum class Status
    {
      success,
      failure,
      iterate
    };

    explicit BFGSControl(const unsigned int max_iter = 100,
                          const double tolerance = 1e-6);
      
    Status check(const unsigned int step, const double check_value);

    unsigned int max_iter;
    double tolerance;
    double initial_val;
  };
 
 
  explicit SolverKokkosBFGS(BFGSControl        &residual_control,
                      const AdditionalData &data = AdditionalData());
 
  void
  solve(
    const std::function<double(
      const Kokkos::View<double*> x, Kokkos::View<double*> g)> &compute,
    Kokkos::View<double*> x);
  
  double line_search(
    const std::function<double(
      const Kokkos::View<double*> x, Kokkos::View<double*> g)> &compute,
    bool   &first_step,
    double &f,
    double &f_prev,
    Kokkos::View<double*> x,
    Kokkos::View<double*> g,
    Kokkos::View<double*> p);
 
 
protected:
  const AdditionalData additional_data;
  BFGSControl solver_control;
};
 
 
// -------------------  inline and template functions ----------------
SolverKokkosBFGS::AdditionalData::AdditionalData(
  const unsigned int max_history_size_,
  const bool         debug_output_)
  : max_history_size(max_history_size_)
  , debug_output(debug_output_)
{}

SolverKokkosBFGS::BFGSControl::BFGSControl(
  const unsigned int max_iter_, const double tolerance_)
  : max_iter(max_iter_)
  , tolerance(tolerance_)
{}
 
 
SolverKokkosBFGS::SolverKokkosBFGS(
  BFGSControl        &solver_control, const AdditionalData &data)
  : additional_data(data),
    solver_control(solver_control)
{}
 
double SolverKokkosBFGS::line_search(
  const std::function<double(
    const Kokkos::View<double*> x, Kokkos::View<double*> g)> &compute,
  bool   &first_step,
  double &f,
  double &f_prev,
  Kokkos::View<double*> x,
  Kokkos::View<double*> g,
  Kokkos::View<double*> p) {

  Kokkos::View<double*> x0("x0", x.extent(0));
  init(x0, x);

  const double f0 = f;
  double g0 = dot(g, p);

  Assert(g0 < 0,
          ExcMessage(
            "Function does not decrease along the current direction"));

  // see scipy implementation
  // https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.line_search.html#scipy.optimize.line_search
  // and Eq. 2.6.8 in Fletcher 2013, Practical methods of optimization
  double df = f_prev - f;

  Assert(first_step || df >= 0.,
          ExcMessage("Function value is not decreasing"));

  df = std::max(df, 100. * std::numeric_limits<double>::epsilon());

  // guess a reasonable first step:
  const double a1 =
    (first_step ? 1. : std::min(1., -1.01 * 2. * df / g0));

  Assert(a1 > 0., ExcInternalError());

  f_prev = f;

  // 1d line-search function
  Kokkos::View<double*> x1("x1", x.extent(0));
  const auto line_func =
    [x, x0, g, p, &f, &compute](const double &x_line) -> std::pair<double, double> {
    init(x, x0);
    add(x, x_line, p);
    f = compute(x, g);
    double g_line = dot(g, p);
    return std::make_pair(f, g_line);
  };

  // loose line search:
  std::pair<double, int> res = LineMinimization::line_search<double>(
    line_func,
    f0,
    g0,
    LineMinimization::poly_fit<double>,
    a1,
    0.9,
    0.001);

  if (first_step)
    first_step = false;

  return res.first;
}

SolverKokkosBFGS::BFGSControl::Status SolverKokkosBFGS::BFGSControl::check(const unsigned int step, const double check_value)
{
  // if this is the first time we
  // come here, then store the
  // residual for later comparisons
  if (step == 0)
    initial_val = check_value;
    
  if (check_value <= tolerance)
    return BFGSControl::Status::success;
 
  if ((step >= max_iter) || numbers::is_nan(check_value))
    return BFGSControl::Status::failure;
 
  return BFGSControl::Status::iterate;
}

void
SolverKokkosBFGS::solve(
  const std::function<double(
    const Kokkos::View<double*> x,
    Kokkos::View<double*> f)> &compute,
  Kokkos::View<double*> x)
{
  // Also see scipy Fortran implementation
  // https://github.com/scipy/scipy/blob/master/scipy/optimize/lbfgsb_src/lbfgsb.f
  // and Octave-optim implementation:
  // https://sourceforge.net/p/octave/optim/ci/default/tree/src/__bfgsmin.cc
 
  // default line search:
  bool   first_step = true;
  double f_prev     = 0.;  
 
  // FIXME: Octave has convergence in terms of:
  // function change tolerance, default 1e-12
  // parameter change tolerance, default 1e-6
  // gradient tolerance, default 1e-5
  // SolverBase and/or BFGSControl need extension
 
  Kokkos::View<double*> g("g", x.extent(0));
  Kokkos::View<double*> p("p", x.extent(0));
  Kokkos::View<double*> y_k("y_k", x.extent(0));
  Kokkos::View<double*> s_k("s_k", x.extent(0));
 
  std::vector<double> c1;
  c1.reserve(additional_data.max_history_size);
 
  // limited history
  std::vector<Kokkos::View<double*>> y(additional_data.max_history_size);
  std::vector<Kokkos::View<double*>> s(additional_data.max_history_size);
  std::vector<double>     rho(additional_data.max_history_size);
 
  unsigned int m = 0;
  double       f;
 
  unsigned int         k    = 0;
 
  f = compute(x, g);
 
  double g_l2_norm = l2norm(g);
  double g_l1_norm = l1norm(g);
  auto conv = solver_control.check(k, g_l1_norm);
  
  if (conv != BFGSControl::Status::iterate)
    return;
 
  while (conv == BFGSControl::Status::iterate)
  {
    Kokkos::Timer timer;
    // 1. Two loop recursion to calculate p = - H*g
    c1.resize(m);
    init(p, g);

    // first loop:
    for (unsigned int i = 0; i < m; ++i)
    {
      c1[i] = dot(s[i], p);
      c1[i] *= rho[i];
      add(p, c1[i], y[i]);
    }

    // second loop:
    for (int i = m - 1; i >= 0; --i)
    {
      Assert(i >= 0, ExcInternalError());
      double c2 = 0.0;
      c2 = dot(y[i], p);
      c2 *= rho[i];
      add(p, c1[i] - c2, s[i]);
    }
    negate(p);

    // 2. Line search
    init(s_k, x);
    init(y_k, g);

    double alpha = line_search(compute, first_step, f, f_prev, x, g, p);

    sadd(s_k, x, -1, 1);
    sadd(y_k, g, -1, 1);

    // 3. Check convergence
    ++k;
    double g_l2 = l2norm(g);
    double g_l1 = l1norm(g);
    conv = solver_control.check(k, g_l1);
    if (conv != BFGSControl::Status::iterate)
      break;

    // 4. Store s, y, rho
    double curvature = dot(s_k, y_k);

    if (curvature > 0. && additional_data.max_history_size > 0)
    {
      Kokkos::View<double*> s_k_copy("s_k_copy", s_k.extent(0));
      Kokkos::View<double*> y_k_copy("y_k_copy", y_k.extent(0));

      init(s_k_copy, s_k);
      init(y_k_copy, y_k);

      if (s.size() < additional_data.max_history_size)
      {
        s.push_back(s_k_copy);
        y.push_back(y_k_copy);
        rho.push_back(1. / curvature);
      }
      else
      {
        s.erase(s.begin());
        y.erase(y.begin());
        rho.erase(rho.begin());

        s.push_back(s_k_copy);
        y.push_back(y_k_copy);
        rho.push_back(1. / curvature);
      }
      m = s.size();

      Assert(y.size() == m, ExcInternalError());
      Assert(rho.size() == m, ExcInternalError());
    }

    Assert(m <= additional_data.max_history_size, ExcInternalError());

    g_l2_norm = l2norm(g);
    g_l1_norm = l1norm(g);
    double elapsed = timer.seconds();
    std::cout << "Iteration " << k << " in " << elapsed*1e6 << " mus, "
              << " L2-norm grad: " << g_l2_norm
              << ", L1-norm grad: " << g_l1_norm << std::endl;
  }
}
   
#endif