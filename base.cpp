#include <iostream>
#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <array>
#include <unordered_map>
#include <cstdint>
#include <stdexcept>
#include <cstddef>  // 为 std::ptrdiff_t 准备
#include <limits>
#include <numeric>
#include "BSP-OT_header_only.h"
#define _USE_MATH_DEFINES
#include <functional>
#include <queue>
#include <stack>
#include <unordered_map>

// split
class comp
{
    friend class Hilbert_sort;
    public:

      int current_axis1;
      bool sign1;
      const Eigen::ArrayXXd & X1;


    public:

      comp (int current_axis11, bool sign11, const Eigen::ArrayXXd & X11): current_axis1(current_axis11), sign1(sign11), X1(X11){}
      
      bool operator()(const int &i1, const int &i2)
      {
        return (sign1 ? ( X1(i1,current_axis1) >  X1(i2,current_axis1) )
                      : ( X1(i1,current_axis1) <  X1(i2,current_axis1) ));
      }


};


//using recursive sort algorithm from CGAL library
class Hilbert_sort
{
  public:

    const Eigen::ArrayXXd & X;
    int d;
    int n;
    ptrdiff_t pow_d;


  public:

    Hilbert_sort (const Eigen::ArrayXXd & X1):X(X1){}


    void hilbert_median_sort_d(std::vector<ptrdiff_t>::iterator begin, std::vector<ptrdiff_t>::iterator end, int first_axis, std::vector<bool> signs){
        
        if (end - begin <= 1) return;

        int i = 0, ii = 0, j = 0, jj = 0;
        bool sign = false;
        int _d = d;
        ptrdiff_t _pow_d = pow_d;

        if ((end - begin) < (_pow_d/2)) { 
          _pow_d = 1;
          _d = 0;
          while ( (end-begin) > _pow_d) {
            _d++;
            _pow_d *= 2;
          }
        }

        
        // split at 1+2+4+...+2^{_d-1} index and assign the first axis
        std::vector<std::vector<ptrdiff_t>::iterator> split_index(_pow_d+1);
        split_index[0] = begin;
        split_index[_pow_d] = end;

        std::vector<int> axiss(_pow_d+1);
        int current_axis = first_axis;
        int current_step = _pow_d;
        int last_step = 0;

        for(i=0; i<_d; i++){
          last_step = current_step;
          current_step = current_step/2;
          sign = signs[current_axis]; 
          for(j=0; j<pow(2,i); j++){
            jj = current_step + last_step * j;
            axiss[jj] = current_axis;
            if(split_index[jj-current_step] >= split_index[jj+current_step])split_index[jj] = split_index[jj - current_step];
            else{                        
              std::vector<ptrdiff_t>::iterator _med = split_index[jj-current_step] + (split_index[jj+current_step] - split_index[jj-current_step]) / 2;
              comp cmp(current_axis, sign, X);
              std::nth_element (split_index[jj-current_step], _med, split_index[jj+current_step], cmp);
              split_index[jj] = _med;
            }
            sign = !sign;
          }
          current_axis = (current_axis+1)%d;
        }


        if((end-begin)<pow_d) return;

        // perform recursive sort
        int last_axis = (first_axis+d-1)%d;
        hilbert_median_sort_d(split_index[0], split_index[1], last_axis, signs);

        for(i=1; i<pow_d-1; i=i+2){
          ii = axiss[i+1];
          hilbert_median_sort_d(split_index[i], split_index[i+1], ii, signs);
          hilbert_median_sort_d(split_index[i+1], split_index[i+2], ii, signs);
          signs[ii] = !signs[ii];
          signs[last_axis] = !signs[last_axis];
        }

       hilbert_median_sort_d(split_index[pow_d-1], split_index[pow_d], last_axis, signs);

    }

    void Hilbert_sort_median_d(std::vector<ptrdiff_t>::iterator begin, std::vector<ptrdiff_t>::iterator end)
    {
      n = X.rows() * 2;
      d = X.cols();
      pow_d = 1;
      int i=0;
      std::vector<bool> direction(d,false);

      for (i=0; i<d; i++) {
        pow_d *= 2;        
        n/=2;
        if(n==0)
          break;
      }

      hilbert_median_sort_d (begin, end, 0, direction);
    }

};


// =====  ==========  ==========  ==========  ==========  =====

// ==========================================
// -------- NEW general-d H ------------


using index_t = std::ptrdiff_t;

static void RRM_md_rec(
    const Eigen::ArrayXXd& X,
    std::vector<index_t>& idx,
    index_t lo,
    index_t hi,
    int axis,
    std::vector<bool>& signs)
{
    index_t n = hi - lo;
    if (n <= 1) return;

    const int d = X.cols();
    bool sign = signs[axis];
    index_t mid = lo + n / 2;

    auto first = idx.begin() + lo;
    auto middle = idx.begin() + mid;
    auto last = idx.begin() + hi;

    std::nth_element(first, middle, last,
        [&](index_t a, index_t b)
        {
            double va = X((int)a, axis);
            double vb = X((int)b, axis);
            if (va < vb) return !sign;
            if (va > vb) return  sign;
            return a < b;
        });

    int next_axis = (axis + 1) % d;

    if (!sign) {
        RRM_md_rec(X, idx, lo, mid, next_axis, signs);
        signs[next_axis] = !signs[next_axis];
        RRM_md_rec(X, idx, mid, hi, next_axis, signs);
    }
    else {
        RRM_md_rec(X, idx, mid, hi, next_axis, signs);
        signs[next_axis] = !signs[next_axis];
        RRM_md_rec(X, idx, lo, mid, next_axis, signs);
    }
}

Eigen::ArrayXi RRM(const Eigen::ArrayXXd& X)
{
    int n = X.rows();
    int d = X.cols();
    if (d <= 0) throw std::runtime_error("dimension must be >= 1");

    std::vector<index_t> idx(n);
    for (int i = 0; i < n; i++) idx[i] = i;

    std::vector<bool> signs(d, false);

    RRM_md_rec(X, idx, 0, (index_t)n, 0, signs);

    Eigen::ArrayXi out(n);
    for (int i = 0; i < n; i++) out(i) = (int)idx[i];
    return out;
}

// ==========================================// ==========================================// ==========================================
using index_t = std::ptrdiff_t;

static void RRM_md_rec_axes(
    const Eigen::ArrayXXd& X,
    const std::vector<int>& axes,      // length q, values in [0, d-1]
    std::vector<index_t>& idx,
    index_t lo,
    index_t hi,
    int pos,                           // position in axes (0..q-1)
    std::vector<bool>& signs)          // length q
{
    index_t n = hi - lo;
    if (n <= 1) return;

    const int q = (int)axes.size();
    const int axis = axes[pos];
    const bool sign = signs[pos];
    const index_t mid = lo + n / 2;

    auto first  = idx.begin() + lo;
    auto middle = idx.begin() + mid;
    auto last   = idx.begin() + hi;

    std::nth_element(first, middle, last,
        [&](index_t a, index_t b)
        {
            double va = X((int)a, axis);
            double vb = X((int)b, axis);
            if (va < vb) return !sign;
            if (va > vb) return  sign;

            // tie-break: 用剩余选定轴继续比，避免只用q轴时顺序抖动
            for (int t = 1; t < q; ++t) {
                int ax2 = axes[(pos + t) % q];
                double a2 = X((int)a, ax2);
                double b2 = X((int)b, ax2);
                if (a2 < b2) return !sign;
                if (a2 > b2) return  sign;
            }
            return a < b;
        });

    int next_pos = (pos + 1) % q;

    if (!sign) {
        RRM_md_rec_axes(X, axes, idx, lo,  mid, next_pos, signs);
        signs[next_pos] = !signs[next_pos];
        RRM_md_rec_axes(X, axes, idx, mid, hi, next_pos, signs);
    } else {
        RRM_md_rec_axes(X, axes, idx, mid, hi, next_pos, signs);
        signs[next_pos] = !signs[next_pos];
        RRM_md_rec_axes(X, axes, idx, lo,  mid, next_pos, signs);
    }
}

Eigen::ArrayXi RRM_Order_Axes(const Eigen::ArrayXXd& X,
                                        const Eigen::ArrayXi& axes_in)
{
    const int n = (int)X.rows();
    const int d = (int)X.cols();
    const int q = (int)axes_in.size();

    if (d <= 0) throw std::runtime_error("dimension must be >= 1");
    if (q <= 0) throw std::runtime_error("axes must be non-empty");

    std::vector<int> axes(q);
    for (int i = 0; i < q; ++i) {
        int ax = axes_in(i);
        if (ax < 0 || ax >= d) throw std::runtime_error("axis out of range");
        axes[i] = ax;
    }

    std::vector<index_t> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;

    std::vector<bool> signs(q, false);
    RRM_md_rec_axes(X, axes, idx, 0, (index_t)n, 0, signs);

    Eigen::ArrayXi out(n);
    for (int i = 0; i < n; ++i) out(i) = (int)idx[i];
    return out;
}
// ---------------------weight---------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------
struct TripletMass {
    int x;
    int y;
    double mass;
};

struct Atom {
    int id = -1;      // point index
    double mass = 0;  // weight
};

struct CDFSplit {
    index_t id;       // pivot position in atoms array
    double rho;       // mass strictly before pivot within current recursion path
};

struct AtomSplit {
    index_t pos;      // pivot position
    double mass_left; // part assigned to left
    double mass_right;// part assigned to right
};

// ---------------- Hilbert-style comparator with tie-break on remaining axes ----------------
// sign=false => ascending, sign=true => descending (matches your comparator behavior)
static inline bool less_axes_point(
    const Eigen::ArrayXXd& P,        // (n,d)
    int ida,
    int idb,
    const std::vector<int>& axes,
    int pos,
    bool sign
) {
    const int q = (int)axes.size();
    const int axis = axes[pos];

    const double va = P(ida, axis);
    const double vb = P(idb, axis);

    if (va < vb) return !sign;
    if (va > vb) return  sign;

    // tie-break: continue compare on remaining chosen axes (stable like your code)
    for (int t = 1; t < q; ++t) {
        int ax2 = axes[(pos + t) % q];
        double a2 = P(ida, ax2);
        double b2 = P(idb, ax2);
        if (a2 < b2) return !sign;
        if (a2 > b2) return  sign;
    }
    return ida < idb;
}

// ---------------- random pivot ----------------
static inline index_t rand_pivot(index_t lo, index_t hi_inclusive, std::mt19937_64& rng) {
    std::uniform_int_distribution<index_t> dist(lo, hi_inclusive);
    return dist(rng);
}

// ---------------- partition atoms[beg:end) around pivot atoms[idx] ----------------
// Reorders atoms so that "before pivot" are on the left w.r.t. less_axes_point.
// Returns boundary p and mass_left = sum of masses of items moved to left.
static CDFSplit partition_atoms(
    const Eigen::ArrayXXd& P,
    std::vector<Atom>& atoms,
    index_t beg,
    index_t end,
    index_t pivot_idx,
    const std::vector<int>& axes,
    int pos,
    bool sign
) {
    const int pivot_id = atoms[pivot_idx].id;
    index_t i = beg;
    index_t j = end - 1;

    double sum_left = 0.0;

    while (i < j) {
        while (i < end && less_axes_point(P, atoms[i].id, pivot_id, axes, pos, sign)) {
            sum_left += atoms[i].mass;
            ++i;
        }
        while (j >= beg && less_axes_point(P, pivot_id, atoms[j].id, axes, pos, sign)) {
            --j;
        }
        if (i >= j) break;
        std::swap(atoms[i], atoms[j]);
    }

    return { i, sum_left };
}

// ---------------- QuickCDF: find pivot position where cumulative mass crosses rho ----------------
// Similar spirit to your GeneralBSPMatching::quickCDF + splitCDF.
static CDFSplit quickCDF_atoms(
    const Eigen::ArrayXXd& P,
    std::vector<Atom>& atoms,
    index_t beg,
    index_t end,
    const std::vector<int>& axes,
    int pos,
    bool sign,
    double rho_target,
    double mass_prefix,
    std::mt19937_64& rng
) {
    const index_t n = end - beg;
    if (n <= 1) return { beg, mass_prefix };

    index_t piv = rand_pivot(beg, end - 1, rng);
    auto part = partition_atoms(P, atoms, beg, end, piv, axes, pos, sign);

    // part.rho is mass strictly before boundary part.id in THIS partition step
    // We recurse into side where rho_target lies.
    if (part.rho >= rho_target) {
        return quickCDF_atoms(P, atoms, beg, part.id, axes, pos, sign, rho_target, mass_prefix, rng);
    } else {
        return quickCDF_atoms(P, atoms, part.id, end, axes, pos, sign, rho_target - part.rho, mass_prefix + part.rho, rng);
    }
}

static CDFSplit quickCDF_atoms(
    const Eigen::ArrayXXd& P,
    std::vector<Atom>& atoms,
    index_t beg,
    index_t end,
    const std::vector<int>& axes,
    int pos,
    bool sign,
    double rho_target,
    std::mt19937_64& rng
) {
    return quickCDF_atoms(P, atoms, beg, end, axes, pos, sign, rho_target, 0.0, rng);
}

// Split an atom so left side total mass equals rho exactly (may split pivot atom).
static AtomSplit splitCDF_atoms(
    const Eigen::ArrayXXd& P,
    std::vector<Atom>& atoms,
    index_t beg,
    index_t end,
    const std::vector<int>& axes,
    int pos,
    bool sign,
    double rho,
    std::mt19937_64& rng
) {
    CDFSplit sel = quickCDF_atoms(P, atoms, beg, end, axes, pos, sign, rho, rng);
    const index_t p = sel.id;
    const double mass_before = sel.rho;

    // Invariant: mass_before < rho <= mass_before + atoms[p].mass
    double mass_left = rho - mass_before;
    if (mass_left < 0.0) mass_left = 0.0;
    if (mass_left > atoms[p].mass) mass_left = atoms[p].mass;

    double mass_right = atoms[p].mass - mass_left;

    return { p, mass_left, mass_right };
}

// Utility: sum mass in atoms[beg:end)
static double sum_mass(const std::vector<Atom>& atoms, index_t beg, index_t end) {
    double s = 0.0;
    for (index_t i = beg; i < end; ++i) s += atoms[i].mass;
    return s;
}

// ---------------- recursive coupling builder ----------------
static void hilbert_axes_coupling_rec(
    const Eigen::ArrayXXd& X,
    std::vector<Atom>& mu,
    index_t begA,
    index_t endA,
    const Eigen::ArrayXXd& Y,
    std::vector<Atom>& nu,
    index_t begB,
    index_t endB,
    const std::vector<int>& axes,
    int pos,
    std::vector<bool>& signs,
    std::vector<TripletMass>& out,
    std::mt19937_64& rng
) {
    const index_t gapA = endA - begA;
    const index_t gapB = endB - begB;
    if (gapA <= 0 || gapB <= 0) return;

    if (gapA == 1) {
        const int xid = mu[begA].id;
        for (index_t j = begB; j < endB; ++j) {
            if (nu[j].mass <= 1e-15) continue;
            out.push_back({ xid, nu[j].id, nu[j].mass });
        }
        return;
    }
    if (gapB == 1) {
        const int yid = nu[begB].id;
        for (index_t i = begA; i < endA; ++i) {
            if (mu[i].mass <= 1e-15) continue;
            out.push_back({ mu[i].id, yid, mu[i].mass });
        }
        return;
    }

    const int q = (int)axes.size();
    const bool sign = signs[pos];
    const int next_pos = (pos + 1) % q;

    const double totalA = sum_mass(mu, begA, endA);
    const double totalB = sum_mass(nu, begB, endB);
    if (std::abs(totalA - totalB) > 1e-8 * (1.0 + std::max(std::abs(totalA), std::abs(totalB)))) {
        throw std::runtime_error("hilbert_axes_coupling_rec_stable: mass mismatch in subtree");
    }

    // ---- A side: choose a cut near half mass but DO NOT split mu ----
    const double target = 0.5 * totalA;
    CDFSplit cdfsA = quickCDF_atoms(X, mu, begA, endA, axes, pos, sign, target, rng);
    index_t p = cdfsA.id;
    double rho = cdfsA.rho; // mass strictly before p

    // guard to ensure progress (no empty side)
    if (p <= begA) {
        p = begA + 1;
        rho = mu[begA].mass; // left takes first atom
    }
    if (p >= endA) {
        p = endA - 1;
        rho = totalA - mu[endA - 1].mass;
    }

    // ---- B side: split nu exactly to match rho ----
    AtomSplit splitB = splitCDF_atoms(Y, nu, begB, endB, axes, pos, sign, rho, rng);

    const int savedB_id   = nu[splitB.pos].id;
    const double savedB_m = nu[splitB.pos].mass;

    const index_t midB = splitB.pos + 1;

    auto rec_left = [&]() {
        // left part of nu pivot
        nu[splitB.pos].id = savedB_id;
        nu[splitB.pos].mass = splitB.mass_left;

        hilbert_axes_coupling_rec(
            X, mu, begA, p,
            Y, nu, begB, midB,
            axes, next_pos, signs, out, rng
        );
    };

    auto rec_right = [&]() {
        // right part of nu pivot
        nu[splitB.pos].id = savedB_id;
        nu[splitB.pos].mass = splitB.mass_right;

        hilbert_axes_coupling_rec(
            X, mu, p, endA,
            Y, nu, splitB.pos, endB,
            axes, next_pos, signs, out, rng
        );
    };

    // Hilbert recursion order & sign flip
    if (!sign) {
        rec_left();
        signs[next_pos] = !signs[next_pos];
        rec_right();
    } else {
        rec_right();
        signs[next_pos] = !signs[next_pos];
        rec_left();
    }

    // restore
    nu[splitB.pos].id = savedB_id;
    nu[splitB.pos].mass = savedB_m;
}

// ---------------- public API ----------------
// X: (m,d), Y: (n,d), mu_mass: (m,), nu_mass: (n,)
// axes_in: (q,), values in [0,d-1]
// Output: triplets (x_id, y_id, mass), a mass-balanced coupling with possible splitting.
static std::vector<TripletMass> Hilbert_Axes_Coupling_MassBalanced(
    const Eigen::ArrayXXd& X,
    const Eigen::ArrayXd& mu_mass,
    const Eigen::ArrayXXd& Y,
    const Eigen::ArrayXd& nu_mass,
    const Eigen::ArrayXi& axes_in,
    std::uint64_t seed = 0
) {
    const int m = (int)X.rows();
    const int d = (int)X.cols();
    const int n = (int)Y.rows();
    if ((int)mu_mass.size() != m) throw std::runtime_error("mu_mass must have length X.rows()");
    if ((int)nu_mass.size() != n) throw std::runtime_error("nu_mass must have length Y.rows()");
    if ((int)axes_in.size() <= 0) throw std::runtime_error("axes must be non-empty");
    if (Y.cols() != d) throw std::runtime_error("X and Y must have same dimension");

    // axes
    std::vector<int> axes((int)axes_in.size());
    for (int i = 0; i < (int)axes.size(); ++i) {
        int ax = axes_in(i);
        if (ax < 0 || ax >= d) throw std::runtime_error("axis out of range");
        axes[i] = ax;
    }

    // build atoms
    std::vector<Atom> mu((size_t)m), nu((size_t)n);
    for (int i = 0; i < m; ++i) {
        double w = mu_mass(i);
        if (!(w >= 0.0)) throw std::runtime_error("mu_mass must be >= 0");
        mu[(size_t)i] = { i, w };
    }
    for (int j = 0; j < n; ++j) {
        double w = nu_mass(j);
        if (!(w >= 0.0)) throw std::runtime_error("nu_mass must be >= 0");
        nu[(size_t)j] = { j, w };
    }

    double sumA = mu_mass.sum();
    double sumB = nu_mass.sum();
    if (std::abs(sumA - sumB) > 1e-8 * (1.0 + std::max(std::abs(sumA), std::abs(sumB)))) {
        throw std::runtime_error("Total mass mismatch: sum(mu) != sum(nu)");
    }

    std::mt19937_64 rng(seed ? seed : (std::uint64_t)std::random_device{}());
    std::vector<bool> signs(axes.size(), false);

    std::vector<TripletMass> out;
    out.reserve((size_t)(m + n)); // rough; final edges <= m+n-1 typically

    hilbert_axes_coupling_rec(
        X, mu, 0, (index_t)m,
        Y, nu, 0, (index_t)n,
        axes, 0, signs, out, rng
    );

    return out;
}




// -------------------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------
//Hilbert order
Eigen::ArrayXi Hilbert_Curve_Order(const Eigen::ArrayXXd & X)
{

    std::vector<ptrdiff_t> indices(X.rows());
    int i = 0;

    for(i=0; i<X.rows(); i++){
      indices[i] = i;
    }

    Hilbert_sort Hs(X);
    Hs.Hilbert_sort_median_d(indices.begin(),indices.end());

    Eigen::ArrayXi res = Eigen::ArrayXi::Zero(X.rows());
    for (int i = 0; i < X.rows(); i++) {
      res(i) = indices[i];
    }

    return res;
}




//North-West Corner Algorithm
Eigen::ArrayXXd general_plan(const double * a_weight, int n, const double * b_weight, int m){
    
    int i = 0, j = 0, l = n+m-1, c_id = 0, s = 0;
    double w_i = a_weight[0], w_j = b_weight[0];

    std::vector<double> idx1(l);
    std::vector<int> idx2(l);
    std::vector<int> idx3(l);

    while(true){

        if(w_i < w_j || j == m - 1){
        
            idx1[c_id] = w_i;
            idx2[c_id] = i;
            idx3[c_id] = j;
            i++;
            if(i == n){break;}
            w_j -= w_i;
            w_i = a_weight[i];
        }
        else{
            idx1[c_id] = w_j;
            idx2[c_id] = i;
            idx3[c_id] = j;
            j++;
            if(j == m){break;}
            w_i -= w_j;
            w_j = b_weight[j];
        }
        c_id++;
            
    }

    c_id++;
    Eigen::ArrayXXd G(3, c_id);

    for(s=0; s<c_id; s++){
      G(0,s) = idx1[s];
      G(1,s) = idx2[s];
      G(2,s) = idx3[s];
    }

    return G;
}

Eigen::ArrayXXd General_Plan(const Eigen::ArrayXd & X, const Eigen::ArrayXd & Y)
{
    int n1 = X.rows();
    int m1 = Y.rows();

    return  general_plan(X.data(), n1, Y.data(), m1);
}



namespace py = pybind11;

static BSPOT::Points<-1> to_points_d_by_n(const Eigen::Ref<const Eigen::MatrixXd>& X_nd) {
    // X_nd: N x d  ->  d x N
    return X_nd.transpose().eval();
}

static py::tuple py_computeBijectiveBSPOT(
    const Eigen::Ref<const Eigen::MatrixXd>& A_nd,
    const Eigen::Ref<const Eigen::MatrixXd>& B_nd,
    int nb_plans,
    double radial_prob
) {
    if (A_nd.rows() != B_nd.rows() || A_nd.cols() != B_nd.cols()) {
        throw std::runtime_error("A and B must have same shape (N,d).");
    }
    const int N = (int)A_nd.rows();
    if (N <= 0) throw std::runtime_error("Empty point set.");

    const bool use_override = (radial_prob >= 0.0);
    const BSPOT::scalar old_p = BSPOT::get_radial_prob();
    if (use_override) BSPOT::set_radial_prob((BSPOT::scalar)radial_prob);

    auto A = to_points_d_by_n(A_nd); // d x N
    auto B = to_points_d_by_n(B_nd); // d x N

    BSPOT::cost_function cost = [&](size_t i, size_t j) -> BSPOT::scalar {
        return (A.col((int)i) - B.col((int)j)).squaredNorm();
    };

    BSPOT::BijectiveMatching T =
        BSPOT::computeBijectiveBSPOT<-1>(A, B, nb_plans, cost);

    const std::vector<int>& plan = T.getPlan();
    double avg_cost = T.evalMatching(cost);

    if (use_override) BSPOT::set_radial_prob(old_p);

    return py::make_tuple(plan, avg_cost);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

// Random orthogonal matrix via QR
static Eigen::MatrixXd random_orthogonal_matrix(int d, std::mt19937_64& rng) {
    std::normal_distribution<double> nd(0.0, 1.0);

    Eigen::MatrixXd A(d, d);
    for (int r = 0; r < d; ++r)
        for (int c = 0; c < d; ++c)
            A(r, c) = nd(rng);

    Eigen::HouseholderQR<Eigen::MatrixXd> decomp(A);
    Eigen::MatrixXd Q = decomp.householderQ() * Eigen::MatrixXd::Identity(d, d);

    // Optional: force det(Q)=+1 (pure rotation)
    if (Q.determinant() < 0.0) Q.col(0) *= -1.0;
    return Q;
}


// NEW: random DPQ transform + mergeplans
static Eigen::VectorXi RRM_merge_random_dpq_impl(
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const Eigen::Ref<const Eigen::MatrixXd>& Y,
    int p,
    std::vector<int> axes,     // same meaning as before (axis cycle used by Hilbert_Curve_Order_Axes)
    bool cycle,
    std::uint64_t seed
) {
    const int n = static_cast<int>(X.rows());
    const int d = static_cast<int>(X.cols());

    if (Y.rows() != X.rows() || Y.cols() != X.cols())
        throw std::runtime_error("X and Y must have the same shape (n, d).");
    if (n <= 0 || d <= 0)
        throw std::runtime_error("X/Y must be non-empty.");
    if (p <= 0)
        throw std::runtime_error("p must be >= 1.");

    // axes default: 0..d-1
    if (axes.empty()) {
        axes.resize(d);
        std::iota(axes.begin(), axes.end(), 0);
    }

    // build Eigen axes_in for Hilbert_Curve_Order_Axes
    Eigen::ArrayXi axes_in(static_cast<int>(axes.size()));
    for (int i = 0; i < static_cast<int>(axes.size()); ++i) {
        if (axes[i] < 0 || axes[i] >= d)
            throw std::runtime_error("axis out of range in axes.");
        axes_in(i) = axes[i];
    }

    std::mt19937_64 rng(seed);

    // cost: squared euclidean in ORIGINAL space
    BSPOT::cost_function cost = [&](int i, int j) {
        return static_cast<BSPOT::scalar>((X.row(i) - Y.row(j)).squaredNorm());
    };

    std::vector<BSPOT::BijectiveMatching> plans;
    plans.reserve(static_cast<std::size_t>(p));

    std::uniform_int_distribution<int> bit01(0, 1);

    // buffers for perm and sign
    std::vector<int> perm(d);
    std::vector<double> sgn(d);

    for (int t = 0; t < p; ++t) {
        // Q: random orthogonal
        Eigen::MatrixXd Q = random_orthogonal_matrix(d, rng);

        // Apply Q first
        Eigen::MatrixXd XQ = X * Q;
        Eigen::MatrixXd YQ = Y * Q;

        // P: random permutation of axes
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);

        // D: random sign flips (+1/-1)
        for (int j = 0; j < d; ++j) {
            sgn[j] = bit01(rng) ? 1.0 : -1.0;
        }

        // Compose (P + D) without building matrices:
        Eigen::MatrixXd Xr(n, d), Yr(n, d);
        for (int j = 0; j < d; ++j) {
            Xr.col(j) = XQ.col(perm[j]) * sgn[j];
            Yr.col(j) = YQ.col(perm[j]) * sgn[j];
        }

        // Hilbert order on transformed coordinates
        Eigen::ArrayXi orderX = RRM_Order_Axes(Xr.array(), axes_in);
        Eigen::ArrayXi orderY = RRM_Order_Axes(Yr.array(), axes_in);

        // build plan: plan[orderX[k]] = orderY[k]
        Eigen::VectorXi T(n);
        for (int k = 0; k < n; ++k)
            T(orderX(k)) = orderY(k);

        plans.emplace_back(T);
    }

    BSPOT::BijectiveMatching merged =
        BSPOT::MergePlans(plans, cost, BSPOT::BijectiveMatching(), cycle);

//    BSPOT::BijectiveMatching merged =
//        BSPOT::MergePlansTopM(plans, cost, BSPOT::BijectiveMatching(), cycle, /*m=*/16, /*iters=*/4);

    Eigen::VectorXi out(n);
    const auto& mp = merged.getPlan();
    for (int i = 0; i < n; ++i) out(i) = static_cast<int>(mp[i]);
    return out;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ----- helper: for each row in P (m,d) generate per_point samples around it, clip [0,1] -----
// returns Z of shape (m*per_point, d)

static void sample_XY_global_deltas_apply_to_all(
    const Eigen::Ref<const Eigen::MatrixXd>& Xsub,   // (m,d)
    const Eigen::Ref<const Eigen::MatrixXd>& Ysub,   // (m,d)
    int per_point,
    std::mt19937_64& rng,
    Eigen::MatrixXd& Xextra,                         // out: (m*per_point,d)
    Eigen::MatrixXd& Yextra                          // out: (m*per_point,d)
) {
    const int m = (int)Xsub.rows();
    const int d = (int)Xsub.cols();
    if ((int)Ysub.rows() != m || (int)Ysub.cols() != d)
        throw std::runtime_error("sample_XY_global_deltas_apply_to_all: Xsub/Ysub shape mismatch");
    if (m <= 0) throw std::runtime_error("sample_XY_global_deltas_apply_to_all: m must be > 0");
    if (per_point <= 0) per_point = 1;


    Eigen::RowVectorXd mn = Xsub.colwise().minCoeff().cwiseMin(Ysub.colwise().minCoeff());
    Eigen::RowVectorXd mx = Xsub.colwise().maxCoeff().cwiseMax(Ysub.colwise().maxCoeff());
    Eigen::RowVectorXd span = (mx - mn).cwiseMax(1e-6);

    double radius_scale = 0.05 * span.mean();
    if (radius_scale < 1e-3) radius_scale = 1e-3;

    std::normal_distribution<double> nd(0.0, 1.0);


    std::vector<Eigen::RowVectorXd> deltas;
    deltas.reserve((size_t)per_point);

    for (int kk = 0; kk < per_point; ++kk) {

        Eigen::RowVectorXd dir(d);
        for (int j = 0; j < d; ++j) dir(j) = nd(rng);
        double norm = dir.norm();
        if (norm < 1e-12) {
            dir.setZero();
            dir(0) = 1.0;
        } else {
            dir /= norm;
        }


        double r = radius_scale;

        deltas.push_back(r * dir);
    }

    Xextra.resize(m * per_point, d);
    Yextra.resize(m * per_point, d);


    // t = kk*m + i
    for (int kk = 0; kk < per_point; ++kk) {
        const Eigen::RowVectorXd& delta = deltas[(size_t)kk];
        for (int i = 0; i < m; ++i) {
            int t = kk * m + i;
            Eigen::RowVectorXd vx = Xsub.row(i) + delta;
            Eigen::RowVectorXd vy = Ysub.row(i) + delta;


            for (int j = 0; j < d; ++j) {
                if (vx(j) < 0.0) vx(j) = 0.0;
                if (vx(j) > 1.0) vx(j) = 1.0;
                if (vy(j) < 0.0) vy(j) = 0.0;
                if (vy(j) > 1.0) vy(j) = 1.0;
            }

            Xextra.row(t) = vx;
            Yextra.row(t) = vy;
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



static Eigen::MatrixXd sample_Z_near_points(
    const Eigen::Ref<const Eigen::MatrixXd>& P,
    int per_point,
    std::mt19937_64& rng
) {
    const int m = (int)P.rows();
    const int d = (int)P.cols();
    if (m <= 0) throw std::runtime_error("sample_Z_near_points: m must be > 0");
    if (per_point <= 0) per_point = 1;

    // scale from data span (cheap & stable)
    Eigen::RowVectorXd mn = P.colwise().minCoeff();
    Eigen::RowVectorXd mx = P.colwise().maxCoeff();
    Eigen::RowVectorXd span = (mx - mn).cwiseMax(1e-6);

    // std = 5% span, with floor
    Eigen::RowVectorXd sigma(d);
    for (int j = 0; j < d; ++j) {
        double s = 0.05 * span(j);
        if (s < 1e-3) s = 1e-3;
        sigma(j) = s;
    }

    std::normal_distribution<double> nd(0.0, 1.0);

    Eigen::MatrixXd Z(m * per_point, d);
    int t = 0;
    for (int i = 0; i < m; ++i) {
        for (int kk = 0; kk < per_point; ++kk, ++t) {
            for (int j = 0; j < d; ++j) {
                double v = P(i, j) + nd(rng) * sigma(j);
                if (v < 0.0) v = 0.0;
                if (v > 1.0) v = 1.0;
                Z(t, j) = v;
            }
        }
    }
    return Z;
}
// ----- helper: Eigen::VectorXi -> numpy int64 -----
static py::array_t<long long> veci_to_i64_array(const Eigen::VectorXi& v) {
    py::array_t<long long> out(v.size());
    auto b = out.mutable_unchecked<1>();
    for (int i = 0; i < v.size(); ++i) b(i) = (long long)v(i);
    return out;
}

static py::array_t<long long> idx_to_i64_array(const std::vector<int>& idx) {
    py::array_t<long long> out((ssize_t)idx.size());
    auto b = out.mutable_unchecked<1>();
    for (ssize_t i = 0; i < (ssize_t)idx.size(); ++i) b(i) = (long long)idx[(size_t)i];
    return out;
}

// ----- helper: Hungarian (min cost) for square m x m -----
static std::vector<int> hungarian_min_cost(const std::vector<double>& cost, int m) {
    // potentials
    const double INF = 1e300;
    std::vector<double> u(m + 1, 0.0), v(m + 1, 0.0), minv(m + 1, INF);
    std::vector<int> p(m + 1, 0), way(m + 1, 0);
    std::vector<char> used(m + 1, false);

    for (int i = 1; i <= m; ++i) {
        p[0] = i;
        int j0 = 0;
        std::fill(minv.begin(), minv.end(), INF);
        std::fill(used.begin(), used.end(), false);

        do {
            used[j0] = true;
            int i0 = p[j0];
            double delta = INF;
            int j1 = 0;

            for (int j = 1; j <= m; ++j) if (!used[j]) {
                double cur = cost[(i0 - 1) * m + (j - 1)] - u[i0] - v[j];
                if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                if (minv[j] < delta) { delta = minv[j]; j1 = j; }
            }

            for (int j = 0; j <= m; ++j) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else { minv[j] -= delta; }
            }
            j0 = j1;
        } while (p[j0] != 0);

        // augment
        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }

    // row -> col assignment, 0-indexed
    std::vector<int> col_of_row(m, -1);
    for (int j = 1; j <= m; ++j) {
        int i = p[j];
        col_of_row[i - 1] = j - 1;
    }
    return col_of_row;
}

// ----- helper: sample Z (kZ,d) -----
static Eigen::MatrixXd sample_Z(int kZ, int d, const std::string& z_mode, std::mt19937_64& rng) {
    Eigen::MatrixXd Z(kZ, d);

    if (z_mode == "uniform01") {
        std::uniform_real_distribution<double> ud(0.0, 1.0);
        for (int i = 0; i < kZ; ++i)
            for (int j = 0; j < d; ++j)
                Z(i, j) = ud(rng);
        return Z;
    }

    if (z_mode == "gauss_clip01") {
        std::normal_distribution<double> nd(0.5, 0.2);
        for (int i = 0; i < kZ; ++i)
            for (int j = 0; j < d; ++j) {
                double v = nd(rng);
                if (v < 0.0) v = 0.0;
                if (v > 1.0) v = 1.0;
                Z(i, j) = v;
            }
        return Z;
    }

    if (z_mode == "gauss_minmax01") {
        std::normal_distribution<double> nd(0.0, 1.0);
        for (int i = 0; i < kZ; ++i)
            for (int j = 0; j < d; ++j)
                Z(i, j) = nd(rng);

        // per-column minmax -> [0,1]
        for (int j = 0; j < d; ++j) {
            double mn = Z.col(j).minCoeff();
            double mx = Z.col(j).maxCoeff();
            double denom = std::max(mx - mn, 1e-12);
            Z.col(j) = (Z.col(j).array() - mn) / denom;
        }
        return Z;
    }

    throw std::runtime_error("z_mode must be: uniform01 / gauss_clip01 / gauss_minmax01");
}

// ----- finalize: make plan a permutation (bijection) -----
static Eigen::VectorXi finalize_bijection_impl(
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const Eigen::Ref<const Eigen::MatrixXd>& Y,
    const Eigen::Ref<const Eigen::VectorXi>& plan_in,
    bool verbose
) {
    const int n = (int)X.rows();
    Eigen::VectorXi plan = plan_in;

    // 1) resolve duplicate Y: keep closer, others -> -1
    std::unordered_map<int,int> owner;
    owner.reserve((size_t)n);

    for (int i = 0; i < n; ++i) {
        int j = plan(i);
        if (j < 0) continue;
        auto it = owner.find(j);
        if (it == owner.end()) {
            owner.emplace(j, i);
        } else {
            int i0 = it->second;
            double d0 = (X.row(i0) - Y.row(j)).squaredNorm();
            double d1 = (X.row(i ) - Y.row(j)).squaredNorm();
            if (d1 < d0) {
                plan(i0) = -1;
                it->second = i;
            } else {
                plan(i) = -1;
            }
        }
    }

    // 2) collect unX and freeY
    std::vector<int> unX;
    unX.reserve((size_t)n);

    std::vector<char> usedY((size_t)n, false);
    for (int i = 0; i < n; ++i) {
        int j = plan(i);
        if (j >= 0) usedY[(size_t)j] = true;
        else unX.push_back(i);
    }

    std::vector<int> freeY;
    freeY.reserve(unX.size());
    for (int j = 0; j < n; ++j) if (!usedY[(size_t)j]) freeY.push_back(j);

    if (unX.size() != freeY.size())
        throw std::runtime_error("finalize_bijection failed: |unX| != |freeY|");

    const int m = (int)unX.size();
    if (verbose) py::print("[Finalize] unassigned=", m);
    if (m == 0) {
        // verify permutation
        std::vector<char> seen((size_t)n, false);
        for (int i = 0; i < n; ++i) {
            int j = plan(i);
            if (j < 0 || j >= n) throw std::runtime_error("final plan out of range");
            if (seen[(size_t)j]) throw std::runtime_error("final plan not bijective (duplicate)");
            seen[(size_t)j] = true;
        }
        return plan;
    }

    // 3) build cost (m x m)
    std::vector<double> cost((size_t)m * (size_t)m);
    for (int a = 0; a < m; ++a) {
        int ix = unX[a];
        for (int b = 0; b < m; ++b) {
            int jy = freeY[b];
            cost[(size_t)a * (size_t)m + (size_t)b] = (X.row(ix) - Y.row(jy)).squaredNorm();
        }
    }

    // 4) Hungarian
    std::vector<int> col_of_row = hungarian_min_cost(cost, m);
    for (int a = 0; a < m; ++a) {
        plan(unX[a]) = freeY[col_of_row[a]];
    }

    // 5) verify permutation
    std::vector<char> seen((size_t)n, false);
    for (int i = 0; i < n; ++i) {
        int j = plan(i);
        if (j < 0 || j >= n) throw std::runtime_error("final plan out of range");
        if (seen[(size_t)j]) throw std::runtime_error("final plan not bijective (duplicate)");
        seen[(size_t)j] = true;
    }
    return plan;
}

// ----- main: iter_match using hilbert_merge_random_dpq_impl -----
static py::object py_iter_match_dpq(
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const Eigen::Ref<const Eigen::MatrixXd>& Y,
    int num_rounds,
    const std::string& z_mode,
    bool verbose,
    bool return_history,
    int z_count,                 // <=0 -> n
    int p,                       // dpq p
    std::vector<int> axes,        // optional
    bool cycle,
    std::uint64_t seed0,          // base seed, round r uses seed0 + r
    bool finalize
) {
    const int n = (int)X.rows();
    const int d = (int)X.cols();
    if ((int)Y.rows() != n || (int)Y.cols() != d)
        throw std::runtime_error("X and Y must have same shape (n,d).");
    if (num_rounds < 0)
        throw std::runtime_error("num_rounds must be >= 0.");
    if (p <= 0)
        throw std::runtime_error("p must be >= 1.");

    const int per_bad = (z_count <= 0 ? 5 : z_count);

    py::list hist;
    Eigen::VectorXi plan(n);
    plan.setConstant(-1);

    auto push_hist = [&](const std::string& name,
                         const Eigen::VectorXi& plan_snapshot,
                         const std::vector<int>& badX,
                         const std::vector<int>& badY) {
        py::dict h;
        h["name"] = name;
        h["plan"] = veci_to_i64_array(plan_snapshot);
        h["badX"] = idx_to_i64_array(badX);
        h["badY"] = idx_to_i64_array(badY);
        hist.append(h);
    };

    // -------- num_rounds == 0: baseline once --------
    if (num_rounds == 0) {
        Eigen::VectorXi plan0 = RRM_merge_random_dpq_impl(X, Y, p, axes, cycle, seed0);

        if (finalize) plan0 = finalize_bijection_impl(X, Y, plan0, verbose);

        if (!return_history) return veci_to_i64_array(plan0);
        push_hist("Base", plan0, {}, {});
        return py::make_tuple(veci_to_i64_array(plan0), hist);
    }

    // start with full indices
    std::vector<int> curX(n), curY(n);
    std::iota(curX.begin(), curX.end(), 0);
    std::iota(curY.begin(), curY.end(), 0);

    for (int r = 0; r < num_rounds; ++r) {
        const int m = (int)curX.size();
        if (m == 0) break;

        // build sub matrices
        Eigen::MatrixXd Xsub(m, d), Ysub(m, d);
        for (int i = 0; i < m; ++i) {
            Xsub.row(i) = X.row(curX[i]);
            Ysub.row(i) = Y.row(curY[i]);
        }

std::mt19937_64 rng(seed0 + (std::uint64_t)r);
///////////////////////////////////////////////////////////////
// round 1 and later: ALWAYS local Z around current set (treat round 1 as "all bad")
Eigen::MatrixXd Zx = sample_Z_near_points(Xsub, per_bad, rng); // (m*per_bad, d)
Eigen::MatrixXd Zy = sample_Z_near_points(Ysub, per_bad, rng); // (m*per_bad, d)

const int kZ_round = (int)(Zx.rows() + Zy.rows()); // = 2*m*per_bad

Eigen::MatrixXd Z(kZ_round, d);
Z.topRows(Zx.rows()) = Zx;
Z.bottomRows(Zy.rows()) = Zy;

Eigen::MatrixXd Xaug(m + kZ_round, d), Yaug(m + kZ_round, d);
Xaug.topRows(m) = Xsub;
Xaug.bottomRows(kZ_round) = Z;
Yaug.topRows(m) = Ysub;
Yaug.bottomRows(kZ_round) = Z;


//Eigen::MatrixXd Xextra, Yextra;
//sample_XY_global_deltas_apply_to_all(Xsub, Ysub, per_bad, rng, Xextra, Yextra);
//
//const int kZ_round = (int)Xextra.rows(); // = m * per_bad
//
//Eigen::MatrixXd Xaug(m + kZ_round, d), Yaug(m + kZ_round, d);
//Xaug.topRows(m) = Xsub;
//Xaug.bottomRows(kZ_round) = Xextra;
//Yaug.topRows(m) = Ysub;
//Yaug.bottomRows(kZ_round) = Yextra;
///////////////////////////////////////////////////////////////


Eigen::VectorXi matchY = RRM_merge_random_dpq_impl(
    Xaug, Yaug, p, axes, cycle, seed0 + (std::uint64_t)r
);

const int N = m + kZ_round;

        // invY[y] = x
        std::vector<int> invY((size_t)N, -1);
        for (int i = 0; i < N; ++i) invY[(size_t)matchY(i)] = i;

        // classify good/bad
        std::vector<int> badX, badY;
        badX.reserve(curX.size());
        badY.reserve(curY.size());

        // good X: i<m and matchY(i)<m
        for (int i = 0; i < m; ++i) {
            int j = matchY(i);
            if (j < m) {
                plan(curX[i]) = curY[j];
            } else {
                badX.push_back(curX[i]);
            }
        }

        // bad Y: y in [0,m) with invY[y] >= m (matched from Z)
        for (int j = 0; j < m; ++j) {
            if (invY[(size_t)j] >= m) badY.push_back(curY[j]);
        }

        curX.swap(badX);
        curY.swap(badY);

        if (verbose) {
    py::print("[Round", r + 1, "] badX=", (int)curX.size(), " badY=", (int)curY.size(),
              " Z=", kZ_round, " per_bad=", per_bad);
}

        if (return_history) {
            push_hist("Round" + std::to_string(r + 1), plan, curX, curY);
        }

        if (curX.empty() && curY.empty()) break;
    }

    // finalize to full bijection
    Eigen::VectorXi plan_final = plan;
    if (finalize) {
        plan_final = finalize_bijection_impl(X, Y, plan_final, verbose);
        if (return_history) {
            push_hist("Final", plan_final, curX, curY);
        }
    } else {
        if (return_history) {
            push_hist("End", plan_final, curX, curY);
        }
    }

    if (!return_history) return veci_to_i64_array(plan_final);
    return py::make_tuple(veci_to_i64_array(plan_final), hist);
}




////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
} // anonymous namespace

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////





PYBIND11_MODULE(base, m)
{
    m.doc() = "source code";
    m.def("hilbert_order", &Hilbert_Curve_Order);
    m.def("RRM", &RRM);

    m.def("general_Plan", &General_Plan);

    m.def("computeBijectiveBSPOT",
      &py_computeBijectiveBSPOT,
      py::arg("A"),
      py::arg("B"),
      py::arg("nb_plans"),
      py::arg("radial_prob") = -1.0);


    m.def(
    "RRM_merge_random_dpq",
    &RRM_merge_random_dpq_impl,
    py::arg("X"),
    py::arg("Y"),
    py::arg("p") = 32,
    py::arg("axes") = std::vector<int>{},
    py::arg("cycle") = true,
    py::arg("seed") = 1ULL
);


    m.def(
    "iter_match_dpq",
    &py_iter_match_dpq,
    py::arg("X"),
    py::arg("Y"),
    py::arg("num_rounds") = 2,
    py::arg("z_mode") = "uniform01",
    py::arg("verbose") = true,
    py::arg("return_history") = false,
    py::arg("z_count") = -1,               // <=0 => n
    py::arg("p") = 10,                     // dpq plans count
    py::arg("axes") = std::vector<int>{},  // empty => default 0..d-1
    py::arg("cycle") = true,
    py::arg("seed0") = 0ULL,
    py::arg("finalize") = true
);

}
