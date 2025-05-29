from sympy import symbols, simplify, solve, Symbol, S
from typing import List, Dict, Tuple, Any, Union

"""simplex_patch.py – Fix objective-type handling (MAX/MIN)

* Convert **MAX → MIN** by negating c when objective_type == 'max'.
* Keep c as‑is for MIN.
* Flip sign of z* back to MAX when returning.
* Retains preprocessing of ≥ constraints and step-by-step printing.
"""

# ----------------------------- helpers --------------------------------------

def _print_tableau(title: str, exprs: Dict[str, Any]) -> None:
    ordered = sorted(k for k in exprs if k != 'z') + (['z'] if 'z' in exprs else [])
    print(f"\n{title}")
    for var in ordered:
        print(f"  {var} = {exprs[var]}")


def _preprocess(A: List[List[float]], b: List[float], constraint_types: List[str]) -> Tuple[List[List[float]], List[float]]:
    """Convert every row to ≤ form by flipping ≥ rows."""
    A_new, b_new = [], []
    for row, bi, t in zip(A, b, constraint_types):
        if t.strip() == '>=':
            A_new.append([-a for a in row])
            b_new.append(-bi)
        else:
            A_new.append(row[:])
            b_new.append(bi)
    return A_new, b_new

# --------------------------- core simplex -----------------------------------

def _simplex_min(A: List[List[float]], b: List[float], c: List[float]) -> Tuple[str, Union[float, None], Dict[str, float], Dict[str, Any]]:
    m, n = len(A), len(A[0])
    x_syms = list(symbols(f"x1:{n+1}"))
    w_syms = list(symbols(f"w1:{m+1}"))

    step = 0
    steps: Dict[str, Dict[str, Any]] = {}
    cur: Dict[str, Any] = {}

    # Step 0
    for i in range(m):
        cur[str(w_syms[i])] = simplify(b[i] - sum(A[i][j] * x_syms[j] for j in range(n)))
    cur['z'] = simplify(-sum(c[j] * x_syms[j] for j in range(n)))
    steps['Step 0'] = cur.copy()
    _print_tableau('Step 0', cur)

    basic = [str(w) for w in w_syms]
    non_basic = [str(x) for x in x_syms]

    while True:
        z_expr = cur['z']
        entering, most_neg = None, S.Zero
        for v in non_basic:
            coeff = z_expr.coeff(Symbol(v))
            if coeff < most_neg:
                most_neg = coeff
                entering = v
        if entering is None:
            status = 'Optimal'
            break

        leaving, min_ratio = None, float('inf')
        for w in basic:
            a = cur[w].coeff(Symbol(entering))
            if a < 0:
                const = float(cur[w].subs({Symbol(v): 0 for v in basic + non_basic}))
                if const >= 0:
                    ratio = const / -a
                    if ratio < min_ratio:
                        min_ratio, leaving = ratio, w
        if leaving is None:
            title = f"Step {step+1} (enter {entering}, no leaving — Unbounded)"
            steps[title] = cur.copy()
            _print_tableau(title, cur)
            return 'Unbounded', None, {}, steps

        step += 1
        pivot_expr = solve(cur[leaving], Symbol(entering))[0]
        new_cur = {var: simplify(expr.subs(Symbol(entering), pivot_expr)) for var, expr in cur.items()}
        new_cur[entering] = simplify(pivot_expr)
        del new_cur[leaving]

        basic[basic.index(leaving)] = entering
        non_basic[non_basic.index(entering)] = leaving
        cur = new_cur

        title = f'Step {step} ({entering} in, {leaving} out)'
        steps[title] = cur.copy()
        _print_tableau(title, cur)

    all_vars = [str(x) for x in x_syms] + [str(w) for w in w_syms]
    subs0 = {Symbol(v): 0 for v in all_vars if v in non_basic}
    z_star = -float(cur['z'].subs(subs0))
    opt_vals = {v: (float(cur[v].subs(subs0)) if v in cur else 0.0) for v in all_vars}

    return 'Optimal', z_star, opt_vals, steps

# --------------------------- public wrapper ---------------------------------

def auto_simplex(
    A: List[List[float]],
    b: List[float],
    c: List[float],
    constraint_types: List[str],
    objective_type: str = 'max',
    variable_types: List[str] | None = None,
) -> Dict[str, Any]:
    A_std, b_std = _preprocess(A, b, constraint_types)

    # MAX → MIN by negating c
    if objective_type.lower() == 'max':
        c_eff = [-ci for ci in c]
        flip_back = True
    else:
        c_eff = c[:]
        flip_back = False

    status, z_star, opt_vals, steps = _simplex_min(A_std, b_std, c_eff)

    # flip sign for MAX objective value
    if flip_back and z_star is not None:
        z_star = -z_star

    return {
        'status': {'Optimal': 'Tối ưu (Optimal)', 'Unbounded': 'Không giới nội (Unbounded)'}[status],
        'z': z_star,
        'solution': {k: v for k, v in opt_vals.items() if k.startswith('x')},
        'steps': steps,
    }

# ------------------------------- tests --------------------------------------
if __name__ == '__main__':
    # Feasible MIN
    A1 = [[-3, 1], [1, 2]]
    b1 = [6, 4]
    c1 = [-1, 4]
    cons1 = ['<=', '<=']
    print('\nMIN example:')
    print(auto_simplex(A1, b1, c1, cons1, 'min'))

    # Same LP as MAX
    print('\nMAX example:')
    print(auto_simplex(A1, b1, c1, cons1, 'max'))
