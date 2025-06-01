from sympy import simplify, solve, Symbol, S
from typing import List, Dict, Tuple, Any, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def format_expression_for_printing(expression):
    """Format a SymPy expression with constant term first and sorted variables."""
    expression = simplify(expression)
    vars_in_expr = expression.free_symbols
    if vars_in_expr:
        const_term, var_terms = expression.as_coeff_add(*vars_in_expr)
    else:
        const_term = expression
        var_terms = []
    const_str = f"{float(const_term):.2f}" if const_term != 0 else "0.00"
    if not var_terms:
        return const_str
    var_term_dict = {}
    for term in var_terms:
        coeff, var = term.as_coeff_Mul()
        var_str = str(var) if var != 1 else ""
        if var_str:
            var_term_dict[var_str] = coeff
    sorted_vars = sorted(var_term_dict.keys(), key=lambda x: (
        0 if x.startswith('x') else 
        1 if x.startswith('y') else 
        2 if x.startswith('w') else 3, x
    ))
    var_str_list = []
    for var in sorted_vars:
        coeff = var_term_dict[var]
        if coeff == 1:
            var_str_list.append(f"{var}")
        elif coeff == -1:
            var_str_list.append(f"-{var}")
        else:
            sign = "" if coeff > 0 else "-"
            var_str_list.append(f"{sign}{abs(float(coeff)):.2f}*{var}")
    var_part = " + ".join(var_str_list).replace("+ -", "- ")
    return f"{const_str} + {var_part}" if var_part else const_str

def _print_tableau(title: str, exprs: Dict[str, Any], basic_vars: List[str] = None, non_basic_vars: List[str] = None) -> None:
    """Print the Simplex tableau with improved formatting."""
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print(f"{'-' * 60}")
    ordered_var_names = ['z'] if 'z' in exprs else []
    decision_vars = sorted([
        k for k in exprs if k != 'z' and (k.startswith('x') or k.startswith('y'))
    ])
    slack_vars = sorted([
        k for k in exprs if k != 'z' and k.startswith('w')
    ])
    ordered_var_names.extend(decision_vars + slack_vars)
    print(f"{'Biến':<10} | {'Biểu thức':<50}")
    print(f"{'-' * 10} | {'-' * 50}")
    for var_name in ordered_var_names:
        if var_name in exprs:
            formatted_str = format_expression_for_printing(exprs[var_name])
            print(f"{var_name:<10} | {formatted_str:<50}")
    if basic_vars or non_basic_vars:
        print(f"{'-' * 60}")
        if basic_vars:
            print(f"Biến cơ bản: {', '.join(basic_vars)}")
        if non_basic_vars:
            print(f"Biến không cơ bản: {', '.join(non_basic_vars)}")
    print(f"{'=' * 60}")

def _simplex_min(A: List[List[float]], b: List[float], c: List[float], constraint_types: List[str], variable_types: List[str], objective_type: str) -> Tuple[str, Union[float, None], Dict[str, float], Dict[str, Any]]:
    """Core Simplex algorithm for minimization with robust URS and multiple-optima handling."""
    m, n_original = len(A), len(A[0])

    # Input validation
    if len(b) != m or len(constraint_types) != m or len(c) != n_original or len(variable_types) != n_original:
        raise ValueError("Kích thước đầu vào không khớp")
    for i, ct in enumerate(constraint_types):
        if ct.strip() not in ['<=', '>=', '=']:
            raise ValueError(f"Loại ràng buộc không hợp lệ: {ct}")
        if ct == '<=' and b[i] < 0:
            raise ValueError(f"Ràng buộc <= không thể có b[{i}] âm: {b[i]}")

    # 1. Standardize constraints to <= form (handled in auto_simplex)
    A_std = [row[:] for row in A]
    b_std = b[:]
    logger.info("Constraints standardized to <= form")

    # 2. Process variables to build tableau
    tableau_decision_symbols = []
    c_tableau = []
    A_tableau = [[] for _ in range(m)]
    original_var_info_map = []

    for i in range(n_original):
        original_name = f"x{i+1}"
        var_type = variable_types[i].strip()
        if var_type == '<=0':
            tableau_name = f"y{i+1}"
            symbol = Symbol(tableau_name)
            tableau_decision_symbols.append(symbol)
            c_tableau.append(-c[i])
            for j_row in range(m):
                A_tableau[j_row].append(-A_std[j_row][i])
            original_var_info_map.append({
                'original_name': original_name,
                'tableau_name': tableau_name,
                'symbol': symbol,
                'is_transformed': True,
                'original_idx': i
            })
        elif var_type == 'URS':
            pos_name = f"x{i+1}_pos"
            neg_name = f"x{i+1}_neg"
            pos_symbol = Symbol(pos_name)
            neg_symbol = Symbol(neg_name)
            tableau_decision_symbols.extend([pos_symbol, neg_symbol])
            c_tableau.extend([c[i], -c[i]])
            for j_row in range(m):
                A_tableau[j_row].extend([A_std[j_row][i], -A_std[j_row][i]])
            original_var_info_map.append({
                'original_name': original_name,
                'pos_name': pos_name,
                'neg_name': neg_name,
                'pos_symbol': pos_symbol,
                'neg_symbol': neg_symbol,
                'is_urs': True,
                'original_idx': i
            })
            logger.info(f"URS variable {original_name} split into {pos_name} and {neg_name}")
        else:  # '>=0'
            tableau_name = original_name
            symbol = Symbol(tableau_name)
            tableau_decision_symbols.append(symbol)
            c_tableau.append(c[i])
            for j_row in range(m):
                A_tableau[j_row].append(A_std[j_row][i])
            original_var_info_map.append({
                'original_name': original_name,
                'tableau_name': tableau_name,
                'symbol': symbol,
                'is_transformed': False,
                'original_idx': i
            })

    # 3. Add slack variables for <= constraints
    slack_symbols = [Symbol(f"w{i+1}") for i in range(m)]
    for constraint_idx in range(m):
        for r_idx in range(m):
            A_tableau[r_idx].append(1 if r_idx == constraint_idx else 0)
        c_tableau.append(0)

    all_tableau_symbols = tableau_decision_symbols + slack_symbols
    num_vars_in_tableau = len(all_tableau_symbols)
    logger.info(f"Tableau initialized with {num_vars_in_tableau} variables")

    # 4. Initialize tableau correctly
    step = 0
    steps: Dict[str, Dict[str, Any]] = {}
    cur: Dict[str, Any] = {}
    basic_var_names = [f"w{i+1}" for i in range(m)]
    non_basic_vars = [str(s) for s in tableau_decision_symbols]

    # Objective function
    z_expr = sum(float(c_tableau[j_col]) * all_tableau_symbols[j_col] for j_col in range(num_vars_in_tableau))
    cur['z'] = simplify(z_expr)

    # Constraint equations: skip slack in RHS
    for i in range(m):
        expr = S.Zero
        for j_col, sym in enumerate(tableau_decision_symbols):
            expr += float(A_tableau[i][j_col]) * sym
        cur[f"w{i + 1}"] = simplify(S(b_std[i]) - expr)

    steps['Step 0'] = cur.copy()
    _print_tableau('Step 0 (Khởi tạo - Initial Tableau)', cur, basic_var_names, non_basic_vars)

    # 5. Simplex loop with enhanced pivot logic
    while True:
        z_expr = cur['z']
        entering_var_name, most_neg_coeff = None, float(0)

        # Choose entering: most negative in z for min
        for v_name_str in non_basic_vars:
            v_sym = Symbol(v_name_str)
            coeff = z_expr.coeff(v_sym)
            coeff_val = float(coeff.evalf())
            is_urs_component = any(
                v_name_str == info.get('pos_name', '') or v_name_str == info.get('neg_name', '')
                for info in original_var_info_map if info.get('is_urs', False)
            )
            adjusted_coeff = coeff_val * 1.1 if is_urs_component else coeff_val
            if adjusted_coeff < most_neg_coeff:
                most_neg_coeff = coeff_val
                entering_var_name = v_name_str

        # Check optimality and multiple-optima
        tol = 1e-12
        if entering_var_name is None or most_neg_coeff >= -tol:
            # Check for alternative optima: any non-basic var with reduced cost ~0
            alt_optima = False
            for v_name_str in non_basic_vars:
                v_sym = Symbol(v_name_str)
                coeff = z_expr.coeff(v_sym)
                if abs(float(coeff.evalf())) < tol:
                    alt_optima = True
                    break
            if alt_optima:
                status = 'Multiple'
                logger.info("Multiple optimal solutions detected")
            else:
                status = 'Optimal'
                logger.info("Optimal solution found")
            break

        leaving_var_name, min_ratio = None, float('inf')
        entering_var_symbol = Symbol(entering_var_name)
        no_positive = True

        for w_name_str in basic_var_names:
            expr_w = cur.get(w_name_str)
            if expr_w is None:
                continue
            coeff_entering = expr_w.coeff(entering_var_symbol)
            coeff_entering_val = float(coeff_entering.evalf())
            if coeff_entering_val < 0:
                no_positive = False
                # Evaluate constant part by setting all symbols to zero
                const_subs = {sym: 0 for sym in expr_w.free_symbols}
                const_val = float(expr_w.subs(const_subs).evalf())
                ratio = const_val / (-coeff_entering_val)
                if ratio < min_ratio or (abs(ratio - min_ratio) < tol and (leaving_var_name is None or w_name_str < leaving_var_name)):
                    min_ratio = ratio
                    leaving_var_name = w_name_str

        if no_positive:
            status = 'Unbounded'
            title = f"Step {step+1} (Biến vào {entering_var_name}, không có biến ra — Không giới nội)"
            steps[title] = cur.copy()
            _print_tableau(title, steps[title], basic_var_names, non_basic_vars)
            logger.warning(f"Unbounded solution detected at step {step+1}")
            break

        if leaving_var_name is None:
            status = 'Infeasible'
            title = f"Step {step+1} (Biến vào {entering_var_name}, không có biến ra hợp lệ — Vô nghiệm)"
            steps[title] = cur.copy()
            _print_tableau(title, steps[title], basic_var_names, non_basic_vars)
            logger.warning(f"Infeasible solution detected at step {step+1}")
            break

        step += 1
        pivot_eq_lhs = Symbol(leaving_var_name)
        pivot_eq_rhs = cur[leaving_var_name]
        sol_list = solve(pivot_eq_lhs - pivot_eq_rhs, entering_var_symbol)
        if not sol_list:
            status = 'Error'
            logger.error(f"Pivot solving failed for {entering_var_name}")
            break
        pivot_expr = sol_list[0]

        new_cur = {entering_var_name: simplify(pivot_expr)}
        for var_name, old_expr in cur.items():
            if var_name == leaving_var_name:
                continue
            new_cur[var_name] = simplify(old_expr.subs(entering_var_symbol, pivot_expr))

        # Validate URS non-negativity
        for info in original_var_info_map:
            if info.get('is_urs', False):
                pos_expr = new_cur.get(info['pos_name'], S.Zero)
                neg_expr = new_cur.get(info['neg_name'], S.Zero)
                pos_val = float(pos_expr.subs({sym:0 for sym in pos_expr.free_symbols}).evalf())
                neg_val = float(neg_expr.subs({sym:0 for sym in neg_expr.free_symbols}).evalf())
                if pos_val < -tol or neg_val < -tol:
                    logger.warning(f"URS non-negativity violation for {info['original_name']}")

        basic_var_names.remove(leaving_var_name)
        basic_var_names.append(entering_var_name)
        non_basic_vars.remove(entering_var_name)
        non_basic_vars.append(leaving_var_name)

        cur = new_cur
        title = f"Step {step} (Biến vào: {entering_var_name}, Biến ra: {leaving_var_name})"
        steps[title] = cur.copy()
        _print_tableau(title, cur, basic_var_names, non_basic_vars)

    # 6. Extract results
    z_star = None
    sol_final = {}

    if status in ['Optimal', 'Multiple']:
        subs_nb_zero = {Symbol(nb): 0 for nb in non_basic_vars}
        tabla_vals = {}
        for v in basic_var_names:
            tabla_vals[v] = float(cur[v].subs(subs_nb_zero))
        for v in non_basic_vars:
            tabla_vals[v] = 0.0

        for info in original_var_info_map:
            name = info['original_name']
            if info.get('is_urs', False):
                p = tabla_vals.get(info['pos_name'], 0.0)
                n = tabla_vals.get(info['neg_name'], 0.0)
                val = p - n
                sol_final[name] = 0.0 if abs(val) < tol else val
                logger.info(f"URS variable {name} = {p} - {n} = {sol_final[name]}")
            elif info.get('is_transformed', False):
                vname = info['tableau_name']
                vval = tabla_vals.get(vname, 0.0)
                sol_final[name] = -vval if abs(vval) > tol else 0.0
            else:
                vname = info['tableau_name']
                vval = tabla_vals.get(vname, 0.0)
                sol_final[name] = 0.0 if abs(vval) < tol else vval

        z_current = sum(float(c[i]) * sol_final.get(original_var_info_map[i]['original_name'], 0.0) for i in range(n_original))
        z_star = 0.0 if abs(z_current) < tol else z_current
    elif status == 'Unbounded':
        z_star = float('-inf')
    elif status == 'Infeasible':
        z_star = float('inf')

    return status, z_star, sol_final, steps

def auto_simplex(
    A: List[List[float]],
    b: List[float],
    c: List[float], 
    constraint_types: List[str],
    objective_type: str = 'max',
    variable_types: List[str] | None = None,
) -> Dict[str, Any]:
    """Main function to solve linear programming problems using Simplex with robust URS and multiple-optima support."""
    num_constraints, num_vars = len(A), len(c)
    if not A or not all(len(row) == num_vars for row in A) or len(b) != num_constraints or len(constraint_types) != num_constraints:
        raise ValueError("Đầu vào không hợp lệ")
    if variable_types is None:
        variable_types = ['>=0'] * num_vars
    elif len(variable_types) != num_vars:
        raise ValueError("Độ dài variable_types phải bằng số biến")

    # Standardize constraints
    A_std = [row[:] for row in A]
    b_std = b[:]
    constraint_types_std = constraint_types[:]
    for i in range(num_constraints):
        if constraint_types[i].strip() == '>=':
            for j in range(num_vars):
                A_std[i][j] = -A[i][j]
            b_std[i] = -b[i]
            constraint_types_std[i] = '<='
        elif constraint_types[i].strip() == '=':
            raise ValueError(f"Ràng buộc = chưa được hỗ trợ: {constraint_types[i]}")
    logger.info("Constraints standardized in auto_simplex")

    c_orig = c[:]
    flip_z = objective_type.strip().lower().startswith('max')
    c_eff = [-ci for ci in c_orig] if flip_z else c_orig[:]
    logger.info(f"Objective type: {objective_type}, c_eff: {c_eff}")

    try:
        status, z_star_min, sol_vals, steps = _simplex_min(
            A_std, b_std, c_eff, constraint_types_std, variable_types, objective_type
        )
    except Exception as e:
        logger.error(f"Simplex failed: {str(e)}")
        return {
            'status': 'Lỗi (Error)',
            'z': None,
            'solution': {},
            'steps': {},
            'error_message': str(e)
        }

    z_final = z_star_min
    if status == 'Optimal' and flip_z and z_star_min is not None:
        z_final = -z_star_min
    elif status == 'Unbounded':
        z_final = float('inf') if flip_z else float('-inf')
    elif status == 'Infeasible':
        z_final = None
    elif status == 'Multiple':
        z_final = z_star_min if not flip_z else -z_star_min

    sol_return = sol_vals if status in ['Optimal', 'Multiple'] else {}
    status_map = {
        'Optimal': 'Tối ưu (Optimal)',
        'Multiple': 'Vô số nghiệm (Multiple Optima)',
        'Unbounded': 'Không giới nội (Unbounded)',
        'Infeasible': 'Vô nghiệm (Infeasible)',
        'Error': 'Lỗi (Error)'
    }
    formatted_steps = {}
    for title, tab in steps.items():
        fmt_tab = {}
        for var, expr in tab.items():
            fmt_tab[var] = format_expression_for_printing(expr)
        formatted_steps[title] = fmt_tab

    return {
        'status': status_map.get(status, status),
        'z': z_final,
        'solution': sol_return,
        'steps': formatted_steps,
    }

if __name__ == '__main__':
    # Test cases
    print("Test case 1: MIN Z = 5x1 - 10x2 with x1 >= 0, x2 URS")
    A1 = [[-2.0, 1.0], [1.0, -1.0], [3.0, 1.0], [-2.0, 3.0]]
    b1 = [1.0, -2.0, 8.0, -9.0]
    c1 = [5.0, -10.0]
    constraint_types1 = ['<=', '>=', '<=', '>=']
    variable_types1 = ['>=0', 'URS']
    objective_type1 = 'min'

    result1 = auto_simplex(A1, b1, c1, constraint_types1, objective_type1, variable_types1)
    print(f"Status: {result1['status']}, z = {result1['z']}, Solution = {result1['solution']}\n")

    print("Test case 2: MAX Z = 3x1 + 2x2 subject to x1 + x2 <= 4, x1 <= 2, x2 <= 3, x1,x2>=0")
    A2 = [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    b2 = [4.0, 2.0, 3.0]
    c2 = [3.0, 2.0]
    constraint_types2 = ['<=', '<=', '<=']
    variable_types2 = ['>=0', '>=0']
    objective_type2 = 'max'

    result2 = auto_simplex(A2, b2, c2, constraint_types2, objective_type2, variable_types2)
    print(f"Status: {result2['status']}, z = {result2['z']}, Solution = {result2['solution']}\n")
